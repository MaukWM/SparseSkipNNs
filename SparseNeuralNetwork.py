import math
import time

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import random

from Config import SparseModelConfig
from LayerType import LayerType
from LogLevel import LogLevel
from item_keys import ItemKey

MAX_REGROW_ITER_RATIO = 4


class SparseNeuralNetwork(nn.Module):

    """
    max_connection_depth=1 means NO skip connections
    skip_sequential_ratio=0.8 means 80% of the connections are aimed to be sequential
    prune_rate is the % of active connections we want to prune per evolutionary step
    """
    def __init__(self, input_size, output_size, model_config: SparseModelConfig, l):
        super(SparseNeuralNetwork, self).__init__()
        self.model_config = model_config

        # Set variables
        self.sparsity = model_config.sparsity
        self.max_connection_depth = model_config.max_connection_depth
        self.n_hidden_layers = model_config.n_hidden_layers
        self.network_width = model_config.network_width
        self.input_size = input_size
        self.output_size = output_size
        self.skip_sequential_ratio = model_config.skip_sequential_ratio

        # Set logging
        self.l = l

        # Set evolution parameters
        self.pruning_type = model_config.pruning_type
        self.prune_rate = model_config.prune_rate
        self.cutoff = model_config.cutoff
        # A regrowth type on a ratio implies we keep the sparsity fixed
        self.regrowth_type = model_config.regrowth_type
        if self.regrowth_type == "fixed_sparsity":
            self.regrowth_ratio = model_config.skip_sequential_ratio
        else:
            self.regrowth_ratio = model_config.regrowth_ratio
        self.regrowth_percentage = model_config.regrowth_percentage

        # Evolution checks
        if self.pruning_type is not None and self.pruning_type != "bottom_k" and self.pruning_type != "cutoff" and self.pruning_type != "no_pruning":
            raise ValueError(f"Invalid pruning type specified: {self.pruning_type}")
        if self.pruning_type == "bottom_k" and self.prune_rate is None:
            raise ValueError("If pruning type \"bottom_k\" is used, a prune rate must be specified")
        if self.pruning_type == "cutoff" and self.cutoff is None:
            raise ValueError("If pruning type \"cutoff\" is used, a cutoff must be specified")
        if self.regrowth_type == "percentage" and self.regrowth_percentage is None:
            raise ValueError("If regrowth type \"percentage\" is used, a regrowth_percentage must be specified")

        # Model checks
        if self.max_connection_depth < 1:
            raise ValueError(f"max_connection_depth must be >=1")
        if self.max_connection_depth > self.n_hidden_layers + 1:
            raise ValueError(f"It's not possible to have a higher max_connection_depth than there are hidden_layers: {self.max_connection_depth}>{self.n_hidden_layers + 1}")
        if self.max_connection_depth == 1 and self.skip_sequential_ratio != 1:
            print(f"If the max_connection_depth is 1 (meaning we have a sequential only network), it is not possible to specify a skip-sequential ratio: {self.skip_sequential_ratio} != 1. Setting to 1.")
            self.skip_sequential_ratio = 1
        if self.sparsity is None:
            self.l(message="Sparsity not set, treating the network as fully dense.", level=LogLevel.SIMPLE)
        elif self.sparsity >= 1 or self.sparsity < 0:
            raise ValueError(f"Invalid sparsity {self.sparsity}, must be 0 <= sparsity < 1")

        # Initialize network
        self.layers = nn.ModuleDict()
        self.initialize_network()

        # Define activation functions used, None is a linear summation of inputs
        # TODO: Implement Mish/Swish?
        self.hidden_activation_function = F.mish
        self.final_activation_function = None

        # Calculate max amount of sequential and skip connections, to be used for calculating global sparsity target and
        # in order to initialize the target ratio
        self.n_max_sequential_connections = self.calculate_n_max_sequential_connections()
        self.n_max_skip_connections = self.calculate_n_max_skip_connections()

        # Calculate target n active connections for sequential and skip networks
        if self.sparsity:
            self.n_target_active_connections = self.n_max_sequential_connections - round(self.sparsity * self.n_max_sequential_connections)
            self.n_target_sequential_connections = round(self.n_target_active_connections * self.skip_sequential_ratio)
            self.n_target_skip_connections = round(self.n_target_active_connections * (1 - self.skip_sequential_ratio))
        else:
            self.n_target_active_connections = self.n_max_sequential_connections + self.n_max_skip_connections
            self.n_target_sequential_connections = self.n_max_sequential_connections
            self.n_target_skip_connections = self.n_max_skip_connections

        # Calculate target sparsities
        self.sequential_target_sparsity = 1 - self.n_target_sequential_connections / self.n_max_sequential_connections
        self.skip_target_sparsity = 1
        if self.max_connection_depth > 1:
            self.skip_target_sparsity = 1 - self.n_target_skip_connections / self.n_max_skip_connections

        # Log initial state of network
        self.l(message=f'\n\n[Max Connections] Max seq con={self.n_max_sequential_connections}, Max skip con={self.n_max_skip_connections}', level=LogLevel.SIMPLE)
        self.l(message=f'[Target Connections] Target act con={self.n_target_active_connections}, Target seq con={self.n_target_sequential_connections}, Target skip con={self.n_target_skip_connections}', level=LogLevel.SIMPLE)
        self.l(message=f'[Target Sparsity] Target seq con sparsity={self.sequential_target_sparsity:.3f}, Target skip con sparsity={self.skip_target_sparsity:.3f}, Target skip con sparsity (by max seq)={1 - (self.n_target_skip_connections / self.n_max_sequential_connections):.3f}', level=LogLevel.SIMPLE)

        # Initialize mask for sparsity
        self.masks = {}
        self.initialize_mask()
        self.apply_mask()

        # FLOP calculation
        self.dense_inferencing_flops = self.calculate_dense_inferencing_flops()
        # print(f"{self.dense_inferencing_flops * model_config.sparsity}")
        self.sparse_inferencing_flops = self.calculate_sparse_inferencing_flops()

        # Initialize n active connection trackers
        self.n_active_seq_connections = None
        self.n_active_skip_connections = None
        self.n_active_connections = None
        self.update_active_connection_info()

        # Initialize lists of skip and sequential layers names, useful for regrowing connections
        self.sequential_layer_names = []
        self.skip_layer_names = []
        for name in self.masks.keys():
            if LayerType.layer_name_to_layer_type(name) == LayerType.SEQUENTIAL:
                self.sequential_layer_names.append(name)
            else:
                self.skip_layer_names.append(name)

    def calculate_dense_inferencing_flops(self):
        """
        Calculate how much flops a forward pass of a dense counterpart of the network requires.
        """
        result = 0
        for name, param in self.named_parameters():
            k = name.split(".")[1]
            if k == "1":
                if "bias" in name:
                    result += param.shape[0]
                else:
                    result += param.shape[0] * param.shape[1]
        return result

    def calculate_sparse_inferencing_flops(self):
        """
        Calculate how much flops a forward pass of a dense counterpart of the network requires.
        """
        result = 0
        for name, param in self.named_parameters():
            if "bias" in name:
                result += param.shape[0]
            else:
                result += param.shape[0] * param.shape[1] * (torch.count_nonzero(param) / param.numel())
        return result.item()

    def move_weights_outside_cutoff(self):
        """
        This method moves all weights outside of the cutoff range. This is used so that the first evolution step does
        not remove a large amount of weights initialized in the pruning range.
        """
        with torch.no_grad():
            for name in self.masks.keys():
                current_k = name.split(".")[1]
                current_layer = int(name.split(".")[2])
                for neuron_idx in range(len(self.layers[current_k][current_layer].weight)):
                    for weight_idx in range(len(self.layers[current_k][current_layer].weight[neuron_idx])):
                        if self.masks[name][neuron_idx][weight_idx]:
                            _weight_value = self.layers[current_k][current_layer].weight[neuron_idx][weight_idx]
                            if self.cutoff > _weight_value > 0:
                                self.layers[current_k][current_layer].weight[neuron_idx][weight_idx] += self.cutoff
                            if -self.cutoff < _weight_value < 0:
                                self.layers[current_k][current_layer].weight[neuron_idx][weight_idx] -= self.cutoff

    def calculate_n_max_sequential_connections(self):
        """
        Calculate the maximum amount of sequential connections. Used for sparsity calculations.
        """
        result = self.input_size * self.network_width

        for j in range(self.n_hidden_layers - 1):
            result += self.network_width * self.network_width

        result += self.network_width * self.output_size

        return result

    def calculate_n_max_skip_connections(self):
        """
        Calculate the maximum amount of skip connections. Used for sparsity calculations.
        """
        result = 0
        for i in range(2, self.max_connection_depth + 1):

            if i > self.n_hidden_layers:
                result += self.input_size * self.output_size
                break

            result += self.input_size * self.network_width

            for j in range(self.n_hidden_layers - i):
                result += self.network_width * self.network_width

            result += self.network_width * self.output_size

        return result

    def update_active_connection_info(self):
        """
        Update information tracking N active connections, used for sparsity calculations and evolutionary steps.
        """
        self.n_active_seq_connections = 0
        self.n_active_skip_connections = 0
        for name, param in self.named_parameters():
            if "bias" in name:
                continue

            # If this is a layer named layers.1.x.weight, it is a sequential layer
            if LayerType.layer_name_to_layer_type(name) == LayerType.SEQUENTIAL:
                self.n_active_seq_connections += np.count_nonzero(self.masks[name])
            # If this is not a layer named layers.1.x.weight, it is a skip layer
            else:
                self.n_active_skip_connections += np.count_nonzero(self.masks[name])

        self.n_active_connections = self.n_active_seq_connections + self.n_active_skip_connections

    def get_k_distribution_info(self):
        """
        Get the distribution of n sequential and skip connections by k depth (By N connections and sparsity)
        """
        result_by_n = dict()
        result_by_sparsity = dict()
        result_by_sparsity_by_max_seq = dict()
        max_connections_by_k = dict()

        for name, param in self.named_parameters():
            if "bias" in name:
                continue

            current_k = name.split(".")[1]
            if current_k not in result_by_n.keys():
                result_by_n[current_k] = 0
                result_by_sparsity[current_k] = 0
                result_by_sparsity_by_max_seq[current_k] = 0
                max_connections_by_k[current_k] = 0

            result_by_n[current_k] += torch.count_nonzero(self.masks[name]).item()
            max_connections_by_k[current_k] += self.masks[name].numel()

        for k in result_by_n.keys():
            result_by_sparsity[k] = 1 - result_by_n[k] / max_connections_by_k[k]
            result_by_sparsity_by_max_seq[k] = 1 - result_by_n[k] / self.n_max_sequential_connections
        return result_by_n, result_by_sparsity, result_by_sparsity_by_max_seq

    def get_layer_distribution_info(self):
        # TODO: Extend this to keep track of all layer to layer connections into a dict. This allows us to analyze the ratio of skip and sequential connections on different incoming and outgoing ratios
        result_outgoing = dict()
        result_incoming = dict()
        max_outgoing_connections_per_layer = dict()
        max_incoming_connections_per_layer = dict()

        for name, param in self.named_parameters():
            if "bias" in name:
                continue

            # Extract layer information
            current_k = name.split(".")[1]
            current_layer = name.split(".")[2]
            target_layer = str(int(current_layer) + int(current_k))

            if current_layer not in result_outgoing.keys():
                result_outgoing[current_layer] = 0

            if target_layer not in result_incoming.keys():
                result_incoming[target_layer] = 0

            # Make sure we also keep track of the max amount of outgoing connections, we we can calculate the ratio
            if current_k == "1":
                max_outgoing_connections_per_layer[current_layer] = self.masks[name].numel()
                max_incoming_connections_per_layer[target_layer] = self.masks[name].numel()

            # Count amount of active connections for this k
            # print(f"{name} - LAYER{current_layer} - {torch.count_nonzero(self.masks[name]).item()}")
            result_outgoing[current_layer] += torch.count_nonzero(self.masks[name]).item()
            result_incoming[target_layer] += torch.count_nonzero(self.masks[name]).item()

        # print("preres", result)
        # print("maxout", max_outgoing_connections_per_layer)

        for key in result_outgoing.keys():
            result_outgoing[key] = result_outgoing[key] / max_outgoing_connections_per_layer[key]

        # print("preres", result_incoming)
        # print("maxinc", max_incoming_connections_per_layer)

        for key in result_incoming.keys():
            result_incoming[key] = result_incoming[key] / max_incoming_connections_per_layer[key]
        # print("postres", result_incoming)

        # print("postres", result)

        return result_outgoing, result_incoming

    def get_and_update_sparsity_information(self):
        """
        Update connectivity information in the network and track this data.
        :return: Current sparsity information
        """
        # Update n sequential and skip connections
        self.update_active_connection_info()

        # Get information on k depth connections
        k_n_distribution, k_sparsity_distribution, k_sparsity_distribution_by_max_seq = self.get_k_distribution_info()

        # Get information on incoming/outgoing connections per layer
        layer_outgoing_remaining_ratio, layer_incoming_remaining_ratio = self.get_layer_distribution_info()

        # TODO: Retrieve information on node connectivity, generate distribution of how connective the most connective nodes are and see how this might increase in networks with skip connections

        # Calculate the actualized sparsity levels in the model
        actualized_overall_sparsity = 1 - (self.n_active_seq_connections + self.n_active_skip_connections) / self.n_max_sequential_connections
        actualized_sequential_sparsity = 1 - self.n_active_seq_connections / self.n_max_sequential_connections
        actualized_skip_sparsity = 1
        actualized_skip_sparsity_by_max_seq = 1
        if self.n_max_skip_connections > 0:
            actualized_skip_sparsity = 1 - self.n_active_skip_connections / self.n_max_skip_connections
            actualized_skip_sparsity_by_max_seq = 1 - self.n_active_skip_connections / self.n_max_sequential_connections
        actualized_sparsity_ratio = self.n_active_seq_connections / (self.n_active_skip_connections + self.n_active_seq_connections)

        result = dict()
        result[ItemKey.N_ACTIVE_CONNECTIONS.value] = self.n_active_connections
        result[ItemKey.N_ACTIVE_SEQ_CONNECTIONS.value] = self.n_active_seq_connections
        result[ItemKey.N_ACTIVE_SKIP_CONNECTIONS.value] = self.n_active_skip_connections
        result[ItemKey.ACTUALIZED_OVERALL_SPARSITY.value] = actualized_overall_sparsity
        result[ItemKey.ACTUALIZED_SEQUENTIAL_SPARSITY.value] = actualized_sequential_sparsity
        result[ItemKey.ACTUALIZED_SKIP_SPARSITY.value] = actualized_skip_sparsity
        result[ItemKey.ACTUALIZED_SKIP_SPARSITY_BY_MAX_SEQ.value] = actualized_skip_sparsity_by_max_seq
        result[ItemKey.ACTUALIZED_SPARSITY_RATIO.value] = actualized_sparsity_ratio
        result[ItemKey.K_N_DISTRIBUTION.value] = k_n_distribution
        result[ItemKey.K_SPARSITY_DISTRIBUTION.value] = k_sparsity_distribution
        result[ItemKey.K_SPARSITY_DISTRIBUTION_BY_MAX_SEQ.value] = k_sparsity_distribution_by_max_seq
        result[ItemKey.LAYER_OUTGOING_REMAINING_RATIO.value] = layer_outgoing_remaining_ratio
        result[ItemKey.LAYER_INCOMING_REMAINING_RATIO.value] = layer_incoming_remaining_ratio

        # Update amount of flops required for new calculations
        self.sparse_inferencing_flops = self.calculate_sparse_inferencing_flops()

        # Log information
        self.l(
            message=f"[NetworkInfo]        N max active connections={self.n_max_skip_connections + self.n_max_sequential_connections}, N max seq connections={self.n_max_sequential_connections}, N max skip connections={self.n_max_skip_connections}",
            level=LogLevel.SIMPLE)
        self.l(
            message=f"[TargetN]            N active connections={self.n_target_active_connections}, N active sequential connections={self.n_target_sequential_connections}, N active skip connections={self.n_target_skip_connections}",
            level=LogLevel.SIMPLE)
        self.l(
            message=f"[ActualizedN]        N active connections={self.n_active_connections}, N active sequential connections={self.n_active_seq_connections}, N active skip connections={self.n_active_skip_connections}",
            level=LogLevel.SIMPLE)
        if self.sparsity:
            self.l(
                message=f"[TargetSparsity]     OverallSparsity={self.sparsity:.3f}, SequentialSparsity={self.sequential_target_sparsity:.3f}, SkipSparsity={self.skip_target_sparsity:.3f}, SparsityRatio={self.skip_sequential_ratio:.3f}, SkipSparsityByMaxSeq={1 - (self.sequential_target_sparsity - self.sparsity):.3f}",
                level=LogLevel.SIMPLE)
        else:
            self.l(
                message=f"[TargetSparsity]     OverallSparsity=None, SequentialSparsity=None, SkipSparsity=None, SparsityRatio=None, SkipSparsityByMaxSeq=None",
                level=LogLevel.SIMPLE)
        self.l(
            message=f"[ActualizedSparsity] OverallSparsity={actualized_overall_sparsity:.3f}, SequentialSparsity={actualized_sequential_sparsity:.3f}, SkipSparsity={actualized_skip_sparsity:.3f}, SparsityRatio={actualized_sparsity_ratio:.3f}, SkipSparsityByMaxSeq={actualized_skip_sparsity_by_max_seq:.3f}",
            level=LogLevel.SIMPLE)

        return result

    def prune_network(self):
        """
        Prune the network according to different methods.
        """
        if self.pruning_type == "no_pruning":
            return 0

        _start_pruning = time.time()
        self.l(
            message=f"[EvolveNetwork - Pre-pruning] Pre-pruning Overall Sparsity: {1 - self.n_active_connections / self.n_max_sequential_connections:.3f}, Pre-pruning Skip Sparsity: {1 - self.n_active_skip_connections / self.n_max_sequential_connections:.3f}, Pre-pruning Sequential Sparsity: {1 - self.n_active_seq_connections / self.n_max_sequential_connections:.3f}",
            level=LogLevel.SIMPLE)
        self.eval()

        weight_coordinates = []

        _prune_flops = 0

        # Retrieve a list of all the weights and their exact coordinates
        for name in self.masks.keys():
            current_k = name.split(".")[1]
            current_layer = int(name.split(".")[2])
            # print(name, self.masks[name], self.layers[current_k][current_layer].weight)
            for neuron_idx in range(len(self.layers[current_k][current_layer].weight)):
                for weight_idx in range(len(self.layers[current_k][current_layer].weight[neuron_idx])):
                    if self.masks[name][neuron_idx][weight_idx]:
                        weight_coordinates.append((name, current_k, current_layer, neuron_idx, weight_idx,
                            self.layers[current_k][current_layer].weight[neuron_idx][weight_idx].data))

        if len(weight_coordinates) == 0:
            raise ValueError("The entire mask is False! This means sparsity=1, which should never happen.")

        n_sequential_pruned = 0
        n_skip_pruned = 0

        if self.pruning_type == "bottom_k":
            _start_sorting = time.time()
            weight_coordinates.sort(key=lambda x: abs(x[5]))
            # O(n log n) complexity for sorting
            _prune_flops += len(weight_coordinates) * int(math.log2(len(weight_coordinates)))
            _end_sorting = time.time()
            self.l(
                message=f"[EvolveNetwork - PruneNetwork - Bottom K] len(weight_coordinates)={len(weight_coordinates)}, time_to_sort={_end_sorting - _start_sorting:.3f}s",
                level=LogLevel.SIMPLE)
            self.l(
                message=f"[EvolveNetwork - PruneNetwork - Bottom K] sorted_weight_coordinates={weight_coordinates}",
                level=LogLevel.VERBOSE)

            n_to_prune = round(self.n_active_connections * self.prune_rate)
            # Each prune costs a flop
            _prune_flops += n_to_prune

            # Prune the first n weight from this list
            for i in range(n_to_prune):
                name, current_k, current_layer, neuron_idx, weight_idx, weight_value = weight_coordinates[i]
                self.masks[name][neuron_idx][weight_idx] = False

                # Current k is the depth of the connection
                current_k = name.split(".")[1]

                if current_k == "1":
                    n_sequential_pruned += 1
                else:
                    n_skip_pruned += 1

        if self.pruning_type == "cutoff":
            # Each read costs a flop
            _prune_flops += len(weight_coordinates)
            for weight_coordinate in weight_coordinates:
                name, current_k, current_layer, neuron_idx, weight_idx, weight_value = weight_coordinate
                # Prune by cutoff
                if abs(weight_value) < self.cutoff:
                    self.masks[name][neuron_idx][weight_idx] = False
                    # Current k is the depth of the connection
                    current_k = name.split(".")[1]

                    if current_k == "1":
                        n_sequential_pruned += 1
                    else:
                        n_skip_pruned += 1

        self.n_sequential_pruned = n_sequential_pruned
        self.n_skip_pruned = n_skip_pruned
        self.n_active_connections_pruned = (n_sequential_pruned + n_skip_pruned)
        wandb.log({"n_skip_pruned": n_skip_pruned,
                   "n_seq_pruned": n_sequential_pruned})
        # Each prune costs a flop
        _prune_flops += self.n_active_connections_pruned

        # Update n_active connections
        self.n_active_seq_connections -= n_sequential_pruned
        self.n_active_skip_connections -= n_skip_pruned
        self.n_active_connections -= (n_sequential_pruned + n_skip_pruned)

        # Reapply mask
        _prune_flops += self.apply_mask()

        _end_pruning = time.time()

        self.l(
            message=f"[EvolveNetwork - Post-pruning] Sequential connections pruned: {n_sequential_pruned}, Skip connections pruned: {n_skip_pruned}, [Time taken:{_end_pruning - _start_pruning:.2f}s]",
            level=LogLevel.SIMPLE)
        self.l(
            message=f"[EvolveNetwork - Post-pruning] Post-pruning Overall Sparsity: {1 - self.n_active_connections / self.n_max_sequential_connections:.3f}, Post-pruning Skip Sparsity: {1 - self.n_active_skip_connections / self.n_max_sequential_connections:.3f}, Post-pruning Sequential Sparsity: {1 - self.n_active_seq_connections / self.n_max_sequential_connections:.3f}",
            level=LogLevel.SIMPLE)

        return _prune_flops, self.n_active_connections_pruned

    def evolve_network(self):
        """
        Evolve the network
        """
        _evolution_flops = 0
        self.l(message="\n=============== [EvolveNetwork - Start] ===============", level=LogLevel.SIMPLE)
        _prune_flops, n_pruned = self.prune_network()
        _regrowth_flops = self.regrow_network(n_pruned)
        _evolution_flops += _prune_flops + _regrowth_flops
        self.l(message=f"[EvolveNetwork - Floppa] Pruning flops: {_prune_flops} Regrowth flops: {_regrowth_flops}", level=LogLevel.VERBOSE)
        self.l(message="=============== [EvolveNetwork - End] =================", level=LogLevel.SIMPLE)

        return _evolution_flops

    def regrow_network(self, n_pruned):
        """
        Regrow connections in the network depending on the method.
        """
        _regrow_flops = 0
        # fixed_sparsity == fixed_ratio
        if self.regrowth_type == "fixed_sparsity":
            # n_new_connections = np.clip(self.n_target_active_connections - self.n_active_connections, 0, None)
            _regrow_flops += self.regrow_exactly(self.sequential_layer_names, self.skip_layer_names)
            return _regrow_flops
        elif self.regrowth_type == "percentage":
            # TODO: Check if clipping works as expected, we never want sparsity > 1
            n_new_connections = np.clip(round(self.n_active_connections * self.regrowth_percentage), 0, self.n_max_sequential_connections - self.n_active_connections)
        elif self.regrowth_type == "no_regrowth":
            return _regrow_flops
        elif self.regrowth_type == "by_ratio":
            return self.regrow_by_ratio(n_pruned, self.sequential_layer_names, self.skip_layer_names)
        else:
            raise ValueError(f"No valid regrowth type was given: {self.regrowth_type}")

        if self.max_connection_depth > 1:
            _regrow_flops += self.regrow_by_ratio(n_new_connections, self.sequential_layer_names, self.skip_layer_names)
        else:
            _regrow_flops += self.regrow_on_layer_name_list(n_new_connections, self.sequential_layer_names)

        return _regrow_flops

    def regrow_exactly(self, sequential_layer_names, skip_layer_names, max_iter_ratio=MAX_REGROW_ITER_RATIO,
                       max_iter_connection_growth=20):
        """
        Regrow connections by a given ratio.
        :param n_to_regrow: N connections to regrow
        :param sequential_layer_names: The names of the sequential layers available for regrowth
        :param skip_layer_names: The names of the skip layers available for regrowth
        :param max_iter_ratio: max amount of iterations by ratio before stopping regrowth
        :param max_iter_connection_growth: max amount of iterations when attempting to regrow in a specific layer, high numbers here will lead to very long evolution times on dense networks

        self.regrow_ratio: Regrowth ratio. 0.8 means 80% of regrowth will take place in sequential layers.
        """
        _seq_to_regrow = self.n_sequential_pruned
        _skip_to_regrow = self.n_skip_pruned
        _n_to_regrow = _seq_to_regrow + _skip_to_regrow

        _regrow_flops = 0

        max_iter = int((_seq_to_regrow + _skip_to_regrow) * max_iter_ratio)
        n_weights_activated = 0
        n_k_activated = dict()
        total_iter = 0
        _regrow_start = time.time()

        for i in range(max_iter):
            total_iter += 1
            # Stop when we've regrown everything we need to regrow or when we reach the max amount of iterations
            if n_weights_activated >= _n_to_regrow or i == max_iter - 1:
                _regrow_end = time.time()
                self.l(
                    message=f"[EvolveNetwork - RegrowNetwork] Activated {n_weights_activated}/{_n_to_regrow} after {total_iter}/{max_iter * max_iter_connection_growth} iterations. [Time taken:{_regrow_end - _regrow_start:.3f}s]",
                    level=LogLevel.SIMPLE)
                # Every iteration is a flop
                _regrow_flops += total_iter
                break

            if "1" in n_k_activated.keys():
                _sequential_activated = n_k_activated["1"]
            else:
                _sequential_activated = 0
            _skip_activated = sum([n_k_activated[_k] for _k in n_k_activated.keys() if int(_k) >= 2])

            # Select layer type to regrow depending on type
            if _seq_to_regrow > 0:
                _layer_name_list = sequential_layer_names
            else:
                _layer_name_list = skip_layer_names

            _mask_name = np.random.choice(_layer_name_list)
            _current_k = _mask_name.split(".")[1]
            _mask = self.masks[_mask_name]

            for j in range(max_iter_connection_growth):
                total_iter += 1
                ix, iy = random.randrange(0, _mask.shape[0]), random.randrange(0, _mask.shape[1])

                if _mask[ix][iy]:
                    continue
                else:
                    self.l(message=f"[EvolveNetwork - RegrowNetwork] Regrowing W{ix},{iy} in {_mask_name}",
                           level=LogLevel.VERBOSE)
                    self.masks[_mask_name][ix][iy] = True
                    n_weights_activated += 1
                    # TODO: Initialize outside of loop so we don't have to check every entry
                    if _current_k not in n_k_activated.keys():
                        n_k_activated[_current_k] = 0
                    n_k_activated[_current_k] += 1
                    if int(_current_k) == 1:
                        _seq_to_regrow -= 1
                    else:
                        _skip_to_regrow -= 1
                    break

        if "1" in n_k_activated:
            sequential_activated = n_k_activated["1"]
        else:
            sequential_activated = 0

        skip_activated = sum([n_k_activated[_k] for _k in n_k_activated.keys() if int(_k) >= 2])

        self.l(message=f"[EvolveNetwork - RegrowNetwork] Sequential regrown: {sequential_activated}, Skip regrown: {skip_activated}", level=LogLevel.SIMPLE)
        self.l(message=f"[EvolveNetwork - RegrowNetwork] N K's regrown: {n_k_activated}", level=LogLevel.VERBOSE)

        return _regrow_flops

    def regrow_by_ratio(self, n_to_regrow, sequential_layer_names, skip_layer_names, max_iter_ratio=MAX_REGROW_ITER_RATIO,
                        max_iter_connection_growth=20):
        """
        Regrow connections by a given ratio.
        :param n_to_regrow: N connections to regrow
        :param sequential_layer_names: The names of the sequential layers available for regrowth
        :param skip_layer_names: The names of the skip layers available for regrowth
        :param max_iter_ratio: max amount of iterations by ratio before stopping regrowth
        :param max_iter_connection_growth: max amount of iterations when attempting to regrow in a specific layer, high numbers here will lead to very long evolution times on dense networks

        self.regrow_ratio: Regrowth ratio. 0.8 means 80% of regrowth will take place in sequential layers.
        """
        max_iter = int(n_to_regrow * max_iter_ratio)
        n_weights_activated = 0
        n_k_activated = dict()
        total_iter = 0
        _regrow_start = time.time()

        _regrow_flops = 0

        # These variables are only used in fixed_sparsity networks
        _seq_to_regrow = self.n_sequential_pruned
        _skip_to_regrow = self.n_skip_pruned

        # print(n_to_regrow, _seq_to_regrow, _skip_to_regrow)

        for i in range(max_iter):
            total_iter += 1
            # Stop when we've regrown everything we need to regrow or when we reach the max amount of iterations
            if n_weights_activated >= n_to_regrow or i == max_iter - 1:
                _regrow_end = time.time()
                self.l(
                    message=f"[EvolveNetwork - RegrowNetwork] Activated {n_weights_activated}/{n_to_regrow} after {total_iter}/{max_iter * max_iter_connection_growth} iterations. [Time taken:{_regrow_end - _regrow_start:.3f}s]",
                    level=LogLevel.SIMPLE)
                # Every iteration is a flop
                _regrow_flops += total_iter
                break

            if "1" in n_k_activated.keys():
                _sequential_activated = n_k_activated["1"]
            else:
                _sequential_activated = 0
            _skip_activated = sum([n_k_activated[_k] for _k in n_k_activated.keys() if int(_k) >= 2])

            # Decide whether to regrow a sequential or skip connection depending on ratio and n weight types activated
            if n_weights_activated > 0:
                # Select layer type to regrow depending on type
                if self.regrowth_type == "fixed_sparsity":
                    # if (_sequential_activated + self.n_active_seq_connections - self.n_sequential_pruned) / (_skip_activated + _sequential_activated + self.n_active_connections - self.n_active_connections_pruned) < self.regrowth_ratio:
                    #     _layer_name_list = sequential_layer_names
                    # else:
                    #     _layer_name_list = skip_layer_names
                    print("regrowth type is fixed_sparsity but we're not regrowing exactly: this should never happen!")
                    if _seq_to_regrow > 0:
                        _layer_name_list = sequential_layer_names
                        _seq_to_regrow -= 1
                    else:
                        _layer_name_list = skip_layer_names
                else:
                    if _sequential_activated / n_weights_activated > self.regrowth_ratio:
                        _layer_name_list = skip_layer_names
                    elif _sequential_activated / n_weights_activated < self.regrowth_ratio:
                        _layer_name_list = sequential_layer_names
                    else:
                        if random.random() < self.regrowth_ratio:
                            _layer_name_list = sequential_layer_names
                        else:
                            _layer_name_list = skip_layer_names
            else:
                if random.random() < self.regrowth_ratio:
                    _layer_name_list = sequential_layer_names
                else:
                    _layer_name_list = skip_layer_names

            _mask_name = np.random.choice(_layer_name_list)
            _current_k = _mask_name.split(".")[1]
            _mask = self.masks[_mask_name]

            for j in range(max_iter_connection_growth):
                total_iter += 1
                ix, iy = random.randrange(0, _mask.shape[0]), random.randrange(0, _mask.shape[1])

                if _mask[ix][iy]:
                    continue
                else:
                    self.l(message=f"[EvolveNetwork - RegrowNetwork] Regrowing W{ix},{iy} in {_mask_name}",
                           level=LogLevel.VERBOSE)
                    self.masks[_mask_name][ix][iy] = True
                    n_weights_activated += 1
                    # TODO: Initialize outside of loop so we don't have to check every entry
                    if _current_k not in n_k_activated.keys():
                        n_k_activated[_current_k] = 0
                    n_k_activated[_current_k] += 1
                    break

        if "1" in n_k_activated:
            sequential_activated = n_k_activated["1"]
        else:
            sequential_activated = 0

        skip_activated = sum([n_k_activated[_k] for _k in n_k_activated.keys() if int(_k) >= 2])

        self.l(message=f"[EvolveNetwork - RegrowNetwork] Sequential regrown: {sequential_activated}, Skip regrown: {skip_activated}", level=LogLevel.SIMPLE)
        self.l(message=f"[EvolveNetwork - RegrowNetwork] N K's regrown: {n_k_activated}", level=LogLevel.VERBOSE)

        return _regrow_flops

    def regrow_on_layer_name_list(self, n_to_regrow, layer_name_list, max_iter_ratio=MAX_REGROW_ITER_RATIO):
        """
        Given a list of layer names, regrow connections in these layers.
        :param n_to_regrow: Amount of connections to regrow
        :param layer_name_list: List of the layers in which we regrow
        :param max_iter_ratio: Max iteration attempts per layer before moving on (higher values can mean larger computation times in denser networks)
        """
        max_iter = int(n_to_regrow * max_iter_ratio)
        n_weights_activated = 0
        n_k_activated = dict()

        _regrow_flops = 0

        for i in range(max_iter):
            # Stop when we've regrown everything we need to regrow or when we reach the max amount of iterations
            if n_weights_activated >= n_to_regrow or i == max_iter - 1:
                self.l(message=f"[EvolveNetwork - RegrowNetwork] Activated {n_weights_activated}/{n_to_regrow} after {i}/{max_iter} iterations.",
                       level=LogLevel.SIMPLE)
                # Every iteration is a flop
                _regrow_flops += i
                break

            _mask_name = np.random.choice(layer_name_list)
            _current_k = _mask_name.split(".")[1]
            _mask = self.masks[_mask_name]
            ix, iy = random.randrange(0, _mask.shape[0]), random.randrange(0, _mask.shape[1])

            if _mask[ix][iy]:
                continue
            else:
                self.l(message=f"[EvolveNetwork - RegrowNetwork] Regrowing W{ix},{iy} in {_mask_name}", level=LogLevel.VERBOSE)
                self.masks[_mask_name][ix][iy] = True
                n_weights_activated += 1
                # TODO: Initialize outside of loop so we don't have to check every entry
                if _current_k not in n_k_activated.keys():
                    n_k_activated[_current_k] = 0
                n_k_activated[_current_k] += 1

        self.l(message=f"[EvolveNetwork - RegrowNetwork] N K's regrown: {n_k_activated}", level=LogLevel.SIMPLE)

        return _regrow_flops

    def apply_mask(self):
        """
        Apply the mask to the network (must be performed after weight update iteration to guarantee sparsity)
        :return:
        """
        # Make sure we don't track the gradient from mask application!!!
        _mask_application_flops = 0
        _is_training = self.training
        if _is_training:
            self.eval()
        for name, param in self.named_parameters():
            if "bias" in name:
                continue
            param.data[~self.masks[name]] = 0
            _mask_application_flops += self.masks[name].numel()
            # print(f"applying{name} {self.masks[name]} to {old_param_data} -> {param.data}")
        if _is_training:
            self.train()
        return _mask_application_flops

    def initialize_mask(self):
        """
        Initialize a sparse topology
        """
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                continue

            # If this is a layer named layers.1.x.weight, it is a sequential layer
            if LayerType.layer_name_to_layer_type(name) == LayerType.SEQUENTIAL:
                self.masks[name] = np.random.rand(*param.shape) > self.sequential_target_sparsity
            # If this is not a layer named layers.1.x.weight, it is a skip layer
            else:
                self.masks[name] = np.random.rand(*param.shape) > self.skip_target_sparsity
            self.masks[name] = torch.tensor(self.masks[name])

    def initialize_network(self):
        """
        Initialize the layers
        """
        # Each iteration of this loop initializes connections for connection depth i.
        for i in range(1, self.max_connection_depth + 1):

            # If i is larger (or equal) than the amount of hidden layers it's impossible to continue, so add a skip
            # connection from start to end and break out of the loop, we're done!
            if i > self.n_hidden_layers:
                self.layers[str(i)] = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.output_size)])
                break

            _layers = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.network_width)])

            for j in range(self.n_hidden_layers - i):
                _layers.append(nn.Linear(in_features=self.network_width, out_features=self.network_width))

            _layers.append(nn.Linear(in_features=self.network_width, out_features=self.output_size))
            self.layers[str(i)] = _layers

    def forward(self, _x):
        """
        Forward pass
        :param _x: Input
        :return Output
        """
        # Depending on skip_depth we keep track of all previously calculated xs
        # This is x_0 that is received by the first layer
        _xs = {0: _x}

        for i in range(self.n_hidden_layers + 1):
            # print(f'performing calculatios on layer{i}')
            if i == self.n_hidden_layers:
                activation_function = self.final_activation_function
            else:
                activation_function = self.hidden_activation_function

            _new_x = 0
            for j in range(0, i + 1):
                # print(f"updating _new_x from {_new_x}", end=" ")

                # Can't have more skip connection networks than max_connection_depth
                if i + 1 - j > self.max_connection_depth:
                    continue

                # print(f'[{i + 1 - j}][{j}]')
                if activation_function is None:
                    _new_x = _new_x + self.layers[str(i + 1 - j)][j](_xs[j])
                else:
                    _new_x = _new_x + activation_function(self.layers[str(i + 1 - j)][j](_xs[j]))
                # print(f"to {_new_x}")
            _xs[i + 1] = _new_x
            # print(f'adding {_new_x} to {i + 1} key in _xs')

        # print(_xs)
        return _xs[len(_xs) - 1]

    def __repr__(self):
        return f"SparseNN={{sparsity={self.sparsity}, skip_depth={self.max_connection_depth}, network_depth={self.n_hidden_layers}, " \
               f"network_width={self.network_width}}}"
