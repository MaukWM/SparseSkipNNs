import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import random

from LogLevel import LogLevel


class SparseNeuralNetwork(nn.Module):

    """
    max_connection_depth=1 means NO skip connections
    skip_sequential_ratio=0.8 means 80% of the connections are aimed to be sequential
    """
    def __init__(self, sparsity=0.5, max_connection_depth=4, amount_hidden_layers=3, network_width=3, input_size=1,
                 output_size=1, skip_sequential_ratio=0.5, log_level=LogLevel.VERBOSE):
        # TODO: Add regularization L1/L2 to drive weights down

        super(SparseNeuralNetwork, self).__init__()
        if max_connection_depth < 1:
            raise ValueError(f"max_connection_depth must be >=1")
        if max_connection_depth > amount_hidden_layers + 1:
            raise ValueError(f"It's not possible to have a higher max_connection_depth than there are hidden_layers: {max_connection_depth}>{amount_hidden_layers + 1}")
        if max_connection_depth == 1 and skip_sequential_ratio == 0:
            raise ValueError(f"If the max_connection_depth is 1 (meaning we have a sequential only network), it is not possible to specify a skip-sequential ratio: {skip_sequential_ratio} !=0")
        if sparsity >= 1 or sparsity < 0:
            raise ValueError(f"Invalid sparsity {sparsity}, must be 0 <= sparsity < 1")

        # Set variables
        self.sparsity = sparsity
        self.max_connection_depth = max_connection_depth
        self.amount_hidden_layers = amount_hidden_layers
        self.network_width = network_width
        self.input_size = input_size
        self.output_size = output_size
        self.skip_sequential_ratio = skip_sequential_ratio

        # Set logging
        self.log_level = log_level
        self.l = lambda level, message: print(message) if level >= self.log_level else None

        # Set evolution variables
        self.keep_skip_sequential_ratio_same = True

        # Initialize network
        self.layers = nn.ModuleDict()
        self.initialize_network()

        # Calculate amount of skip layer instances. TODO: Figure out the formula, it's not difficult but the current solution is inelegant
        # print([name for name, _ in self.named_parameters() if name.split(".")[1] != "1" and "bias" not in name])
        self.n_skip_instances = len([name for name, _ in self.named_parameters() if name.split(".")[1] != "1" and "bias" not in name])

        # print("start")
        # for name, param in self.named_parameters():
        #     print(name, param)
        # print("end")

        # Define activation functions used
        self.hidden_activation_function = F.relu
        self.final_activation_function = None

        # Calculate max amount of sequential and skip connections, to be used for calculating global sparsity target and
        # in order to initialize the target ratio
        self.n_max_sequential_connections = self.calculate_n_max_sequential_connections()
        self.n_max_skip_connections = self.calculate_n_max_skip_connections()

        # Calculate target n active connections for sequential and skip networks
        self.n_target_active_connections = self.n_max_sequential_connections - round(self.sparsity * self.n_max_sequential_connections)
        self.n_target_sequential_connections = round(self.n_target_active_connections * self.skip_sequential_ratio)
        self.n_target_skip_connections = round(self.n_target_active_connections * (1 - self.skip_sequential_ratio))

        self.sequential_sparsity = 1 - self.n_target_sequential_connections / self.n_max_sequential_connections
        self.skip_sparsity = 1
        if self.max_connection_depth > 1:
            self.skip_sparsity = 1 - self.n_target_skip_connections / self.n_max_skip_connections

        self.l(message=f'\n\n[Max Connections] Max seq con={self.n_max_sequential_connections}, Max skip con={self.n_max_skip_connections}', level=LogLevel.SIMPLE)
        self.l(message=f'[Target Connections] Target act con={self.n_target_active_connections}, Target seq con={self.n_target_sequential_connections}, Target skip con={self.n_target_skip_connections}', level=LogLevel.SIMPLE)
        self.l(message=f'[Target Sparsity] Target seq con sparsity={self.sequential_sparsity}, Target skip con sparsity={self.skip_sparsity}', level=LogLevel.SIMPLE)

        # Initialize mask for sparsity
        self.masks = {}
        self.initialize_mask()
        self.apply_mask()

        # Initialize n active connection trackers
        self.n_seq_connections = None
        self.n_skip_connections = None
        self.n_active_connections = None
        self.update_active_connection_info()

        # Initialize lists of skip and sequential layers names, useful for regrowing connections
        self.sequential_layer_names = []
        self.skip_layer_names = []
        for name in self.masks.keys():
            if name.split(".")[1] == "1":
                self.sequential_layer_names.append(name)
            else:
                self.skip_layer_names.append(name)

    def calculate_n_max_sequential_connections(self):
        result = self.input_size * self.network_width

        for j in range(self.amount_hidden_layers - 1):
            result += self.network_width * self.network_width

        result += self.network_width * self.output_size

        return result

    def calculate_n_max_skip_connections(self):
        result = 0
        for i in range(2, self.max_connection_depth + 1):

            if i > self.amount_hidden_layers:
                result += self.input_size * self.output_size
                break

            result += self.input_size * self.network_width

            for j in range(self.amount_hidden_layers - i):
                result += self.network_width * self.network_width

            result += self.network_width * self.output_size

        return result

    def update_active_connection_info(self):
        self.n_seq_connections = 0
        self.n_skip_connections = 0
        for name, param in self.named_parameters():
            if "bias" in name:
                continue

            # Check whether the parameter is from sequential or skip layers
            name_split = name.split(".")

            # If this is a layer named layers.1.x.weight, it is a sequential layer
            if name_split[1] == "1":
                self.n_seq_connections += np.count_nonzero(self.masks[name])
            # If this is not a layer named layers.1.x.weight, it is a skip layer
            else:
                self.n_skip_connections += np.count_nonzero(self.masks[name])

        self.n_active_connections = self.n_seq_connections + self.n_skip_connections

    def get_and_update_sparsity_information(self):
        # Update n sequential and skip connections
        self.update_active_connection_info()

        # Calculate the actualized sparsity levels in the model
        actualized_overall_sparsity = 1 - (self.n_seq_connections + self.n_skip_connections) / self.n_max_sequential_connections
        actualized_sequential_sparsity = 1 - self.n_seq_connections / self.n_max_sequential_connections
        actualized_skip_sparsity = 1
        if self.n_max_skip_connections > 0:
            actualized_skip_sparsity = 1 - self.n_skip_connections / self.n_max_skip_connections
        actualized_sparsity_ratio = self.n_seq_connections / (self.n_skip_connections + self.n_seq_connections)

        self.l(message=f"[Raw N Data] N max seq connections={self.n_max_sequential_connections}, N active connections={self.n_active_connections}, N active seq connections={self.n_seq_connections}, N active skip connections={self.n_skip_connections}", level=LogLevel.SIMPLE)
        self.l(message=f"[TargetSparsity] OverallSparsity={self.sparsity}, SequentialSparsity={self.sequential_sparsity}, SkipSparsity={self.skip_sparsity}, SparsityRatio={self.skip_sequential_ratio}", level=LogLevel.SIMPLE)
        self.l(message=f"[ActualizedSparsity] OverallSparsity={actualized_overall_sparsity}, SequentialSparsity={actualized_sequential_sparsity}, SkipSparsity={actualized_skip_sparsity}, SparsityRatio={actualized_sparsity_ratio}", level=LogLevel.SIMPLE)

        # Collect result TODO: Change this to just a dict we map in, cause now we have to convert between list of dicts and dict of lists, unncessary
        result = dict()
        result["n_active_connections"] = self.n_active_connections
        result["n_seq_connections"] = self.n_seq_connections
        result["n_skip_connections"] = self.n_skip_connections
        result["actualized_overall_sparsity"] = actualized_overall_sparsity
        result["actualized_sequential_sparsity"] = actualized_sequential_sparsity
        result["actualized_skip_sparsity"] = actualized_skip_sparsity
        result["actualized_sparsity_ratio"] = actualized_sparsity_ratio
        return result

    def evolve_network(self):
        self.l(message="\n\n=============== [EvolveNetwork] ===============", level=LogLevel.SIMPLE)
        self.eval()
        # --- Prune n smallest weights ---
        n = 1
        weight_coordinates = []

        # First get a sorted list of all the weights and their exact coordinates
        for name in self.masks.keys():
            current_k = name.split(".")[1]
            current_layer = int(name.split(".")[2])
            # print(name, self.masks[name], self.layers[current_k][current_layer].weight)
            for neuron_idx in range(len(self.layers[current_k][current_layer].weight)):
                for weight_idx in range(len(self.layers[current_k][current_layer].weight[neuron_idx])):
                    if self.masks[name][neuron_idx][weight_idx]:
                        weight_coordinates.append((name, current_k, current_layer, neuron_idx, weight_idx, torch.abs(self.layers[current_k][current_layer].weight[neuron_idx][weight_idx].data)))

        weight_coordinates.sort(key=lambda x: x[5])
        # print(len(weight_coordinates), weight_coordinates)
        #TODO: Add logging here for amount of weight coordinates, or add logging for currenty sparsity and targets

        if len(weight_coordinates) == 0:
            raise ValueError("The entire mask is False! This means sparsity=1, which should never happen.")

        n_sequential_pruned = 0
        n_skip_pruned = 0

        # Prune the first n weight from this list
        for i in range(n):
            name, current_k, current_layer, neuron_idx, weight_idx, weight_value = weight_coordinates[i]
            self.masks[name][neuron_idx][weight_idx] = False

            # Current k is the depth of the connection
            current_k = name.split(".")[1]

            if current_k == "1":
                n_sequential_pruned += 1
            else:
                n_skip_pruned += 1

        # Calculate how many new connections we need to meet the target sparsity
        n_new_sequential_connections = np.clip(self.n_target_sequential_connections - self.n_seq_connections, 0, None)
        n_new_skip_connections = np.clip(self.n_target_skip_connections - self.n_skip_connections, 0, None)
        n_new_connections = n_new_sequential_connections + n_new_skip_connections

        self.l(message=f"[EvolveNetwork - PruneNetwork] Sequential connections pruned: {n_sequential_pruned}, Skip connections pruned: {n_skip_pruned}", level=LogLevel.SIMPLE)

        # --- Regrow n weights ---
        if self.keep_skip_sequential_ratio_same:
            self.regrow_connection_by_ratio(n_new_sequential_connections, n_new_skip_connections)
        else:
            # TODO: Implement
            self.regrow_connections_anywhere(n_new_connections)

        # Reapply mask
        self.apply_mask()

        self.l(message="=============== [EvolveNetwork] ===============", level=LogLevel.SIMPLE)

    def regrow_connection_by_ratio(self, n_sequential_to_regrow, n_skip_to_regrow):
        # Uniformly select new layers to grow in
        seq_layers = random.choices(self.sequential_layer_names, k=n_sequential_to_regrow)
        skip_layers = random.choices(self.skip_layer_names, k=n_skip_to_regrow)

        self.regrow_on_layer_list(n_sequential_to_regrow, seq_layers)
        self.regrow_on_layer_list(n_skip_to_regrow, skip_layers)

    def regrow_on_layer_list(self, n_to_regrow, layer_list, max_iter_ratio=4):
        max_iter = n_to_regrow * max_iter_ratio
        current_layer_to_regrow = 0

        for i in range(max_iter):
            # Stop when we've regrown everything we need to regrow or when we reach the max amount of iterations
            if current_layer_to_regrow >= n_to_regrow or i == max_iter - 1:
                self.l(message=f"[EvolveNetwork - RegrowNetwork] Activated {current_layer_to_regrow}/{n_to_regrow} after {i} iterations.",
                       level=LogLevel.SIMPLE)
                break

            _mask_name = layer_list[current_layer_to_regrow]
            _mask = self.masks[_mask_name]
            ix, iy = random.randrange(0, _mask.shape[0]), random.randrange(0, _mask.shape[1])

            if _mask[ix][iy]:
                continue
            else:
                self.l(message=f"[EvolveNetwork - RegrowNetwork] Regrowing W{ix},{iy} in {_mask_name}", level=LogLevel.VERBOSE)
                self.masks[_mask_name][ix][iy] = True
                current_layer_to_regrow += 1

    def apply_mask(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                continue
            param.data[~self.masks[name]] = 0
            # print(f"applying{name} {self.masks[name]} to {old_param_data} -> {param.data}")

    def initialize_mask(self):
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                continue

            # Check whether the parameter is from sequential or skip layers
            name_split = name.split(".")

            # If this is a layer named layers.1.x.weight, it is a sequential layer
            if name_split[1] == "1":
                self.masks[name] = np.random.rand(*param.shape) > self.sequential_sparsity
            # If this is not a layer named layers.1.x.weight, it is a skip layer
            else:
                self.masks[name] = np.random.rand(*param.shape) > self.skip_sparsity
            self.masks[name] = torch.tensor(self.masks[name])

    def initialize_network(self):
        # Each iteration of this loop initializes connections for connection depth i.
        for i in range(1, self.max_connection_depth + 1):
            # print(f'creating {i} skip layer')
            # TODO: This will not work on skip depths equal to network depth

            # If i is larger (or equal) than the amount of hidden layers it's impossible to continue, so add a skip
            # connection from start to end and break out of the loop, we're done!
            if i > self.amount_hidden_layers:
                # print(f'size 1')
                self.layers[str(i)] = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.output_size)])
                break

            _layers = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.network_width)])

            for j in range(self.amount_hidden_layers - i):
                _layers.append(nn.Linear(in_features=self.network_width, out_features=self.network_width))

            _layers.append(nn.Linear(in_features=self.network_width, out_features=self.output_size))
            # print(f'size {len(_layers)}')
            self.layers[str(i)] = _layers
        # print(self.layers)

    def forward(self, _x):
        # TODO: Verify correctness of forward pass
        # Depending on skip_depth we keep track of all previously calculated xs
        # This is x_0 that is received by the first layer
        _xs = {0: _x}

        for i in range(self.amount_hidden_layers + 1):
            # print(f'performing calculatios on layer{i}')
            if i == self.amount_hidden_layers:
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
                # _new_x = _new_x + F.relu(self.layers[j](_xs[i - j]))
            _xs[i + 1] = _new_x
            # print(f'adding {_new_x} to {i + 1} key in _xs')

        # print(_xs)
        return _xs[len(_xs) - 1]

    def __repr__(self):
        return f"SparseNN={{sparsity={self.sparsity}, skip_depth={self.max_connection_depth}, network_depth={self.amount_hidden_layers}, " \
               f"network_width={self.network_width}}}"


# if __name__ == "__main__":
#
#     input_size = 1
#     snn = SparseNeuralNetwork(input_size=input_size, amount_hidden_layers=30, max_connection_depth=14, network_width=30,
#                               sparsity=0, skip_sequential_ratio=0.5)
#
#     x = torch.rand((10, input_size))
#
#     snn.get_sparsities()

    # print(snn.n_max_sequential_connections)

    # print(x, x.dtype)
    # print(snn(x))
    # print(snn.layers)
    # for name, param in snn.named_parameters():
    #     # if param.requires_grad:
    #     print(name, param.data)
    # print(snn)
