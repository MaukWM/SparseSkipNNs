import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import random


class SparseNeuralNetwork(nn.Module):

    """
    max_connection_depth=1 means NO skip connections
    skip_sequential_ratio=0.8 means 80% of the connections are aimed to be sequential
    """
    def __init__(self, sparsity=0.5, max_connection_depth=4, amount_hidden_layers=3, network_width=3, input_size=1,
                 output_size=1, skip_sequential_ratio=0.5):
        super(SparseNeuralNetwork, self).__init__()
        # TODO: Check max_connection_depth not > network depth and check max_connection_depth=>1
        # Set variables
        self.sparsity = sparsity
        self.max_connection_depth = max_connection_depth
        self.amount_hidden_layers = amount_hidden_layers
        self.network_width = network_width
        self.input_size = input_size
        self.output_size = output_size
        self.skip_sequential_ratio = skip_sequential_ratio

        # Initialize network
        self.layers = nn.ModuleDict()
        self.initialize_network()

        # Initialize mask for sparsity
        self.masks = dict()
        self.initialize_mask()
        self.apply_mask()

        # Define activation functions used
        self.hidden_activation_function = F.relu
        self.final_activation_function = None

        # Calculate max amount of sequential connections, to be used for calculating global sparsity target
        self.n_max_sequential_connections = self.calculate_n_max_sequential_connections()

    def calculate_n_max_sequential_connections(self):
        result = self.input_size * self.network_width

        for j in range(self.amount_hidden_layers - 1):
            result += self.network_width * self.network_width

        result += self.network_width * self.output_size

        return result

    def get_sparsities(self):
        sparsities_by_k = []

        n_active_connections = dict()
        n_total_connections = dict()

        for k_layer in self.layers.keys():
            # k_layer_n_activated_connection = 0
            # k_layer_n_total_connection = 0
            n_active_connections[k_layer] = dict()
            n_total_connections[k_layer] = dict()

            for name, layer in self.layers[k_layer].named_parameters():
                if 'bias' in name:
                    continue
                n_activated_connection = sum([np.count_nonzero(mask) for mask in self.masks[k_layer]])
                n_total_connection = sum([mask.size for mask in self.masks[k_layer]])
                # k_layer_n_activated_connection += n_activated_connection
                # k_layer_n_total_connection += n_total_connection
                n_active_connections[k_layer][name] = n_activated_connection
                n_total_connections[k_layer][name] = n_total_connection

        print(n_active_connections, n_total_connections)

        # true_sequential_sparsity = sparsities_by_k[0]
        # true_skip_sparsity = np.sum(sparsities_by_k[1:]) / (self.max_connection_depth - 1)
        # true_sequential_skip_ratio = true_sequential_sparsity / true_skip_sparsity

        # print(f'[TrueSequentialSparsity]={true_sequential_sparsity}\n'
        #       f'[TrueSkipSparsity]={true_skip_sparsity}\n'
        #       f'[TrueSequentialSkipRatio]={true_sequential_skip_ratio}')

        # return true_sequential_skip_ratio, true_skip_sparsity, true_sequential_skip_ratio

    def apply_mask(self):
        for k_layer in self.layers.keys():
            for layer in self.layers[k_layer].keys():
                # Does multiplying a list work with modulelist? We'll see later
                self.layers[k_layer][layer] = self.layers[k_layer][layer] * self.masks[k_layer][layer]

    @staticmethod
    def create_single_mask(shape, sparsity):
        return np.random.rand(*shape) > sparsity
        # return np.random.rand((self.amount_hidden_layers, self.network_width)) * self.sparsity < self.skip_sequential_ratio

    def initialize_mask(self):
        # TODO: Rewrite to similar dict method emiel uses, build a bit on top for retrieving sparsities ratios
        # Each iteration of this loop initializes connections for connection depth i.
        # Create mask for sequential layers
        self.masks["1"] = self.create_single_mask(shape=(self.amount_hidden_layers, self.network_width),
                                                  sparsity=self.skip_sequential_ratio * self.sparsity)

        # Create masks for skip layers
        skip_layer_sparsity = self.skip_sequential_ratio / self.max_connection_depth * self.sparsity

        for i in range(2, self.max_connection_depth + 1):
            # print(f'creating {i} skip mask')
            # TODO: This will not work on skip depths equal to network depth

            # If i is larger (or equal) than the amount of hidden layers it's impossible to continue, so add a skip
            # connection from start to end and break out of the loop, we're done!
            if i > self.amount_hidden_layers:
                # print(f'size 1')
                self.masks[str(i)] = self.create_single_mask((self.input_size, self.output_size), skip_layer_sparsity)
                break

            _layers = [self.create_single_mask((self.input_size, self.network_width), skip_layer_sparsity)]

            for j in range(self.amount_hidden_layers - i):
                _layers.append(self.create_single_mask((self.network_width, self.network_width), skip_layer_sparsity))

            _layers.append(self.create_single_mask((self.network_width, self.output_size), skip_layer_sparsity))
            self.masks[str(i)] = _layers

    def apply_mask(self):
        pass

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


if __name__ == "__main__":

    input_size = 1
    snn = SparseNeuralNetwork(input_size=input_size, amount_hidden_layers=3, max_connection_depth=4, network_width=3)

    x = torch.rand((10, input_size))

    snn.get_sparsities()

    # print(snn.n_max_sequential_connections)

    # print(x, x.dtype)
    # print(snn(x))
    # print(snn.layers)
    for name, param in snn.named_parameters():
        # if param.requires_grad:
        print(name, param.data)
    # print(snn)
