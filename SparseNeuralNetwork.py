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
        # TODO: Dissallow ratio if max_conn_depth = 1
        # TODO: dissallow sparsity=1
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
        self.skip_sparsity = 1 - self.n_target_skip_connections / self.n_max_skip_connections

        print(f'Max seq con={self.n_max_sequential_connections}, Max skip con={self.n_max_skip_connections}')
        print(f'Target act con={self.n_target_active_connections}, Target seq con={self.n_target_sequential_connections}, Target skip con={self.n_target_skip_connections}')
        print(f'Target seq con sparsity={self.sequential_sparsity}, Target skip con sparsity={self.skip_sparsity}')

        # Initialize mask for sparsity
        self.masks = {}
        self.initialize_mask()
        self.apply_mask()

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

    def get_sparsities(self):
        # Calculate n sequential connections
        # Calculate n skip connections
        n_seq_connections = 0
        n_skip_connections = 0

        for name, param in self.named_parameters():
            if "bias" in name:
                continue

            # Check whether the parameter is from sequential or skip layers
            name_split = name.split(".")

            # If this is a layer named layers.1.x.weight, it is a sequential layer
            if name_split[1] == "1":
                n_seq_connections += np.count_nonzero(self.masks[name])
            # If this is not a layer named layers.1.x.weight, it is a skip layer
            else:
                n_skip_connections += np.count_nonzero(self.masks[name])

        overall_sparsity = 1 - (n_seq_connections + n_skip_connections) / self.n_max_sequential_connections
        sequential_sparsity = 1 - n_seq_connections / self.n_max_sequential_connections
        skip_sparsity = 1 - n_skip_connections / self.n_max_sequential_connections
        sparsity_ratio = n_seq_connections / (n_skip_connections + n_seq_connections)

        print(f"[Sparsity] N active connections={n_seq_connections + n_skip_connections}, N active seq connections={n_seq_connections}, N active skip connections={n_skip_connections}")
        print(f"[Sparsity] OverallSparsity={overall_sparsity}, SequentialSparsity={sequential_sparsity}, SkipSparsity={skip_sparsity}, SparsityRatio={sparsity_ratio}")

        return overall_sparsity, sequential_sparsity, skip_sparsity, sparsity_ratio

    def apply_mask(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                continue
            # old_param_data = param.data
            param.data = param.data * self.masks[name]
            # print(f"applying{name} {self.masks[name]} to {old_param_data} -> {param.data}")

    def initialize_mask(self):
        # First calculate the sparsity for each individual layer, as we must incorporate both global sparsity and
        # We know the maximum amount of sequential and skip connections. We must simply uphold the ratio now
        # individual_sequential_layer_sparsity = (self.n_max_sequential_connections * self.sparsity * self.skip_sequential_ratio) / self.n_max_sequential_connections
        # sparsity ratio
        # individual_sequential_layer_sparsity = self.sparsity * (1 - self.skip_sequential_ratio)

        # We can't simply calculate one value for this right?
        # individual_skip_layer_sparsity = self.sparsity * self.skip_sequential_ratio / self.n_skip_instances

        # print(individual_sequential_layer_sparsity, individual_skip_layer_sparsity)

        # We need to calculate how many skip connection instances there will be of [layers.x.x.weights], this is not hard
        # Then we know how many initialization rounds there will be and we can calculate the probability each initialization round needs
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
    snn = SparseNeuralNetwork(input_size=input_size, amount_hidden_layers=30, max_connection_depth=14, network_width=30,
                              sparsity=0.8, skip_sequential_ratio=0)

    x = torch.rand((10, input_size))

    snn.get_sparsities()

    # print(snn.n_max_sequential_connections)

    # print(x, x.dtype)
    # print(snn(x))
    # print(snn.layers)
    # for name, param in snn.named_parameters():
    #     # if param.requires_grad:
    #     print(name, param.data)
    # print(snn)
