import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import random

from Config import DenseModelConfig


class DenseNeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size, model_config: DenseModelConfig, l):
        super(DenseNeuralNetwork, self).__init__()
        self.model_config = model_config

        self.n_hidden_layers = model_config.n_hidden_layers
        self.network_width = model_config.network_width

        self.hidden_activation_function = F.relu
        self.final_activation_function = None

        # Set logging
        self.l = l

        # Initialize network
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, self.network_width))
        for i in range(self.n_hidden_layers - 1):
            self.layers.append(nn.Linear(self.network_width, self.network_width))
        self.layers.append(nn.Linear(self.network_width, output_size))

        print(self.layers)

    def forward(self, x):
        x = torch.flatten(x, 1)

        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)

        return x


