import math
import time

import PIL
import numpy as np
import torch
import torchvision.datasets
import tqdm as tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms

from DataLoaderInitializer import DataLoaderInitializer
from DatasetEnum import DatasetEnum
from LogLevel import LogLevel
from SineWave import SineWave
from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey
import Visualizer

from PIL import Image, ImageDraw


class SparseTrainer:

    def __init__(self, train_dataset, test_dataset, trainloader, testloader,
                 epochs: int, model: SparseNeuralNetwork, evolution_interval, batch_size=64,
                 prune_rate=0.05, keep_skip_sequential_ratio_same=False, lr=1e-3, early_stopping_threshold=None, train_test_split_ratio=0.8):
        self.epochs = epochs
        self.evolution_interval = evolution_interval
        self.lr = lr
        self.early_stopping_threshold = early_stopping_threshold
        self.train_test_split_ratio = train_test_split_ratio
        self.batch_size = batch_size

        # Initialize dataset and dataloaders
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.trainloader, self.testloader = trainloader, testloader

        # Set model and initialize model evolution parameters
        self.model = model
        self.model.prune_rate = prune_rate
        self.model.keep_skip_sequential_ratio_same = keep_skip_sequential_ratio_same

        # Initialize dict that keeps track of data over training TODO: Make dynamic
        self.items = dict()
        for item_key in ItemKey:
            self.items[item_key.value] = []

        # Distribution images, used with SineWave dataset. Handy for getting a historic overview of model performance
        self.images = []
        self.train_progress_image_interval = 100

    def write_train_progress(self):
        self.images[0].save('out/gif.gif', save_all=True, append_images=self.images[1:], optimize=False, duration=0.5)

    def train(self):
        _train_start = time.time()
        # For prediction
        # criterion = nn.MSELoss()

        # For classification
        criterion = nn.CrossEntropyLoss()

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        early_stopping_counter = 0
        lowest_val_loss = math.inf

        for epoch in range(self.epochs):
            with tqdm.tqdm(total=len(self.trainloader) + len(self.testloader)) as pbar:
                pbar.set_description(f"epoch {epoch}/{self.epochs}")
                train_loss = 0
                val_loss = 0
                train_accuracy = 0
                average_train_accuracy = 0
                val_accuracy = 0
                average_val_accuracy = 0

                # Train
                # with tqdm.tqdm(self.train_generator, unit="batch") as tepoch:
                i = 0
                self.model.train()
                for batch in self.trainloader:
                    optimizer.zero_grad()

                    inp_xs, true_ys = batch

                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)
                    loss.backward()
                    optimizer.step()

                    self.model.apply_mask()

                    # print statistics
                    i += 1
                    batch_loss = loss.item()
                    train_loss += batch_loss

                    train_accuracy += torch.count_nonzero(torch.eq(true_ys, torch.argmax(pred_ys, dim=1))).item() / pred_ys.size()[0] * 100
                    average_train_accuracy = train_accuracy / i

                    pbar.set_postfix(train_loss=f"{(train_loss / i):5f}",
                                     train_accuracy=f"{average_train_accuracy:2f}%",
                                     val_loss=f"0",
                                     val_accuracy=f"0%")
                    pbar.update(1)

                self.items[ItemKey.TRAINING_LOSS.value].append(train_loss / i)
                self.items[ItemKey.TRAINING_ACCURACY.value].append(average_train_accuracy)

            # Calculate validation
            # with tqdm.tqdm(self.test_generator, unit="batch") as tepoch:
                i = 0
                self.model.eval()
                for batch in self.testloader:
                    inp_xs, true_ys = batch

                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)

                    # print statistics
                    i += 1
                    val_loss += loss.item()

                    val_accuracy += torch.count_nonzero(torch.eq(true_ys, torch.argmax(pred_ys, dim=1))).item() / pred_ys.size()[0] * 100
                    average_val_accuracy = val_accuracy / i

                    pbar.set_postfix(train_loss=f"{self.items[ItemKey.TRAINING_LOSS.value][epoch]:5f}",
                                     train_accuracy=f"{average_train_accuracy:2f}%",
                                     val_loss=f"{(val_loss / i):5f}",
                                     val_accuracy=f"{average_val_accuracy:2f}%",)
                    pbar.update(1)

                self.items[ItemKey.VALIDATION_LOSS.value].append(val_loss / i)
                self.items[ItemKey.VALIDATION_ACCURACY.value].append(average_val_accuracy)

                # Early stopping policy
                if self.early_stopping_threshold is not None:
                    if val_loss < lowest_val_loss:
                        early_stopping_counter = 0
                        lowest_val_loss = val_loss
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter > self.early_stopping_threshold:
                            break

                if self.evolution_interval is not None:
                    if epoch % self.evolution_interval == 0:
                        if epoch > 0:
                            self.model.evolve_network()

                        sparsity_information = self.model.get_and_update_sparsity_information()

                        for item_key in sparsity_information.keys():
                            self.items[item_key].append(sparsity_information[item_key])

        _train_end = time.time()

        print(f"Total training time: {_train_end - _train_start}s")


if __name__ == "__main__":
    _train_test_split_ratio = 0.8
    _batch_size = 512 * 4
    _dataset_enum = DatasetEnum.CIFAR10
    data_loader_initializer = DataLoaderInitializer(_dataset_enum, _train_test_split_ratio, _batch_size)

    # Load datasets
    _train_dataset, _test_dataset, _trainloader, _testloader = data_loader_initializer.get_datasets_and_dataloaders()

    # Find input and output sizes from dataset
    _input_size = np.prod(_train_dataset.data.shape[1:])
    _output_size = len(_train_dataset.classes)

    # TODO: Investigate adding a convolutional layer, or perhaps doing some sparse densenet beforehand
    # TODO: Add analysis for sparsity for k

    # TODO: Add feature which makes it possible to specify each layers width
    # snn = SparseNeuralNetwork(input_size=_input_size, output_size=_output_size, amount_hidden_layers=12, max_connection_depth=13, network_width=5,
    #                           sparsity=0.3, skip_sequential_ratio=0.5, log_level=LogLevel.SIMPLE)
    snn = SparseNeuralNetwork(input_size=_input_size, output_size=_output_size, amount_hidden_layers=1, max_connection_depth=1, network_width=1,
                              sparsity=0.3, skip_sequential_ratio=1, log_level=LogLevel.SIMPLE)

    trainer = SparseTrainer(_train_dataset, _test_dataset, _trainloader, _testloader,
                            epochs=3, model=snn, batch_size=_batch_size, evolution_interval=1,
                            prune_rate=0.3, keep_skip_sequential_ratio_same=False, lr=2e-3, early_stopping_threshold=10)

    trainer.train()
    trainer.model.eval()
    # for name, param in training.model.named_parameters():
    #     print(name, param)

    visualizer = Visualizer.Visualizer(trainer)
    visualizer.visualize_all()

    # Investigate with up to max k skip connections, to what distribution of k's the network prunes itself

    # print(training.images)

    # SineWave.plot_model_distribution(trainer.model)

    # trainer.write_train_progress()

    # SineWave.plot_model_distribution(training.model)
