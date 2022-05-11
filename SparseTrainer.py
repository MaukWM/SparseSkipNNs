import math

import PIL
import numpy as np
import torch
import tqdm as tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from LogLevel import LogLevel
from SineWave import SineWave
from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey

from PIL import Image, ImageDraw


class Training:

    def __init__(self, epochs: int, model: SparseNeuralNetwork, plot_interval=None, batch_size=64, evolution_interval=10):
        self.plot_interval = plot_interval
        self.epochs = epochs
        self.evolution_interval = evolution_interval

        _data_set = SineWave()
        self.batch_size = batch_size

        # 0.8 means 80% train 20% test
        self.train_test_split_ratio = 0.8
        self.train_dataset, self.test_dataset = random_split(_data_set,
                                                             [round(_data_set.dps * self.train_test_split_ratio),
                                                              round(_data_set.dps * (1 - self.train_test_split_ratio))])

        self.train_generator = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_generator = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = model

        # Initialize dict that keeps track of data over training
        self.items = dict()
        self.items[ItemKey.TRAINING_LOSS.value] = []
        self.items[ItemKey.VALIDATION_LOSS.value] = []
        self.items[ItemKey.SPARSITIES] = []

        self.images = []
        self.train_progress_image_interval = 10

    def write_train_progress(self):
        self.images[0].save('out/gif.gif', save_all=True, append_images=self.images[1:], optimize=False, duration=2, loop=0)

    def train(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        for epoch in range(self.epochs):
            with tqdm.tqdm(total=len(self.train_generator) + len(self.test_generator)) as pbar:
                pbar.set_description(f"epoch {epoch}/{self.epochs}")
                train_loss = 0
                val_loss = 0

                # Train
                # with tqdm.tqdm(self.train_generator, unit="batch") as tepoch:
                i = 0
                self.model.train()
                for batch in self.train_generator:
                    optimizer.zero_grad()

                    inp_xs = torch.reshape(batch[0], (batch[0].size()[0], 1))
                    true_ys = torch.reshape(batch[1], (batch[1].size()[0], 1))

                    # print("Train", inp_xs.shape)

                    # print(inp_xs.shape)

                    # print(f'training on {inp_xs} and {true_ys}')
                    # print(inp_xs.dtype)
                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)
                    loss.backward()
                    optimizer.step()

                    self.model.apply_mask()

                    # print statistics
                    i += 1
                    batch_loss = loss.item()
                    train_loss += batch_loss

                    pbar.set_postfix(train_loss=f"{(train_loss / i):5f}", val_loss=f"0")
                    pbar.update(1)

                self.items[ItemKey.TRAINING_LOSS.value].append(train_loss)

            # Calculate validation
            # with tqdm.tqdm(self.test_generator, unit="batch") as tepoch:
                i = 0
                self.model.eval()
                for batch in self.test_generator:
                    inp_xs = torch.reshape(batch[0], (batch[0].size()[0], 1))
                    true_ys = torch.reshape(batch[1], (batch[1].size()[0], 1))

                    # print("Validate", inp_xs.shape)

                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)

                    # print statistics
                    i += 1
                    val_loss += loss.item()
                    pbar.set_postfix(train_loss=f"{self.items[ItemKey.TRAINING_LOSS.value][epoch]:5f}",
                                     val_loss=f"{(val_loss / i):5f}")
                    pbar.update(1)

                self.items[ItemKey.VALIDATION_LOSS.value].append(val_loss)

                if epoch % self.evolution_interval == 0:
                    self.model.evolve_network()
                    self.items[ItemKey.SPARSITIES].append(self.model.get_and_update_sparsity_information())

                if epoch % self.train_progress_image_interval == 0:
                    self.images.append(SineWave.get_model_plot_distribution(self.model))
                if self.plot_interval is not None and epoch % self.plot_interval == 0:
                    SineWave.plot_model_distribution(self.model, epoch=epoch)
            # pbar.update(1)
            # pbar.set_postfix(train_loss=f"{self.items[ItemKey.TRAINING_LOSS.value][epoch]:5f}",
            #                  val_loss=f"{self.items[ItemKey.VALIDATION_LOSS.value][epoch]:5f}")


if __name__ == "__main__":
    import visualization

    snn = SparseNeuralNetwork(input_size=1, amount_hidden_layers=3, max_connection_depth=4, network_width=3,
                              sparsity=0.5, skip_sequential_ratio=0.5, log_level=LogLevel.VERBOSE)

    training = Training(epochs=150, model=snn, plot_interval=50, batch_size=256, evolution_interval=10)

    training.train()
    training.model.eval()
    # for name, param in training.model.named_parameters():
    #     print(name, param)

    visualization.plot_train_val_loss(training)

    # Investigate with up to max k skip connections, to what distribution of k's the network prunes itself

    # print(training.images)

    SineWave.plot_model_distribution(training.model)

    training.write_train_progress()

    visualization.plot_sparsity_info(training)

    # SineWave.plot_model_distribution(training.model)

