import math

import PIL
import numpy as np
import torch
import tqdm as tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey

from PIL import Image, ImageDraw


class SineWave(Dataset):

    def __init__(self):
        self.dps = 5000
        self.x = torch.rand(self.dps, dtype=torch.float32)
        self.y = torch.sin(self.x * math.pi * 2)

    def __len__(self):
        return self.dps

    def __getitem__(self, item):
        idx = np.random.randint(0, self.dps)
        return self.x[idx], self.y[idx]

    @staticmethod
    def get_model_plot_distribution(model, x_min=0, x_max=1, resolution=100):
        model.eval()

        xs = torch.linspace(x_min, x_max, resolution)

        ys = model(torch.reshape(xs, (xs.shape[0], 1)))

        ys = ys.detach().numpy()

        # Raw image retrieval from: https://stackoverflow.com/questions/58849953/how-do-get-the-raw-plot-image-data-from-matplotlib-instead-of-saving-to-file-o
        fig, ax = plt.subplots(1, figsize=(4, 4), dpi=300)

        ax.plot(xs, ys, label="pred")
        ax.plot(xs, torch.sin(xs * math.pi * 2), label="real")
        ax.legend()

        fig.canvas.draw()
        temp_canvas = fig.canvas
        plt.close()

        return PIL.Image.frombytes('RGB', temp_canvas.get_width_height(), temp_canvas.tostring_rgb())

    @staticmethod
    def plot_model_distribution(model, x_min=0, x_max=1, resolution=100, epoch=None):
        model.eval()

        xs = torch.linspace(x_min, x_max, resolution)

        ys = model(torch.reshape(xs, (xs.shape[0], 1)))

        ys = ys.detach().numpy()

        if epoch is not None:
            plt.title(f"Epoch {epoch}")

        plt.plot(xs, ys, label="pred")
        plt.plot(xs, torch.sin(xs * math.pi * 2), label="real")
        plt.legend()
        plt.show()


class Training:

    def __init__(self, epochs, model, plot_interval=None, batch_size=64):
        self.plot_interval = plot_interval
        self.epochs = epochs

        _data_set = SineWave()
        self.batch_size = batch_size

        # 0.8 means 80% train 20% test
        self.train_test_split_ratio = 0.8
        self.train_dataset, self.test_dataset = random_split(_data_set,
                                                             [round(_data_set.dps * self.train_test_split_ratio),
                                                              round(_data_set.dps * (1 - self.train_test_split_ratio))])

        self.train_generator = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_generator = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.model = model

        self.items = {}

        self.items[ItemKey.TRAINING_LOSS.value] = []
        self.items[ItemKey.VALIDATION_LOSS.value] = []

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

                if epoch % self.train_progress_image_interval == 0:
                    self.images.append(SineWave.get_model_plot_distribution(self.model))
                if self.plot_interval is not None and epoch % self.plot_interval == 0:
                    SineWave.plot_model_distribution(self.model, epoch=epoch)
                    self.model.get_sparsities()
            # pbar.update(1)
            # pbar.set_postfix(train_loss=f"{self.items[ItemKey.TRAINING_LOSS.value][epoch]:5f}",
            #                  val_loss=f"{self.items[ItemKey.VALIDATION_LOSS.value][epoch]:5f}")


if __name__ == "__main__":
    import visualization

    # TODO: This works for max_conn_depth>2 but for 1 it's broken, fix
    snn = SparseNeuralNetwork(input_size=1, amount_hidden_layers=8, max_connection_depth=8, network_width=15,
                              sparsity=0.5, skip_sequential_ratio=0.5)
    # TODO: figure out why batch sizes of 256 lead to no learning and batch sizes of 1 converge super fast
    training = Training(epochs=1500, model=snn, plot_interval=50, batch_size=32)

    training.train()
    training.model.eval()
    for name, param in training.model.named_parameters():
        print(name, param)

    visualization.plot_train_val_loss(training.items)

    # Investigate with up to max k skip connections, to what distribution of k's the network prunes itself

    # print(training.images)

    SineWave.plot_model_distribution(training.model)

    training.write_train_progress()

    # SineWave.plot_model_distribution(training.model)

