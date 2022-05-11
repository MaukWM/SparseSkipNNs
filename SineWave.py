import math

import PIL
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


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