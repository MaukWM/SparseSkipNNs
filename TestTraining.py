import numpy as np
import torch
import tqdm as tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from SparseNeuralNetwork import SparseNeuralNetwork


class SineWave(Dataset):

    def __init__(self):
        self.dps = 500
        self.x = np.random.uniform(-1, 1, self.dps)
        self.y = np.sin(self.x)

    def __len__(self):
        return self.dps

    def __getitem__(self, item):
        idx = np.random.randint(0, self.dps)
        return torch.tensor([self.x[idx]]), torch.tensor([self.y[idx]])


class Training:

    def __init__(self, epochs):
        self.epochs = epochs

        _data_set = SineWave()
        # TODO: Fix batch sizes over 1
        self.batch_size = 1
        self.data_generator = DataLoader(_data_set, batch_size=self.batch_size)

        self.model = SparseNeuralNetwork(amount_hidden_layers=10, network_width=50)
        self.model.double()

    def train(self):
        self.model.train()
        criterion = nn.MSELoss()
        optim = torch.optim.SGD(self.model.parameters(), lr=1e-5)

        for epoch in range(self.epochs):

            train_loss = 0
            i = 0
            with tqdm.tqdm(self.data_generator, unit="batch") as tepoch:
                for inp_xs, true_ys in self.data_generator:
                    optim.zero_grad()

                    inp_xs = torch.reshape(inp_xs, (self.batch_size, 1))

                    pred_y = self.model(inp_xs)
                    loss = criterion(pred_y, true_ys)
                    loss.backward()
                    optim.step()

                    # print statistics
                    i += 1
                    batch_loss = loss.item()
                    train_loss += batch_loss
                    tepoch.update(1)
                    tepoch.set_postfix(loss=f"{(train_loss / i):5f}")


if __name__ == "__main__":

    training = Training(epochs=10)

    training.train()
