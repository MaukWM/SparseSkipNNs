import numpy as np
import torch
import tqdm as tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from SparseNeuralNetwork import SparseNeuralNetwork


class SineWave(Dataset):

    def __init__(self):
        self.dps = 500
        self.x = torch.rand(self.dps, dtype=torch.float32)
        self.y = torch.sin(self.x)

    def __len__(self):
        return self.dps

    def __getitem__(self, item):
        idx = np.random.randint(0, self.dps)
        return self.x[idx], self.y[idx]


class Training:

    def __init__(self, epochs):
        self.epochs = epochs

        _data_set = SineWave()
        # TODO: Fix batch sizes over 1
        self.batch_size = 256
        self.data_generator = DataLoader(_data_set, batch_size=self.batch_size)

        self.model = SparseNeuralNetwork(input_size=1, output_size=1, amount_hidden_layers=3, network_width=50)
        # self.model.double()

    def train(self):
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5)
        for epoch in range(self.epochs):
            train_loss = 0
            i = 0
            with tqdm.tqdm(self.data_generator, unit="batch") as tepoch:
                for batch in self.data_generator:
                    optimizer.zero_grad()

                    inp_xs = torch.reshape(batch[0], (batch[0].size()[0], 1))
                    true_ys = torch.reshape(batch[1], (batch[1].size()[0], 1))

                    # print(inp_xs.shape)

                    # print(f'training on {inp_xs} and {true_ys}')
                    # print(inp_xs.dtype)
                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    i += 1
                    batch_loss = loss.item()
                    train_loss += batch_loss
                    tepoch.update(1)
                    tepoch.set_postfix(loss=f"{(train_loss / i):5f}")


if __name__ == "__main__":

    training = Training(epochs=10)

    training.train()
