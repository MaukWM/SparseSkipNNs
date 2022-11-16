import math
import time

import numpy as np
import torch
import tqdm as tqdm
from torch import nn

from Config import SparseTrainerConfig, SparseModelConfig, DenseTrainerConfig, DenseModelConfig
from DataLoaderInitializer import DataLoaderInitializer
from DatasetEnum import DatasetEnum
from DenseNeuralNetwork import DenseNeuralNetwork
from LogLevel import LogLevel
from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey
import Visualizer


class DenseTrainer:

    def __init__(self, train_dataset, test_dataset, trainloader, testloader,
                 model: DenseNeuralNetwork, trainer_config: DenseTrainerConfig, l):
        self.trainer_config = trainer_config

        # Set training parameters
        self.epochs = trainer_config.epochs
        self.lr = trainer_config.lr
        self.early_stopping_threshold = trainer_config.early_stopping_threshold
        self.batch_size = trainer_config.batch_size
        self.decay_type = trainer_config.decay_type
        self.weight_decay_lambda = trainer_config.weight_decay_lambda

        # Initialize dataset and dataloaders
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.trainloader, self.testloader = trainloader, testloader

        # Set logging
        self.l = l

        # Set gpu use
        self.use_gpu = False

        # # Move to gpu
        if self.use_gpu:
            self.trainloader.cuda(), self.testloader.cuda()

        # Set model and initialize model evolution parameters
        self.model = model

        if self.decay_type is not None and self.weight_decay_lambda is None:
            raise ValueError("If weight decay is used, a weight decay lambda must be specified")

        # Initialize dict that keeps track of data over training
        self.items = dict()
        for item_key in ItemKey:
            self.items[item_key.value] = []

        # FLOP Calculation
        self.training_flops = 0

        # Keep track of data at peak performance for evaluation
        self.peak_epoch = 0
        self.validation_accuracy_at_peak = 0

        # Total training time
        self.total_train_time = None

    def train(self):
        _train_start = time.time()

        # Classification loss
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Early stopping variables
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
                i = 0
                self.model.train()
                for batch in self.trainloader:
                    optimizer.zero_grad()

                    inp_xs, true_ys = batch

                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)

                    if self.weight_decay_lambda is not None:
                        if self.decay_type == "l1":
                            norm = sum(p.abs().sum() for p in self.model.parameters())
                        elif self.decay_type == "l2":
                            norm = sum(p.pow(2.0).sum()for p in self.model.parameters())
                        else:
                            raise ValueError(f"Weight decay lambda was specified but no valid decay type was specified: {self.decay_type}")
                        loss += self.weight_decay_lambda * norm

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    i += 1
                    batch_loss = loss.item()
                    train_loss += batch_loss

                    train_accuracy += torch.count_nonzero(torch.eq(true_ys, torch.argmax(pred_ys, dim=1))).item() / pred_ys.size()[0] * 100
                    average_train_accuracy = train_accuracy / i

                    pbar.set_postfix(train_loss=f"{(train_loss / i):.5f}",
                                     train_accuracy=f"{average_train_accuracy:.2f}%",
                                     val_loss=f"0",
                                     val_accuracy=f"0%")
                    pbar.update(1)

                self.items[ItemKey.TRAINING_LOSS.value].append(train_loss / i)
                self.items[ItemKey.TRAINING_ACCURACY.value].append(average_train_accuracy)

                # Calculate validation
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

                    pbar.set_postfix(train_loss=f"{self.items[ItemKey.TRAINING_LOSS.value][epoch]:.5f}",
                                     train_accuracy=f"{average_train_accuracy:.2f}%",
                                     val_loss=f"{(val_loss / i):.5f}",
                                     val_accuracy=f"{average_val_accuracy:.2f}%",)
                    pbar.update(1)

            if average_val_accuracy > self.validation_accuracy_at_peak:
                self.l(message=f"Model improved [{self.peak_epoch}, {self.validation_accuracy_at_peak:.2f}%] -> ", end="", level=LogLevel.SIMPLE)
                self.validation_accuracy_at_peak = average_val_accuracy
                self.peak_epoch = epoch
                self.l(message=f"[{self.peak_epoch}, {self.validation_accuracy_at_peak:.2f}%]", level=LogLevel.SIMPLE)

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

        _train_end = time.time()
        self.total_train_time = _train_end - _train_start

        self.l(message=f"Total training time: {_train_end - _train_start:.2f}s", level=LogLevel.SIMPLE)
        self.l(message=f"Final performance at epoch {self.peak_epoch}: Val_acc={self.validation_accuracy_at_peak:.2f}%", level=LogLevel.SIMPLE)


if __name__ == "__main__":
    _log_level = LogLevel.SIMPLE

    trainer_config = DenseTrainerConfig(
        batch_size=512,
        dataset="CIFAR10",
        epochs=100,
        lr=5e-3,
        early_stopping_threshold=4,
        # Options: l1, l2
        decay_type="l1",
        weight_decay_lambda=0.00005
    )

    model_config = DenseModelConfig(
        n_hidden_layers=3,
        network_width=100,
        log_level=_log_level
    )

    l = lambda level, message, end="\n": print(message, end=end) if level >= _log_level else None

    data_loader_initializer = DataLoaderInitializer(trainer_config.dataset, trainer_config.batch_size)

    # Load datasets
    _train_dataset, _test_dataset, _trainloader, _testloader = data_loader_initializer.get_datasets_and_dataloaders()

    # Find input and output sizes from dataset
    _input_size = np.prod(_train_dataset.data.shape[1:])
    _output_size = len(_train_dataset.classes)

    dnn = DenseNeuralNetwork(input_size=_input_size,
                             output_size=_output_size,
                             model_config=model_config,
                             l=l)

    print(sum(p.numel() for p in dnn.parameters()))

    trainer = DenseTrainer(_train_dataset, _test_dataset, _trainloader, _testloader,
                           model=dnn,
                           trainer_config=trainer_config,
                           l=l)

    trainer.train()
    trainer.model.eval()

    visualizer = Visualizer.Visualizer(trainer)
    visualizer.visualize_all()
