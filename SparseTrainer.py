import math
import time

import numpy as np
import torch
import tqdm as tqdm
from torch import nn

from DataLoaderInitializer import DataLoaderInitializer
from DatasetEnum import DatasetEnum
from LogLevel import LogLevel
from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey
import Visualizer


class SparseTrainer:

    def __init__(self, train_dataset, test_dataset, trainloader, testloader,
                 epochs: int, model: SparseNeuralNetwork, evolution_interval, batch_size=64,
                 prune_rate=None, lr=1e-3, early_stopping_threshold=None, train_test_split_ratio=0.8,
                 decay_type=None, weight_decay_lambda=None, pruning_type="bottom_k", cutoff=None,
                 regrowth_type=None, regrowth_ratio=None, regrowth_percentage=None):
        self.epochs = epochs
        self.evolution_interval = evolution_interval
        self.lr = lr
        self.early_stopping_threshold = early_stopping_threshold
        self.train_test_split_ratio = train_test_split_ratio
        self.batch_size = batch_size
        self.decay_type = decay_type
        self.weight_decay_lambda = weight_decay_lambda

        if decay_type is not None and weight_decay_lambda is None:
            raise ValueError("If weight decay is used, a weight decay lambda must be specified")
        if pruning_type == "bottom_k" and prune_rate is None:
            raise ValueError("If pruning type \"bottom_k\" is used, a prune rate must be specified")
        if pruning_type == "cutoff" and cutoff is None:
            raise ValueError("If pruning type \"cutoff\" is used, a cutoff must be specified")
        if regrowth_type == "percentage" and regrowth_percentage is None:
            raise ValueError("If regrowth type \"percentage\" is used, a regrowth_percentage must be specified")

        # Initialize dataset and dataloaders
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.trainloader, self.testloader = trainloader, testloader

        # Set model and initialize model evolution parameters
        self.model = model
        self.model.pruning_type = pruning_type
        self.model.prune_rate = prune_rate
        self.model.cutoff = cutoff
        self.model.regrowth_type = regrowth_type
        self.model.regrowth_ratio = regrowth_ratio
        self.model.regrowth_percentage = regrowth_percentage

        # If we set a cutoff move all the weights outside if the cutoff range
        if cutoff is not None:
            self.model.move_weights_outside_cutoff()

        # Initialize dict that keeps track of data over training
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

                    # TODO: Add a weight decay type "l1"/"l2" and None for nothing
                    # Apply weight decay (L1/L2)
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
                        sparsity_information = self.model.get_and_update_sparsity_information()

                        for item_key in sparsity_information.keys():
                            self.items[item_key].append(sparsity_information[item_key])

                        self.model.evolve_network()

        # Track the final sparsity state
        sparsity_information = self.model.get_and_update_sparsity_information()

        for item_key in sparsity_information.keys():
            self.items[item_key].append(sparsity_information[item_key])

        _train_end = time.time()

        print(f"Total training time: {_train_end - _train_start}s")


if __name__ == "__main__":
    _train_test_split_ratio = 0.8
    _batch_size = 512 * 4
    _dataset_enum = DatasetEnum.MNIST
    data_loader_initializer = DataLoaderInitializer(_dataset_enum, _train_test_split_ratio, _batch_size)

    # Load datasets
    _train_dataset, _test_dataset, _trainloader, _testloader = data_loader_initializer.get_datasets_and_dataloaders()

    # Find input and output sizes from dataset
    _input_size = np.prod(_train_dataset.data.shape[1:])
    _output_size = len(_train_dataset.classes)

    # TODO: Investigate adding a convolutional layer, or perhaps doing some sparse densenet beforehand
    # TODO: Add analysis for sparsity for k
    # TODO: Calculate when a given ratio will max out the skip connections with a specific sparsity, handy to give the use a heads up cause the skip connections can flatline
    # TODO: Add feature which makes it possible to specify each layers width
    snn = SparseNeuralNetwork(input_size=_input_size,
                              output_size=_output_size,
                              amount_hidden_layers=5,
                              max_connection_depth=6,
                              network_width=30,
                              sparsity=0.99,
                              skip_sequential_ratio=0.5,
                              log_level=LogLevel.SIMPLE)
    # snn = SparseNeuralNetwork(input_size=_input_size, output_size=_output_size, amount_hidden_layers=1, max_connection_depth=1, network_width=1,
    #                           sparsity=0.3, skip_sequential_ratio=1, log_level=LogLevel.SIMPLE)

    trainer = SparseTrainer(_train_dataset, _test_dataset, _trainloader, _testloader,
                            epochs=200,
                            model=snn,
                            batch_size=_batch_size,
                            evolution_interval=1,
                            # Options: bottom_k, fixed_cutoff
                            pruning_type="cutoff",
                            cutoff=0.005,
                            prune_rate=0.1,
                            # Options: fixed_sparsity, percentage, no_regrowth
                            regrowth_type="percentage",
                            regrowth_ratio=0.5,
                            regrowth_percentage=0.1,
                            lr=3e-3,
                            early_stopping_threshold=10,
                            # Options: l1, l2
                            decay_type="l1",
                            weight_decay_lambda=0.0001)

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
