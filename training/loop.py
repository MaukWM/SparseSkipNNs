import itertools
import math
import random
import time

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, BatchNorm2d, AvgPool2d
import torch.nn.functional as F

# from torchsummary import summary
# from torchsummary import summary
import Config
from Config import TrainerConfig
from DataLoaderInitializer import DataLoaderInitializer
from LogLevel import LogLevel

import wandb

from SparseNeuralNetwork import SparseNeuralNetwork

# Gpu
from item_keys import ItemKey

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_devices = torch.cuda.device_count()  # returns 1 in my case

print([torch.cuda.get_device_name(i) for i in range(n_devices)])


class Trainer:

    def __init__(self, train_dataset, test_dataset, trainloader, testloader,
                 model: SparseNeuralNetwork, trainer_config: TrainerConfig, l, evolution_interval, prune_rate):
        self.trainer_config = trainer_config

        # Set training parameters
        self.epochs = trainer_config.epochs
        self.lr = trainer_config.lr
        self.early_stopping_threshold = trainer_config.early_stopping_threshold
        self.batch_size = trainer_config.batch_size
        self.decay_type = trainer_config.decay_type
        self.weight_decay_lambda = trainer_config.weight_decay_lambda
        self.evolution_interval = evolution_interval
        self.prune_rate = prune_rate

        # Initialize dataset and dataloaders
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.trainloader, self.testloader = trainloader, testloader

        # Set logging
        self.l = l

        # Set model and initialize model evolution parameters
        self.model = model

        self.n_active_connections = model.n_active_connections

        if self.decay_type is not None and self.weight_decay_lambda is None:
            raise ValueError("If weight decay is used, a weight decay lambda must be specified")

        # Keep track of data at peak performance for evaluation
        self.peak_epoch = 0
        self.validation_accuracy_at_peak = 0

        # Total training time
        self.total_train_time = None

    def evolve_network(self):
        self.model.evolve_network()

    def train(self):
        _train_start = time.time()

        # Classification loss
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Early stopping variables
        early_stopping_counter = 0
        lowest_val_loss = math.inf

        # self.evolve_network()

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
                max_i_train = 0
                self.model.train()
                for batch in self.trainloader:
                    optimizer.zero_grad()

                    inp_xs, true_ys = batch
                    inp_xs = inp_xs.to(device)
                    true_ys = true_ys.to(device)

                    pred_ys = self.model(inp_xs)
                    loss = criterion(pred_ys, true_ys)

                    # from torchviz import make_dot
                    #
                    # make_dot(pred_ys, params=dict(list(self.model.named_parameters()))).render("dynamicallycoded2", format="jpg")

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
                    max_i_train = i
                    batch_loss = loss.item()
                    train_loss += batch_loss
                    average_train_loss = train_loss / i

                    train_accuracy += torch.count_nonzero(torch.eq(true_ys, torch.argmax(pred_ys, dim=1))).item() / pred_ys.size()[0] * 100
                    average_train_accuracy = train_accuracy / i

                    pbar.set_postfix(train_loss=f"{(train_loss / i):.5f}",
                                     train_accuracy=f"{average_train_accuracy:.2f}%",
                                     val_loss=f"0",
                                     val_accuracy=f"0%")
                    pbar.update(1)

                wandb.log({"training_loss": average_train_loss})
                wandb.log({"training_accuracy": average_train_accuracy})

                if wandb.config["max_connection_depth"] > 1:
                    result = self.model.get_and_update_sparsity_information()

                    wandb.log({"ratio": result[ItemKey.ACTUALIZED_SPARSITY_RATIO.value]})
                    wandb.log({"n_seq": result[ItemKey.N_ACTIVE_SEQ_CONNECTIONS.value]})
                    wandb.log({"n_skip": result[ItemKey.N_ACTIVE_SKIP_CONNECTIONS.value]})

                # Calculate validation
                i = 0

                with torch.no_grad():
                    self.model.eval()
                    for batch in self.testloader:
                        inp_xs, true_ys = batch
                        inp_xs = inp_xs.to(device)
                        true_ys = true_ys.to(device)

                        pred_ys = self.model(inp_xs)
                        loss = criterion(pred_ys, true_ys)

                        # print statistics
                        i += 1
                        val_loss += loss.item()
                        average_val_loss = val_loss / i

                        val_accuracy += torch.count_nonzero(torch.eq(true_ys, torch.argmax(pred_ys, dim=1))).item() / pred_ys.size()[0] * 100
                        average_val_accuracy = val_accuracy / i

                        pbar.set_postfix(train_loss=f"{train_loss / max_i_train:.5f}",
                                         train_accuracy=f"{average_train_accuracy:.2f}%",
                                         val_loss=f"{(val_loss / i):.5f}",
                                         val_accuracy=f"{average_val_accuracy:.2f}%",)
                        pbar.update(1)

                    wandb.log({"validation_loss": average_val_loss})
                    wandb.log({"validation_accuracy": average_val_accuracy})

            if average_val_accuracy > self.validation_accuracy_at_peak:
                self.l(message=f"Model improved [{self.peak_epoch}, {self.validation_accuracy_at_peak:.2f}%] -> ", end="", level=LogLevel.SIMPLE)
                self.validation_accuracy_at_peak = average_val_accuracy
                self.peak_epoch = epoch
                self.l(message=f"[{self.peak_epoch}, {self.validation_accuracy_at_peak:.2f}%]", level=LogLevel.SIMPLE)

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
                    self.evolve_network()

        _train_end = time.time()
        self.total_train_time = _train_end - _train_start

        self.l(message=f"Total training time: {_train_end - _train_start:.2f}s", level=LogLevel.SIMPLE)
        self.l(message=f"Final performance at epoch {self.peak_epoch}: Val_acc={self.validation_accuracy_at_peak:.2f}%", level=LogLevel.SIMPLE)


def train(config, project, run_name):
    print("running", project, run_name)
    _log_level = LogLevel.SIMPLE

    wandb.init(project=project)
    wandb.config.update(config)

    trainer_config = TrainerConfig(
        batch_size=wandb.config["batch_size"],
        dataset=wandb.config["dataset"],
        epochs=wandb.config["epochs"],
        lr=wandb.config["learning_rate"],
        early_stopping_threshold=wandb.config["early_stopping_threshold"],
        # Options: l1, l2
        decay_type=wandb.config["decay_type"],
        weight_decay_lambda=wandb.config["weight_decay_lambda"],
        evolution_interval=wandb.config["evolution_interval"]
    )

    l = lambda level, message, end="\n": print(message, end=end) if level >= _log_level else None

    data_loader_initializer = DataLoaderInitializer(trainer_config.dataset, trainer_config.batch_size)

    # Load datasets
    _train_dataset, _test_dataset, _trainloader, _testloader = data_loader_initializer.get_datasets_and_dataloaders()

    # Find input and output sizes from dataset
    _input_size = np.prod(_train_dataset.data.shape[1:])
    _output_size = len(_train_dataset.classes)

    torch.manual_seed(wandb.config["seed"])

    model_config = Config.SparseModelConfig(n_hidden_layers=wandb.config["n_hidden_layers"],
                                          max_connection_depth=wandb.config["max_connection_depth"],
                                          network_width=wandb.config["network_width"],
                                          sparsity=wandb.config["sparsity"],
                                          skip_sequential_ratio=wandb.config["skip_sequential_ratio"],
                                          log_level=wandb.config["log_level"],
                                          pruning_type=wandb.config["pruning_type"],
                                          cutoff=wandb.config["cutoff"],
                                          prune_rate=wandb.config["prune_rate"],
                                          regrowth_type=wandb.config["regrowth_type"],
                                          regrowth_ratio=wandb.config["regrowth_ratio"],
                                          regrowth_percentage=wandb.config["regrowth_percentage"]
                                          )

    net = SparseNeuralNetwork(input_size=_input_size, output_size=_output_size, model_config=model_config, l=l)

    net.to(device)

    # model_summary = summary(dense_net, (3, 32, 32))

    n_max_params = net.calculate_n_max_sequential_connections() + net.calculate_n_max_skip_connections()
    n_params = net.n_active_connections

    print(f"n_max_params: {n_max_params}, n_params: {n_params}")

    # Optional
    wandb.watch(net)

    trainer = Trainer(_train_dataset, _test_dataset, _trainloader, _testloader,
                      model=net,
                      trainer_config=trainer_config,
                      l=l,
                      evolution_interval=wandb.config["evolution_interval"],
                      prune_rate=wandb.config["prune_rate"])

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    import torchsummary

    model = SETSkipNet_CIFAR10(n_dense_blocks=3,
                               k=3,
                               ch=8,
                               sparsity=0.5,
                               sparsify_classification=True,
                               config=None)

    model.to(device)

    torchsummary.summary(model, (3, 32, 32))



