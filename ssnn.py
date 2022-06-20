import argparse

import Visualizer

from DataLoaderInitializer import DataLoaderInitializer
from DatasetEnum import DatasetEnum
from LogLevel import LogLevel
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer


def parseArguments():
    parser = argparse.ArgumentParser()

    # Trainer arguments
    parser.add_argument("-bs", "--batch_size", help="Batch sized used during training", type=int, default=512)

    # Model arguments

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parseArguments()

    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))

    # _batch_size = 512 # * 4
    # _dataset_enum = DatasetEnum.CIFAR10
    # data_loader_initializer = DataLoaderInitializer(_dataset_enum, _train_test_split_ratio, _batch_size)
    #
    # # Load datasets
    # _train_dataset, _test_dataset, _trainloader, _testloader = data_loader_initializer.get_datasets_and_dataloaders()
    #
    # # Find input and output sizes from dataset
    # _input_size = np.prod(_train_dataset.data.shape[1:])
    # _output_size = len(_train_dataset.classes)
    #
    # # TODO: Investigate adding a convolutional layer, or perhaps doing some sparse densenet beforehand
    # # TODO: Add analysis for sparsity for k
    # # TODO: Calculate when a given ratio will max out the skip connections with a specific sparsity, handy to give the use a heads up cause the skip connections can flatline
    # # TODO: Add feature which makes it possible to specify each layers width
    # snn = SparseNeuralNetwork(input_size=_input_size,
    #                           output_size=_output_size,
    #                           amount_hidden_layers=4,
    #                           max_connection_depth=3,
    #                           network_width=100,
    #                           sparsity=0,
    #                           skip_sequential_ratio=0.5,
    #                           log_level=LogLevel.SIMPLE)
    #
    # trainer = SparseTrainer(_train_dataset, _test_dataset, _trainloader, _testloader,
    #                         epochs=100,
    #                         model=snn,
    #                         batch_size=_batch_size,
    #                         evolution_interval=1,
    #                         # Options: bottom_k, cutoff
    #                         pruning_type="cutoff",
    #                         cutoff=0.001,
    #                         prune_rate=0.1,
    #                         # Options: fixed_sparsity, percentage, no_regrowth
    #                         regrowth_type="percentage",
    #                         regrowth_ratio=0.5,
    #                         regrowth_percentage=0.10,
    #                         lr=5e-3,
    #                         early_stopping_threshold=4,
    #                         # Options: l1, l2
    #                         decay_type="l1",
    #                         weight_decay_lambda=0.00005)
    #
    # # from torchsummary import summary
    # # summary(snn, (snn.input_size, ))
    #
    # trainer.train()
    # trainer.model.eval()
    #
    # visualizer = Visualizer.Visualizer(trainer)
    # visualizer.visualize_all()

    # Investigate with up to max k skip connections, to what distribution of k's the network prunes itself
