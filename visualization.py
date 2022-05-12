from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

import SparseTrainer
from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey

# TODO: Make this a drawer class with drawer components so that we can draw multiple subplots or draw in the same plot


def plot_train_val_loss(trainer: SparseTrainer):
    items = trainer.items

    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(items[ItemKey.TRAINING_LOSS.value], label=ItemKey.TRAINING_LOSS.value)
    plt.plot(items[ItemKey.VALIDATION_LOSS.value], label=ItemKey.VALIDATION_LOSS.value)
    plt.grid()

    plt.legend()

    plt.show()


def plot_k_evolution_graphs(trainer: SparseTrainer):
    items = trainer.items
    _xs = np.arange(start=0, stop=len(items[ItemKey.N_ACTIVE_CONNECTIONS.value]) * trainer.evolution_interval, step=trainer.evolution_interval)

    k_n_dists_LD = items[ItemKey.K_N_DISTRIBUTION.value]
    k_sparsity_dists_LD = items[ItemKey.K_SPARSITY_DISTRIBUTION.value]

    # List of Dicts to Dicts of Lists from: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    k_n_dists_DL = {k: [dic[k] for dic in k_n_dists_LD] for k in k_n_dists_LD[0]}
    k_sparsity_dists_DL = {k: [dic[k] for dic in k_sparsity_dists_LD] for k in k_sparsity_dists_LD[0]}
    print(k_n_dists_DL)
    print(k_sparsity_dists_DL)

    plt.title("N change by k")
    plt.xlabel("Epoch")
    plt.ylabel("N")
    plt.grid()
    for k in k_n_dists_DL.keys():
        plt.plot(_xs, k_n_dists_DL[k], label=f"k={k}")
    plt.legend()
    plt.show()

    plt.title("Sparsity change by k")
    plt.xlabel("Epoch")
    plt.ylabel("Sparsity")
    plt.grid()
    for k in k_n_dists_DL.keys():
        plt.plot(_xs, k_sparsity_dists_DL[k], label=f"k={k}")
    plt.legend()
    plt.show()


def plot_k_distribution(trainer: SparseTrainer):
    items = trainer.items

    plt.title("Initial/Final K N Distribution")
    k_n_dists = items[ItemKey.K_N_DISTRIBUTION.value]
    final_k_n_dist = k_n_dists[len(k_n_dists) - 1]
    initial_k_n_dist = k_n_dists[0]

    plt.grid()
    plt.bar(final_k_n_dist.keys(), final_k_n_dist.values(), label="Final distribution", width=0.8)
    plt.bar(initial_k_n_dist.keys(), initial_k_n_dist.values(), label="Initial distribution", width=0.7)
    plt.legend()
    plt.show()

    plt.title("Initial/Final K Sparsity Distribution")
    k_sparsity_dists = items[ItemKey.K_SPARSITY_DISTRIBUTION.value]
    final_k_sparsity_dist = k_sparsity_dists[len(k_sparsity_dists) - 1]
    initial_k_sparsity_dist = k_sparsity_dists[0]

    plt.grid()
    plt.bar(final_k_sparsity_dist.keys(), final_k_sparsity_dist.values(), label="Final distribution", width=0.8)
    plt.bar(initial_k_sparsity_dist.keys(), initial_k_sparsity_dist.values(), label="Initial distribution", width=0.7)
    plt.legend()
    plt.show()


def plot_accuracies(trainer: SparseTrainer):
    items = trainer.items

    plt.title("Training and validation accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(items[ItemKey.VALIDATION_ACCURACY.value], label=ItemKey.VALIDATION_ACCURACY.value)
    plt.plot(items[ItemKey.TRAINING_ACCURACY.value], label=ItemKey.TRAINING_ACCURACY.value)
    plt.grid()

    plt.legend()

    plt.show()


def plot_sparsity_info(trainer: SparseTrainer):
    items = trainer.items
    _xs = np.arange(start=0, stop=len(items[ItemKey.N_ACTIVE_CONNECTIONS.value]) * trainer.evolution_interval, step=trainer.evolution_interval)

    plt.title("Active connections")
    plt.xticks(_xs)
    plt.xlabel("Normalize to epoch")
    plt.ylabel("N Active connections")
    plt.ylim(0, trainer.model.n_active_connections * 1.1)

    plt.plot(_xs, items[ItemKey.N_ACTIVE_CONNECTIONS.value], label=ItemKey.N_ACTIVE_CONNECTIONS.value)
    plt.plot(_xs, items[ItemKey.N_ACTIVE_SEQ_CONNECTIONS.value], label=ItemKey.N_ACTIVE_SEQ_CONNECTIONS.value)
    plt.plot(_xs, items[ItemKey.N_ACTIVE_SKIP_CONNECTIONS.value], label=ItemKey.N_ACTIVE_SKIP_CONNECTIONS.value)

    plt.grid()
    plt.legend()
    plt.show()

    plt.title("Actualized sparsities")
    plt.xticks(np.arange(start=0, stop=len(items[ItemKey.N_ACTIVE_CONNECTIONS.value]) * trainer.evolution_interval, step=trainer.evolution_interval))
    plt.xlabel("Normalize to epoch")
    plt.ylabel("Sparsity %")
    plt.ylim(0, 1)

    plt.plot(_xs, items[ItemKey.ACTUALIZED_OVERALL_SPARSITY.value], label=ItemKey.ACTUALIZED_OVERALL_SPARSITY.value)
    plt.plot(_xs, items[ItemKey.ACTUALIZED_SEQUENTIAL_SPARSITY.value], label=ItemKey.ACTUALIZED_SEQUENTIAL_SPARSITY.value)
    plt.plot(_xs, items[ItemKey.ACTUALIZED_SKIP_SPARSITY.value], label=ItemKey.ACTUALIZED_SKIP_SPARSITY.value)
    plt.plot(_xs, items[ItemKey.ACTUALIZED_SPARSITY_RATIO.value], label=ItemKey.ACTUALIZED_SPARSITY_RATIO.value)

    plt.grid()
    plt.legend()
    plt.show()
