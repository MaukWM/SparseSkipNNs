from abc import ABC

import matplotlib.pyplot as plt

import SparseTrainer
from SparseNeuralNetwork import SparseNeuralNetwork
from item_keys import ItemKey


def plot_train_val_loss(trainer: SparseTrainer):
    # TODO: Add validation and return warnings if invalid
    items = trainer.items

    plt.plot(items["training_loss"], label="training_loss")
    plt.plot(items["validation_loss"], label="validation_loss")
    plt.grid()

    plt.legend()

    plt.show()


def plot_sparsity_info(trainer: SparseTrainer):
    # Convert List of dicts to dict of lists
    items = trainer.items[ItemKey.SPARSITIES]

    dict_items = {k: [dic[k] for dic in items] for k in items[0]}
    # TODO: add normalizer for epochs compared to evolve speed
    plt.title("Active connections")
    plt.grid()
    plt.xlabel("Evolution step TODO: Normalize to epoch")
    plt.ylabel("N Active connections")
    plt.ylim(0, trainer.model.n_active_connections)
    plt.plot(dict_items["n_active_connections"], label="n_active_connections")
    plt.plot(dict_items["n_seq_connections"], label="n_seq_connections")
    plt.plot(dict_items["n_skip_connections"], label="n_skip_connections")
    plt.legend()
    plt.show()

    plt.title("Actualized sparsities")
    plt.grid()
    plt.xlabel("Evolution step TODO: Normalize to epoch")
    plt.ylabel("Sparsity %")
    plt.ylim(0, 1)
    plt.plot(dict_items["actualized_overall_sparsity"], label="actualized_overall_sparsity")
    plt.plot(dict_items["actualized_sequential_sparsity"], label="actualized_sequential_sparsity")
    plt.plot(dict_items["actualized_skip_sparsity"], label="actualized_skip_sparsity")
    plt.plot(dict_items["actualized_sparsity_ratio"], label="actualized_sparsity_ratio")
    plt.legend()
    plt.show()
