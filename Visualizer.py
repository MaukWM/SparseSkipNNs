import matplotlib.pyplot as plt
import numpy as np

import SparseTrainer
import util
from item_keys import ItemKey

# TODO: Make this a drawer class with drawer components so that we can draw multiple subplots or draw in the same plot


class Visualizer:

    def __init__(self, trainer: SparseTrainer):
        self.trainer = trainer
        self.trainer_items = trainer.items

    def visualize_all(self):
        self.plot_train_val_loss()
        self.plot_accuracies()

        if self.trainer.evolution_interval is not None:
            self.plot_sparsity_info()
            self.plot_k_distributions()
            self.plot_k_evolution_graphs()

    def plot_train_val_loss(self):
        plt.title("Training and validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self.trainer_items[ItemKey.TRAINING_LOSS.value], label=ItemKey.TRAINING_LOSS.value)
        plt.plot(self.trainer_items[ItemKey.VALIDATION_LOSS.value], label=ItemKey.VALIDATION_LOSS.value)
        plt.grid()

        plt.legend()

        plt.show()

    def plot_k_evolution_graphs(self):
        # TODO: Split function into generalized method for 3 different distributions
        _xs = np.arange(start=0, stop=len(self.trainer_items[ItemKey.K_N_DISTRIBUTION.value]) * self.trainer.evolution_interval, step=self.trainer.evolution_interval)

        k_n_dists = util.ld_to_dl(self.trainer_items[ItemKey.K_N_DISTRIBUTION.value])
        k_sparsity_dists = util.ld_to_dl(self.trainer_items[ItemKey.K_SPARSITY_DISTRIBUTION.value])
        k_sparsity_by_max_seq_dists = util.ld_to_dl(self.trainer_items[ItemKey.K_SPARSITY_DISTRIBUTION_BY_MAX_SEQ.value])

        plt.title("N change by k")
        plt.xlabel("Epoch")
        plt.ylabel("N")
        plt.grid()
        for k in k_n_dists.keys():
            plt.plot(_xs, k_n_dists[k], label=f"k={k}")
        plt.legend()
        plt.show()

        plt.title("Sparsity (by k) change for k")
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity")
        plt.grid()
        for k in k_sparsity_dists.keys():
            plt.plot(_xs, k_sparsity_dists[k], label=f"k={k}")
        plt.legend()
        plt.show()

        plt.title("Sparsity (by k) change for k")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity")
        plt.grid()
        for k in k_sparsity_dists.keys():
            plt.plot(_xs, k_sparsity_dists[k], label=f"k={k}")
        plt.legend()
        plt.show()

        plt.title("Sparsity (by max seq) change for k")
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity")
        plt.grid()
        for k in k_sparsity_by_max_seq_dists.keys():
            plt.plot(_xs, k_sparsity_by_max_seq_dists[k], label=f"k={k}")
        plt.legend()
        plt.show()

        plt.title("Sparsity (by max seq) change for k")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity")
        plt.grid()
        for k in k_sparsity_by_max_seq_dists.keys():
            plt.plot(_xs, k_sparsity_by_max_seq_dists[k], label=f"k={k}")
        plt.legend()
        plt.show()

    # TODO: Add a tracker and plot the amount of connections per layer
    # TODO: Add a tracker and plot the amount of connections per neuron(?)
    @staticmethod
    def plot_k_distribution(k_n_dist_values, plot_title):
        plt.title(plot_title)
        final_k_n_dist = k_n_dist_values[len(k_n_dist_values) - 1]
        initial_k_n_dist = k_n_dist_values[0]

        plt.grid()
        plt.bar(initial_k_n_dist.keys(), initial_k_n_dist.values(), label="Initial distribution", width=0.7)
        plt.bar(final_k_n_dist.keys(), final_k_n_dist.values(), label="Final distribution", width=0.6)
        plt.legend()
        plt.show()

        plt.title(plot_title)
        final_k_n_dist = k_n_dist_values[len(k_n_dist_values) - 1]
        initial_k_n_dist = k_n_dist_values[0]

        plt.grid()
        plt.plot(final_k_n_dist.keys(), final_k_n_dist.values(), label="Final distribution")
        plt.plot(initial_k_n_dist.keys(), initial_k_n_dist.values(), label="Initial distribution")
        plt.legend()
        plt.show()

    def plot_k_distributions(self):
        # TODO: Split function into generalized method for 3 different distributions
        self.plot_k_distribution(k_n_dist_values=self.trainer_items[ItemKey.K_N_DISTRIBUTION.value],
                                 plot_title="Initial/Final K N Distribution")

        self.plot_k_distribution(k_n_dist_values=self.trainer_items[ItemKey.K_SPARSITY_DISTRIBUTION.value],
                                 plot_title="Initial/Final K Sparsity Distribution By K Sparsity")

        self.plot_k_distribution(k_n_dist_values=self.trainer_items[ItemKey.K_SPARSITY_DISTRIBUTION_BY_MAX_SEQ.value],
                                 plot_title="Initial/Final K Sparsity Distribution By Max Seq Sparsity")

    def plot_accuracies(self):
        plt.title("Training and validation accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(self.trainer_items[ItemKey.VALIDATION_ACCURACY.value], label=ItemKey.VALIDATION_ACCURACY.value)
        plt.plot(self.trainer_items[ItemKey.TRAINING_ACCURACY.value], label=ItemKey.TRAINING_ACCURACY.value)
        plt.grid()

        plt.legend()

        plt.show()

    def plot_sparsity_info(self):
        _xs = np.arange(start=0, stop=len(self.trainer_items[ItemKey.N_ACTIVE_CONNECTIONS.value]) * self.trainer.evolution_interval, step=self.trainer.evolution_interval)

        plt.title("Active connections")
        plt.xticks(_xs)
        plt.xlabel("Epoch")
        plt.ylabel("N Active connections")
        # plt.ylim(0, None)

        plt.plot(_xs, self.trainer_items[ItemKey.N_ACTIVE_CONNECTIONS.value], label=ItemKey.N_ACTIVE_CONNECTIONS.value)
        plt.plot(_xs, self.trainer_items[ItemKey.N_ACTIVE_SEQ_CONNECTIONS.value], label=ItemKey.N_ACTIVE_SEQ_CONNECTIONS.value)
        plt.plot(_xs, self.trainer_items[ItemKey.N_ACTIVE_SKIP_CONNECTIONS.value], label=ItemKey.N_ACTIVE_SKIP_CONNECTIONS.value)

        plt.grid()
        plt.legend()
        plt.show()

        plt.title("Actualized sparsities by k sparsity")
        plt.xticks(np.arange(start=0, stop=len(self.trainer_items[ItemKey.N_ACTIVE_CONNECTIONS.value]) * self.trainer.evolution_interval, step=self.trainer.evolution_interval))
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity %")
        plt.ylim(0, 1)

        plt.plot(_xs, self.trainer_items[ItemKey.ACTUALIZED_OVERALL_SPARSITY.value], label=ItemKey.ACTUALIZED_OVERALL_SPARSITY.value)
        plt.plot(_xs, self.trainer_items[ItemKey.ACTUALIZED_SEQUENTIAL_SPARSITY.value], label=ItemKey.ACTUALIZED_SEQUENTIAL_SPARSITY.value)
        plt.plot(_xs, self.trainer_items[ItemKey.ACTUALIZED_SKIP_SPARSITY.value], label=ItemKey.ACTUALIZED_SKIP_SPARSITY.value)
        plt.plot(_xs, self.trainer_items[ItemKey.ACTUALIZED_SKIP_SPARSITY_BY_MAX_SEQ.value], label=ItemKey.ACTUALIZED_SKIP_SPARSITY_BY_MAX_SEQ.value)
        plt.plot(_xs, self.trainer_items[ItemKey.ACTUALIZED_SPARSITY_RATIO.value], label=ItemKey.ACTUALIZED_SPARSITY_RATIO.value)

        plt.grid()
        plt.legend()
        plt.show()
