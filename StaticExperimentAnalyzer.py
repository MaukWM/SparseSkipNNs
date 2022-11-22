import math
import time
from typing import Tuple

import dill
import numpy as np
from tqdm import tqdm

import util
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer
import os

from matplotlib import pyplot as plt

DIRECTORY = "experiments_start_12_10_2022_end_27_10_2022"


def load_experiment(experiment_path) -> Tuple[SparseNeuralNetwork, SparseTrainer]:
    result_dict = dill.load(open(experiment_path, 'rb'))
    return result_dict["snn"], result_dict["trainer"]


def load_trainers(dataset: str):
    result = {}

    for experiment_config_dir in tqdm(os.listdir(f"{DIRECTORY}/static/{dataset}")):
        _experiments = [x for x in os.listdir(os.path.join(f"{DIRECTORY}/static/{dataset}", experiment_config_dir)) if ".result" in x]
        result[experiment_config_dir] = []
        for _experiment in _experiments:
            _snn, _trainer = load_experiment(
                f"{DIRECTORY}/static/{dataset}/{experiment_config_dir}/{_experiment}")
            result[experiment_config_dir].append(_trainer)

    return result


# Given a list of dictionaries, make one dictionary that averages each key (recursively as well for embedded dicts)
def average_ld(ld):
    average_collected_trainer_items = util.ld_to_dl(ld)
    for collected_trainer_item_name in average_collected_trainer_items.keys():
        if type(average_collected_trainer_items[collected_trainer_item_name][0]) is dict:
            average_collected_trainer_items[collected_trainer_item_name] = average_ld(average_collected_trainer_items[collected_trainer_item_name])
        else:
            try:
                average_collected_trainer_items[collected_trainer_item_name] = np.mean(average_collected_trainer_items[collected_trainer_item_name])
            except Exception:
                print(f"fatal error in average_ld(ld), most likely caused by experiments ran with a static topology, continuing...")
    return average_collected_trainer_items


# Given a list of dictionaries, make one dictionary that gets the standard deviation of each key (recursively as well for embedded dicts)
def std_ld(ld):
    std = util.ld_to_dl(ld)
    for collected_trainer_item_name in std.keys():
        if type(std[collected_trainer_item_name][0]) is dict:
            std[collected_trainer_item_name] = std_ld(std[collected_trainer_item_name])
        else:
            try:
                std[collected_trainer_item_name] = np.std(std[collected_trainer_item_name])
            except Exception:
                print(f"fatal error in std_ld(ld), most likely caused by experiments ran with a static topology, continuing...")
    return std


# Given a list of trainers with multiple experiments ran, compile the results
def compile_trainer_results(trainers):
    mean_result = {}
    std_result = {}
    all_result = {}

    extra_items = ["inference_flops_at_peak", "training_flops_at_peak"]

    for trainer_name in trainers.keys():
        mean_result[trainer_name] = {}
        std_result[trainer_name] = {}

        collected_trainer_items = []

        # If there are not 5 results that specific experiment is not done, so continue
        if len(trainers[trainer_name]) != 5:
            continue

        for trainer in trainers[trainer_name]:
            trainer_items = trainer.items
            for extra_item in extra_items:
                trainer_items[extra_item] = math.nan

            trainer_peak_epoch = trainer.peak_epoch

            # for all trainer.items grab the element of trainer.peak_epoch
            for trainer_item_name in trainer_items.keys():
                if trainer_item_name not in extra_items:
                    if trainer_peak_epoch < len(trainer_items[trainer_item_name]):
                        trainer_items[trainer_item_name] = trainer_items[trainer_item_name][trainer_peak_epoch]
                    else:
                        print(f"WARNING: {trainer_item_name}'s trainer peak epoch {trainer_peak_epoch} is less than trainers' item list {len(trainer_items[trainer_item_name])}")

            # Add big floppa
            trainer_items["inference_flops_at_peak"] = trainer.inference_flops_at_peak
            trainer_items["training_flops_at_peak"] = trainer.training_flops_at_peak

            collected_trainer_items.append(trainer_items)

        for item in collected_trainer_items:
            item.pop('k_n_distribution')
            item.pop('k_sparsity_distribution')
            item.pop('k_sparsity_distribution_by_max_seq')
            item.pop('layer_outgoing_remaining_ratio')
            item.pop('layer_incoming_remaining_ratio')
        average_collected_trainer_items = average_ld(collected_trainer_items)
        std_collected_trainer_items = std_ld(collected_trainer_items)

        mean_result[trainer_name] = average_collected_trainer_items
        std_result[trainer_name] = std_collected_trainer_items
        all_result[trainer_name] = collected_trainer_items

    return mean_result, std_result, all_result


class StaticExperimentAnalyzer:

    def __init__(self, mean_compiled_trainers, std_compiled_trainers, all_trainers):
        self.mean_compiled_trainers = mean_compiled_trainers
        self.std_compiled_trainers = std_compiled_trainers
        self.all_trainers = all_trainers

        # Create grouping lists
        # First we create lists on different max_connection_depths
        self.mcd_grouping = {}
        self.ratio_grouping = {}
        self.sparsity_grouping = {}
        for compiled_trainer_name in mean_compiled_trainers.keys():
            _depth = compiled_trainer_name.split("_")[1].split("-")[1]
            _sparsity = compiled_trainer_name.split("_")[2].split("-")[1]
            _ratio = compiled_trainer_name.split("_")[3].split("-")[1]
            if _depth not in self.mcd_grouping.keys():
                self.mcd_grouping[_depth] = []
            if _sparsity not in self.sparsity_grouping.keys():
                self.sparsity_grouping[_sparsity] = []
            if _ratio not in self.ratio_grouping.keys():
                self.ratio_grouping[_ratio] = []
            self.mcd_grouping[_depth].append(compiled_trainer_name)
            self.sparsity_grouping[_sparsity].append(compiled_trainer_name)
            self.ratio_grouping[_ratio].append(compiled_trainer_name)

        self.groupings = {
            "mcd": self.mcd_grouping,
            "ratio": self.ratio_grouping,
            "sparsity": self.sparsity_grouping,
        }

    def get_maxes_by_grouping(self, k="validation_accuracy"):
        results = {}
        for grouping_name in self.groupings.keys():
            if grouping_name not in results:
                results[grouping_name] = {}
            _grouping = self.groupings[grouping_name]
            for _grouping_key in _grouping.keys():
                if _grouping_key not in results[grouping_name]:
                    results[grouping_name][_grouping_key] = []
                _experiments = _grouping[_grouping_key]
                for _experiment in _experiments:
                    if k in self.mean_compiled_trainers[_experiment].keys():
                        results[grouping_name][_grouping_key] = max(self.mean_compiled_trainers[_experiment][k], results[grouping_name][_grouping_key])

        print(results)

    def get_maxes_and_reduce_by_one_grouping(self, k="validation_accuracy"):
        pass

    def plot_grouping(self, grouping_name, k="validation_accuracy"):
        to_plot_categories = list(self.groupings.keys())
        # print(to_plot_categories)
        mean_compiled_mcd_ratio_groupings = {}
        mean_compiled_mcd_sparsity_groupings = {}
        std_compiled_mcd_sparsity_groupings = {}
        all_compiled_mcd_ratio_groupings = {}
        all_compiled_mcd_sparsity_groupings = {}

        # Collect information according to group (currently separate grouping per max connection depth, x axis ratio by sparsity)
        for _grouping_key in self.groupings[grouping_name]:
            _sub_groupings = list(self.sparsity_grouping.keys())
            _mean_sub_grouping_dict = {}
            _all_sub_grouping_dict = {}
            for _sub_grouping in _sub_groupings:
                _mean_sub_grouping_dict[_sub_grouping] = []
                _all_sub_grouping_dict[_sub_grouping] = []
            for _experiment in self.groupings[grouping_name][_grouping_key]:
                # Group by sparsity by hand
                _sparsity = _experiment.split("_")[2].split("-")[1]
                _ratio = _experiment.split("_")[3].split("-")[1]

                if k in self.mean_compiled_trainers[_experiment].keys():
                    _mean_sub_grouping_dict[_sparsity].append((_ratio, self.mean_compiled_trainers[_experiment][k]))
                    _all_sub_grouping_dict[_sparsity].append((_ratio, [x[k] for x in self.all_trainers[_experiment]]))
            mean_compiled_mcd_ratio_groupings[_grouping_key] = _mean_sub_grouping_dict
            all_compiled_mcd_ratio_groupings[_grouping_key] = _all_sub_grouping_dict

        # print(mean_compiled_mcd_ratio_groupings)

        generalized_compiled_mcd_ratio_groupings_mean = {}
        generalized_compiled_mcd_ratio_groupings_max = {}
        generalized_compiled_mcd_ratio_groupings_std = {}

        # Compile mcd_ratio_grouping even further
        for _grouping_key in mean_compiled_mcd_ratio_groupings.keys():
            generalized_compiled_mcd_ratio_groupings_mean[_grouping_key] = {}
            generalized_compiled_mcd_ratio_groupings_max[_grouping_key] = {}
            generalized_compiled_mcd_ratio_groupings_std[_grouping_key] = {}
            for _sub_grouping_key in mean_compiled_mcd_ratio_groupings[_grouping_key].keys():
                _mean_sub_grouping = mean_compiled_mcd_ratio_groupings[_grouping_key][_sub_grouping_key]
                _all_sub_grouping = all_compiled_mcd_ratio_groupings[_grouping_key][_sub_grouping_key]
                if len(_mean_sub_grouping) > 0:
                    generalized_compiled_mcd_ratio_groupings_mean[_grouping_key][_sub_grouping_key] = np.mean([x[1] for x in _mean_sub_grouping])
                if len(_all_sub_grouping) > 0:
                    generalized_compiled_mcd_ratio_groupings_std[_grouping_key][_sub_grouping_key] = np.std([x[1] for x in _all_sub_grouping])
                    generalized_compiled_mcd_ratio_groupings_max[_grouping_key][_sub_grouping_key] = np.max([x[1] for x in _all_sub_grouping])

        # print(generalized_compiled_mcd_ratio_groupings_mean)
        # print(generalized_compiled_mcd_ratio_groupings_std)

        # Collect information according to group (currently separate grouping per max connection depth, x axis ratio by sparsity)
        for _grouping_key in self.groupings[grouping_name]:
            _sub_groupings = list(self.ratio_grouping.keys())
            _sub_grouping_dict = {}
            _sub_grouping_dict_std = {}
            for _sub_grouping in _sub_groupings:
                _sub_grouping_dict[_sub_grouping] = []
                _sub_grouping_dict_std[_sub_grouping] = []
            for _experiment in self.groupings[grouping_name][_grouping_key]:
                # Group by sparsity by hand
                _sparsity = _experiment.split("_")[2].split("-")[1]
                _ratio = _experiment.split("_")[3].split("-")[1]

                if k in self.mean_compiled_trainers[_experiment].keys():
                    _sub_grouping_dict[_ratio].append((_sparsity, self.mean_compiled_trainers[_experiment][k]))
                    _sub_grouping_dict_std[_ratio].append((_sparsity, self.std_compiled_trainers[_experiment][k]))

            mean_compiled_mcd_sparsity_groupings[_grouping_key] = _sub_grouping_dict
            std_compiled_mcd_sparsity_groupings[_grouping_key] = _sub_grouping_dict_std

            # print("subgroupdict", _sub_grouping_dict)

        # print(compiled_mcd_sparsity_groupings)

        # # Plot the sub groupings in separate graphs
        # for _compiled_mcd_grouping_key in compiled_mcd_groupings.keys():
        #     _sub_grouping = compiled_mcd_groupings[_compiled_mcd_grouping_key]
        #     for _sparsity_key in _sub_grouping.keys():
        #         _results = _sub_grouping[_sparsity_key]
        #         xs = [float(x[0]) for x in _results]
        #         ys = [float(x[1]) for x in _results]
        #         plt.plot(xs, ys)
        #         plt.xlabel("ratio")
        #         plt.ylabel(k)
        #         plt.title(f"Connection depth {_compiled_mcd_grouping_key}, sparsity {_sparsity_key}")
        #         plt.show()
        #         plt.grid()

        # Plot detailed graphs
        # sparsity_colors = {}
        # for sparsity_key in self.sparsity_grouping.keys():
        #     sparsity_colors[sparsity_key] = (float(sparsity_key) * 0.5, 0, (float(sparsity_key) - 0.6) * 2)

        # Plot the sub groupings in a single graphs
        for _compiled_mcd_grouping_key in mean_compiled_mcd_ratio_groupings.keys():
            plt.gcf().clear()
            fig = plt.figure(1)
            ax = fig.add_subplot(111)

            # Order the labels, full experiment
            order = [0, 2, 1, 8, 3, 4, 7, 5, 6]
            # Order for first experiment round
            # order = [0, 2, 1, 5, 3, 4]
            _sub_grouping = mean_compiled_mcd_ratio_groupings[_compiled_mcd_grouping_key]
            _sparsity_keys = list(_sub_grouping.keys())
            print(order, _sparsity_keys)
            _sparsity_keys = [_sparsity_keys[idx] for idx in order]

            # Setup colors
            n_colors = len(order)
            sparsity_colors = []
            for i in range(n_colors):
                sparsity_colors.append((i * 0.8/n_colors, 0.5-i*0.4/n_colors, i * 0.4/n_colors))

            for i in range(len(_sparsity_keys)):
                _sparsity_key = _sparsity_keys[i]
                _results = _sub_grouping[_sparsity_key]
                if len(_results) > 0:
                    xs = [float(x[0]) for x in _results]
                    ys = [float(x[1]) for x in _results]
                    _density_key = 1 - float(_sparsity_key)
                    if _density_key > 0.01:
                        _density_key = f"{_density_key:.2f}"
                    else:
                        _density_key = f"{_density_key:.3f}"
                    ax.plot(xs, ys, label=f"{_density_key}", color=sparsity_colors[i], linewidth=1)
            if k=="actualized_sparsity_ratio":
                ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), label="static", color=(0.2, 0.3, 0.7), linewidth=1)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel("Ratio")
            ax.set_ylabel("Validation Accuracy %")
            ax.set_title(f"Accuracy by ratio, D={_compiled_mcd_grouping_key}")
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.75))
            ax.grid('on', which='both')
            ax.set_xlim(0, 0.9)
            fig.savefig(f"out/{time.time()}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")

        print(generalized_compiled_mcd_ratio_groupings_mean)
        print(generalized_compiled_mcd_ratio_groupings_std)

        # Nice printing
        for _generalized_compiled_mcd_grouping_key in generalized_compiled_mcd_ratio_groupings_mean.keys():
            print("Connection depth: " + _generalized_compiled_mcd_grouping_key)
            _sub_grouping = generalized_compiled_mcd_ratio_groupings_mean[_generalized_compiled_mcd_grouping_key]
            _sub_grouping_keys = generalized_compiled_mcd_ratio_groupings_mean[_generalized_compiled_mcd_grouping_key].keys()
            for _sub_grouping_key in _sub_grouping_keys:
                print(f"{_sub_grouping_key}: {_sub_grouping[_sub_grouping_key]:.2e} ± {generalized_compiled_mcd_ratio_groupings_std[_generalized_compiled_mcd_grouping_key][_sub_grouping_key]:.2e}")

        plt.gcf().clear()
        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        # Setup colors
        n_colors = 4
        cd_colors = []
        for i in range(n_colors):
            cd_colors.append((i * 0.8 / n_colors, 0.5 - i * 0.4 / n_colors, i * 0.4 / n_colors))
        _generalized_compiled_mcd_grouping_keys = list(generalized_compiled_mcd_ratio_groupings_mean.keys())
        # Plot the sub groupings in a single graphs
        for i in range(len(_generalized_compiled_mcd_grouping_keys)):
            _generalized_compiled_mcd_grouping_key = _generalized_compiled_mcd_grouping_keys[i]
            _sub_grouping = generalized_compiled_mcd_ratio_groupings_mean[_generalized_compiled_mcd_grouping_key]
            _results = sorted([(_key, _sub_grouping[_key]) for _key in _sub_grouping], key=lambda x: x[0])
            xs = [1 - float(x[0]) for x in _results]
            ys = [float(x[1]) for x in _results]
            ax.plot(xs, ys, label=f"{_generalized_compiled_mcd_grouping_key}", color=cd_colors[i], linewidth=1)
        handles, labels = ax.get_legend_handles_labels()
        ax.set_xlabel("Density")
        ax.set_ylabel(k)
        ax.set_title(f"Average model performance by density")
        lgd = ax.legend(handles, labels)
        ax.grid()
        ax.set_xlim(0.25, 0.001)
        fig.savefig(f"out/{time.time()}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")

        # ratio_colors = {}
        # for ratio_key in self.ratio_grouping.keys():
        #     ratio_colors[ratio_key] = (0.5 + 0.1 * float(ratio_key), 0.5 * float(ratio_key), float(ratio_key))

        print("Ratio tables")
        # Plot the sub groupings in a single graphs
        for _compiled_mcd_grouping_key in mean_compiled_mcd_sparsity_groupings.keys():
            plt.gcf().clear()
            fig = plt.figure(1)
            ax = fig.add_subplot(111)

            # Setup colors
            n_colors = len(list(mean_compiled_mcd_sparsity_groupings[_compiled_mcd_grouping_key].keys()))
            ratio_colors = []
            for i in range(n_colors):
                ratio_colors.append((i * 0.8/n_colors, 0.5-i*0.4/n_colors, i * 0.4/n_colors))

            print("Connection depth:", _compiled_mcd_grouping_key)
            _sub_grouping = mean_compiled_mcd_sparsity_groupings[_compiled_mcd_grouping_key]
            _sub_grouping_std = std_compiled_mcd_sparsity_groupings[_compiled_mcd_grouping_key]
            _ratio_keys = list(_sub_grouping.keys())
            for i in range(len(_ratio_keys)):
                _ratio_key = _ratio_keys[i]
                _results = _sub_grouping[_ratio_key]
                _results = sorted(_results, key=lambda x: x[0])
                _results_std = _sub_grouping_std[_ratio_key]
                _results_std = sorted(_results_std, key=lambda x: x[0])
                # print("ratio_key", _ratio_key, _results, _results_std)
                # Print suitably for an overleaf table
                if _compiled_mcd_grouping_key != "1":
                    print(f"{_ratio_key} & ", end="")
                    for _i in range(len(_results)):
                        _end = " & "
                        if _i == len((_results)) - 1:
                            _end = "\\\\\n"
                        print(f"{_results[_i][1]:.1f}\\pm{_results_std[_i][1]:.2f}", end=_end)
                if len(_results) > 0:
                    xs = [1 - float(x[0]) for x in _results]
                    ys = [float(x[1]) for x in _results]
                    ax.plot(xs, ys, label=f"{_ratio_key}", color=ratio_colors[i], linewidth=1)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel("Density")
            ax.set_ylabel(k)
            ax.set_ylabel("Validation Accuracy %")
            ax.set_title(f"Accuracy by density, D={_compiled_mcd_grouping_key}")
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.75))
            ax.grid('on', which='both')
            ax.set_xlim(0.001, 0.25)
            fig.savefig(f"out/{time.time()}.png", bbox_extra_artists=(lgd,), bbox_inches="tight")

            print(mean_compiled_mcd_sparsity_groupings)


if __name__ == "__main__":
    # snn, trainer = load_experiment("experiments/static/CIFAR10/DS-CIFAR10_MCD-1_S-0.75_R-1/result1.result")

    trainers = load_trainers("CIFAR10")

    mean_compiled_trainers, std_compiled_trainers, all_trainers = compile_trainer_results(trainers)

    print(f"keys: {mean_compiled_trainers[list(mean_compiled_trainers.keys())[0]].keys()}")

    sea = StaticExperimentAnalyzer(mean_compiled_trainers, std_compiled_trainers, all_trainers)

    # sea.get_maxes_by_grouping()
    # Verify sparsity ratios
    # sea.plot_grouping("mcd", k="actualized_sparsity_ratio")
    # Investigate accuracy
    sea.plot_grouping("mcd", k="validation_accuracy")


    # print(trainer.items)
    # visualizer = Visualizer.Visualizer(trainer)
    # visualizer.visualize_all()
