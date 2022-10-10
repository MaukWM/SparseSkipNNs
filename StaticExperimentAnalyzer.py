import math
from typing import Tuple

import dill
import numpy as np
from tqdm import tqdm

import util
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer
import os

from matplotlib import pyplot as plt


def load_experiment(experiment_path) -> Tuple[SparseNeuralNetwork, SparseTrainer]:
    result_dict = dill.load(open(experiment_path, 'rb'))
    return result_dict["snn"], result_dict["trainer"]


def load_trainers(dataset: str):
    result = {}

    for experiment_config_dir in tqdm(os.listdir(f"experiments/static/{dataset}")):
        _experiments = [x for x in os.listdir(os.path.join(f"experiments/static/{dataset}", experiment_config_dir)) if ".result" in x]
        result[experiment_config_dir] = []
        for _experiment in _experiments:
            _snn, _trainer = load_experiment(f"experiments/static/{dataset}/{experiment_config_dir}/{_experiment}")
            result[experiment_config_dir].append(_trainer)

    return result


# Given a list of dictionaries, make one dictionary that averages each key (recursively as well for embedded dicts)
def average_ld(ld):
    average_collected_trainer_items = util.ld_to_dl(ld)
    for collected_trainer_item_name in average_collected_trainer_items.keys():
        if type(average_collected_trainer_items[collected_trainer_item_name][0]) is dict:
            average_collected_trainer_items[collected_trainer_item_name] = average_ld(average_collected_trainer_items[collected_trainer_item_name])
        else:
            average_collected_trainer_items[collected_trainer_item_name] = np.mean(average_collected_trainer_items[collected_trainer_item_name])
    return average_collected_trainer_items


# Given a list of dictionaries, make one dictionary that gets the standard deviation of each key (recursively as well for embedded dicts)
def std_ld(ld):
    std = util.ld_to_dl(ld)
    for collected_trainer_item_name in std.keys():
        if type(std[collected_trainer_item_name][0]) is dict:
            std[collected_trainer_item_name] = std_ld(std[collected_trainer_item_name])
        else:
            std[collected_trainer_item_name] = np.std(std[collected_trainer_item_name])
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
                    trainer_items[trainer_item_name] = trainer_items[trainer_item_name][trainer_peak_epoch]

            # Add big floppa
            trainer_items["inference_flops_at_peak"] = trainer.inference_flops_at_peak
            trainer_items["training_flops_at_peak"] = trainer.training_flops_at_peak

            collected_trainer_items.append(trainer_items)

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
            for _sub_grouping in _sub_groupings:
                _sub_grouping_dict[_sub_grouping] = []
            for _experiment in self.groupings[grouping_name][_grouping_key]:
                # Group by sparsity by hand
                _sparsity = _experiment.split("_")[2].split("-")[1]
                _ratio = _experiment.split("_")[3].split("-")[1]

                if k in self.mean_compiled_trainers[_experiment].keys():
                    _sub_grouping_dict[_ratio].append((_sparsity, self.mean_compiled_trainers[_experiment][k]))

            mean_compiled_mcd_sparsity_groupings[_grouping_key] = _sub_grouping_dict

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
        sparsity_colors = {}
        for sparsity_key in self.sparsity_grouping.keys():
            sparsity_colors[sparsity_key] = (float(sparsity_key) * 0.5, 0, (float(sparsity_key) - 0.6) * 2)

        # Plot the sub groupings in a single graphs
        for _compiled_mcd_grouping_key in mean_compiled_mcd_ratio_groupings.keys():
            _sub_grouping = mean_compiled_mcd_ratio_groupings[_compiled_mcd_grouping_key]
            for _sparsity_key in _sub_grouping.keys():
                _results = _sub_grouping[_sparsity_key]
                if len(_results) > 0:
                    xs = [float(x[0]) for x in _results]
                    ys = [float(x[1]) for x in _results]
                    plt.plot(xs, ys, label=f"{_sparsity_key}")  #, color=sparsity_colors[_sparsity_key])
            plt.xlabel("ratio")
            plt.ylabel(k)
            plt.title(f"Connection depth {_compiled_mcd_grouping_key}")
            plt.legend()
            plt.grid()
            plt.show()

        print(generalized_compiled_mcd_ratio_groupings_mean)
        print(generalized_compiled_mcd_ratio_groupings_std)

        # Nice printing
        for _generalized_compiled_mcd_grouping_key in generalized_compiled_mcd_ratio_groupings_mean.keys():
            print("Connection depth: " + _generalized_compiled_mcd_grouping_key)
            _sub_grouping = generalized_compiled_mcd_ratio_groupings_mean[_generalized_compiled_mcd_grouping_key]
            _sub_grouping_keys = generalized_compiled_mcd_ratio_groupings_mean[_generalized_compiled_mcd_grouping_key].keys()
            for _sub_grouping_key in _sub_grouping_keys:
                print(f"{_sub_grouping_key}: {_sub_grouping[_sub_grouping_key]:.2e} ± {generalized_compiled_mcd_ratio_groupings_std[_generalized_compiled_mcd_grouping_key][_sub_grouping_key]:.2e}")

        # Plot the sub groupings in a single graphs
        for _generalized_compiled_mcd_grouping_key in generalized_compiled_mcd_ratio_groupings_mean.keys():
            _sub_grouping = generalized_compiled_mcd_ratio_groupings_mean[_generalized_compiled_mcd_grouping_key]
            _results = sorted([(_key, _sub_grouping[_key]) for _key in _sub_grouping], key=lambda x: x[0])
            xs = [float(x[0]) for x in _results]
            ys = [float(x[1]) for x in _results]
            plt.plot(xs, ys, label=f"{_generalized_compiled_mcd_grouping_key}")
        plt.xlabel("sparsity")
        plt.ylabel(k)
        plt.title(f"Average model performance by connection depth and sparsity")
        plt.legend()
        plt.grid()
        plt.show()

        ratio_colors = {}
        for ratio_key in self.ratio_grouping.keys():
            ratio_colors[ratio_key] = (0.5 + 0.1 * float(ratio_key), 0.5 * float(ratio_key), float(ratio_key))

        # Plot the sub groupings in a single graphs
        for _compiled_mcd_grouping_key in mean_compiled_mcd_sparsity_groupings.keys():
            _sub_grouping = mean_compiled_mcd_sparsity_groupings[_compiled_mcd_grouping_key]
            for _ratio_key in _sub_grouping.keys():
                _results = _sub_grouping[_ratio_key]
                _results = sorted(_results, key=lambda x: x[0])
                if len(_results) > 0:
                    xs = [float(x[0]) for x in _results]
                    ys = [float(x[1]) for x in _results]
                    plt.plot(xs, ys, label=f"{_ratio_key}") #, color=ratio_colors[_ratio_key])
            plt.xlabel("sparsity")
            plt.ylabel(k)
            plt.title(f"Connection depth {_compiled_mcd_grouping_key}")
            plt.legend()
            plt.grid()
            plt.show()


if __name__ == "__main__":
    # snn, trainer = load_experiment("experiments/static/CIFAR10/DS-CIFAR10_MCD-1_S-0.75_R-1/result1.result")

    trainers = load_trainers("CIFAR10")

    mean_compiled_trainers, std_compiled_trainers, all_trainers = compile_trainer_results(trainers)

    print(f"keys: {mean_compiled_trainers[list(mean_compiled_trainers.keys())[0]].keys()}")

    sea = StaticExperimentAnalyzer(mean_compiled_trainers, std_compiled_trainers, all_trainers)

    # sea.get_maxes_by_grouping()
    sea.plot_grouping("mcd", k="actualized_sparsity_ratio")

    # print(trainer.items)
    # visualizer = Visualizer.Visualizer(trainer)
    # visualizer.visualize_all()
