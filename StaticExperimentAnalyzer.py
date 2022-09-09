from typing import Tuple

import dill
import numpy as np

import Visualizer
import util
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer
import os
from itertools import chain


def load_experiment(experiment_path) -> Tuple[SparseNeuralNetwork, SparseTrainer]:
    result_dict = dill.load(open(experiment_path, 'rb'))
    return result_dict["snn"], result_dict["trainer"]


def load_trainers(dataset: str):
    result = {}

    for experiment_config_dir in os.listdir(f"experiments/static/{dataset}"):
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


# Given a list of trainers with multiple experiments ran, compile the results
def compile_trainer_results(trainers):
    result = {}

    for trainer_name in trainers.keys():
        result[trainer_name] = {}

        collected_trainer_items = []

        # If there are not 5 results that specific experiment is not done, so continue
        if len(trainers[trainer_name]) != 5:
            continue

        for trainer in trainers[trainer_name]:
            trainer_items = trainer.items
            trainer_peak_epoch = trainer.peak_epoch

            # for all trainer.items grab the element of trainer.peak_epoch
            for trainer_item_name in trainer_items.keys():
                trainer_items[trainer_item_name] = trainer_items[trainer_item_name][trainer_peak_epoch]

            collected_trainer_items.append(trainer_items)

        average_collected_trainer_items = average_ld(collected_trainer_items)

        result[trainer_name] = average_collected_trainer_items

    return result


if __name__ == "__main__":
    # snn, trainer = load_experiment("experiments/static/CIFAR10/DS-CIFAR10_MCD-1_S-0.75_R-1/result1.result")

    trainers = load_trainers("CIFAR10")

    compiled_trainers = compile_trainer_results(trainers)

    for compiled_trainer_name in compiled_trainers.keys():
        print(compiled_trainer_name, compiled_trainers[compiled_trainer_name])

    # print(trainer.items)
    # visualizer = Visualizer.Visualizer(trainer)
    # visualizer.visualize_all()
