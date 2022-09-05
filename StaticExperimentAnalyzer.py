from typing import TypeVar, Tuple

import dill
import Visualizer
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer


def load_experiment(experiment_path) -> Tuple[SparseNeuralNetwork, SparseTrainer]:
    result_dict = dill.load(open(experiment_path, 'rb'))
    return result_dict["snn"], result_dict["trainer"]


snn, trainer = load_experiment("experiments/static/CIFAR10/DS-CIFAR10_MCD-1_S-0.75_R-1/result1.result")

# print(trainer.items)
visualizer = Visualizer.Visualizer(trainer)
visualizer.visualize_all()
