import pickle
import os

import numpy as np

from Config import TrainerConfig, ModelConfig
from DataLoaderInitializer import DataLoaderInitializer
from LogLevel import LogLevel
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer

N_EXPERIMENTS_PER_CONFIG = 5

loaded_datasets = {}


def get_configs_from_file(file_path):
    with open(file_path, "rb") as config_file:
        print(file_path)
        config = pickle.load(config_file)
        trainer_config = TrainerConfig(
            batch_size=config["trainer_config"]["batch_size"],
            dataset=config["trainer_config"]["dataset"],
            epochs=config["trainer_config"]["epochs"],
            evolution_interval=config["trainer_config"]["evolution_interval"],
            lr=config["trainer_config"]["lr"],
            early_stopping_threshold=config["trainer_config"]["early_stopping_threshold"],
            # Options: l1, l2
            decay_type=config["trainer_config"]["decay_type"],
            weight_decay_lambda=config["trainer_config"]["weight_decay_lambda"]
        )

        model_config = ModelConfig(
            n_hidden_layers=config["model_config"]["n_hidden_layers"],
            max_connection_depth=config["model_config"]["max_connection_depth"],
            network_width=config["model_config"]["network_width"],
            sparsity=config["model_config"]["sparsity"],
            skip_sequential_ratio=config["model_config"]["skip_sequential_ratio"],
            log_level=LogLevel.SIMPLE,
            # Options: bottom_k, cutoff
            pruning_type=config["model_config"]["pruning_type"],
            cutoff=config["model_config"]["cutoff"],
            prune_rate=config["model_config"]["prune_rate"],
            # Options: fixed_sparsity, percentage, no_regrowth
            regrowth_type=config["model_config"]["regrowth_type"],
            regrowth_ratio=config["model_config"]["regrowth_ratio"],
            regrowth_percentage=config["model_config"]["regrowth_percentage"],
        )

        return trainer_config, model_config


# First map out what experiments still need to be run

experiments = os.listdir("experiments/static/CIFAR10") + os.listdir("experiments/static/CIFAR100")

for _experiment in experiments:
    experiment_dataset = _experiment.split("_")[0].split("-")[1]
    results = [result for result in os.listdir(f"experiments/static/{experiment_dataset}/{_experiment}") if "result" in result]
    n_results = len(results)
    if n_results > N_EXPERIMENTS_PER_CONFIG:
        print(f"WARNING: Experiment {_experiment} has more results than expected: {n_results}>{N_EXPERIMENTS_PER_CONFIG}")
        continue
    if n_results == N_EXPERIMENTS_PER_CONFIG:
        print(f"Experiment {_experiment} has already been completed, continuing...")
        continue

    to_perform_experiments = N_EXPERIMENTS_PER_CONFIG - n_results

    trainer_config, model_config = get_configs_from_file(f"experiments/static/{experiment_dataset}/{_experiment}/config.pkl")

    if trainer_config.dataset not in loaded_datasets.keys():
        data_loader_initializer = DataLoaderInitializer(trainer_config.dataset, trainer_config.batch_size)

        # Load datasets
        loaded_datasets[trainer_config.dataset] = {}
        loaded_datasets[trainer_config.dataset]["train_dataset"], loaded_datasets[trainer_config.dataset]["test_dataset"], loaded_datasets[trainer_config.dataset]["trainloader"], loaded_datasets[trainer_config.dataset]["testloader"] = data_loader_initializer.get_datasets_and_dataloaders()

    _train_dataset, _test_dataset, _trainloader, _testloader = loaded_datasets[trainer_config.dataset]["train_dataset"], loaded_datasets[trainer_config.dataset]["test_dataset"], loaded_datasets[trainer_config.dataset]["trainloader"], loaded_datasets[trainer_config.dataset]["testloader"]

    # Find input and output sizes from dataset
    _input_size = np.prod(_train_dataset.data.shape[1:])
    _output_size = len(_train_dataset.classes)

    for i in range(to_perform_experiments):
        snn = SparseNeuralNetwork(input_size=_input_size,
                                  output_size=_output_size,
                                  model_config=model_config)

        trainer = SparseTrainer(_train_dataset, _test_dataset, _trainloader, _testloader,
                                model=snn,
                                trainer_config=trainer_config)

        trainer.train()

        _pkl_result = {
            "snn": snn,
            "trainer": trainer
        }

        with open(f"experiments/static/{experiment_dataset}/{_experiment}/result{n_results + i + 1}", "wb") as result_file:
            pickle.dump(_pkl_result, result_file)
