# import pickle
import dill
import os

import numpy as np

import StaticExperimentAnalyzer
from Config import SparseTrainerConfig, SparseModelConfig
from DataLoaderInitializer import DataLoaderInitializer
from LogLevel import LogLevel
from SparseNeuralNetwork import SparseNeuralNetwork
from SparseTrainer import SparseTrainer

N_EXPERIMENTS_PER_CONFIG = 5

loaded_datasets = {}

experiment_directory = "experiments"
experiment_top_directory = "experiments_static_vs_dynamic_and_skip_vs_no_skip"


def get_configs_from_file(file_path):
    with open(file_path, "rb") as config_file:
        print(f"Starting experiments for {file_path}")
        config = dill.load(config_file)
        trainer_config = SparseTrainerConfig(
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

        model_config = SparseModelConfig(
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


if __name__ == "__main__":
    # First map out what experiments still need to be run

    experiments = os.listdir(f"{experiment_top_directory}/{experiment_directory}/CIFAR10") # + os.listdir("experiments/static/CIFAR100")

    n_total_experiments_to_be_run = len(experiments) * N_EXPERIMENTS_PER_CONFIG
    n_total_done = 0
    average_training_time = 0
    trainer_times = []

    for _experiment in experiments:
        experiment_dataset = _experiment.split("_")[0].split("-")[1]
        results = [result for result in os.listdir(
            f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}") if ".result" in result]
        n_total_done += len(results)
        for result in results:
            _, _trainer = StaticExperimentAnalyzer.load_experiment(
                f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}/{result}")
            trainer_times.append(_trainer.total_train_time)

    if len(trainer_times) > 0:
        average_training_time = int(np.mean(trainer_times))

    print(f"Launching experiment framework, {n_total_done}/{n_total_experiments_to_be_run} experiment performed, {n_total_experiments_to_be_run - n_total_done} remaining.")
    print(f"Average time per experiment: {average_training_time}s . Estimated time for remaining experiments: {average_training_time * (n_total_experiments_to_be_run - n_total_done)}")

    for _experiment in experiments:
        experiment_dataset = _experiment.split("_")[0].split("-")[1]
        results = [result for result in os.listdir(
            f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}") if ".result" in result]
        n_results = len(results)
        if n_results > N_EXPERIMENTS_PER_CONFIG:
            print(f"WARNING: Experiment {_experiment} has more results than expected: {n_results}>{N_EXPERIMENTS_PER_CONFIG}")
            continue
        if n_results == N_EXPERIMENTS_PER_CONFIG:
            _sub_result_trainer_times = []
            _sub_result_trainer_vals = []
            for result in results:
                _, _trainer = StaticExperimentAnalyzer.load_experiment(
                    f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}/{result}")
                _sub_result_trainer_times.append(_trainer.total_train_time)
                _sub_result_trainer_vals.append(_trainer.validation_accuracy_at_peak)
            print(f"Experiment {_experiment} has already been completed[{np.mean(_sub_result_trainer_vals):.2f}±{np.std(_sub_result_trainer_vals):.2f}] and took ~{int(np.mean(_sub_result_trainer_times))}s, continuing...")
            continue

        if n_results > 0:
            print(f"{n_results} experiments have already been run, continuing the remaining experiments.")

        to_perform_experiments = N_EXPERIMENTS_PER_CONFIG - n_results

        trainer_config, model_config = get_configs_from_file(
            f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}/config.pkl")

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
            # Set logging
            _log_level = model_config.log_level
            _log_file_location = f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}/result{n_results + i + 1}.log"
            _log_file = open(_log_file_location, 'w')
            _l = lambda level, message, end="\n": print(message, end="\n", file=_log_file) if level >= _log_level else None

            snn = SparseNeuralNetwork(input_size=_input_size,
                                      output_size=_output_size,
                                      model_config=model_config,
                                      l=_l)

            trainer = SparseTrainer(_train_dataset, _test_dataset, _trainloader, _testloader,
                                    model=snn,
                                    trainer_config=trainer_config,
                                    l=_l)

            trainer.train()

            _pkl_result = {
                "snn": snn,
                "trainer": trainer
            }

            with open(
                    f"{experiment_top_directory}/{experiment_directory}/{experiment_dataset}/{_experiment}/result{n_results + i + 1}.result", "wb") as result_file:
                trainer.train_dataset, trainer.test_dataset = None, None
                trainer.trainloader, trainer.testloader = None, None
                trainer.model = None
                print(f"Dumping results for {_experiment} {n_results + i + 1}")
                dill.dump(_pkl_result, result_file)
8
