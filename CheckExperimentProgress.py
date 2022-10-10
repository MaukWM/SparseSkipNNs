import os

import numpy as np

import StaticExperimentAnalyzer

N_EXPERIMENTS_PER_CONFIG = 5

loaded_datasets = {}


if __name__ == "__main__":

    # First map out what experiments still need to be run
    experiments = os.listdir("experiments/static/CIFAR10") # + os.listdir("experiments/static/CIFAR100")

    n_total_experiments_to_be_run = len(experiments) * N_EXPERIMENTS_PER_CONFIG
    n_total_done = 0
    trainer_times = []

    for _experiment in experiments:
        experiment_dataset = _experiment.split("_")[0].split("-")[1]
        results = [result for result in os.listdir(
            f"experiments/static/{experiment_dataset}/{_experiment}") if ".result" in result]
        n_total_done += len(results)
        for result in results:
            _, _trainer = StaticExperimentAnalyzer.load_experiment(
                f"experiments/static/{experiment_dataset}/{_experiment}/{result}")
            trainer_times.append(_trainer.total_train_time)

    average_training_time = int(np.mean(trainer_times))

    print(f"Checking amount of experiments remaining: {n_total_done}/{n_total_experiments_to_be_run} experiment performed, {n_total_experiments_to_be_run - n_total_done} remaining.")
    print(f"Average time per experiment: {average_training_time}s . Estimated time for remaining experiments: {average_training_time * (n_total_experiments_to_be_run - n_total_done)}")

