import dill
import os

base_configuration = {
    "trainer_config": {
        "batch_size": 512,
        # Options: CIFAR10/CIFAR100
        "dataset": None,
        "epochs": 100,
        "evolution_interval": None,
        "lr": 5e-3,
        "early_stopping_threshold": 4,
        # Options: l1, l2
        "decay_type": "l1",
        "weight_decay_lambda": 0.00005
    },
    "model_config": {
        "n_hidden_layers": 3,
        "max_connection_depth": None,
        "network_width": 100,
        "sparsity": None,
        "skip_sequential_ratio": None,
        "log_level": "SIMPLE",
        # Options: bottom_k, cutoff
        "pruning_type": "bottom_k",
        "cutoff": 0.001,
        "prune_rate": 0.1,
        # Options: fixed_sparsity, percentage, no_regrowth
        "regrowth_type": "fixed_sparsity",
        "regrowth_ratio": 0.5,
        "regrowth_percentage": None,
    }
}

# To vary:
# dataset
# sparsity
# skip_sequential_ratio
# max_connection_depth
datasets = ["CIFAR10"]
sparsities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
ratios = [0.50]
max_connection_depths = [1, 4]
evolution_interval = [None, 1]


experiment_directory = "experiments"
experiment_top_directory = "experiments_static_vs_dynamic_and_skip_vs_no_skip"

for dataset in datasets:
    if not os.path.isdir(f"{experiment_top_directory}"):
        os.mkdir(f"{experiment_top_directory}")
    if not os.path.isdir(f"{experiment_top_directory}/{experiment_directory}"):
        os.mkdir(f"{experiment_top_directory}/{experiment_directory}")
    if not os.path.isdir(f"{experiment_top_directory}/{experiment_directory}/{dataset}/"):
        os.mkdir(f"{experiment_top_directory}/{experiment_directory}/{dataset}/")
    for sparsity in sparsities:
        for ratio in ratios:
            for max_connection_depth in max_connection_depths:
                for e_i in evolution_interval:
                    _ratio = ratio
                    config = base_configuration.copy()
                    config["trainer_config"]["dataset"] = dataset
                    config["model_config"]["max_connection_depth"] = max_connection_depth
                    config["model_config"]["sparsity"] = sparsity
                    config["model_config"]["evolution_interval"] = e_i
                    if max_connection_depth == 1:
                        _ratio = 1
                    config["model_config"]["skip_sequential_ratio"] = _ratio

                    directory_name = f"DS-{dataset}_MCD-{max_connection_depth}_S-{sparsity}_R-{_ratio}_E-{e_i}"

                    if not os.path.isdir(f"{experiment_top_directory}/{experiment_directory}/{dataset}/{directory_name}"):
                        os.mkdir(f"{experiment_top_directory}/{experiment_directory}/{dataset}/{directory_name}")

                    if not os.path.exists(f"{experiment_top_directory}/{experiment_directory}/{dataset}/{directory_name}/config.pkl"):
                        print(f"creating {experiment_top_directory}/{experiment_directory}/{dataset}/{directory_name}/config.pkl")
                        with open(f"{experiment_top_directory}/{experiment_directory}/{dataset}/{directory_name}/config.pkl", "wb") as file:
                            dill.dump(config, file)


