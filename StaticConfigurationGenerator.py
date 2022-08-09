import pickle as pkl
import os

base_configuration = {
    "trainer_config": {
        "batch_size": 512,
        # Options: CIFAR10/CIFAR100
        "dataset": None,
        "epochs": 100,
        "evolution_interval": 1,
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
        "pruning_type": "cutoff",
        "cutoff": 0.001,
        "prune_rate": 0.1,
        # Options: fixed_sparsity, percentage, no_regrowth
        "regrowth_type": "percentage",
        "regrowth_ratio": 0.5,
        "regrowth_percentage": 0.10,
    }
}

# To vary:
# dataset
# sparsity
# skip_sequential_ratio
# max_connection_depth
datasets = ["CIFAR10", "CIFAR100"]
sparsities = [0, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
ratios = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
max_connection_depths = [1, 2, 3, 4]

for dataset in datasets:
    for sparsity in sparsities:
        for ratio in ratios:
            for max_connection_depth in max_connection_depths:
                config = base_configuration.copy()
                config["trainer_config"]["dataset"] = dataset
                config["model_config"]["max_connection_depth"] = max_connection_depth
                config["model_config"]["sparsity"] = sparsity
                config["model_config"]["skip_sequential_ratio"] = ratio

                directory_name = f"DS-{dataset}_MCD-{max_connection_depth}_S-{sparsity}_R-{ratio}"

                os.mkdir(f"experiments/static/{dataset}/{directory_name}")

                with open(f"experiments/static/{dataset}/{directory_name}/config.pkl", "wb") as file:
                    pkl.dump(config, file)


