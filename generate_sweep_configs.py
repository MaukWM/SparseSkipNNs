import json
import math
import random
from itertools import product

base_configuration = {
    "batch_size": 512,
    # Options: CIFAR10/CIFAR100
    "dataset": None,
    "epochs": 100,
    "evolution_interval": None,
    "learning_rate": 5e-3,
    "early_stopping_threshold": 4,
    # Options: l1, l2
    "decay_type": "l1",
    "weight_decay_lambda": 0.00005,
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
    # Options: fixed_sparsity, percentage, no_regrowth, by_ratio
    "regrowth_type": "by_ratio",
    "regrowth_ratio": 0.5,
    "regrowth_percentage": None
}

# ACTUAL EXPERIMENT
variables = {
    "dataset": ["CIFAR10"],
    "sparsity": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    "evolution_interval": [None, 1],
    "max_connection_depth": [1, 3],
    "ratios": [0.5]
}

# # Test
# variables = {
#     "dataset": ["CIFAR10"],
#     "sparsity": [0.7],
#     "evolution_interval": [1],
#     "max_connection_depth": [3],
#     "skip_sequential_ratio": [0.5]
# }

# # To vary:
# # Skip/no skip - dynamic/static experiment
# variables = {
#     "sparsity": [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999],
#     "evolution_interval": [None, 1],
#     "skip": [False, True]
# }

# variables = {
#     "sparsity": [0.5]
# }

n_experiments = 5

config_file = "training/sweep/configs/dynamic_static_skip_no_skip_experiment_18apr.json"
collected_configs = []

combinations = [dict(zip(variables.keys(), values)) for values in product(*variables.values())]


for combination in combinations:
    _id = random.randint(0, int(math.pow(2, 23)))
    name = _id
    for i in range(n_experiments):
        config = base_configuration.copy()

        # Generate seed, id and name
        seed = random.randint(0, int(math.pow(2, 23)))

        config["seed"] = seed
        config["id"] = _id
        config["name"] = f"{name}-{seed}"

        _keys = combination.keys()
        for _key in _keys:
            config[_key] = combination[_key]

        collected_configs.append(config)

json.dump(obj=collected_configs, fp=open(config_file, "w"))
