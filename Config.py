from dataclasses import dataclass

from DatasetEnum import DatasetEnum
from LogLevel import LogLevel


@dataclass
class ModelConfig:
    n_hidden_layers: int
    max_connection_depth: int
    network_width: int
    sparsity: float
    skip_sequential_ratio: float
    log_level: LogLevel
    pruning_type: str
    cutoff: float
    prune_rate: float
    regrowth_type: str
    regrowth_ratio: float
    regrowth_percentage: float


@dataclass
class TrainerConfig:
    batch_size: int
    dataset: DatasetEnum
    epochs: int
    evolution_interval: int
    lr: float
    early_stopping_threshold: int
    decay_type: str
    weight_decay_lambda: float



