import enum


class ItemKey(enum.Enum):

    VALIDATION_LOSS = "validation_loss"
    VALIDATION_ACCURACY = "validation_accuracy"
    TRAINING_LOSS = "training_loss"
    TRAINING_ACCURACY = "training_accuracy"
    N_ACTIVE_CONNECTIONS = "n_active_connections"
    N_ACTIVE_SEQ_CONNECTIONS = "n_active_seq_connections"
    N_ACTIVE_SKIP_CONNECTIONS = "n_active_skip_connections"
    ACTUALIZED_OVERALL_SPARSITY = "actualized_overall_sparsity"
    ACTUALIZED_SEQUENTIAL_SPARSITY = "actualized_sequential_sparsity"
    ACTUALIZED_SKIP_SPARSITY = "actualized_skip_sparsity"
    ACTUALIZED_SKIP_SPARSITY_BY_MAX_SEQ = "actualized_skip_sparsity_by_max_seq"
    ACTUALIZED_SPARSITY_RATIO = "actualized_sparsity_ratio"
    K_N_DISTRIBUTION = "k_n_distribution"
    K_SPARSITY_DISTRIBUTION = "k_sparsity_distribution"
    K_SPARSITY_DISTRIBUTION_BY_MAX_SEQ = "k_sparsity_distribution_by_max_seq"

    # TODO: Add trackers for amount pruned at epoch
