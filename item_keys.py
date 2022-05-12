import enum


class ItemKey(enum.Enum):

    VALIDATION_LOSS = "validation_loss"
    VALIDATION_ACCURACY = "validation_accuracy"
    TRAINING_LOSS = "training_loss"
    TRAINING_ACCURACY = "training_accuracy"
    SPARSITIES = "sparsities"

