from abc import ABC

import matplotlib.pyplot as plt


def plot_train_val_loss(items: dict):
    # TODO: Add validation and return warnings if invalid
    plt.plot(items["training_loss"], label="training_loss")
    plt.plot(items["validation_loss"], label="validation_loss")

    plt.legend()

    plt.show()
