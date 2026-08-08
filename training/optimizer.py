"""
Optimizer Factory
=================

Creates optimizers from the YAML configuration.

Project:
    Semiconductor Image Restoration
"""

import torch


def create_optimizer(model, config):
    """
    Create optimizer from configuration.
    """

    optimizer_name = config["optimizer"]["name"].lower()
    lr = config["training"]["learning_rate"]
    weight_decay = config["optimizer"]["weight_decay"]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )