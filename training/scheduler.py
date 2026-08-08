"""
Scheduler Factory
=================

Creates learning-rate schedulers from configuration.
"""

import torch


def create_scheduler(optimizer, config):
    """
    Create scheduler from configuration.
    """

    scheduler_name = config["scheduler"]["name"].lower()

    epochs = config["training"]["epochs"]

    if scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

    elif scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )

    return None