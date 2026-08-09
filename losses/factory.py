%%writefile losses/factory.py

"""
Loss Factory
============

Creates loss functions from configuration.
"""

import torch.nn as nn

from losses.losses import (
    CharbonnierLoss,
    GradientL1Loss,
)


def create_loss(config):

    loss_name = config["loss"]["name"].lower()

    if loss_name == "l1loss":
        return nn.L1Loss()

    elif loss_name == "mseloss":
        return nn.MSELoss()

    elif loss_name == "charbonnierloss":
        return CharbonnierLoss()

    elif loss_name == "gradientl1loss":
        lambda_gradient = config["loss"].get(
            "lambda_gradient",
            0.1,
        )

        return GradientL1Loss(
            lambda_gradient=lambda_gradient,
        )

    raise ValueError(
        f"Unknown loss function: {loss_name}"
    )
