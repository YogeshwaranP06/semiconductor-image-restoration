"""
Loss Functions
==============

Collection of loss functions for image restoration.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Robust L1 Loss (Charbonnier Loss)
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()