"""
Common Neural Network Blocks
============================
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Simple residual block used throughout the project.
    """

    def __init__(self, channels: int):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):

        return x + self.block(x)
