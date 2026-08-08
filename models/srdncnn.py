"""
SRDnCNN
=======

Baseline Super-Resolution + Denoising Network
"""

import torch.nn as nn

from models.common.blocks import ResidualBlock
from models.common.layers import UpsampleBlock


class SRDnCNN(nn.Module):
    """
    Super-Resolution DnCNN
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_blocks: int,
        scale: int,
    ):
        super().__init__()

        # Feature Extraction
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Residual Body
        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Upsampling
        self.upsample = UpsampleBlock(
            channels,
            scale=scale,
        )

        # Reconstruction
        self.tail = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        x = self.head(x)

        x = self.body(x)

        x = self.upsample(x)

        x = self.tail(x)

        return x