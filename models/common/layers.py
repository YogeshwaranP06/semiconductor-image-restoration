"""
Reusable Layers
===============
"""

import torch.nn as nn


class UpsampleBlock(nn.Module):

    def __init__(self, channels: int, scale: int = 2):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * scale * scale,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):

        return self.block(x)