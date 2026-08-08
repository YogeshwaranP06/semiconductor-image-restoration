import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn

from utils.config import load_config
from training.optimizer import create_optimizer
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

config = load_config("configs/default.yaml")

# Dummy model
model = nn.Conv2d(1, 64, kernel_size=3, padding=1)

optimizer = create_optimizer(model, config)

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=5,
    loss=0.01234,
    path="checkpoints/test_checkpoint.pth",
)

epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    path="checkpoints/test_checkpoint.pth",
)

print("Checkpoint Loaded")
print("Epoch :", epoch)
print("Loss  :", loss)