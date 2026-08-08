import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.config import load_config
from losses.factory import create_loss

config = load_config("configs/default.yaml")

criterion = create_loss(config)

print(type(criterion).__name__)

prediction = torch.rand(1, 1, 256, 256)
target = torch.rand(1, 1, 256, 256)

loss = criterion(prediction, target)

print("Loss :", loss.item())