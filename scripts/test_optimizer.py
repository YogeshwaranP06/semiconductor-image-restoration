import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn

from utils.config import load_config
from training.optimizer import create_optimizer

config = load_config("configs/default.yaml")

model = nn.Conv2d(1, 64, kernel_size=3, padding=1)

optimizer = create_optimizer(model, config)

print(type(optimizer).__name__)
print("Learning Rate:", optimizer.param_groups[0]["lr"])