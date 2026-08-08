import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn

from utils.config import load_config
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler

config = load_config("configs/default.yaml")

model = nn.Conv2d(1, 64, kernel_size=3, padding=1)

optimizer = create_optimizer(model, config)
scheduler = create_scheduler(optimizer, config)

print(type(scheduler).__name__)
print("Initial LR:", optimizer.param_groups[0]["lr"])

scheduler.step()

print("After Step :", optimizer.param_groups[0]["lr"])