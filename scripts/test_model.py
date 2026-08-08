import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.config import load_config
from models.factory import create_model

config = load_config("configs/default.yaml")

model = create_model(config)

print(model)

x = torch.randn(1, 1, 128, 128)

y = model(x)

print("\nInput Shape :", x.shape)
print("Output Shape:", y.shape)