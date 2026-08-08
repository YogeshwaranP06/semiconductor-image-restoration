import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import random
import numpy as np
import torch

from utils.seed import set_seed

set_seed(42)

print("Python :", random.randint(0, 100))

print("NumPy  :", np.random.randint(0, 100))

print("Torch  :", torch.randint(0, 100, (5,)))