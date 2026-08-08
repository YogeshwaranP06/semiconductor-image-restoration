import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config

config = load_config("configs/default.yaml")

print("=" * 50)

print("Project")
print(config["project"])

print()

print("Dataset")
print(config["dataset"])

print()

print("Training")
print(config["training"])