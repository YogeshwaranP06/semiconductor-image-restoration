import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.datamodule import create_dataloaders

train_loader, val_loader = create_dataloaders(
    gt_dir="data/raw/train/GT",
    noisy_dir="data/raw/train/NoisyLR",
    batch_size=8,
)

print("Training batches :", len(train_loader))
print("Validation batches :", len(val_loader))

batch = next(iter(train_loader))

print("\nBatch Shapes")
print("-" * 30)
print("Noisy :", batch["noisy"].shape)
print("GT    :", batch["gt"].shape)