"""
Semiconductor Image Restoration — Validation Demo
==================================================

Demonstrates the frozen Experiment 2 SRDnCNN model on a
deterministic validation sample.

Pipeline:
    NoisyLR 128x128
        ↓
    SRDnCNN
        ↓
    Restored 256x256

The validation split exactly follows datasets/datamodule.py:
    Total samples : 3200
    Train         : 2880
    Validation    : 320
    Seed          : 42
"""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from models.factory import create_model


# ============================================================
# PATHS
# ============================================================

CONFIG = PROJECT / "configs" / "default.yaml"

CHECKPOINT = (
    PROJECT
    / "checkpoint"
    / "exp2_epoch50_best_model.pth"
)

GT_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "train"
    / "GT"
)

NOISY_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "train"
    / "NoisyLR"
)

OUTPUT_DIR = PROJECT / "demo_output"


# ============================================================
# VALIDATION SPLIT
# ============================================================

TOTAL_SAMPLES = 3200
VAL_SPLIT = 0.1
SEED = 42

VAL_SIZE = int(TOTAL_SAMPLES * VAL_SPLIT)
TRAIN_SIZE = TOTAL_SAMPLES - VAL_SIZE

# Deterministic split matching datasets/datamodule.py
generator = torch.Generator().manual_seed(SEED)

_, val_indices = torch.utils.data.random_split(
    range(TOTAL_SAMPLES),
    [TRAIN_SIZE, VAL_SIZE],
    generator=generator,
)


# Select the first validation sample deterministically.
VALIDATION_POSITION = 0
DATASET_INDEX = val_indices.indices[VALIDATION_POSITION]


# ============================================================
# FILES
# ============================================================

NOISY_FILES = sorted(NOISY_DIR.glob("*.npy"))
GT_FILES = sorted(GT_DIR.glob("*.npy"))

if len(NOISY_FILES) != TOTAL_SAMPLES:
    raise RuntimeError(
        f"Expected {TOTAL_SAMPLES} NoisyLR files, "
        f"found {len(NOISY_FILES)}."
    )

if len(GT_FILES) != TOTAL_SAMPLES:
    raise RuntimeError(
        f"Expected {TOTAL_SAMPLES} GT files, "
        f"found {len(GT_FILES)}."
    )


NOISY_FILE = NOISY_FILES[DATASET_INDEX]
GT_FILE = GT_FILES[DATASET_INDEX]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SEMICONDUCTOR IMAGE RESTORATION — VALIDATION DEMO")
print("=" * 70)

print("Checkpoint :", CHECKPOINT)
print("Config     :", CONFIG)

print()
print("Dataset")
print("---------")
print("Total samples      :", TOTAL_SAMPLES)
print("Training samples   :", TRAIN_SIZE)
print("Validation samples :", VAL_SIZE)
print("Validation seed    :", SEED)
print("Validation index   :", DATASET_INDEX)

print()
print("Selected sample")
print("----------------")
print("NoisyLR :", NOISY_FILE.name)
print("GT      :", GT_FILE.name)


# ============================================================
# VALIDATE PATHS
# ============================================================

for name, path in [
    ("Configuration", CONFIG),
    ("Checkpoint", CHECKPOINT),
    ("GT directory", GT_DIR),
    ("Noisy directory", NOISY_DIR),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG,
    "r",
    encoding="utf-8",
) as f:
    config = yaml.safe_load(f)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device :", device)

if device.type == "cuda":
    print(
        "GPU    :",
        torch.cuda.get_device_name(0),
    )
else:
    print("Running on CPU")


# ============================================================
# LOAD MODEL
# ============================================================

model = create_model(config).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

if "model_state_dict" not in checkpoint:
    raise RuntimeError(
        "Checkpoint does not contain model_state_dict."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print()
print("Checkpoint epoch :", checkpoint.get("epoch"))
print("Checkpoint loss  :", checkpoint.get("loss"))
print(
    "Model parameters :",
    sum(
        p.numel()
        for p in model.parameters()
    ),
)


# ============================================================
# LOAD VALIDATION SAMPLE
# ============================================================

noisy = np.load(
    NOISY_FILE
).astype(np.float32)

gt = np.load(
    GT_FILE
).astype(np.float32)


if noisy.shape != (128, 128):
    raise RuntimeError(
        f"NoisyLR shape is {noisy.shape}; "
        "expected (128,128)."
    )

if gt.shape != (256, 256):
    raise RuntimeError(
        f"GT shape is {gt.shape}; "
        "expected (256,256)."
    )


print()
print("Input")
print("------")
print("Shape :", noisy.shape)
print("Dtype :", noisy.dtype)
print("Min   :", float(noisy.min()))
print("Max   :", float(noisy.max()))


# ============================================================
# INFERENCE
# ============================================================

x = (
    torch.from_numpy(noisy)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)


if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    prediction = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_ms = (
    time.perf_counter() - start
) * 1000.0


restored = (
    prediction
    .squeeze()
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


if restored.shape != (256, 256):
    raise RuntimeError(
        f"Restored output shape is {restored.shape}; "
        "expected (256,256)."
    )

if not np.isfinite(restored).all():
    raise RuntimeError(
        "Restored output contains NaN or Inf."
    )


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print()
print("=" * 70)
print("DEMO COMPLETE")
print("=" * 70)

print("Input shape      :", noisy.shape)
print("Output shape     :", restored.shape)
print("Ground truth     :", gt.shape)
print(
    "Inference time   :",
    f"{elapsed_ms:.3f} ms",
)

print(
    "Output min       :",
    float(restored.min()),
)

print(
    "Output max       :",
    float(restored.max()),
)

print(
    "Output mean      :",
    float(restored.mean()),
)


# ============================================================
# VISUALIZATION
# ============================================================

comparison_path = (
    OUTPUT_DIR
    / f"{NOISY_FILE.stem}_validation_comparison.png"
)


# Normalize only for DISPLAY.
# The saved model output itself is not modified.
def display_normalize(image):
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    minimum = image.min()
    maximum = image.max()

    if maximum <= minimum:
        return np.zeros_like(image)

    return (
        (image - minimum)
        / (maximum - minimum)
    )


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)


axes[0].imshow(
    display_normalize(noisy),
    cmap="gray",
)

axes[0].set_title(
    "NoisyLR\n128 × 128"
)


axes[1].imshow(
    display_normalize(restored),
    cmap="gray",
)

axes[1].set_title(
    "SRDnCNN Restored\n256 × 256"
)


axes[2].imshow(
    display_normalize(gt),
    cmap="gray",
)

axes[2].set_title(
    "Ground Truth\n256 × 256"
)


for ax in axes:
    ax.axis("off")


fig.suptitle(
    "Semiconductor Image Restoration — Validation Sample",
    fontsize=14,
)

plt.tight_layout()

fig.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved comparison:")
print(comparison_path)
print("=" * 70)