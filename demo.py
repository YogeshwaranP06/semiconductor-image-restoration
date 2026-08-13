import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.factory import create_model


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT
    / "demo_checkpoint"
    / "exp2_epoch50_best_model.pth"
)

CONFIG = (
    PROJECT
    / "configs"
    / "default.yaml"
)

INPUT = (
    PROJECT
    / "data"
    / "raw"
    / "Test_NoisyLR"
    / "NoisyLR"
    / "000000.npy"
)

OUTPUT_DIR = PROJECT / "demo_output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT = OUTPUT_DIR / "000000_restoration.png"


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("SEMICONDUCTOR IMAGE RESTORATION — LIVE DEMO")
print("=" * 70)

for name, path in [
    ("Checkpoint", CHECKPOINT),
    ("Config", CONFIG),
    ("Input", INPUT),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name:12}: {path}")


# ============================================================
# LOAD CONFIG
# ============================================================

import yaml

with open(CONFIG, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Device       :", device)

if device.type == "cuda":
    print(
        "GPU          :",
        torch.cuda.get_device_name(0)
    )
else:
    print(
        "Running on CPU"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = create_model(config).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("Checkpoint epoch :", checkpoint["epoch"])
print("Checkpoint loss  :", checkpoint["loss"])
print("Model parameters :", sum(
    p.numel() for p in model.parameters()
))


# ============================================================
# LOAD INPUT
# ============================================================

noisy = np.load(
    INPUT
).astype(np.float32)

if noisy.shape != (128, 128):
    raise RuntimeError(
        f"Expected 128x128 input, got {noisy.shape}"
    )

print()
print("Input shape      :", noisy.shape)
print("Input dtype      :", noisy.dtype)
print("Input min        :", float(noisy.min()))
print("Input max        :", float(noisy.max()))


# ============================================================
# BICUBIC BASELINE
# ============================================================

input_tensor = torch.from_numpy(
    noisy
).unsqueeze(0).unsqueeze(0)

bicubic = F.interpolate(
    input_tensor,
    size=(256, 256),
    mode="bicubic",
    align_corners=False
).squeeze().numpy()


# ============================================================
# MODEL INFERENCE
# ============================================================

x = (
    torch.from_numpy(noisy)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)

# Warm-up
with torch.no_grad():
    _ = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    prediction = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed = (
    time.perf_counter() - start
)

prediction = (
    prediction
    .squeeze()
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# ============================================================
# VALIDATION
# ============================================================

if prediction.shape != (256, 256):
    raise RuntimeError(
        f"Expected 256x256 output, "
        f"got {prediction.shape}"
    )

if not np.isfinite(prediction).all():
    raise RuntimeError(
        "Prediction contains NaN or Inf"
    )


# ============================================================
# DISPLAY RANGE
# ============================================================

combined = np.concatenate([
    bicubic.ravel(),
    prediction.ravel()
])

vmin, vmax = np.percentile(
    combined,
    [1, 99]
)


# ============================================================
# SAVE COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 4.5)
)

axes[0].imshow(
    noisy,
    cmap="gray"
)

axes[0].set_title(
    "Degraded Input\n128 × 128"
)

axes[0].axis("off")


axes[1].imshow(
    bicubic,
    cmap="gray",
    vmin=vmin,
    vmax=vmax
)

axes[1].set_title(
    "Bicubic Upscaling\n256 × 256"
)

axes[1].axis("off")


axes[2].imshow(
    prediction,
    cmap="gray",
    vmin=vmin,
    vmax=vmax
)

axes[2].set_title(
    "SRDnCNN Restoration\n256 × 256"
)

axes[2].axis("off")


fig.suptitle(
    "Semiconductor Image Restoration — Exp2 Demo",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    OUTPUT,
    dpi=180,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("DEMO COMPLETE")
print("=" * 70)

print("Input shape       :", noisy.shape)
print("Output shape      :", prediction.shape)
print(
    "Inference time    :",
    f"{elapsed * 1000:.3f} ms"
)
print(
    "Output min        :",
    float(prediction.min())
)
print(
    "Output max        :",
    float(prediction.max())
)
print(
    "Output mean       :",
    float(prediction.mean())
)

print()
print("Saved comparison:")
print(OUTPUT)

print("=" * 70)
