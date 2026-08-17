import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * scale * scale,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        return self.block(x)


class SRDnCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=64,
        num_blocks=8,
        scale=2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.upsample = UpsampleBlock(
            channels,
            scale=scale,
        )

        self.tail = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


if len(sys.argv) != 3:
    print("Usage: python run.py <input-dir> <output-dir>")
    sys.exit(1)


input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

project_dir = Path(__file__).resolve().parent
weights_path = project_dir / "models" / "model_weights.pth"


if not input_dir.exists():
    raise FileNotFoundError(
        f"Input directory does not exist: {input_dir}"
    )

if not input_dir.is_dir():
    raise NotADirectoryError(
        f"Input path is not a directory: {input_dir}"
    )

if not weights_path.exists():
    raise FileNotFoundError(
        f"Model weights not found: {weights_path}"
    )

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


input_files = sorted(input_dir.glob("*.npy"))

if not input_files:
    raise RuntimeError(
        f"No .npy files found in: {input_dir}"
    )


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("KLA SEMICONDUCTOR IMAGE RESTORATION")
print("=" * 70)
print("Input directory :", input_dir)
print("Output directory:", output_dir)
print("Input files     :", len(input_files))
print("Device          :", device)

if device.type == "cuda":
    print("GPU             :", torch.cuda.get_device_name(0))


model = SRDnCNN().to(device)

state_dict = torch.load(
    weights_path,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("Model parameters:", parameter_count)

if parameter_count != 739777:
    raise RuntimeError(
        f"Unexpected model parameter count: {parameter_count}"
    )


processed = 0

with torch.inference_mode():

    for input_file in input_files:

        arr = np.load(input_file).astype(np.float32)

        if arr.shape != (128, 128):
            raise RuntimeError(
                f"{input_file.name}: expected "
                f"(128, 128), got {arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{input_file.name}: input contains NaN or Inf."
            )

        x = torch.from_numpy(arr)
        x = x.unsqueeze(0).unsqueeze(0).to(device)

        prediction = model(x)

        output = (
            prediction
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if output.shape != (256, 256):
            raise RuntimeError(
                f"{input_file.name}: expected output "
                f"(256, 256), got {output.shape}"
            )

        output = np.clip(
            output,
            0.0,
            1.0,
        ).astype(np.float32)

        if not np.isfinite(output).all():
            raise RuntimeError(
                f"{input_file.name}: output contains NaN or Inf."
            )

        output_file = output_dir / input_file.name
        np.save(output_file, output)

        processed += 1

        if (
            processed <= 5
            or processed % 50 == 0
            or processed == len(input_files)
        ):
            print(
                f"Processed {processed}/{len(input_files)} | "
                f"{input_file.name}"
            )


output_files = sorted(output_dir.glob("*.npy"))

input_names = {f.name for f in input_files}
output_names = {f.name for f in output_files}

missing = sorted(input_names - output_names)
extra = sorted(output_names - input_names)

bad_shapes = []
bad_range = []
bad_finite = []

for output_file in output_files:

    arr = np.load(output_file)

    if arr.shape != (256, 256):
        bad_shapes.append(output_file.name)

    if not np.isfinite(arr).all():
        bad_finite.append(output_file.name)

    if arr.min() < 0.0 or arr.max() > 1.0:
        bad_range.append(output_file.name)


print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("Input files     :", len(input_files))
print("Output files    :", len(output_files))
print("Missing outputs :", len(missing))
print("Extra outputs   :", len(extra))
print("Bad shapes      :", len(bad_shapes))
print("Bad range       :", len(bad_range))
print("NaN/Inf files   :", len(bad_finite))

if (
    missing
    or extra
    or bad_shapes
    or bad_range
    or bad_finite
    or len(output_files) != len(input_files)
):
    raise RuntimeError(
        "FINAL SUBMISSION VERIFICATION FAILED."
    )

print()
print("128x128 -> 256x256 : PASSED")
print("Filename preservation: PASSED")
print("Output range [0,1]  : PASSED")
print("NaN/Inf check       : PASSED")
print("All outputs valid   : PASSED")
print("KLA submission run  : PASSED")
print("=" * 70)
