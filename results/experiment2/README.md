# Experiment 2 — SRDnCNN Baseline

## Scope

Experiment 2 is the frozen baseline for the semiconductor image restoration study.

## Configuration

- Model: SRDnCNN
- Input: 1 × 128 × 128 NoisyLR
- Output: 1 × 256 × 256 restored image
- Loss: L1Loss
- Epochs: 50
- Batch size: 8
- Learning rate: 0.0001
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Seed: 42
- Training device: NVIDIA Tesla T4

## Dataset split

- Paired training samples: 3200
- Training split: 2880
- Validation split: 320
- Test NoisyLR samples: 400

## Training result

- Best epoch: 50
- Best validation L1 loss: 0.0311003166

## Validation result

| Metric | Result |
|---|---:|
| Mean Bicubic PSNR | 22.7294 dB |
| Mean SRDnCNN PSNR | 28.0301 dB |
| Mean PSNR gain | +5.3007 dB |
| Median PSNR gain | +4.1676 dB |
| Mean SSIM | 0.9440 |
| Mean SSIM gain | +0.05885 |
| Validation samples improved over Bicubic | 320 / 320 (100%) |
| Validation samples worse than Bicubic | 0 / 320 (0%) |

## Improvement distribution

| PSNR gain over Bicubic | Samples | Percentage |
|---|---:|---:|
| ≥ 6 dB | 113 | 35.31% |
| 3–6 dB | 106 | 33.12% |
| 1–3 dB | 90 | 28.12% |
| 0–1 dB | 11 | 3.44% |
| < 0 dB | 0 | 0.00% |

## Phase 8

Phase 8 evaluated all 320 validation samples and included:

- per-image PSNR/SSIM evaluation
- Bicubic baseline comparison
- degradation-severity analysis
- PSNR-gain correlations
- best/moderate/low-performing visual comparison panels
- objective failure classification
- final visual evaluation report

The main limitation identified was fine/high-frequency texture reconstruction under difficult degradation.

## Repository artifact policy

The repository keeps experiment metadata, metrics, reports, and selected visualizations under version control. Large binary artifacts such as model checkpoints and prediction arrays should be stored using Git LFS or an external experiment archive rather than ordinary Git blobs.

The complete local archive is `EXP2_RESULTS_FINAL.zip` and contains the complete 360-file Exp2 result package.
