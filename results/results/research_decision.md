Research Decision — Post-Experiment Review

AI-Based Restoration of Degraded Images for Semiconductor Inspection

Project status: Post-Exp3 / Phase 8 reviewPrimary model: SRDnCNNValidation set: 320 imagesRandom seed: 42Completed experiments: Exp1, Exp2, Exp3Phase 8: Evidence-based failure analysis completedOOD dataset: Generated, quantitative frozen-model evaluation not established

1. Current Experimental Position

The project has completed three controlled experiments:

Experiment 1 — SRDnCNN baseline

Experiment 2 — L1Loss ablation

Experiment 3 — Synthetic degradation training

Phase 8 failure analysis has also been completed using the preserved experimental records.

An optional OOD robustness dataset has been generated, but a quantitative OOD evaluation of the frozen models has not been established because the original experiment checkpoints were not recovered from the Kaggle runtime.

2. What Has Been Established

2.1 Learned Restoration Outperforms Bicubic

The restoration model substantially outperforms Bicubic interpolation on the evaluated validation set.

For Exp3:

Metric

Exp3

Bicubic

PSNR

27.899917 dB

22.729405 dB

SSIM

0.739837

0.541810

MAE

0.031635

0.057429

Improvement relative to Bicubic:

PSNR improvement : +5.170512 dB
SSIM improvement : +0.198027
MAE reduction    : +0.025794

Therefore, learned restoration is substantially more effective than simple Bicubic interpolation on the evaluated validation distribution.

3. Current Best Frozen Model

Among the frozen Exp1 and Exp2 results:

Metric

Exp1

Exp2

PSNR

28.0357 dB

28.0315 dB

SSIM

0.749338

0.749582

LPIPS

0.324483

0.319640

Exp2 has:

Slightly higher SSIM

Lower LPIPS

197/320 SSIM wins

290/320 LPIPS wins

PSNR is nearly identical between the two experiments.

Current preferred frozen model

Exp2

This preference is based on the combined frozen validation evidence rather than PSNR alone.

4. What Exp3 Tells Us

Exp3 tested a 50/50 mixture of original and synthetically degraded training data.

The synthetic degradation used:

Speckle

Gaussian noise

Downsampling

Exp3 achieved:

PSNR : 27.899917 dB
SSIM : 0.739837
MAE  : 0.031635
LPIPS: 0.338273

Compared with Exp2:

PSNR : -0.131583 dB
SSIM : -0.009745
MAE  : +0.000548
LPIPS: +0.018633

Therefore, the tested synthetic-degradation configuration did not improve performance on the original validation distribution.

Important interpretation

This does not prove that synthetic degradation is ineffective.

It only shows that the tested 50/50 configuration did not provide an improvement on the original validation distribution.

5. Known Failure Modes

Phase 8 identified the following documented failure characteristics.

F1 — Over-smoothing

Difficult reconstructions can lose fine image detail.

F2 — Texture Reconstruction Mismatch

Some difficult cases show substantial texture/intensity differences.

F3 — Fine-Structure Loss

Fine structural information can be lost during restoration.

F4 — Metric Disagreement

An image can behave differently according to PSNR, SSIM and LPIPS.

Index 272 is a documented example.

F5 — Robustness Uncertainty

The current frozen experiments do not establish how well the model performs when degradation severity or degradation ordering changes.

6. What Has NOT Been Established

The current experiments do not establish:

Statistical significance

Multi-seed reproducibility

Complete frozen Exp3 per-image failure ranking

Exact frozen visual reproduction

Quantitative OOD robustness of the frozen models

General superiority of synthetic degradation

General superiority of Exp2 beyond the tested validation distribution

These remain limitations rather than failed experiments.

7. OOD Dataset Status

An OOD dataset has already been generated.

Source GT images : 100
Conditions       : 4
Generated inputs : 400
Seed             : 42

Conditions:

moderate
strong
alternate_order
mixed

The dataset uses variations of the documented mechanisms:

Speckle

Gaussian noise

Downsampling

The OOD definition represents variation in severity and/or ordering of the documented KLA degradation mechanisms. It does not introduce a new degradation mechanism.

The OOD dataset generation and verification passed.

Therefore:

OOD dataset generation : COMPLETED
OOD model evaluation   : NOT ESTABLISHED

8. Research Gap

The current research gap is:

The restoration model performs well on the original validation distribution, but its robustness to controlled variations in degradation severity and ordering has not been quantitatively established.

A second related gap is:

The tested synthetic-degradation training configuration did not improve original-distribution performance, so a more controlled robustness strategy is required rather than simply adding more synthetic degradation.

9. Decision Before Exp4

A new experiment should only be conducted if it answers a specific research question.

The proposed research question is:

Can robustness to controlled degradation variations be improved while preserving performance on the original validation distribution?

This is a stronger research question than simply asking whether another model can obtain a higher PSNR.

10. Proposed Exp4 Direction

If Exp4 is conducted, it should use the existing SRDnCNN pipeline rather than changing several variables simultaneously.

The experiment should focus on controlled robustness training.

Conceptually:

Original training distribution
             +
Controlled robustness samples
             ↓
           SRDnCNN
             ↓
      ┌──────┴──────┐
      ↓             ↓
Original Val      OOD Test

This allows us to determine whether robustness improves without sacrificing normal-distribution performance.

11. Exp4 Success Criteria

Exp4 should not be considered successful merely because OOD performance improves.

It should satisfy both:

Primary criterion

Improve robustness on the controlled OOD dataset.

Preservation criterion

Do not cause a meaningful degradation on the original validation distribution.

Therefore, Exp4 should report:

Original validation

PSNR

SSIM

MAE

LPIPS

OOD evaluation

PSNR

SSIM

MAE

LPIPS

The comparison should include the current frozen Exp2 reference where the corresponding evaluation is available.

12. Experimental Control

If Exp4 is performed, keep fixed:

Dataset split

Validation set

Random seed

Model architecture

Optimizer

Evaluation protocol

Change only the intended robustness-training component.

This is necessary to make the experiment scientifically interpretable.

13. Artifact Preservation Requirement

The previous Kaggle workflow demonstrated the risk of losing runtime-generated artifacts.

For Exp4, persistent storage must be prepared before training begins.

Recommended structure:

results/
└── experiment4/
    ├── checkpoint/
    ├── metrics/
    ├── per_image/
    ├── outputs/
    ├── logs/
    └── config/

At minimum preserve:

best_model.pth
training_log.csv
validation_metrics.csv
per_image_metrics.csv
experiment_config.yaml
experiment_summary.md

Also preserve:

Test outputs

OOD outputs

OOD manifest

LPIPS results

Random seed

Hardware/software information

Training configuration

All generated artifacts must be copied to persistent storage before the Colab/Kaggle runtime ends.

14. Storage Strategy for Future Experiments

GitHub

Use GitHub for:

Source code

Configuration

Small CSV/JSON/Markdown research records

Documentation

Experiment scripts

Do not rely on GitHub as the primary storage for large model checkpoints or large image-output directories.

Persistent Drive

Use Google Drive or another persistent storage location for:

Model checkpoints

Large arrays

Generated images

Large experiment artifacts

Training logs

The project repository should contain references to the persistent artifact locations.

15. Decision

Current recommendation

Do not immediately start Exp4.

First confirm that the final submission actually benefits from an additional robustness experiment.

If robustness is an important evaluation requirement, proceed with Exp4.

If the existing evidence is sufficient for the submission, finalize the current study rather than adding another experiment without a clear requirement.

16. Current Research State

Baseline restoration             : ESTABLISHED
Loss ablation                    : ESTABLISHED
Synthetic degradation experiment : ESTABLISHED
Failure analysis                 : ESTABLISHED
OOD dataset                      : AVAILABLE
OOD quantitative evaluation      : OPEN
Robustness improvement           : OPEN
Current preferred frozen model   : Exp2

17. Final Research Question

If the project proceeds to another experiment, it should answer:

Can the restoration model become more robust to controlled degradation variations without sacrificing its performance on the original semiconductor inspection image distribution?

This question follows directly from the observed experimental results and Phase 8 failure analysis.

18. Recommended Next Action

Before using the T4 for another training run:

Confirm whether the KLA submission requires quantitative robustness/OOD evidence.

Confirm whether the current Exp1–Exp3 evidence is sufficient.

If another experiment is required, freeze the Exp2 configuration as the reference.

Define the Exp4 robustness intervention before training.

Prepare persistent artifact storage.

Run one controlled experiment.

Evaluate both original validation and OOD data.

Preserve all checkpoints, per-image metrics and outputs immediately.

END OF RESEARCH DECISION