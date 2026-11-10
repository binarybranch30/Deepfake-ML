# Experiment log

**Measurement standard:** Always use the same 80/20 stratified split from metadata (`seed=42`) after rebuilding the training pipeline. Record the exact validation image IDs, confidence outputs, epoch-wise AUC, F1/threshold, runtime, and environment for reproducibility.

## Historical experiments (reported in the uploaded Kaggle notebook)

> These are **reported** results. The exported notebook contains an experiment log and two evaluation cells, **not the full original training loop**. The exact historical split cannot be reconstructed from the export alone; a new stratified split may not match those scores exactly.

### Run 1 — 6 Nov 2026

- Architecture: ImageNet-pretrained ResNet18, single-logit head
- BCEWithLogitsLoss; Adam, LR `1e-4`; batch size 32
- Augmentation: horizontal flip, rotation ~10°, color jitter ~0.2, Gaussian blur p=0.3
- Training loss by epoch: `0.5349`, `0.4115`, `0.3434`, `0.2955`, `0.2393`
- Validation AUC by epoch: `0.8630`, `0.8924`, `0.8853`, `0.8848`, `0.8651`
- Best reported validation AUC: **0.8924 at epoch 2**
- Validation F1 at threshold 0.5 (epoch 2): **0.8490**
- Best validation F1 threshold reported: `0.38`, yielding F1 `0.8523`
- Additional JPEG-40 + blur evaluation: clean AUC `0.8924`; degraded AUC `0.8881`

### Run 2 — available summary only

- Clean validation AUC: `0.8859`
- JPEG-40 + blur validation AUC: `0.8833`
- The exported notebook mentions testing stronger augmentation, but complete code and training curves for Run 2 are absent. Do not claim verified augmentation details.

## Future experiments

Copy this table row for each controlled experiment. Change one major factor at a time and save the best AUC checkpoint.

| Run | Git commit | Model | Split hash/seed | Augmentation | Optimizer / LR | Best val AUC | F1 | Degraded AUC | Inference time | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R3 | TBD | ResNet18 | seed=42 | TBD | AdamW / TBD | TBD | TBD | TBD | TBD | TBD |

Never treat any validation score as a private test-set score. The organizer does not provide a public leaderboard.
