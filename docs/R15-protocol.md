# R15 — ResNet18 label smoothing (preregistered experiment)

## Hypothesis
R13's near-zero training loss and confident joint errors suggest that training-target overconfidence is worth testing. Applying binary label smoothing with epsilon=0.05 may improve unseen-image ranking and/or ensemble diversity. This is only a hypothesis, not an established cause.

## One-variable comparison
Baseline: R13 (`resnet18`, seed 123, backbone_lr 1e-4, head_lr 1e-4, AdamW, 10 epochs, batch 32, unchanged split and augmentation, patience 3).
R15: Identical configuration, with `--label_smoothing 0.05` the only change.
Target transformation is `y_smooth = y*(1-epsilon) + 0.5*epsilon`. For epsilon 0.05: real 0 -> 0.025, fake 1 -> 0.975. Only training targets are smoothed, not validation labels.

## Kaggle command
```
python -m training.train --dataset_dir "$DATASET_DIR" --split_dir data/splits --output_dir runs/r15-resnet18-smoothing --epochs 10 --batch_size 32 --backbone_lr 0.0001 --seed 123 --label_smoothing 0.05
```

## Comparison
Same val.csv and SHA256 split hashes. Compare R15 alone to R13, then prespecified new ensembles:
- Existing: 0.25 R4 + 0.25 R13 + 0.50 R6
- Replace: 0.25 R4 + 0.25 R15 + 0.50 R6
- Four-model: 0.25 R4 + 0.125 R13 + 0.125 R15 + 0.50 R6

Do not edit the existing tested three-model bundle. Keep test images out of training and model selection. All outcomes (including negative results) go into project memory and later technical report.
