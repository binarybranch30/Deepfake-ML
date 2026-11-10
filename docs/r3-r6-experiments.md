# Controlled experiment record — ResNet18 to EfficientNet-B0

Dataset: competition-provided 7,000 training images. Fixed, stratified train/validation split: 80/20 with seed 42 (assuming the split was not regenerated with different settings). Metric: ROC-AUC from probability of `fake=1`.

| Run | Model | Backbone LR | Best epoch | Best validation AUC | Notes |
|---|---|---:|---:|---:|---|
| R3 | ImageNet ResNet18 | 1e-5 | 9 | 0.8649323243 | Improved through epoch 9 |
| R4 | ImageNet ResNet18 | 1e-4 | 2 | **0.8845697002** | Early stopping at epoch 5; selected checkpoint |
| R5 | ImageNet ResNet18 | 5e-5 | 2 | 0.8768000588 | Early stopping at epoch 5; did not beat R4 |
| R6 | ImageNet EfficientNet-B0 | 1e-4 | pending | pending | New architecture; same split and augmentation |

Reported historical R1 best 0.8924 is on an unverified historical split. Do not compare it as a controlled experiment with R3–R6.

The train-loss decline accompanied by non-improving validation AUC in R4/R5 is consistent with overfitting or unstable fine-tuning; the available metrics do not prove which.

## R6 on Kaggle (from repository root)

```python
!python -m training.train \
    --arch efficientnet_b0 \
    --dataset_dir "{DATASET_DIR}" \
    --split_dir data/splits \
    --output_dir runs/r6-efficientnet-b0 \
    --epochs 10 \
    --batch_size 32 \
    --backbone_lr 0.0001
```

For a GPU out-of-memory error, rerun from scratch with a **new** output directory and `--batch_size 16`; document the changed batch size. Preserve R4's `best_model.pt` as the current fallback. Both training and standalone inference now support `resnet18` and `efficientnet_b0`, selected automatically at inference from the checkpoint's `arch` field.

## Inference (once R6 is finished)

```python
!python inference_script.py \
    --test_dir "{DATASET_DIR}/test/images" \
    --weights runs/r6-efficientnet-b0/best_model.pt \
    --output submissions/predictions-r6.csv
```

Only select a final model after comparing validation AUC and test-time efficiency; do not fit or tune on competition test images. Keep run outputs out of Git, but preserve them in Kaggle output or another durable location for the final submission.
