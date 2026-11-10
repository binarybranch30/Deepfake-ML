# Training guide — R3 reproducible ResNet18

This add-on uses the organizer's `train_metadata.csv` mapping rather than `ImageFolder` (which reverses the organizer's `real=0`, `fake=1` label convention).

## Kaggle commands from repository root

```python
# Re-sync code after local GitHub push (with the Kaggle notebook cwd already in repo)
!git pull --ff-only origin main

# If already run, the following split need not be regenerated.
!python scripts/check_dataset.py --dataset_dir "{DATASET_DIR}"
!python scripts/create_split.py --dataset_dir "{DATASET_DIR}" --output_dir data/splits --seed 42

# New run: never reuse an existing non-empty output directory.
!python -m training.train --dataset_dir "{DATASET_DIR}" --split_dir data/splits --output_dir runs/r3-resnet18-adamw --epochs 10 --batch_size 32
```

Make sure `DATASET_DIR` is defined in the Kaggle notebook as in the earlier setup cell; in Jupyter, `{DATASET_DIR}` interpolates the Python variable in `!` shell commands.

If Kaggle cannot fetch ImageNet pretrained weights, turn on notebook Internet (if permitted), or add the exact torchvision ImageNet weight file as a Kaggle input and use an offline cache. **Never substitute deepfake-specific pretrained weights.**

## Files produced in `runs/r3-resnet18-adamw/`

| File | Use |
|---|---|
| `config.json` | Full configuration, environment and split fingerprints |
| `history.csv` | Epoch-wise training loss, validation AUC and F1 |
| `summary.json` | Best validation epoch/AUC and elapsed time |
| `best_model.pt` | Model state dict and inference parameters |
| `best_val_predictions.csv` | Per-image validation probabilities for comparing experiments |

These run files are Git-ignored. Save a Kaggle notebook version (or download the run folder), then write verified summary values into the tracked `docs/experiment-log.md` locally and push the documentation changes.

## Confirm it runs for the judges

```python
!python inference_script.py --test_dir "{DATASET_DIR}/test/images" --weights runs/r3-resnet18-adamw/best_model.pt --output submissions/predictions.csv
!python scripts/validate_submission.py --test_dir "{DATASET_DIR}/test/images" --predictions submissions/predictions.csv
```

Do not upload `predictions.csv` or any checkpoint to GitHub; submit the checkpoint and predictions via the required Google Drive folder. The inference script must be shipped with the model weights and dependencies. Verify it from a clean notebook environment without pretrained ImageNet downloads (inference builds weights=None).

## Important comparison caveat

Historical AUC 0.8924 came from a previous notebook without the original complete training code or split manifest. This run uses the fixed metadata-derived split from `scripts/create_split.py`. AUC differences from the historical number may reflect different validation examples, not necessarily a better or worse model. Always compare new experiments on the *same* saved split.

## Create plots for the report

```python
!python scripts/plot_history.py --history runs/r3-resnet18-adamw/history.csv --out_dir runs/r3-resnet18-adamw/figures
```

The output PNGs are not automatically published in GitHub because `runs/` is ignored; download or copy approved figures and include them in the technical report.
