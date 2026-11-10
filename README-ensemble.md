# R7 ensemble packaging add-on

This is an add-on for the existing `Deepfake-ML` repository. Replace the current `inference_script.py` and add `scripts/pack_ensemble.py`.

## Kaggle commands (run from `/kaggle/working/Deepfake-ML`)

```bash
python scripts/pack_ensemble.py \
  --r4 runs/r4-resnet18-higher-lr/best_model.pt \
  --r6 runs/r6-efficientnet-b0/best_model.pt \
  --output models/ensemble-r4-r6.pt

python inference_script.py \
  --test_dir /path/to/hackathon_dataset/test/images \
  --weights models/ensemble-r4-r6.pt \
  --output submissions/predictions.csv

python scripts/validate_submission.py \
  --predictions submissions/predictions.csv \
  --test_dir /path/to/hackathon_dataset/test/images
```

**Do not commit** the bundle, raw test images, or generated private-test predictions to GitHub. Keep the archive and predictions for the required Drive submission.

The bundle stores the full state dictionaries (no inference-time download) and uses 50/50 arithmetic mean of **probabilities**, not logits or predicted labels. The single-file `--weights` CLI satisfies the organizers' expected interface. The script also accepts a single training checkpoint for fallback.

Training-time claims are different from the inference speed/memory values reported by the script. Record both single-model and ensemble runtime on an NVIDIA T4, with identical batch size and test dataset.

Review/test the script thoroughly before submitting; adhere to organizer originality rules. This add-on contains no training images, test images, trained weights or predictions.
