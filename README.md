# Deepfake Detection — IIIT Bangalore ML Hackathon

A reproducible PyTorch project for binary **real vs deepfake** face-image detection.

> **Status:** Project setup / baseline reconstruction. Existing ResNet18 experiments are documented, but the original full training code was not included in the exported notebook. Training code and final model weights are not yet part of this starter.

## Goal and evaluation

- Class `0` = **real**; class `1` = **fake**.
- The `confidence` submitted for each image is **P(fake)** (not the predicted label or P(real)).
- Private test AUC-ROC: **50 points**; inference efficiency: **20**; robustness: **20**; documentation: **10**.
- Runtime: **Python 3.10**, PyTorch 2.x; GPU evaluation on NVIDIA T4; model under **5 GB**, full inference under **60 minutes**.
- Only the supplied dataset may be used for fine-tuning. ImageNet-pretrained architectures are permitted; external deepfake datasets and pretrained deepfake detectors are not.

The official README also requires a **Google Drive submission folder** containing `predictions.csv`, weights, `inference_script.py`, dependencies, source code (GitHub link or archive), and a technical report (minimum 4 pages). A GitHub repository alone is not a full submission.

## Project layout

```text
deepfake-hackathon/
├── README.md
├── requirements.txt
├── .gitignore
├── .github/workflows/checks.yml
├── docs/
│   ├── competition-notes.md
│   ├── experiment-log.md
│   ├── report-outline.md
│   └── submission-checklist.md
├── data/
│   ├── README.md
│   └── splits/                # Generated locally, not committed
├── notebooks/                # New tracked Kaggle notebooks
├── models/                   # Local checkpoints (ignored)
├── submissions/              # Local outputs (ignored)
├── scripts/
│   ├── check_dataset.py
│   ├── create_split.py
│   └── validate_submission.py
└── tests/
    └── test_validate_submission.py
```

## First-time setup (local Windows PowerShell)

```powershell
# Work in the extracted starter directory.
git init
git branch -M main
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GitHub, create an **empty** repository named `deepfake-hackathon` (skip the auto-generated README and .gitignore), then:

```powershell
git add .
git commit -m "chore: scaffold reproducible ML project"
git remote add origin git@github.com:YOUR_USERNAME/deepfake-hackathon.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username. If using an HTTPS remote, replace the SSH remote accordingly. Do not commit the competition images, test predictions, or trained weights.

## Dataset location

Extract the **inner** `hackathon_dataset.zip` from the organizer package so the resulting directory is:

```text
data/hackathon_dataset/
├── train/real/
├── train/fake/
├── test/images/
└── train_metadata.csv
```

You may also point scripts directly at the dataset in Kaggle, such as:
`/kaggle/input/datasets/binarybranch/dataset/hackathon_dataset`

```bash
python scripts/check_dataset.py --dataset_dir data/hackathon_dataset
python scripts/create_split.py --dataset_dir data/hackathon_dataset --output_dir data/splits --seed 42
```

The split is stratified 80/20. Its manifests are local-only. **Keep the same split and seed across experiments**; do not use the supplied unlabeled test images for training or hyperparameter tuning.

### Important label trap

`torchvision.datasets.ImageFolder` sorts class names, making `fake=0` and `real=1` for `train/fake` and `train/real` folders. This is the **opposite** of the competition convention. Prefer the `train_metadata.csv` labels in all training/validation code, or explicitly remap ImageFolder targets. Always compute AUC using P(fake).

## Kaggle workflow

1. Keep the authoritative source code and documentation in GitHub.
2. Attach the provided competition dataset privately to the Kaggle notebook.
3. Clone the repository in Kaggle (public repo via HTTPS, or use an authorized secure method for a private repo).
4. Point scripts at the Kaggle read-only dataset path. Store training outputs in `/kaggle/working/`.
5. Save each run's configuration, seed, validation AUC/F1, training curves, runtime, and checkpoint path in `docs/experiment-log.md`.
6. Download selected checkpoints/predictions for the final Google Drive submission; do **not** commit them into Git.
7. Implement and smoke-test `inference_script.py` **after** reconstructing the model training pipeline. It must accept `--test_dir`, `--weights`, and `--output` without hardcoded Kaggle paths.

For code syncing, do not paste GitHub personal access tokens into notebooks or commit notebook outputs containing secrets.

## Validate the final predictions

```bash
python scripts/validate_submission.py \
  --test_dir data/hackathon_dataset/test/images \
  --predictions submissions/predictions.csv
```

This checks exact headers, one row per test image, valid UUID-like image IDs as filenames (actual IDs are compared directly), no duplicates, labels in {0,1}, and confidence in [0,1]. It does **not** establish model accuracy or verify that the inference code reproduces the CSV.

Run lightweight project tests without a GPU:

```bash
python -m unittest discover -s tests -v
```

## Existing baseline

An earlier Kaggle notebook reported a ResNet18 run that peaked at **0.8924 local validation AUC (epoch 2)**, and a second run with **0.8859**. These are *reported local validation results*, not the private competition score. See [experiment log](docs/experiment-log.md). The complete original training cells were not supplied, so this starter does not claim to reproduce those results yet.

## Attribution and provenance

Competition requirements summarized from the organizer's participant README and evaluation guide. This repository starter does not redistribute the supplied dataset. Review any assistance or external dependencies for compliance with the organizer's code-originality rules.
