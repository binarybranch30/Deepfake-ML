"""Fine-tune ImageNet-pretrained ResNet18 or EfficientNet-B0 on organizer data.

Usage from repository root:
    python -m training.train --dataset_dir /kaggle/input/.../hackathon_dataset \
        --split_dir data/splits --output_dir runs/r3-resnet18-adamw

Labels: 0 = real; 1 = fake. Metadata-derived labels are used explicitly rather
than torchvision ImageFolder's alphabetical folder labeling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LabeledFaces(Dataset):
    def __init__(self, dataset_dir: Path, manifest: Path, transform):
        self.dataset_dir = Path(dataset_dir)
        self.rows = pd.read_csv(manifest, dtype={"image_id": str})
        if not {"image_id", "label"}.issubset(self.rows.columns):
            raise ValueError(f"Expected image_id,label columns in {manifest}")
        if self.rows.empty or self.rows["label"].isna().any():
            raise ValueError(f"Missing rows or labels in {manifest}")
        if set(self.rows["label"].unique()) - {0, 1}:
            raise ValueError("Manifest labels must be 0 (real) or 1 (fake)")
        self.transform = transform
        self.paths = []
        self.labels = []
        for row in self.rows.itertuples(index=False):
            image_id = str(row.image_id).removesuffix(".jpg")
            label = int(row.label)
            folder = "fake" if label == 1 else "real"
            path = self.dataset_dir / "train" / folder / f"{image_id}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")
            self.paths.append(path)
            self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            pixel_values = self.transform(image)
        return pixel_values, torch.tensor(self.labels[index], dtype=torch.float32)


def get_transforms(train: bool):
    parts = [transforms.Resize((224, 224))]
    if train:
        # Mild augmentation: avoid hiding subtle manipulation artifacts.
        parts += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    parts += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(parts)


def build_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return net


def evaluate(model, loader, device):
    model.eval()
    probabilities, truths = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images).flatten()
            probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
            truths.extend(labels.numpy().astype(int).tolist())
    auc = roc_auc_score(truths, probabilities)
    labels = [int(p >= 0.5) for p in probabilities]
    f1 = f1_score(truths, labels, zero_division=0)
    return float(auc), float(f1), truths, probabilities


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/r3-resnet18-adamw"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Binary target smoothing: 0 = disabled, 0.05 maps targets to 0.025/0.975")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch_size and patience must be positive")
    if not (0.0 <= args.label_smoothing < 1.0):
        parser.error("label_smoothing must be >= 0 and < 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir} (use a fresh run name)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.split_dir / "train.csv"
    val_path = args.split_dir / "val.csv"
    if not train_path.is_file() or not val_path.is_file():
        parser.error("Missing train.csv or val.csv. Run scripts/create_split.py first.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Torch: {torch.__version__}")

    train_ds = LabeledFaces(args.dataset_dir, train_path, get_transforms(train=True))
    val_ds = LabeledFaces(args.dataset_dir, val_path, get_transforms(train=False))
    if set(train_ds.rows.image_id) & set(val_ds.rows.image_id):
        raise ValueError("Train/validation image IDs overlap")
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              generator=train_generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    try:
        model = build_model(args.arch, pretrained=True).to(device)
    except Exception as error:
        raise RuntimeError(
            "Could not load ImageNet weights. Enable Kaggle Internet for the first download "
            "or attach the exact ImageNet pretrained checkpoint for your architecture. Do not replace it with "
            "deepfake-pretrained weights."
        ) from error
    criterion = nn.BCEWithLogitsLoss()
    head_prefix = "fc." if args.arch == "resnet18" else "classifier."
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for name, param in model.named_parameters()
                        if not name.startswith(head_prefix)], "lr": args.backbone_lr},
            {"params": [param for name, param in model.named_parameters()
                        if name.startswith(head_prefix)], "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({"model": args.arch, "pretrained": "ImageNet IMAGENET1K_V1",
                   "class_0": "real", "class_1": "fake", "selection_metric": "val_auc",
                   "image_size": 224, "git_commit": git_commit(),
                   "split_sha256": {"train": file_digest(train_path), "val": file_digest(val_path)},
                   "train_count": len(train_ds), "val_count": len(val_ds),
                   "torch_version": torch.__version__})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_auc = -1.0
    best_epoch = None
    stalled = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images).flatten()
                # Smooth TRAIN labels only; validation labels remain unchanged.
                train_targets = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                loss = criterion(logits, train_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)

        auc, f1, truths, probabilities = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_ds),
               "val_auc": auc, "val_f1_05": f1,
               "backbone_lr": optimizer.param_groups[0]["lr"],
               "head_lr": optimizer.param_groups[1]["lr"]}
        history.append(row)
        print(f"Epoch {epoch:02d} | train loss {row['train_loss']:.4f} | "
              f"val AUC {auc:.4f} | val F1 @0.5 {f1:.4f}", flush=True)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

        if auc > best_auc + 1e-6:
            best_auc, best_epoch, stalled = auc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                        "threshold": 0.5, "image_size": 224,
                        "best_val_auc": auc, "best_epoch": epoch},
                       args.output_dir / "best_model.pt")
            val_ids = [path.stem for path in val_ds.paths]
            pd.DataFrame({"image_id": val_ids, "label": truths,
                          "confidence": probabilities}).to_csv(
                              args.output_dir / "best_val_predictions.csv", index=False)
        else:
            stalled += 1
        scheduler.step()
        if stalled >= args.patience:
            print(f"Early stopping: no AUC improvement for {stalled} epochs.")
            break

    summary = {"best_epoch": best_epoch, "best_val_auc": best_auc,
               "elapsed_seconds": round(time.perf_counter() - start, 1),
               "epochs_completed": len(history),
               "checkpoint": str(args.output_dir / "best_model.pt")}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Best checkpoint:", summary)


if __name__ == "__main__":
    main()
