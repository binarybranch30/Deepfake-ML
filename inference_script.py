"""Standalone deepfake inference for single model or 50/50 model bundle.

Required competition CLI:
  python inference_script.py --test_dir PATH --weights BUNDLE.pt --output predictions.csv

Output labels: 0 = real, 1 = fake; confidence = P(fake).
No training or external downloads occur during inference.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class TestFaces(Dataset):
    def __init__(self, directory: Path):
        self.paths = sorted(
            (p for p in Path(directory).iterdir()
             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            key=lambda p: p.name,
        )
        if not self.paths:
            raise ValueError(f"No test images found in {directory}")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as im:
            pixels = self.transform(im.convert("RGB"))
        return pixels, path.stem


def build_model(arch: str):
    # weights=None is essential: judges must not need internet at inference time.
    if arch == "resnet18":
        network = models.resnet18(weights=None)
        network.fc = nn.Linear(network.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        network = models.efficientnet_b0(weights=None)
        network.classifier[-1] = nn.Linear(network.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")
    return network


def load_models(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint")
    if "ensemble" in checkpoint:
        parts = checkpoint["ensemble"]
        threshold = float(checkpoint.get("threshold", 0.5))
        if len(parts) < 1:
            raise ValueError("Empty ensemble")
        if any(not isinstance(p, dict) or "state_dict" not in p or "arch" not in p for p in parts):
            raise ValueError("Invalid ensemble component")
    elif "state_dict" in checkpoint:
        parts = [{"arch": checkpoint.get("arch", "resnet18"),
                  "state_dict": checkpoint["state_dict"], "weight": 1.0}]
        threshold = float(checkpoint.get("threshold", 0.5))
    else:
        # Legacy standalone ResNet18 state_dict support.
        parts = [{"arch": "resnet18", "state_dict": checkpoint, "weight": 1.0}]
        threshold = 0.5

    weights = [float(part.get("weight", 1.0)) for part in parts]
    if any(w < 0 or not (w < float("inf")) for w in weights) or sum(weights) <= 0:
        raise ValueError("Ensemble weights must be nonnegative, finite, nonzero in total")
    weights = [w / sum(weights) for w in weights]
    loaded = []
    for part, weight in zip(parts, weights):
        net = build_model(str(part["arch"]))
        net.load_state_dict(part["state_dict"], strict=True)
        loaded.append((net.to(device).eval(), weight, str(part["arch"])))
    return loaded, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    if not args.test_dir.is_dir() or not args.weights.is_file():
        parser.error("--test_dir must exist and --weights must be an existing file")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch_size must be positive and --num_workers nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    networks, threshold = load_models(args.weights, device)
    dataset = TestFaces(args.test_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "confidence"])
        with torch.inference_mode():
            for images, ids in loader:
                images = images.to(device, non_blocking=True)
                total_prob = None
                for model, weight, _ in networks:
                    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                        logits = model(images).flatten()
                    part_prob = torch.sigmoid(logits.float()) * weight
                    total_prob = part_prob if total_prob is None else total_prob + part_prob
                for image_id, prob in zip(ids, total_prob.cpu().tolist()):
                    writer.writerow([image_id, int(prob >= threshold), f"{prob:.9f}"])
                    total += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else None
    print("Inference models:", [(arch, w) for _, w, arch in networks])
    print(f"Wrote {total} rows to {args.output} in {elapsed:.2f} s on {device}")
    print(f"Speed: {total / elapsed:.1f} images/s | peak GPU allocated: "
          f"{f'{mem:.1f} MiB' if mem is not None else 'N/A'}")


if __name__ == "__main__":
    main()
