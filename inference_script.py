"""Standalone competition inference script (requires only torch, torchvision, Pillow).

Run:
    python inference_script.py --test_dir /path/to/test/images \
        --weights /path/to/best_model.pt --output predictions.csv

Always outputs image_id,label,confidence where confidence is P(fake).
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class TestFaces(Dataset):
    def __init__(self, folder):
        self.paths = sorted(path for path in Path(folder).iterdir()
                            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.paths:
            raise ValueError(f"No supported image files found in {folder}")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, path.stem


def build_model(arch: str):
    if arch == "resnet18":
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {arch}")
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch_size must be greater than zero")
    if not args.test_dir.is_dir():
        parser.error(f"Not a directory: {args.test_dir}")
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=True avoids the unsafe full-model pickle fallback.
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a weights dict or training/train.py checkpoint")
    if "state_dict" in checkpoint:
        arch = checkpoint.get("arch", "resnet18")
        state_dict = checkpoint["state_dict"]
        threshold = float(checkpoint.get("threshold", 0.5))
    else:
        # Backward compatibility with standalone ResNet18 state dictionaries.
        arch = "resnet18"
        state_dict = checkpoint
        threshold = 0.5
    model = build_model(arch)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    dataset = TestFaces(args.test_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image_id", "label", "confidence"])
        total = 0
        with torch.inference_mode():
            for images, ids in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(images).flatten()
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                for image_id, prob in zip(ids, probs):
                    writer.writerow([image_id, int(prob >= threshold), prob])
                    total += 1
    print(f"Wrote {total} predictions to {args.output} (device={device}, threshold={threshold}).")


if __name__ == "__main__":
    main()
