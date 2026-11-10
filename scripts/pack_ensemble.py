"""Package two independently trained checkpoints into one portable model file.

Example:
  python scripts/pack_ensemble.py \
    --r4 runs/r4-resnet18-higher-lr/best_model.pt \
    --r6 runs/r6-efficientnet-b0/best_model.pt \
    --output models/ensemble-r4-r6.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_component(path: Path, expected_arch: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint at {path}")
    arch = checkpoint.get("arch", "resnet18")
    if arch != expected_arch:
        raise ValueError(f"Checkpoint at {path} arch={arch}, expected={expected_arch}")
    state = checkpoint.get("state_dict", checkpoint)
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Checkpoint state_dict at {path} contains unexpected values")
    return {"arch": arch, "state_dict": state, "weight": 0.5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4", type=Path, required=True)
    parser.add_argument("--r6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = {
        "bundle_version": 1,
        "ensemble": [load_component(args.r4, "resnet18"),
                     load_component(args.r6, "efficientnet_b0")],
        "threshold": 0.5,
        "image_size": 224,
        "label_0": "real",
        "label_1": "fake",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output)
    print(f"Wrote ensemble bundle to {args.output} ({args.output.stat().st_size / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
