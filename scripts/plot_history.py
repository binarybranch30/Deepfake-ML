"""Create two independent figures for the technical report from history.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.history)
    for col in ("epoch", "train_loss", "val_auc"):
        if col not in df:
            raise ValueError(f"Missing {col} in {args.history}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for col, ylabel, filename in (
        ("train_loss", "Training BCE loss", "training_loss.png"),
        ("val_auc", "Validation ROC-AUC", "validation_auc.png"),
    ):
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df[col], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.xticks(df["epoch"].tolist())
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(args.out_dir / filename, dpi=180)
        plt.close()
    print("Saved training_loss.png and validation_auc.png into", args.out_dir)


if __name__ == "__main__":
    main()
