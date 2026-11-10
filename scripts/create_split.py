"""Make a reproducible stratified train/validation split from organizer metadata."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_split(dataset_dir: Path, output_dir: Path, seed: int, val_fraction: float) -> None:
    metadata_file = dataset_dir / "train_metadata.csv"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Missing organizer metadata: {metadata_file}")
    df = pd.read_csv(metadata_file, dtype={"image_id": str})
    if not {"image_id", "label"}.issubset(df.columns):
        raise ValueError("Expected metadata columns image_id and label")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in training metadata")
    if set(df["label"].dropna().unique()) != {0, 1}:
        raise ValueError("Expected class labels exactly {0,1}")
    if df["label"].isna().any():
        raise ValueError("Missing labels")
    df = df.sort_values("image_id").reset_index(drop=True)
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed, stratify=df["label"]
    )
    train_df = train_df.sort_values("image_id")
    val_df = val_df.sort_values("image_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  seed={seed}")
    print("Class counts train:", train_df["label"].value_counts().sort_index().to_dict())
    print("Class counts val:", val_df["label"].value_counts().sort_index().to_dict())
    print(f"Saved local manifests to {output_dir.resolve()} (excluded from Git)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    args = parser.parse_args()
    create_split(args.dataset_dir, args.output_dir, args.seed, args.val_fraction)
