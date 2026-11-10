"""Inspect provided training metadata without modifying competition data."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def check_dataset(dataset_dir: Path) -> None:
    metadata = dataset_dir / "train_metadata.csv"
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test" / "images"
    required = [metadata, train_dir / "real", train_dir / "fake", test_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n" + "\n".join(missing))

    with metadata.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"image_id", "label"}.issubset(rows[0]):
        raise ValueError("Expected train_metadata.csv columns: image_id,label")

    labels = Counter()
    metadata_ids = set()
    for row in rows:
        image_id = row["image_id"].removesuffix(".jpg")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Unexpected label {label}: {image_id}")
        if image_id in metadata_ids:
            raise ValueError(f"Duplicate image_id in metadata: {image_id}")
        metadata_ids.add(image_id)
        labels[label] += 1
        target = train_dir / ("fake" if label == 1 else "real") / f"{image_id}.jpg"
        if not target.is_file():
            raise FileNotFoundError(f"Label or image mismatch: {target}")

    train_images = {p.stem for folder in ("real", "fake") for p in (train_dir / folder).glob("*.jpg")}
    test_images = {p.stem for p in test_dir.glob("*.jpg")}
    print(f"Train metadata rows: {len(rows)}")
    print(f"Real label=0: {labels[0]}; Fake label=1: {labels[1]}")
    print(f"Training JPGs: {len(train_images)}; Test JPGs: {len(test_images)}")
    if metadata_ids != train_images:
        raise ValueError(f"Metadata and train images differ: {len(metadata_ids ^ train_images)} IDs")
    if metadata_ids & test_images:
        raise ValueError("Train/test image IDs overlap")
    print("OK: folder paths, ID sets, and label mapping are consistent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    args = parser.parse_args()
    check_dataset(args.dataset_dir)
