"""Check CSV schema and exact test image coverage before submitting."""

import argparse
import csv
import math
from pathlib import Path

REQUIRED_COLUMNS = ["image_id", "label", "confidence"]


def validate_submission(predictions: Path, test_dir: Path) -> dict:
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not predictions.is_file():
        raise FileNotFoundError(f"Missing predictions: {predictions}")

    expected_ids = {p.stem for p in test_dir.glob("*.jpg")}
    if not expected_ids:
        raise ValueError(f"No JPG test images found in {test_dir}")

    with predictions.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"CSV headers must be exactly {REQUIRED_COLUMNS}; got {reader.fieldnames}")
        seen = set()
        count_fake = 0
        for row_no, row in enumerate(reader, 2):
            image_id = row["image_id"]
            if image_id in seen:
                raise ValueError(f"Duplicate image_id on row {row_no}: {image_id}")
            seen.add(image_id)
            try:
                label = int(row["label"])
                confidence = float(row["confidence"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid label/confidence on row {row_no}") from error
            if row["label"] not in ("0", "1") or label not in (0, 1):
                raise ValueError(f"label must be 0 or 1 on row {row_no}")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be finite in [0, 1] on row {row_no}")
            count_fake += label

    if seen != expected_ids:
        missing = sorted(expected_ids - seen)
        unexpected = sorted(seen - expected_ids)
        raise ValueError(f"Test IDs don't match: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return {"rows": len(seen), "predicted_fake": count_fake, "predicted_real": len(seen) - count_fake}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test_dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_submission(args.predictions, args.test_dir)
    print("PASS — exact test coverage and valid columns:", result)
