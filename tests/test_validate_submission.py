import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_submission import validate_submission  # noqa: E402


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.test_dir = self.root / "images"
        self.test_dir.mkdir()
        for name in ("a", "b"):
            (self.test_dir / f"{name}.jpg").touch()
        self.csv_path = self.root / "predictions.csv"

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, rows):
        with self.csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_id", "label", "confidence"])
            writer.writerows(rows)

    def test_valid_file(self):
        self.write_csv([["a", 0, 0.1], ["b", 1, 0.9]])
        self.assertEqual(validate_submission(self.csv_path, self.test_dir)["rows"], 2)

    def test_missing_image(self):
        self.write_csv([["a", 0, 0.1]])
        with self.assertRaisesRegex(ValueError, "don't match"):
            validate_submission(self.csv_path, self.test_dir)

    def test_duplicate_image(self):
        self.write_csv([["a", 0, 0.1], ["a", 1, 0.9]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_submission(self.csv_path, self.test_dir)

    def test_invalid_probability(self):
        self.write_csv([["a", 0, 0.1], ["b", 1, float("nan")]])
        with self.assertRaisesRegex(ValueError, "confidence"):
            validate_submission(self.csv_path, self.test_dir)


if __name__ == "__main__":
    unittest.main()
