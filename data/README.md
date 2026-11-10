# Dataset directory

Do **not** commit competition image data or generated train/validation split CSVs.

Expected extracted path: `data/hackathon_dataset/` containing `train/real`, `train/fake`, `test/images`, and `train_metadata.csv`.

The source archive supplied by the organizer contains a second, nested `hackathon_dataset.zip`; extract that inner ZIP to obtain this directory.

Use `scripts/check_dataset.py` to confirm folder structure, label conventions, image counts, and metadata IDs. Use `scripts/create_split.py` for a reproducible stratified split.
