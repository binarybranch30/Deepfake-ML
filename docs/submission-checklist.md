# Final submission checklist

Organizer's required **Google Drive folder** must be set to **Anyone with the link can view**.

- [ ] `predictions.csv` with exact `image_id,label,confidence` headers and one row per test image
- [ ] Weights for the **exact** model used to generate the CSV (`.pt` / `.pth`)
- [ ] `inference_script.py`: `--test_dir`, `--weights`, `--output` CLI works without absolute paths or network calls
- [ ] `requirements.txt` or `environment.yml` plus exact tested versions
- [ ] Source code included via accessible GitHub link **or** source zip
- [ ] Technical report PDF or Markdown, **at least 4 pages**
- [ ] Verify output with `scripts/validate_submission.py`
- [ ] Fresh Kaggle/Colab environment end-to-end inference smoke test
- [ ] Confirm weight loading, threshold, image normalization, and label convention
- [ ] Time and measure peak GPU memory of T4 inference, total runtime < 60 minutes
- [ ] Confirm model size ≤ 5 GB
- [ ] Match hashes/checkpoints with final submitted CSV (inference reproducibility)
- [ ] Double-check Drive sharing and deadline

Optional: training logs / curves, Grad-CAM, attention visualization. The scoring guide describes bonuses and penalties; confirm official rules before submission.
