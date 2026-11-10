# Competition requirements — working summary

**Source:** IIIT Bangalore Deepfake ML Hackathon, `README.md` and `evaluation_metric.md` from the provided organizer ZIP. Refer to the official files if rules change.

| Category | Requirement |
|---|---|
| Task | Face image real (0) / fake (1) binary classification |
| Inputs | 224×224 JPEG face crops |
| Train | 7,000 labeled images: 3,478 real, 3,522 fake |
| Test | 3,000 unlabeled image files |
| Score | AUC (50), inference efficiency (20), robustness (20), documentation (10) |
| Input rules | Supplied dataset only for fine-tuning; ImageNet-pretrained models allowed |
| Prohibited | External deepfake data/pretrained detectors; using test images for training |
| Environment | Python 3.10, PyTorch 2.x or TensorFlow 2.x |
| Limits | Model ≤5 GB; inference ≤60 minutes on NVIDIA T4 |
| Required inference CLI | `python inference_script.py --test_dir ... --weights ... --output ...` |
| Required CSV | `image_id,label,confidence`, one row per test ID |
| Submission | Shareable Google Drive folder, with source via GitHub link or zip |
| Report | Technical report PDF or Markdown, at least 4 pages |

Evaluator computes AUC from **confidence** as P(fake), so don't optimize or report AUC from hard class predictions. AUC depends on score ranking rather than the chosen classification threshold.

The official scoring guide's efficiency table awards full 20 points for inference under five minutes, with lower scores for longer runtimes. Robustness is tested against undisclosed variants.

Documentation rubric: problem/rationale 2; architecture 2; hyperparameters 2; training pipeline 2; challenges/limitations 1; reproducibility 1. Possible bonuses are described in the official scoring guide.
