# Technical report outline (minimum four pages)

Create the final report as Markdown or PDF in the required Google Drive folder. This is an outline, **not** a completed report.

## 1. Problem and approach rationale
- Real vs deepfake, label definitions, task constraints
- Why chosen architecture balances AUC, robustness, runtime

## 2. Dataset and preprocessing
- Provided dataset composition and image properties
- Train/validation split method, random seed and duplication safeguards
- Augmentation experiments and validation-only test variants
- ImageFolder class-order trap and explicit P(fake) label mapping

## 3. Model and training
- Model architecture and classifier head
- ImageNet initialization and train data restriction
- Loss, optimizer, learning-rate schedule, epochs, early stopping
- Hardware, library versions, runtime

## 4. Results and ablations
- Experiment comparison table and AUC/F1 curves
- Selected decision threshold (F1) and its distinction from AUC
- Robustness results and efficiency (T4) measurements
- Any Grad-CAM examples, where useful and permitted

## 5. Reproduction and limitations
- Repo commit, environment install, dataset placement
- Exact commands to train and run `inference_script.py`
- Explicit limitations, unexpected failures and uncertainty

Do not invent numbers: mark unmeasured values as pending until verified.
