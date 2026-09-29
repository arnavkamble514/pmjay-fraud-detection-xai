# ClaimGuard-PMJAY — Stage 1: Training the three sub-components

Stage 1 trains three independent components on three unrelated public
datasets. None of them talk to each other yet — that happens in Stage 2
(synthetic claim generation), once all three exist.

## One-time setup

```bash
python -m venv claimguard-env
# Windows: claimguard-env\Scripts\activate
# Linux/Mac: source claimguard-env/bin/activate

pip install -r requirements.txt
```

Kaggle API token: Kaggle account -> Settings -> Create New Token, downloads
`kaggle.json`. Place it at:
- Windows: `C:\Users\<you>\.kaggle\kaggle.json`
- Linux/Mac: `~/.kaggle/kaggle.json`

Verify your GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run order

1. `component_b_ocr/run.py` — no real training, just tunes + benchmarks
   EasyOCR. Fastest to get a working result.
2. `component_c_tamper_cnn/train.py` — fine-tunes ResNet-18 on CASIA 2.0.
   Actual GPU training happens here.
3. `component_a_features/analyze.py` — baseline XGBoost + SHAP on the
   Medicare dataset, purely to identify which engineered feature types
   carry real fraud signal. Produces a written feature blueprint, not a
   deployable model.

Each folder's script downloads its own dataset via the Kaggle CLI the
first time it's run — nothing to download by hand.

## What Stage 1 hands off to Stage 2

- `component_b_ocr/ocr_pipeline.py` — importable `run_ocr()` function
- `component_c_tamper_cnn/tamper_cnn.pth` — trained weights, plus
  `tamper_cnn.py` with an importable `score_tamper()` function
- `component_a_features/feature_blueprint.md` — written list of which
  engineered features to build into the synthetic PM-JAY dataset


## Progress

## Progress

| Component | Status | Result |
|---|---|---|
| A — Feature blueprint | Done | Baseline AUC 0.958 on Medicare provider-fraud data; fraud-class precision 0.60 / recall 0.67 (imbalanced, 9.4% fraud rate). SHAP-ranked features written to `feature_blueprint.md`. |
| B — OCR pipeline | Done | Mean confidence 0.844, low-confidence line ratio 0.049, field recovery 0.731 (40-image sample, with preprocessing). |
| C — Tamper CNN | Done | ResNet-18 + Error Level Analysis (ELA) preprocessing on CASIA 2.0. Best validation: **93.66% accuracy, 0.9866 AUC, F1 0.9227** — exceeds the 92% synopsis target. |

**Stage 1 complete.** All three components trained/analyzed independently. Next: Stage 2 — synthetic PM-JAY claim dataset generation and fusion into a single feature table for final XGBoost + SHAP training.

**Dataset notes:**
- The OCR dataset originally cited in the synopsis was removed from Kaggle. Component B now uses
  [Noisy Medical Document Images](https://www.kaggle.com/datasets/devp1866/noisy-medical-document-images-ocr)
  (500 hospital bills + 500 discharge summaries with JSON annotations).
- The original CASIA 2.0 Kaggle mirror was also inaccessible; Component C uses the
  [divg07 CASIA 2.0 mirror](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset) instead.

**Technique note:** Component C's initial run (plain ResNet-18 fine-tuning, no preprocessing) plateaued at ~77% accuracy / 0.84 AUC — overfitting, evidenced by falling training loss with flat validation accuracy. Adding Error Level Analysis (ELA) preprocessing — re-compressing each image and amplifying the JPEG residual, so tampered regions surface as a visible compression-artifact difference — plus light augmentation and early stopping closed the gap to 93.66% accuracy.