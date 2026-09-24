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
