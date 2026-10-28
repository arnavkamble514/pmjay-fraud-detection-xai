# ClaimGuard-PMJAY

A multimodal, explainable fraud-screening prototype for PM-JAY-style health insurance claims. It combines structured claim data, OCR on the bill image, and image-forensics signals in one XGBoost classifier, and explains each score with SHAP.

> **Read this first**
> - **All results here come from synthetic data.** No real PM-JAY claims were used; row-level PM-JAY claims are not publicly available. The metrics show that the pipeline works end to end, not how well it would work on real claims.
> - **Not affiliated with, endorsed by, or connected to the National Health Authority or the Government of India.**
> - It is an academic prototype for prioritising claims for human review. It is not a tool for deciding or rejecting real claims.

## Results at a glance (Stage 2, 3,000 synthetic claims, 19.4% fraud)

| Measure | Value |
|---|---|
| 5-fold cross-validated ROC AUC | **0.916 +/- 0.006** |
| PR-AUC (out-of-fold) | 0.856 |
| Image-tampering detection, field edits (full model vs without forensics) | recall 0.10 -> 0.55 |
| Image-tampering detection, amount edits | recall 0.89 |
| False-alarm rate on genuine claims (threshold 0.40) | 5.8% |

**Ablation** (cumulative, 5-fold CV AUC): structured data only 0.822 -> + OCR 0.887 (+0.065) -> + tamper CNN 0.885 (-0.002, no effect) -> + noise forensics 0.916 (+0.031).

**Recall by fraud type** (threshold 0.40, out-of-fold):

| Group | n | Structured + OCR | + tamper CNN | + noise forensics |
|---|---|---|---|---|
| Amount-edited bills | 71 | 0.887 | 0.887 | 0.887 |
| Other-field-edited bills | 62 | 0.081 | 0.097 | **0.548** |
| Naive forgeries | 73 | 0.562 | 0.548 | 0.932 |
| Careful forgeries | 60 | 0.450 | 0.483 | 0.483 |
| Other fraud (no image edit) | 449 | 0.802 | 0.800 | 0.786 |
| Genuine claims (false-alarm rate) | 2418 | 0.084 | 0.079 | 0.058 |

**Threshold trade-off** (out-of-fold; the right choice depends on reviewer capacity):

| Threshold | Precision | Recall | Claims flagged (of 3,000) |
|---|---|---|---|
| 0.25 (recall >= 0.80 policy) | 0.605 | 0.820 | 788 |
| 0.40 | 0.753 | 0.775 | 599 |
| 0.50 | 0.828 | 0.751 | 528 |
| 0.70 | 0.927 | 0.716 | 450 |

Full tables are in `stage2/evaluation_report.md`, `stage2/ablation_report.md` and `stage2/subgroup_report.md`.

**What the results do and do not show**
- The pre-set success test (an image signal must raise field-edit recall by at least 20 points) was **passed by noise forensics (+45 points)** and **failed by the tamper CNN (+1.6 points)**.
- The CASIA-trained tamper CNN does not transfer from natural-photo splices to scanned text bills; this is reported as a negative result.
- **Careful forgeries** (edits that imitate the scan noise) are still largely missed.
- The forgeries and the forensics detector were both designed in this project. Noise inconsistency is a documented forensic cue, but this is still a synthetic test, and the figures carry wide uncertainty on small groups (about 60 cases each).
- Precision depends on how common fraud is. The synthetic data is 19.4% fraud; real fraud is almost certainly rarer, so real precision would be lower.

## Repository layout

| Folder | Contents |
|---|---|
| `component_a_features/` | Stage 1: baseline XGBoost + SHAP on public Medicare provider-fraud data, producing a feature blueprint |
| `component_b_ocr/` | Stage 1: EasyOCR pipeline, benchmarked on a public noisy medical-document dataset |
| `component_c_tamper_cnn/` | Stage 1: ResNet-18 with Error Level Analysis, trained on CASIA 2.0 |
| `stage2/` | Synthetic claims, feature fusion, final model, evaluation, serving |

## Setup

```bash
python -m venv claimguard-env
# Windows: claimguard-env\Scripts\activate
# Linux/Mac: source claimguard-env/bin/activate

pip install -r requirements.txt
```

Kaggle API token (needed only to download the Stage 1 datasets): Kaggle account -> Settings -> Create New Token, then place `kaggle.json` in your user's `.kaggle` folder (`%USERPROFILE%\.kaggle` on Windows, `~/.kaggle` on Linux/Mac).

A CUDA GPU is only needed to train the tamper CNN in Stage 1. Check with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Not included in this repository:** datasets, generated claims and bill images (`data/`), and trained model weights for the tamper CNN. Datasets are downloaded by the scripts; weights are produced by running Stage 1 training. The final XGBoost model (`stage2/final_model.json`) and its lookup files are included.

## Stage 1 - three independent components

Each component is trained or benchmarked separately on an unrelated public dataset. Run order:

1. `component_b_ocr/run.py` - tunes and benchmarks EasyOCR (no training).
2. `component_c_tamper_cnn/train.py` - fine-tunes ResNet-18 on CASIA 2.0 with ELA preprocessing (GPU recommended).
3. `component_a_features/analyze.py` - baseline XGBoost + SHAP on the Medicare data, to find which feature types carry fraud signal.

Each script downloads its own dataset through the Kaggle CLI on first run.

| Component | Result |
|---|---|
| A - feature blueprint | AUC 0.958 on Medicare provider-fraud data; fraud precision 0.60 / recall 0.67 (9.4% fraud) |
| B - OCR | mean confidence 0.844, low-confidence line ratio 0.049, field recovery 0.731 (40-image sample) |
| C - tamper CNN | best validation 93.66% accuracy, AUC 0.9866, F1 0.9227 on CASIA 2.0 |

Datasets: [Noisy Medical Document Images](https://www.kaggle.com/datasets/devp1866/noisy-medical-document-images-ocr) (OCR) and the [divg07 CASIA 2.0 mirror](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset) (tamper CNN). The plain ResNet-18 baseline plateaued at about 77% accuracy; ELA preprocessing, light augmentation and early stopping closed the gap.

## Stage 2 - synthetic data, fusion, final model

Stage 1 components are reused as scoring tools; they are not retrained.

| Step | Script | What it does |
|---|---|---|
| 1 | `stage2/generate_synthetic_claims.py` | 3,000 claims with matching bill images |
| 2 | `stage2/fuse_features.py` | OCR (at 3x size), tamper CNN and noise-forensics features, merged with the claim fields |
| 3 | `stage2/train_final_model.py` | Final XGBoost model + SHAP importance |
| 4 | `stage2/evaluate_model.py` | 5-fold CV, threshold sweep, ROC / PR / confusion plots |
| 5 | `stage2/ablation_study.py` | Contribution of OCR, tamper CNN and noise forensics |
| 6 | `stage2/subgroup_analysis.py` | Recall by fraud type; the pre-set 20-point test |
| - | `stage2/predict.py` | `score_claim()`: score, risk band and plain-language reasons for one claim |

Supporting files: `stage2/forensics.py` (noise-forensics features), `stage2/ela_diagnostic.py`, `stage2/inspect_ocr.py`, `stage2/upscale_test.py` and `stage2/refresh_ocr.py` (diagnostics and a shortcut to refresh only the OCR columns of an existing table).

Run from `stage2/`, in order: `generate_synthetic_claims.py`, `fuse_features.py` (slow: several hours on a laptop CPU, mostly OCR), `train_final_model.py`, `evaluate_model.py`, `ablation_study.py`, `subgroup_analysis.py`.

### Synthetic fraud patterns

- **Upcoding** - claimed amount inflated relative to the package rate.
- **Phantom billing** - amount inflated while the documented stay is shorter than typical.
- **Impossible length of stay** - stay far outside the range for the procedure.
- **Document tampering** - the bill image is edited after generation, either the **amount** or another **field** (patient name, diagnosis, diagnosis code), in a **naive** style (flat pasted patch) or a **careful** style (patch blurred and re-noised to imitate the scan).

Genuine and fraudulent claims deliberately overlap in amount, stay length and image compression. Bills are rendered with a scanned look (tint, blur, noise) and saved with the same JPEG-quality range for genuine and tampered bills.

### How the design was corrected along the way

- An early version had a clean numeric gap between genuine and fraud, which allowed a meaningless 100% accuracy; the overlap was added deliberately.
- A later version let JPEG save quality alone separate tampered from genuine bills (AUC 0.85), and its edits left almost no image trace. Both were fixed in the current generator, and the success criterion was fixed before the results were seen.
- OCR read the amount exactly on only 28% of the scanned-style bills. Upscaling the image 3x before OCR raised that to 93% (60-bill test); the amount is now found on 99.3% of bills, up from 73%.

## Not done

- Wiring the model into a dashboard (a review queue for flagged claims).
- Validation on real, confirmed claims, time- and provider-separated splits, and calibration.
- Testing the forensics features on forgeries made by other people (for example public receipt-forgery datasets).
