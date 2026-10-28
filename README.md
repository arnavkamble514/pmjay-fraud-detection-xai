# ClaimGuard-PMJAY

A fraud-screening prototype for PM-JAY-style health insurance claims. It looks at three things together: the structured claim data, the text OCR pulls off the bill image, and some image-forensics signals. All of that goes into one XGBoost classifier, and SHAP explains why each claim got the score it did.

> **Read this first**
> - **Everything here runs on synthetic data.** Row-level PM-JAY claims aren't public, so we generated our own. The metrics show the pipeline works end to end. They don't tell you how it would do on real claims.
> - **This has nothing to do with the National Health Authority or the Government of India.** No affiliation, no endorsement.
> - It's an academic prototype for deciding which claims a human should look at first. Don't use it to approve or reject real claims.

## Results at a glance

These are from Stage 2: 3,000 synthetic claims, 19.4% of them fraud.

| Measure | Value |
|---|---|
| 5-fold cross-validated ROC AUC | **0.916 +/- 0.006** |
| PR-AUC (out-of-fold) | 0.856 |
| Tampering detection, field edits (full model vs without forensics) | recall 0.10 -> 0.55 |
| Tampering detection, amount edits | recall 0.89 |
| False-alarm rate on genuine claims (threshold 0.40) | 5.8% |

**Ablation** (each step adds to the last, 5-fold CV AUC): structured data only gives 0.822. Adding OCR takes it to 0.887 (+0.065). Adding the tamper CNN gets 0.885, so no real effect (-0.002). Adding noise forensics lifts it to 0.916 (+0.031).

**Recall by fraud type** (threshold 0.40, out-of-fold):

| Group | n | Structured + OCR | + tamper CNN | + noise forensics |
|---|---|---|---|---|
| Amount-edited bills | 71 | 0.887 | 0.887 | 0.887 |
| Other-field-edited bills | 62 | 0.081 | 0.097 | **0.548** |
| Naive forgeries | 73 | 0.562 | 0.548 | 0.932 |
| Careful forgeries | 60 | 0.450 | 0.483 | 0.483 |
| Other fraud (no image edit) | 449 | 0.802 | 0.800 | 0.786 |
| Genuine claims (false-alarm rate) | 2418 | 0.084 | 0.079 | 0.058 |

**Threshold trade-off** (out-of-fold). Which one to pick depends on how many claims your reviewers can handle.

| Threshold | Precision | Recall | Claims flagged (of 3,000) |
|---|---|---|---|
| 0.25 (recall >= 0.80 policy) | 0.605 | 0.820 | 788 |
| 0.40 | 0.753 | 0.775 | 599 |
| 0.50 | 0.828 | 0.751 | 528 |
| 0.70 | 0.927 | 0.716 | 450 |

The full tables live in `stage2/evaluation_report.md`, `stage2/ablation_report.md` and `stage2/subgroup_report.md`.

**What these numbers do and don't tell you**
- Before running anything, we set a bar: an image signal had to raise field-edit recall by at least 20 points. Noise forensics cleared it (+45 points). The tamper CNN didn't (+1.6 points).
- The CNN was trained on CASIA, which is natural-photo splices, and it doesn't carry over to scanned text bills. We're reporting that as a negative result.
- Careful forgeries, the ones that imitate the scan noise, are still mostly missed.
- We designed both the forgeries and the forensics detector, so this is still a synthetic test. Noise inconsistency is a known forensic cue, but the groups are small (about 60 cases each), so treat those figures as rough.
- Precision depends on how common fraud is. Our data is 19.4% fraud. Real fraud is almost certainly rarer, which means real precision would be lower.

## Repository layout

| Folder | What's in it |
|---|---|
| `component_a_features/` | Stage 1: baseline XGBoost + SHAP on public Medicare provider-fraud data, used to get a feature blueprint |
| `component_b_ocr/` | Stage 1: EasyOCR pipeline, benchmarked on a public noisy medical-document dataset |
| `component_c_tamper_cnn/` | Stage 1: ResNet-18 with Error Level Analysis, trained on CASIA 2.0 |
| `stage2/` | Synthetic claims, feature fusion, the final model, evaluation, and serving |

## Setup

```bash
python -m venv claimguard-env
# Windows: claimguard-env\Scripts\activate
# Linux/Mac: source claimguard-env/bin/activate

pip install -r requirements.txt
```

You only need a Kaggle API token if you want to download the Stage 1 datasets. Go to your Kaggle account, then Settings, then Create New Token, and put `kaggle.json` in your `.kaggle` folder (`%USERPROFILE%\.kaggle` on Windows, `~/.kaggle` on Linux/Mac).

A CUDA GPU is only needed to train the tamper CNN in Stage 1. To check whether PyTorch can see yours:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Not in this repo:** the datasets, the generated claims and bill images (`data/`), and the tamper CNN weights. The scripts download the datasets, and the weights come out of Stage 1 training. The final XGBoost model (`stage2/final_model.json`) and its lookup files are included.

## Stage 1: three independent components

Each component is trained or benchmarked on its own, using an unrelated public dataset. Run them in this order:

1. `component_b_ocr/run.py` tunes and benchmarks EasyOCR. No training involved.
2. `component_c_tamper_cnn/train.py` fine-tunes ResNet-18 on CASIA 2.0 with ELA preprocessing. A GPU helps.
3. `component_a_features/analyze.py` runs a baseline XGBoost + SHAP on the Medicare data to see which kinds of features carry fraud signal.

Each script downloads its own dataset through the Kaggle CLI the first time you run it.

| Component | Result |
|---|---|
| A: feature blueprint | AUC 0.958 on Medicare provider-fraud data; fraud precision 0.60 / recall 0.67 (9.4% fraud) |
| B: OCR | mean confidence 0.844, low-confidence line ratio 0.049, field recovery 0.731 (40-image sample) |
| C: tamper CNN | best validation 93.66% accuracy, AUC 0.9866, F1 0.9227 on CASIA 2.0 |

Datasets: [Noisy Medical Document Images](https://www.kaggle.com/datasets/devp1866/noisy-medical-document-images-ocr) for OCR, and the [divg07 CASIA 2.0 mirror](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset) for the tamper CNN. A plain ResNet-18 got stuck at about 77% accuracy. ELA preprocessing, light augmentation and early stopping closed the gap.

## Stage 2: synthetic data, fusion, final model

Stage 1 components get reused here as scoring tools. They aren't retrained.

| Step | Script | What it does |
|---|---|---|
| 1 | `stage2/generate_synthetic_claims.py` | Makes 3,000 claims with matching bill images |
| 2 | `stage2/fuse_features.py` | Pulls OCR (at 3x size), tamper CNN and noise-forensics features and merges them with the claim fields |
| 3 | `stage2/train_final_model.py` | Trains the final XGBoost model and computes SHAP importance |
| 4 | `stage2/evaluate_model.py` | 5-fold CV, threshold sweep, ROC / PR / confusion plots |
| 5 | `stage2/ablation_study.py` | Measures what OCR, the tamper CNN and noise forensics each add |
| 6 | `stage2/subgroup_analysis.py` | Recall by fraud type, plus the 20-point test |
| n/a | `stage2/predict.py` | `score_claim()` gives a score, a risk band and plain-language reasons for one claim |

Some supporting files: `stage2/forensics.py` holds the noise-forensics features. `stage2/ela_diagnostic.py`, `stage2/inspect_ocr.py`, `stage2/upscale_test.py` and `stage2/refresh_ocr.py` are diagnostics, and the last one is a shortcut for refreshing just the OCR columns of an existing table.

Run these from `stage2/`, in order: `generate_synthetic_claims.py`, `fuse_features.py`, `train_final_model.py`, `evaluate_model.py`, `ablation_study.py`, `subgroup_analysis.py`. Heads up that `fuse_features.py` is slow. On a laptop CPU it takes several hours, mostly because of OCR.

### Synthetic fraud patterns

- **Upcoding:** the claimed amount is inflated compared to the package rate.
- **Phantom billing:** the amount is inflated while the documented stay is shorter than typical.
- **Impossible length of stay:** the stay is way outside the normal range for the procedure.
- **Document tampering:** the bill image is edited after it's generated, either the **amount** or another **field** (patient name, diagnosis, diagnosis code). The edit is either **naive** (a flat pasted patch) or **careful** (the patch is blurred and re-noised to match the scan).

Genuine and fraudulent claims overlap on purpose in amount, stay length and image compression. Bills are rendered to look scanned (tint, blur, noise) and saved with the same JPEG-quality range whether they're genuine or tampered.

### Mistakes we fixed along the way

- Our first version had a clean numeric gap between genuine and fraud, which gave a meaningless 100% accuracy. We added the overlap deliberately to fix that.
- A later version let JPEG save quality alone separate tampered from genuine bills (AUC 0.85), and its edits left almost no trace in the image. Both problems are fixed in the current generator, and we set the success criterion before seeing results.
- OCR read the amount exactly on only 28% of the scanned-style bills. Upscaling the image 3x before OCR raised that to 93% (on a 60-bill test). The amount is now found on 99.3% of bills, up from 73%.

## Not done yet

- Wiring the model into a dashboard, meaning a review queue for flagged claims.
- Validating on real, confirmed claims, with time-based and provider-based splits, and calibrating the scores.
- Testing the forensics features on forgeries made by other people, for example public receipt-forgery datasets.
