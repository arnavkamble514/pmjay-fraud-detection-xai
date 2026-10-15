"""
Stage 2, step 5 — ablation study.

The project's core claim is that fusing structured claim data + OCR +
tamper-CNN output beats any one signal alone. This script tests that
claim directly instead of assuming it: trains the same model three
times, each with a different feature subset, and compares cross-
validated AUC across them.

  A. structured only       (amounts, dates, provider history — no
                             OCR, no image)
  B. structured + OCR      (+ ocr_confidence, ocr_amount_mismatch_pct,
                             ocr_amount_found)
  C. structured + OCR +    (+ tamper_score — the full model, same as
     tamper                 train_final_model.py)

If C isn't meaningfully better than A, the image pipeline isn't
pulling its weight and that's worth saying plainly, not hiding.

Needs data/fused_features.csv.

Run:  python ablation_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).parent
FUSED_CSV = HERE / "data" / "fused_features.csv"
ABLATION_REPORT = HERE / "ablation_report.md"
ABLATION_CHART = HERE / "ablation_chart.png"

STRUCTURED = [
    "patient_age", "package_rate", "claimed_amount", "length_of_stay",
    "typical_los", "amount_to_package_ratio", "los_ratio",
    "provider_mean_amount", "provider_claim_count",
]
OCR_FEATURES = ["ocr_confidence", "ocr_amount_mismatch_pct", "ocr_amount_found"]
TAMPER_FEATURES = ["tamper_score"]
CATEGORICAL = ["diagnosis_code", "procedure", "state"]
LABEL = "is_fraud"
N_FOLDS = 5

VARIANTS = [
    ("A. Structured only", STRUCTURED),
    ("B. Structured + OCR", STRUCTURED + OCR_FEATURES),
    ("C. Structured + OCR + tamper (full model)",
     STRUCTURED + OCR_FEATURES + TAMPER_FEATURES),
]


def make_model(scale_pos_weight):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42,
    )


def main():
    df = pd.read_csv(FUSED_CSV)
    print(f"Loaded {len(df)} claims, fraud rate {df[LABEL].mean():.1%}")

    df_enc = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    y = df[LABEL].values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    results = []
    print(f"\nRunning {N_FOLDS}-fold CV for each feature set...")
    for name, feature_cols in VARIANTS:
        X = pd.concat([df[feature_cols], df_enc], axis=1)
        scores = cross_val_score(
            make_model(scale_pos_weight), X, y, cv=skf, scoring="roc_auc"
        )
        results.append((name, scores.mean(), scores.std(), len(feature_cols)))
        print(f"  {name}: AUC {scores.mean():.4f} +/- {scores.std():.4f} "
              f"({len(feature_cols)} numeric features)")

    auc_a, auc_b, auc_c = results[0][1], results[1][1], results[2][1]
    gain_ocr = auc_b - auc_a
    gain_tamper = auc_c - auc_b
    gain_total = auc_c - auc_a

    print(f"\nOCR features added: {gain_ocr:+.4f} AUC")
    print(f"Tamper CNN added:   {gain_tamper:+.4f} AUC")
    print(f"Total image gain:   {gain_total:+.4f} AUC (over structured-only)")

    def judge(gain):
        if gain > 0.02:
            return "a real, meaningful improvement"
        elif gain > 0.005:
            return "a small improvement — present but modest"
        elif gain > -0.005:
            return "essentially no difference — within noise"
        else:
            return "the feature set actually performed worse here"

    ocr_verdict = judge(gain_ocr)
    tamper_verdict = judge(gain_tamper)
    total_verdict = judge(gain_total)
    print(f"\nOCR verdict: {ocr_verdict}")
    print(f"Tamper verdict: {tamper_verdict}")

    # ---- chart ----
    names = [r[0].split(". ", 1)[1] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color=["#9fb3c8", "#5e8ab4", "#2a5783"])
    plt.ylabel("Cross-validated AUC")
    plt.title("Ablation: does fusing OCR + tamper signal actually help?")
    plt.ylim(min(means) - 0.1, 1.0)
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.01,
                  f"{mean:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(ABLATION_CHART, dpi=150)
    plt.close()
    print(f"\nSaved {ABLATION_CHART.name}")

    # ---- report ----
    lines = [
        "# Ablation Study — Does Multimodal Fusion Help?\n",
        f"5-fold cross-validated AUC, {len(df)} claims, "
        f"fraud rate {df[LABEL].mean():.1%}.\n",
        "| Feature set | AUC | Std dev | # numeric features |",
        "|---|---|---|---|",
    ]
    for name, mean, std, n in results:
        lines.append(f"| {name} | {mean:.4f} | {std:.4f} | {n} |")
    lines += [
        f"\n**Adding OCR features: {gain_ocr:+.4f} AUC — {ocr_verdict}.**\n",
        f"**Adding the tamper CNN score: {gain_tamper:+.4f} AUC — {tamper_verdict}.**\n",
        f"**Total gain from structured-only to the full model: {gain_total:+.4f} AUC.**\n",
        "## What this means\n",
    ]
    if gain_total > 0.02:
        lines.append(
            "The multimodal fusion claim holds up on this synthetic dataset: "
            "combining OCR and tamper-detection signal with structured claim "
            "data measurably outperforms structured data alone.\n"
        )
    elif gain_total > 0.005:
        lines.append(
            "The multimodal fusion helps, but modestly. The structured "
            "features (mainly `amount_to_package_ratio`) carry most of the "
            "signal; OCR and tamper detection add a real but smaller "
            "contribution. This should be stated honestly rather than "
            "oversold as the project's headline result.\n"
        )
    else:
        lines.append(
            "On this synthetic dataset, the image-derived features "
            "(OCR, tamper score) add little to no measurable improvement "
            "over structured fields alone. This doesn't mean the OCR and "
            "tamper pipeline are worthless — it may mean the synthetic "
            "fraud patterns are too easy to catch from amounts and dates "
            "alone, which is itself a limitation of synthetic data worth "
            "stating plainly in the report rather than hiding behind the "
            "AUC of the full model.\n"
        )
    lines += [
        "## Caveat\n",
        "This test is on synthetic data from generate_synthetic_claims.py. "
        "The synthetic document-tamper fraud type is specifically designed "
        "to be caught by the tamper CNN, so any apparent benefit here may "
        "not transfer to real-world documents with more varied tampering "
        "patterns.\n",
        f"\n![Ablation chart]({ABLATION_CHART.name})\n",
    ]
    ABLATION_REPORT.write_text("\n".join(lines))
    print(f"Report written to {ABLATION_REPORT}")


if __name__ == "__main__":
    main()
