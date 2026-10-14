"""
Stage 2, step 4 — proper evaluation.

train_final_model.py reports metrics from ONE 80/20 split, so a lucky
split could be inflating the numbers. This script fixes that:

  1. 5-fold stratified cross-validation — same model, same features,
     retrained fresh on each fold — to see whether AUC holds up across
     different splits instead of just the one we happened to report.
  2. Out-of-fold predictions (every claim scored only by a model that
     never saw it in training) used to sweep classification thresholds
     from 0.30 to 0.70, so you can justify whatever threshold you pick
     instead of defaulting to 0.5 with no reasoning.
  3. ROC curve, precision-recall curve, and a confusion matrix at the
     chosen threshold, saved as PNGs for the report.

Needs data/fused_features.csv (same as train_final_model.py).

Run:  python evaluate_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)

HERE = Path(__file__).parent
FUSED_CSV = HERE / "data" / "fused_features.csv"
EVAL_REPORT = HERE / "evaluation_report.md"

FEATURES = [
    "patient_age", "package_rate", "claimed_amount", "length_of_stay",
    "typical_los", "amount_to_package_ratio", "los_ratio",
    "tamper_score", "ocr_confidence", "ocr_amount_mismatch_pct",
    "ocr_amount_found",
    "provider_mean_amount", "provider_claim_count",
]
CATEGORICAL = ["diagnosis_code", "procedure", "state"]
LABEL = "is_fraud"
N_FOLDS = 5
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def make_model(scale_pos_weight):
    # same hyperparameters as train_final_model.py — this is an
    # evaluation of that model, not a different one
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42,
    )


def main():
    df = pd.read_csv(FUSED_CSV)
    print(f"Loaded {len(df)} claims, fraud rate {df[LABEL].mean():.1%}")

    df_enc = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    X = pd.concat([df[FEATURES], df_enc], axis=1)
    y = df[LABEL].values

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    # ---- 1. cross-validated AUC, fold by fold ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_aucs = []
    print(f"\nRunning {N_FOLDS}-fold cross-validation...")
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        model = make_model(scale_pos_weight)
        model.fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        fold_aucs.append(auc)
        print(f"  fold {i}: AUC {auc:.4f}")

    mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
    print(f"\nCV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    stable = std_auc < 0.03
    single_split_note = (
        "holds up across folds" if stable
        else "varies more across folds than it should — the single-split "
             "number may have been optimistic or pessimistic by chance"
    )
    print(f"  -> {single_split_note}")

    # ---- 2. out-of-fold predictions for honest threshold tuning ----
    print("\nGenerating out-of-fold predictions for threshold sweep...")
    oof_proba = cross_val_predict(
        make_model(scale_pos_weight), X, y, cv=skf, method="predict_proba"
    )[:, 1]

    print(f"\n{'threshold':>9}  {'precision':>9}  {'recall':>7}  {'f1':>6}  {'flagged':>8}")
    rows = []
    for t in THRESHOLDS:
        preds = (oof_proba >= t).astype(int)
        p = precision_score(y, preds, zero_division=0)
        r = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        flagged = preds.sum()
        rows.append((t, p, r, f1, flagged))
        print(f"{t:>9.2f}  {p:>9.3f}  {r:>7.3f}  {f1:>6.3f}  {flagged:>8d}")

    # pick the threshold that maximizes F1 among those with recall >= 0.80 —
    # for fraud screening we'd rather over-flag than under-flag, since a
    # human reviews the flagged queue anyway (never auto-reject)
    candidates = [row for row in rows if row[2] >= 0.80] or rows
    best = max(candidates, key=lambda row: row[3])
    best_threshold = best[0]
    print(f"\nRecommended threshold: {best_threshold:.2f} "
          f"(precision {best[1]:.3f}, recall {best[2]:.3f}, f1 {best[3]:.3f})")
    print("  -> chosen to keep recall >= 0.80: for fraud screening, a "
          "human reviews flagged claims, so missing fraud is worse than "
          "a false alarm")

    # ---- 3. plots at the recommended threshold ----
    fpr, tpr, _ = roc_curve(y, oof_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC {mean_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "roc_curve.png", dpi=150)
    plt.close()

    prec, rec, _ = precision_recall_curve(y, oof_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve (out-of-fold)")
    plt.tight_layout()
    plt.savefig(HERE / "pr_curve.png", dpi=150)
    plt.close()

    preds_at_best = (oof_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y, preds_at_best)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["genuine", "fraud"])
    plt.yticks([0, 1], ["genuine", "fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix (threshold {best_threshold:.2f})")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(HERE / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nSaved roc_curve.png, pr_curve.png, confusion_matrix.png")

    # ---- report ----
    lines = [
        "# Model Evaluation — Cross-Validation and Threshold Tuning\n",
        f"5-fold stratified cross-validation, {len(df)} claims, "
        f"fraud rate {df[LABEL].mean():.1%}.\n",
        f"**Cross-validated AUC: {mean_auc:.4f} +/- {std_auc:.4f}** "
        f"(per-fold: {', '.join(f'{a:.4f}' for a in fold_aucs)})\n",
        f"This {single_split_note}, compared to the single-split AUC "
        f"reported by train_final_model.py.\n",
        "## Threshold sweep (out-of-fold predictions)\n",
        "| Threshold | Precision | Recall | F1 | Claims flagged |",
        "|---|---|---|---|---|",
    ]
    for t, p, r, f1, flagged in rows:
        marker = " **<- recommended**" if t == best_threshold else ""
        lines.append(f"| {t:.2f} | {p:.3f} | {r:.3f} | {f1:.3f} | {flagged} |{marker}")
    lines += [
        f"\n**Recommended threshold: {best_threshold:.2f}** — chosen to "
        f"keep recall at or above 0.80, since a human reviews every "
        f"flagged claim before any action is taken; missing real fraud "
        f"is a worse outcome than an extra claim going to review.\n",
        "## Plots\n",
        "- `roc_curve.png`\n- `pr_curve.png`\n- `confusion_matrix.png` "
        f"(at threshold {best_threshold:.2f})\n",
        "## Caveat\n",
        "This evaluation is entirely on synthetic data generated by "
        "generate_synthetic_claims.py. Cross-validation confirms the "
        "model performs consistently *on this dataset* — it does not "
        "confirm performance on real PM-JAY claims, which have not "
        "been used anywhere in this project.\n",
    ]
    EVAL_REPORT.write_text("\n".join(lines))
    print(f"Report written to {EVAL_REPORT}")


if __name__ == "__main__":
    main()
