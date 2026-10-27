"""
Stage 2, step 6 — subgroup analysis: does each image signal catch the
fraud it is supposed to catch?

Overall AUC averages over every claim, so ~70 field-edit bills out of
3000 can barely move it even if a detector catches them all. This script
looks at each fraud subgroup separately:

  1. Raw signal: how well does each image feature ALONE (tamper CNN score,
     noise-forensics features) separate each subgroup from genuine
     claims? No model involved — tests the detector itself.
  2. Recall per subgroup, out-of-fold, for three cumulative models at one
     fixed threshold:
        B  structured + OCR
        C  + tamper CNN
        D  + noise forensics (full model)
  3. ROC AUC and PR-AUC (average precision) overall for B, C, D. PR-AUC
     is more sensitive to rare positives than ROC AUC.

Subgroups:
  genuine            is_fraud == 0
  amount_tamper      bill amount edited (OCR can catch this)
  field_tamper       other printed field edited (only image forensics can)
  other_fraud        fraud with an untouched bill (upcoding etc.)
  naive / careful    tampered bills split by forger style (generator v3)

Success criterion, fixed before results: an image signal counts as useful
if it raises field-edit recall by at least 20 points over the model
without it.

Needs data/fused_features.csv from fuse_features.py.

Run:  python subgroup_analysis.py [threshold]     (default 0.40)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).parent
FUSED_CSV = HERE / "data" / "fused_features.csv"
REPORT = HERE / "subgroup_report.md"

STRUCTURED = [
    "patient_age", "package_rate", "claimed_amount", "length_of_stay",
    "typical_los", "amount_to_package_ratio", "los_ratio",
    "provider_mean_amount", "provider_claim_count",
]
OCR_FEATURES = ["ocr_confidence", "ocr_amount_mismatch_pct", "ocr_amount_found"]
TAMPER_FEATURES = ["tamper_score"]
FORENSIC_FEATURES = ["noise_block_min_z", "noise_low_frac"]
CATEGORICAL = ["diagnosis_code", "procedure", "state"]
LABEL = "is_fraud"
N_FOLDS = 5
USEFUL_GAIN = 0.20   # recall points on field edits — fixed before results


def make_model(spw):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="auc", random_state=42,
    )


def assign_group(row):
    if row[LABEL] == 0:
        return "genuine"
    if row["image_tampered"] == 1:
        return "amount_tamper" if row["tamper_field"] == "Amount Claimed" else "field_tamper"
    return "other_fraud"


def raw_separation(df, feature, positives):
    """AUC of one feature for positive rows vs genuine, direction-agnostic."""
    keep = (df.group == "genuine") | positives
    sub = df.loc[keep, [feature]].assign(label=positives[keep].astype(int)).dropna()
    if sub.label.nunique() < 2:
        return np.nan
    auc = roc_auc_score(sub.label, sub[feature])
    return max(auc, 1 - auc)


def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else FUSED_CSV

    df = pd.read_csv(path)
    df["tamper_field"] = df["tamper_field"].fillna("")
    df["group"] = df.apply(assign_group, axis=1)
    y = df[LABEL].values
    print(f"Loaded {len(df)} claims")
    print(df["group"].value_counts().to_string(), "\n")

    tampered = df.image_tampered == 1
    groups = [(g, df.group == g) for g in ["amount_tamper", "field_tamper", "other_fraud"]]
    if "tamper_style" in df.columns and df["tamper_style"].fillna("").ne("").any():
        style = df["tamper_style"].fillna("")
        groups += [("naive", tampered & (style == "naive")),
                   ("careful", tampered & (style == "careful"))]
    has_forensics = all(f in df.columns for f in FORENSIC_FEATURES)
    raw_features = TAMPER_FEATURES + (FORENSIC_FEATURES if has_forensics else [])

    # ---- 1. raw signal, no model ----
    print("1. Raw signal — each image feature alone vs genuine claims (AUC, 0.5 = none)")
    print(f"{'feature':<20}" + "".join(f"{g:>15}" for g, _ in groups))
    raw_rows = []
    for f in raw_features:
        seps = [raw_separation(df, f, mask) for _, mask in groups]
        raw_rows.append((f, seps))
        print(f"{f:<20}" + "".join(f"{s:>15.3f}" for s in seps))

    # ---- 2 & 3. out-of-fold cumulative models ----
    models = [("B", "+OCR", STRUCTURED + OCR_FEATURES),
              ("C", "+tamper CNN", STRUCTURED + OCR_FEATURES + TAMPER_FEATURES)]
    if has_forensics:
        models.append(("D", "+forensics",
                       STRUCTURED + OCR_FEATURES + TAMPER_FEATURES + FORENSIC_FEATURES))
    enc = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = {}
    for key, _, cols in models:
        X = pd.concat([df[cols], enc], axis=1)
        oof[key] = cross_val_predict(make_model(spw), X, y, cv=skf,
                                     method="predict_proba")[:, 1]

    print(f"\n2. Recall per subgroup at threshold {threshold:.2f} (out-of-fold)")
    print(f"{'group':<15}{'n':>5}" + "".join(f"{k + ': ' + lab:>16}" for k, lab, _ in models))
    rec_rows = []
    for g, mask in groups + [("genuine (FPR)", df.group == "genuine")]:
        m = mask.values
        if m.sum() == 0:
            continue
        recalls = [(oof[k][m] >= threshold).mean() for k, _, _ in models]
        rec_rows.append((g, int(m.sum()), recalls))
        print(f"{g:<15}{m.sum():>5}" + "".join(f"{r:>16.3f}" for r in recalls))

    print("\n3. Overall (out-of-fold)")
    overall = []
    for k, lab, _ in models:
        auc, ap = roc_auc_score(y, oof[k]), average_precision_score(y, oof[k])
        overall.append((k, lab, auc, ap))
        print(f"   {k} {lab:<14} ROC AUC {auc:.4f}   PR-AUC {ap:.4f}")

    # ---- verdicts against the fixed criterion ----
    field = next(r for r in rec_rows if r[0] == "field_tamper")[2]
    cnn_raw = next(r for r in raw_rows if r[0] == "tamper_score")[1][1]
    verdicts = []
    cnn_gain = field[1] - field[0]
    verdicts.append(
        f"Tamper CNN: field-edit recall {field[0]:.3f} -> {field[1]:.3f} "
        f"({cnn_gain:+.3f}); raw AUC on field edits {cnn_raw:.2f}. "
        + ("Meets the 20-point criterion." if cnn_gain >= USEFUL_GAIN
           else "Does NOT meet the 20-point criterion.")
    )
    if has_forensics:
        f_gain = field[2] - field[1]
        verdicts.append(
            f"Noise forensics: field-edit recall {field[1]:.3f} -> {field[2]:.3f} "
            f"({f_gain:+.3f}). "
            + ("Meets the 20-point criterion." if f_gain >= USEFUL_GAIN
               else "Does NOT meet the 20-point criterion.")
        )
    print("\nVerdict (criterion: +20 recall points on field edits, fixed before results)")
    for v in verdicts:
        print(" ", v)

    lines = [
        "# Subgroup Analysis — does each image signal catch its fraud?\n",
        f"{len(df)} claims, 5-fold out-of-fold predictions, threshold {threshold:.2f}.\n",
        "## 1. Raw signal vs genuine claims (no model; AUC, 0.5 = none)\n",
        "| Feature | " + " | ".join(g for g, _ in groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for f, seps in raw_rows:
        lines.append(f"| `{f}` | " + " | ".join(f"{s:.3f}" for s in seps) + " |")
    lines += ["\n## 2. Recall per subgroup (cumulative models)\n",
              "| Group | n | " + " | ".join(f"{k} ({lab})" for k, lab, _ in models) + " |",
              "|---|---|" + "---|" * len(models)]
    for g, n, recalls in rec_rows:
        lines.append(f"| {g} | {n} | " + " | ".join(f"{r:.3f}" for r in recalls) + " |")
    lines += ["\n## 3. Overall\n",
              "| Model | ROC AUC | PR-AUC |", "|---|---|---|"]
    for k, lab, auc, ap in overall:
        lines.append(f"| {k} ({lab}) | {auc:.4f} | {ap:.4f} |")
    lines += ["\n## Verdict\n",
              "Criterion (fixed before results): an image signal is useful if it "
              "raises field-edit recall by at least 20 points.\n"]
    lines += [f"- {v}" for v in verdicts]
    lines += ["\n## Caveat\n",
              "Synthetic data from generate_synthetic_claims.py; the forgeries are "
              "generated by the same project that designed the detectors."]
    REPORT.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT.name}")


if __name__ == "__main__":
    main()
