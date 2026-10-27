"""
Stage 2, step 3 — final model.

Trains the deployable XGBoost fraud classifier on the fused feature
table (structured claim data + OCR results + tamper CNN score), then
runs SHAP for per-claim and global explainability. This is the model
the ClaimGuard dashboard actually calls.

Run:  python train_final_model.py
"""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

HERE = Path(__file__).parent
FUSED_CSV = HERE / "data" / "fused_features.csv"
MODEL_PATH = HERE / "final_model.json"
SHAP_SUMMARY = HERE / "shap_summary.md"
FEATURE_COLUMNS_PATH = HERE / "feature_columns.json"
PROCEDURE_LOOKUP_PATH = HERE / "procedure_lookup.json"
PROVIDER_STATS_PATH = HERE / "provider_stats.json"

FEATURES = [
    "patient_age", "package_rate", "claimed_amount", "length_of_stay",
    "typical_los", "amount_to_package_ratio", "los_ratio",
    "tamper_score", "ocr_confidence", "ocr_amount_mismatch_pct",
    "ocr_amount_found",
    "noise_block_min_z", "noise_low_frac",
    "provider_mean_amount", "provider_claim_count",
]
CATEGORICAL = ["diagnosis_code", "procedure", "state"]
LABEL = "is_fraud"


def main():
    df = pd.read_csv(FUSED_CSV)
    print(f"Loaded {len(df)} fused claims, fraud rate {df[LABEL].mean():.1%}")

    df_enc = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    X = pd.concat([df[FEATURES], df_enc], axis=1)
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    print(f"\nFinal model AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["genuine", "fraud"]))

    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Everything predict.py needs to score a brand-new claim the same
    # way this training run encoded one, saved alongside the model.
    FEATURE_COLUMNS_PATH.write_text(json.dumps(list(X.columns)))

    procedure_lookup = (
        df.groupby("procedure")[["package_rate", "typical_los"]]
        .first()
        .to_dict(orient="index")
    )
    PROCEDURE_LOOKUP_PATH.write_text(json.dumps(procedure_lookup))

    provider_stats = df.groupby("provider_id")["claimed_amount"].agg(
        provider_mean_amount="mean", provider_claim_count="count"
    )
    provider_lookup = provider_stats.to_dict(orient="index")
    provider_lookup["__global__"] = {
        "provider_mean_amount": float(df["claimed_amount"].mean()),
        "provider_claim_count": 0,
    }
    PROVIDER_STATS_PATH.write_text(json.dumps(provider_lookup))
    print(f"Saved feature_columns.json, procedure_lookup.json, provider_stats.json "
          f"for predict.py")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    print("\nTop features (mean |SHAP|):")
    for name, val in mean_abs_shap.head(15).items():
        print(f"  {val:.4f}  {name}")

    lines = ["# Final Model — SHAP Feature Importance\n",
             f"AUC: {auc:.4f}\n", "| Feature | Mean |SHAP| |", "|---|---|"]
    for name, val in mean_abs_shap.head(15).items():
        lines.append(f"| {name} | {val:.4f} |")
    SHAP_SUMMARY.write_text("\n".join(lines))
    print(f"\nSHAP summary written to {SHAP_SUMMARY}")


if __name__ == "__main__":
    main()
