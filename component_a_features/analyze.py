"""
Component A — feature blueprint from real fraud data.

This component produces no deployable model. It answers one question:
which engineered features actually carry fraud signal on real claims
data? The answer becomes the blueprint for which columns to build into
the synthetic PM-JAY dataset in Stage 2.

Uses the Medicare provider fraud dataset. Note the label there is
per-PROVIDER, not per-claim, so we aggregate claims up to the provider
level before training — matching how the dataset is actually labelled.

Run:  python analyze.py
"""

import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

DATASET = "rohitrox/healthcare-provider-fraud-detection-analysis"
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
BLUEPRINT = HERE / "feature_blueprint.md"


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.csv")):
        print(f"Dataset already present at {DATA_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )


def find_csv(pattern):
    hits = [p for p in DATA_DIR.rglob("*.csv")
            if pattern.lower() in p.name.lower()]
    if not hits:
        raise FileNotFoundError(
            f"No CSV matching '{pattern}' under {DATA_DIR}. "
            f"Found: {[p.name for p in DATA_DIR.rglob('*.csv')]}"
        )
    # prefer training files over test files
    hits.sort(key=lambda p: ("test" in p.name.lower(), len(p.name)))
    return hits[0]


def load_data():
    train = pd.read_csv(find_csv("Train-"))
    inpatient = pd.read_csv(find_csv("Train_Inpatient"))
    outpatient = pd.read_csv(find_csv("Train_Outpatient"))
    beneficiary = pd.read_csv(find_csv("Train_Beneficiary"))

    print(f"labels      : {train.shape}")
    print(f"inpatient   : {inpatient.shape}")
    print(f"outpatient  : {outpatient.shape}")
    print(f"beneficiary : {beneficiary.shape}")
    return train, inpatient, outpatient, beneficiary


def engineer_features(train, inpatient, outpatient, beneficiary):
    """
    Build provider-level features. Each block below maps to one of the
    fraud hypotheses we want to test — the ones that survive become
    the blueprint for the synthetic PM-JAY generator.
    """
    inpatient["is_inpatient"] = 1
    outpatient["is_inpatient"] = 0
    claims = pd.concat([inpatient, outpatient], ignore_index=True, sort=False)

    claims["ClaimStartDt"] = pd.to_datetime(claims["ClaimStartDt"], errors="coerce")
    claims["ClaimEndDt"] = pd.to_datetime(claims["ClaimEndDt"], errors="coerce")
    claims["claim_duration_days"] = (
        claims["ClaimEndDt"] - claims["ClaimStartDt"]
    ).dt.days

    claims = claims.merge(
        beneficiary[["BeneID", "DOB", "Gender", "RenalDiseaseIndicator",
                     "State", "County"]],
        on="BeneID", how="left",
    )
    claims["DOB"] = pd.to_datetime(claims["DOB"], errors="coerce")
    claims["patient_age"] = (
        (claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.25
    )

    g = claims.groupby("Provider")

    feats = pd.DataFrame({
        # volume — phantom billing shows up as abnormal claim counts
        "total_claims": g.size(),
        "unique_patients": g["BeneID"].nunique(),
        "claims_per_patient": g.size() / g["BeneID"].nunique(),

        # money — upcoding and inflation show up in the amount profile
        "mean_reimbursed": g["InscClaimAmtReimbursed"].mean(),
        "std_reimbursed": g["InscClaimAmtReimbursed"].std(),
        "max_reimbursed": g["InscClaimAmtReimbursed"].max(),
        "total_reimbursed": g["InscClaimAmtReimbursed"].sum(),

        # duration — impossibly short or long stays
        "mean_claim_duration": g["claim_duration_days"].mean(),
        "std_claim_duration": g["claim_duration_days"].std(),

        # mix — providers billing only high-value inpatient claims
        "inpatient_ratio": g["is_inpatient"].mean(),

        # concentration — fraud rings reuse the same doctors
        "unique_attending_physicians": g["AttendingPhysician"].nunique(),
        "claims_per_physician": (
            g.size() / g["AttendingPhysician"].nunique().replace(0, np.nan)
        ),

        # diagnosis diversity — narrow code reuse is a known red flag
        "unique_diagnosis_codes": g["ClmDiagnosisCode_1"].nunique(),

        # demographics of the patient pool
        "mean_patient_age": g["patient_age"].mean(),
        "unique_states": g["State"].nunique(),
    }).reset_index()

    df = feats.merge(train, on="Provider", how="inner")
    df["label"] = (df["PotentialFraud"] == "Yes").astype(int)
    df = df.drop(columns=["PotentialFraud"]).fillna(0)

    print(f"\nProviders: {len(df)}  |  fraud rate: {df['label'].mean():.3f}")
    return df


def train_and_explain(df):
    X = df.drop(columns=["Provider", "label"])
    y = df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc", tree_method="hist",
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"\nBaseline AUC: {roc_auc_score(y_te, proba):.4f}")
    print(classification_report(y_te, model.predict(X_te),
                                target_names=["genuine", "fraud"]))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)
    importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
        .sort_values(ascending=False)
    )

    print("\nFeature importance (mean |SHAP|):")
    for name, val in importance.items():
        print(f"  {val:8.4f}  {name}")

    return importance, roc_auc_score(y_te, proba)


def write_blueprint(importance, auc):
    top = importance.head(10)
    lines = [
        "# Feature blueprint for the synthetic PM-JAY dataset",
        "",
        "Derived from a baseline XGBoost + SHAP analysis on the Medicare",
        "provider fraud dataset. This is Component A's only deliverable —",
        "no model is deployed from here.",
        "",
        f"Baseline provider-level AUC on real fraud labels: **{auc:.4f}**",
        "",
        "## Features to build into the synthetic generator",
        "",
        "Ranked by mean absolute SHAP value — the higher the value, the",
        "more that feature moved the fraud prediction on real data.",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| |",
        "|---|---|---|",
    ]
    for i, (name, val) in enumerate(top.items(), 1):
        lines.append(f"| {i} | `{name}` | {val:.4f} |")

    lines += [
        "",
        "## How this carries into Stage 2",
        "",
        "The synthetic PM-JAY generator should produce claims where these",
        "same relationships exist, so the final classifier has real signal",
        "to learn rather than noise. Fraudulent synthetic claims should be",
        "generated by perturbing these specific quantities — not by adding",
        "a random flag — otherwise the model learns an artifact.",
        "",
        "Adapt units and coding to PM-JAY: rupee amounts rather than USD,",
        "PM-JAY package codes rather than HCPCS, and empanelled-hospital",
        "identifiers rather than Medicare provider IDs.",
    ]
    BLUEPRINT.write_text("\n".join(lines))
    print(f"\nBlueprint written to {BLUEPRINT}")


def main():
    download_dataset()
    train, inpatient, outpatient, beneficiary = load_data()
    df = engineer_features(train, inpatient, outpatient, beneficiary)
    importance, auc = train_and_explain(df)
    write_blueprint(importance, auc)


if __name__ == "__main__":
    main()
