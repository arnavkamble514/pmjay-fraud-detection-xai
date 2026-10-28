import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_PATH = HERE / "final_model.json"
FEATURE_COLUMNS_PATH = HERE / "feature_columns.json"
PROCEDURE_LOOKUP_PATH = HERE / "procedure_lookup.json"
PROVIDER_STATS_PATH = HERE / "provider_stats.json"

CATEGORICAL = ["diagnosis_code", "procedure", "state"]
IMAGE_FEATURES = {"tamper_score", "ocr_confidence", "ocr_amount_mismatch_pct", "ocr_amount_found",
                  "noise_block_min_z", "noise_low_frac"}

STRUCTURED_MODEL_PATH = HERE / "final_model_structured.json"
STRUCTURED_COLUMNS_PATH = HERE / "feature_columns_structured.json"

# score -> band, set from the out-of-fold threshold analysis (evaluation_report.md):
# below 0.25 = low; 0.25-0.50 = medium (the recall >= 0.80 review threshold);
# 0.50 and above = high (precision 0.83, recall 0.75 at 0.50).
# These cutoffs were tuned on the full model; for claims scored without a
# bill image (structured-only model) they are indicative only.
RISK_BANDS = [(0.25, "low"), (0.50, "medium"), (1.01, "high")]

# technical feature name -> (human label, sentence template).
# {value} is filled with the claim's own value for that feature.
FEATURE_EXPLANATIONS = {
    "amount_to_package_ratio": "claimed amount is {value:.2f}x the standard package rate",
    "los_ratio": "length of stay is {value:.2f}x the typical stay for this procedure",
    "tamper_score": "the bill image scores {value:.2f} on the tamper detector",
    "ocr_confidence": "OCR could read the bill with {value:.2f} confidence",
    "ocr_amount_mismatch_pct": "the OCR-read amount differs from the claimed amount by {value:.0%}",
    "ocr_amount_found": "an amount could {not_}be read directly off the bill image",
    "noise_block_min_z": "noise check: the quietest patch of the bill scores {value:.1f} (pasted patches score far below normal)",
    "noise_low_frac": "noise check: {value:.1%} of the bill's background is unusually quiet",
    "claimed_amount": "claimed amount is ₹{value:,.0f}",
    "patient_age": "patient age is {value:.0f}",
    "provider_mean_amount": "this provider's average claim is ₹{value:,.0f}",
    "provider_claim_count": "this provider has {value:.0f} claims on record",
    "package_rate": "standard package rate is ₹{value:,.0f}",
    "length_of_stay": "length of stay is {value:.0f} days",
    "typical_los": "typical stay for this procedure is {value:.0f} days",
}


def _load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in {path.parent}. Run train_final_model.py "
            f"first — it writes this file alongside final_model.json."
        )
    return json.loads(path.read_text())


class ClaimScorer:
    """Loads the model once; call .score() per claim. score_claim() below
    is a module-level convenience that reuses one shared instance."""

    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_PATH)
        self.feature_columns = _load_json(FEATURE_COLUMNS_PATH)
        self.procedure_lookup = _load_json(PROCEDURE_LOOKUP_PATH)
        self.provider_lookup = _load_json(PROVIDER_STATS_PATH)
        self.explainer = shap.TreeExplainer(self.model)
        # structured-only fallback for claims with no bill image (written by
        # train_final_model.py); without it, no-image claims are refused
        self.model_s = self.columns_s = self.explainer_s = None
        if STRUCTURED_MODEL_PATH.exists() and STRUCTURED_COLUMNS_PATH.exists():
            self.model_s = xgb.XGBClassifier()
            self.model_s.load_model(STRUCTURED_MODEL_PATH)
            self.columns_s = _load_json(STRUCTURED_COLUMNS_PATH)
            self.explainer_s = shap.TreeExplainer(self.model_s)
        self._tamper_model = None
        self._tamper_img_size = None

    # -- image scoring, loaded lazily so score_claim() works with no
    #    GPU / no torch installed as long as you don't pass image_path --
    def _score_image(self, image_path):
        if self._tamper_model is None:
            sys.path.insert(0, str(ROOT / "stage2"))
            sys.path.insert(0, str(ROOT / "component_b_ocr"))
            from fuse_features import (  # noqa: E402
                load_tamper_model, score_tamper, parse_amount, DEVICE,
                run_ocr_upscaled,
            )
            from ocr_pipeline import run_ocr  # noqa: E402
            from forensics import forensic_features  # noqa: E402
            self._forensic_features = forensic_features
            self._tamper_model, self._tamper_img_size = load_tamper_model()
            self._run_ocr = run_ocr_upscaled
            self._parse_amount = parse_amount
            self._score_tamper = score_tamper
            self._device = DEVICE

        tamper_score = self._score_tamper(
            self._tamper_model, self._tamper_img_size, image_path
        )
        ocr_result = self._run_ocr(image_path)
        forensics = self._forensic_features(image_path)
        return tamper_score, ocr_result["mean_confidence"], ocr_result["text"], forensics

    def _build_row(self, claim, image_path=None):
        row = dict(claim)

        proc_info = self.procedure_lookup.get(row.get("procedure"), {})
        row.setdefault("package_rate", proc_info.get("package_rate"))
        row.setdefault("typical_los", proc_info.get("typical_los"))
        if row.get("package_rate") is None or row.get("typical_los") is None:
            raise ValueError(
                f"Unknown procedure {row.get('procedure')!r} and no "
                f"package_rate/typical_los given — can't compute ratios."
            )

        row["amount_to_package_ratio"] = row["claimed_amount"] / row["package_rate"]
        row["los_ratio"] = row["length_of_stay"] / row["typical_los"]

        provider = self.provider_lookup.get(
            row.get("provider_id"), self.provider_lookup["__global__"]
        )
        row.setdefault("provider_mean_amount", provider["provider_mean_amount"])
        row.setdefault("provider_claim_count", provider["provider_claim_count"])

        # Image-derived features: use what's given, else compute from an
        # image, else fall back to "no evidence either way" — never a
        # value that pretends the bill was checked when it wasn't.
        image_checked = True
        if "tamper_score" not in row or "ocr_confidence" not in row:
            if image_path is not None:
                tamper_score, ocr_confidence, ocr_text, forensics = self._score_image(image_path)
                row["tamper_score"] = tamper_score
                row["ocr_confidence"] = ocr_confidence
                for key, value in forensics.items():
                    row.setdefault(key, value)
                if "ocr_amount_mismatch_pct" not in row or "ocr_amount_found" not in row:
                    ocr_amount = self._parse_amount(ocr_text)
                    if ocr_amount is None:
                        row["ocr_amount_mismatch_pct"] = np.nan
                        row["ocr_amount_found"] = 0
                    else:
                        row["ocr_amount_mismatch_pct"] = abs(
                            ocr_amount - row["claimed_amount"]
                        ) / max(row["claimed_amount"], 1)
                        row["ocr_amount_found"] = 1
            else:
                # No image given and no precomputed image features: tell
                # the model "unknown", not "clean". 0.0 is itself a real
                # value real genuine bills score, so defaulting to it
                # would silently look like a checked, clean bill. NaN
                # lets XGBoost's native missing-value handling route it
                # instead of faking evidence.
                image_checked = False
                row.setdefault("tamper_score", np.nan)
                row.setdefault("ocr_confidence", np.nan)
                row.setdefault("ocr_amount_mismatch_pct", np.nan)
                row.setdefault("ocr_amount_found", np.nan)
                row.setdefault("noise_block_min_z", np.nan)
                row.setdefault("noise_low_frac", np.nan)

        return row, image_checked

    def score(self, claim, image_path=None):
        """claim: dict with at least patient_age, procedure,
        diagnosis_code, state, provider_id, claimed_amount,
        length_of_stay. package_rate/typical_los are filled in from the
        training data's procedure lookup if you don't pass them.
        Pass image_path to run OCR + tamper scoring on a real bill, or
        pass tamper_score/ocr_confidence/etc. directly if you already
        have them (e.g. from fuse_features.py)."""
        row, image_checked = self._build_row(claim, image_path)

        # The full model was trained only on claims WITH a bill image, so it
        # never saw missing image features; its score for a no-image claim
        # would be arbitrary. Use the structured-only model instead.
        if image_checked:
            model, explainer, columns = self.model, self.explainer, self.feature_columns
            model_used = "full model (structured + bill image)"
        elif self.model_s is not None:
            model, explainer, columns = self.model_s, self.explainer_s, self.columns_s
            model_used = "structured-only model (no bill image provided)"
        else:
            raise ValueError(
                "No bill image given and no structured-only model found. Re-run "
                "train_final_model.py (it writes final_model_structured.json), or "
                "pass image_path / the image features."
            )

        df_row = pd.DataFrame([row])
        df_enc = pd.get_dummies(df_row[CATEGORICAL], prefix=CATEGORICAL)
        X = pd.concat([df_row, df_enc], axis=1)
        # missing one-hot category columns mean "not this category" -> 0,
        # but a missing NUMERIC feature means "unknown" -> NaN, never 0
        # (0 is a real value that would look like a checked, clean bill)
        X = X.reindex(columns=columns)
        dummy_cols = [c for c in X.columns
                      if any(c.startswith(p + "_") for p in CATEGORICAL)]
        X[dummy_cols] = X[dummy_cols].fillna(0)
        X = X.infer_objects()

        proba = float(model.predict_proba(X)[0, 1])
        risk_band = next(band for threshold, band in RISK_BANDS if proba < threshold)

        shap_values = explainer.shap_values(X)[0]
        order = np.argsort(-np.abs(shap_values))
        top_reasons = []
        for i in order[:3]:
            feat = columns[i]
            contribution = float(shap_values[i])
            if FEATURE_EXPLANATIONS.get(feat):
                template = FEATURE_EXPLANATIONS[feat]
                value = row.get(feat, X.iloc[0][feat])
                text = template.format(
                    value=0 if pd.isna(value) else value,
                    not_="" if value else "not ",
                )
            else:
                # a one-hot category column, e.g. "procedure_Angioplasty"
                text = feat.replace("_", " ", 1).replace("_", ": ", 1)
            top_reasons.append({
                "feature": feat,
                "direction": "raises risk" if contribution > 0 else "lowers risk",
                "impact": round(abs(contribution), 3),
                "detail": text,
            })

        return {
            "score": round(proba, 4),
            "risk_band": risk_band,
            "top_reasons": top_reasons,
            "image_checked": image_checked,
            "model_used": model_used,
        }


_shared_scorer = None


def score_claim(claim, image_path=None):
    """Module-level convenience — loads the model once on first call."""
    global _shared_scorer
    if _shared_scorer is None:
        _shared_scorer = ClaimScorer()
    return _shared_scorer.score(claim, image_path=image_path)


def _smoke_test():
    samples = [
        {
            "patient_age": 45, "procedure": "Laparoscopic cholecystectomy",
            "diagnosis_code": "K80.2", "state": "Maharashtra",
            "provider_id": "H003", "claimed_amount": 52000,
            "length_of_stay": 9,
        },
        {
            "patient_age": 30, "procedure": "Appendectomy",
            "diagnosis_code": "K35.8", "state": "Gujarat",
            "provider_id": "H007", "claimed_amount": 19000,
            "length_of_stay": 3,
        },
    ]
    for claim in samples:
        print(f"\nClaim: {claim}")
        result = score_claim(claim)
        print(f"  score={result['score']}  risk={result['risk_band']}  "
              f"image_checked={result['image_checked']}")
        print(f"  model: {result['model_used']}")
        for r in result["top_reasons"]:
            print(f"    [{r['direction']:>11}, impact {r['impact']}] {r['detail']}")


if __name__ == "__main__":
    _smoke_test()
