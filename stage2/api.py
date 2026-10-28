import json
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from predict import CATEGORICAL, FEATURE_EXPLANATIONS, RISK_BANDS, score_claim
from train_final_model import FEATURES

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
FUSED_CSV = DATA_DIR / "fused_features.csv"
IMG_DIR = DATA_DIR / "images"
CACHE = DATA_DIR / "review_scores.json"
REVIEW_LOG = DATA_DIR / "review_log.jsonl"
FRONTEND_DIST = HERE.parent / "frontend" / "dist"
N_FOLDS = 5

# What the system does on its own for each risk band. A person confirms
# every high and medium case; nothing is rejected automatically.
ACTIONS = {
    "low": "Approve for payment (5% random audit)",
    "medium": "Send to a reviewer before payment",
    "high": "Hold payment and request the original bill",
}
DECISIONS = {"cleared", "fraud_confirmed", "escalated"}


def band_of(score):
    return next(band for edge, band in RISK_BANDS if score < edge)


def reason_text(feat, value):
    template = FEATURE_EXPLANATIONS.get(feat)
    if template:
        try:
            return template.format(value=0 if pd.isna(value) else value,
                                   not_="" if value else "not ")
        except (ValueError, TypeError):
            return f"{feat} = {value}"
    for prefix in CATEGORICAL:
        if feat.startswith(prefix + "_"):
            return f"{prefix.replace('_', ' ')} is {feat[len(prefix) + 1:]}"
    return feat.replace("_", " ")


def fraud_label(row):
    """Plain-language ground-truth label (known because the data is synthetic)."""
    if row["is_fraud"] == 0:
        return "Genuine"
    kind = row.get("fraud_type", "")
    if kind == "document_tamper":
        field = row.get("tamper_field") or "a field"
        style = row.get("tamper_style") or ""
        return f"Bill edited: {field.lower()} ({style})" if style else f"Bill edited: {field.lower()}"
    return {"upcoding": "Upcoding", "phantom_billing": "Phantom billing",
            "impossible_los": "Impossible length of stay"}.get(kind, "Fraud")


def file_fingerprint(path):
    st = path.stat()
    return hashlib.sha1(f"{st.st_size}-{st.st_mtime_ns}".encode()).hexdigest()


def compute_scores(df):
    """Out-of-fold scores and SHAP reasons: every claim is scored by a model
    that never saw it in training (same folds and settings as evaluate_model.py),
    so the queue shows honest scores, not the training fit."""
    enc = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    X = pd.concat([df[FEATURES], enc], axis=1).astype(float)
    y = df["is_fraud"].values
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = np.zeros(len(df))
    reasons = [None] * len(df)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=spw, eval_metric="auc", random_state=42)
        model.fit(X.iloc[tr], y[tr])
        scores[te] = model.predict_proba(X.iloc[te])[:, 1]
        sv = shap.TreeExplainer(model).shap_values(X.iloc[te])
        for j, idx in enumerate(te):
            order = np.argsort(-np.abs(sv[j]))[:4]
            out = []
            for k in order:
                feat = X.columns[k]
                value = df.iloc[idx][feat] if feat in df.columns else X.iloc[idx][feat]
                out.append({"feature": feat,
                            "impact": round(float(abs(sv[j][k])), 3),
                            "direction": "raises" if sv[j][k] > 0 else "lowers",
                            "detail": reason_text(feat, value)})
            reasons[idx] = out
        print(f"  scored fold {fold}/{N_FOLDS}")
    return scores, reasons


def load_state():
    if not FUSED_CSV.exists():
        raise SystemExit(f"{FUSED_CSV} not found. Run fuse_features.py first.")
    df = pd.read_csv(FUSED_CSV)
    df["tamper_field"] = df.get("tamper_field", "").fillna("")
    df["tamper_style"] = df.get("tamper_style", "").fillna("")
    fp = file_fingerprint(FUSED_CSV)
    cached = None
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("fingerprint") != fp:
            cached = None
    if cached is None:
        print("Scoring all claims out-of-fold (first run only, about a minute)...")
        scores, reasons = compute_scores(df)
        cached = {"fingerprint": fp, "scores": scores.tolist(), "reasons": reasons}
        CACHE.write_text(json.dumps(cached))
    df["score"] = cached["scores"]
    df["band"] = df["score"].map(band_of)
    df["label"] = df.apply(fraud_label, axis=1)
    return df, cached["reasons"]


DF, REASONS = load_state()
INDEX = {cid: i for i, cid in enumerate(DF["claim_id"])}
print(f"Loaded {len(DF)} claims. Open http://localhost:8000")


def read_reviews():
    log = {}
    if REVIEW_LOG.exists():
        for line in REVIEW_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                log.setdefault(e["claim_id"], []).append(e)
    return log


def num(v, nd=4):
    return None if pd.isna(v) else round(float(v), nd)


app = FastAPI(title="ClaimGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/claims")
def list_claims():
    reviews = read_reviews()
    rows = []
    for _, r in DF.iterrows():
        last = reviews.get(r.claim_id, [None])[-1]
        rows.append({
            "claim_id": r.claim_id, "hospital": r.hospital, "provider_id": r.provider_id,
            "state": r.state, "procedure": r.procedure, "patient_age": int(r.patient_age),
            "claimed_amount": int(r.claimed_amount), "package_rate": int(r.package_rate),
            "length_of_stay": int(r.length_of_stay), "typical_los": int(r.typical_los),
            "score": round(float(r.score), 4), "band": r.band, "action": ACTIONS[r.band],
            "is_fraud": int(r.is_fraud), "label": r.label,
            "decision": last["decision"] if last else None,
        })
    return rows


@app.get("/api/claims/{claim_id}")
def claim_detail(claim_id: str):
    if claim_id not in INDEX:
        raise HTTPException(404, f"No claim {claim_id}")
    i = INDEX[claim_id]
    r = DF.iloc[i]
    return {
        "claim_id": r.claim_id, "hospital": r.hospital, "provider_id": r.provider_id,
        "state": r.state, "procedure": r.procedure, "diagnosis_code": r.diagnosis_code,
        "patient_age": int(r.patient_age),
        "claimed_amount": int(r.claimed_amount), "package_rate": int(r.package_rate),
        "amount_ratio": num(r.amount_to_package_ratio, 2),
        "length_of_stay": int(r.length_of_stay), "typical_los": int(r.typical_los),
        "los_ratio": num(r.los_ratio, 2),
        "ocr_amount_found": None if pd.isna(r.ocr_amount_found) else int(r.ocr_amount_found),
        "ocr_amount_mismatch_pct": num(r.ocr_amount_mismatch_pct),
        "ocr_confidence": num(r.ocr_confidence, 3),
        "noise_block_min_z": num(r.noise_block_min_z, 2),
        "tamper_score": num(r.tamper_score, 3),
        "score": round(float(r.score), 4), "band": r.band, "action": ACTIONS[r.band],
        "reasons": REASONS[i],
        "is_fraud": int(r.is_fraud), "label": r.label,
        "image_url": f"/images/{Path(r.image_path).name}",
        "history": read_reviews().get(r.claim_id, []),
    }


class Review(BaseModel):
    decision: str
    note: str = ""
    reviewer: str = "Reviewer"


@app.post("/api/claims/{claim_id}/review")
def add_review(claim_id: str, review: Review):
    if claim_id not in INDEX:
        raise HTTPException(404, f"No claim {claim_id}")
    if review.decision not in DECISIONS:
        raise HTTPException(400, f"decision must be one of {sorted(DECISIONS)}")
    entry = {"claim_id": claim_id, "decision": review.decision,
             "note": review.note.strip()[:500], "reviewer": review.reviewer.strip()[:60] or "Reviewer",
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with REVIEW_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return read_reviews().get(claim_id, [])


@app.get("/api/summary")
def summary():
    y, s = DF["is_fraud"].values, DF["score"].values
    return {
        "n_claims": int(len(DF)), "n_fraud": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "bands": [{"band": b, "upper": e, "action": ACTIONS[b],
                   "count": int((DF["band"] == b).sum())} for e, b in RISK_BANDS],
        "scores": [{"score": round(float(a), 4), "is_fraud": int(b), "label": c}
                   for a, b, c in zip(s, y, DF["label"])],
    }


@app.post("/api/score")
async def score_one(patient_age: int = Form(...), procedure: str = Form(...),
                    diagnosis_code: str = Form(...), state: str = Form(...),
                    provider_id: str = Form(...), claimed_amount: float = Form(...),
                    length_of_stay: int = Form(...), bill: Optional[UploadFile] = File(None)):
    claim = {"patient_age": patient_age, "procedure": procedure, "diagnosis_code": diagnosis_code,
             "state": state, "provider_id": provider_id, "claimed_amount": claimed_amount,
             "length_of_stay": length_of_stay}
    image_path = None
    if bill is not None and bill.filename:
        suffix = Path(bill.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await bill.read())
            image_path = tmp.name
    try:
        result = score_claim(claim, image_path=image_path)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except (ImportError, OSError, RuntimeError) as e:
        if image_path is None:
            raise
        raise HTTPException(503, "Scoring a bill image needs the OCR and tamper-detection "
                                 "components (EasyOCR, PyTorch and component_c_tamper_cnn/"
                                 f"tamper_cnn.pth). They could not be loaded: {e}")
    result["action"] = ACTIONS[result["risk_band"]]
    return result


@app.get("/api/options")
def options():
    procs = (DF.groupby("procedure")[["diagnosis_code", "package_rate", "typical_los"]]
             .first().reset_index().to_dict(orient="records"))
    return {"procedures": procs, "states": sorted(DF["state"].unique().tolist()),
            "providers": sorted(DF["provider_id"].unique().tolist())}


app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = FRONTEND_DIST / path
        if path and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
