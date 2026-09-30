"""
Stage 2, step 2 — fusion.

Runs every synthetic claim's bill image through Component B (OCR) and
Component C (tamper CNN), cross-checks OCR output against the claim
record, and writes ONE merged feature table. This table is what trains
the final model in train_final_model.py.

Loads the tamper CNN directly from tamper_cnn.pth (same architecture +
ELA preprocessing as train.py) rather than importing tamper_cnn.py, so
this script doesn't depend on that module's internal class API.

Run:  python fuse_features.py
"""

import io
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
CLAIMS_CSV = DATA_DIR / "claims.csv"
FUSED_CSV = DATA_DIR / "fused_features.csv"

TAMPER_WEIGHTS = ROOT / "component_c_tamper_cnn" / "tamper_cnn.pth"
sys.path.insert(0, str(ROOT / "component_b_ocr"))
from ocr_pipeline import run_ocr, extract_claim_fields  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- tamper scoring (mirrors component_c_tamper_cnn/train.py) ----

def ela_loader(path, quality=90, scale=15):
    original = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    orig_arr = np.asarray(original, dtype=np.int16)
    resaved_arr = np.asarray(resaved, dtype=np.int16)
    diff = np.abs(orig_arr - resaved_arr) * scale
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def load_tamper_model():
    checkpoint = torch.load(TAMPER_WEIGHTS, map_location=DEVICE)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    img_size = checkpoint.get("img_size", 224)
    return model, img_size


def score_tamper(model, img_size, image_path):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = ela_loader(image_path)
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 1].item()
    return prob


def parse_amount(ocr_text):
    """Find the amount on the bill straight from the OCR text: the
    number after 'Amount Claimed'. Falls back to the largest
    comma-formatted number. Returns None if nothing is found."""
    text = ocr_text or ""

    def to_int(s):
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else None

    # 1) the number right after the "Amount Claimed" label
    m = re.search(r"Amount\s*Claimed[^\d]*(\d[\d,\.]*\d)", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        v = to_int(m.group(1))
        if v and v >= 1000:
            return v

    # 2) fallback: any plausible amount on the page. OCR often turns
    # commas into periods or drops them, so accept all of those, and
    # ignore claim IDs (CLM000123).
    cleaned = re.sub(r"CLM\s*\d+", " ", text, flags=re.IGNORECASE)
    vals = [to_int(x) for x in re.findall(r"\d[\d,\.]{3,}\d|\d{4,}", cleaned)]
    vals = [v for v in vals if v and 1000 <= v <= 1_000_000]
    return max(vals) if vals else None


def main():
    df = pd.read_csv(CLAIMS_CSV)
    print(f"Loaded {len(df)} claims")

    print("Loading tamper CNN...")
    tamper_model, img_size = load_tamper_model()

    tamper_scores = []
    ocr_confidences = []
    ocr_amount_mismatches = []
    ocr_found = []

    for i, row in df.iterrows():
        image_path = DATA_DIR / row["image_path"]

        tamper_scores.append(score_tamper(tamper_model, img_size, image_path))

        ocr_result = run_ocr(str(image_path), use_preprocessing=True,
                              use_gpu=(DEVICE.type == "cuda"))
        ocr_confidences.append(ocr_result["mean_confidence"])

        ocr_amount = parse_amount(ocr_result["text"])
        claimed = row["claimed_amount"]
        if ocr_amount is None:
            mismatch = np.nan   # unreadable: say so, don't pretend "matches"
        else:
            mismatch = abs(ocr_amount - claimed) / max(claimed, 1)
        ocr_amount_mismatches.append(mismatch)
        ocr_found.append(0 if ocr_amount is None else 1)

        if (i + 1) % 250 == 0:
            print(f"  processed {i + 1}/{len(df)}")

    df["tamper_score"] = tamper_scores
    df["ocr_confidence"] = ocr_confidences
    df["ocr_amount_mismatch_pct"] = ocr_amount_mismatches
    df["ocr_amount_found"] = ocr_found

    # provider-level rolling stats, same spirit as Component A's features
    provider_stats = df.groupby("provider_id")["claimed_amount"].agg(
        provider_mean_amount="mean", provider_claim_count="count"
    )
    df = df.merge(provider_stats, on="provider_id", how="left")

    df.to_csv(FUSED_CSV, index=False)
    print(f"\nFused feature table written to {FUSED_CSV}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
