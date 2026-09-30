"""
Quick sanity check (about 2 minutes) — run BEFORE the full fusion.

Scores 60 tampered and 60 clean bills with the tamper CNN, and reads
the amount off 20 tampered bills with OCR. Tells you whether the two
image signals actually work on the generated bills.

Run:  python check_signal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "component_b_ocr"))
import fuse_features as ff  # noqa: E402
from ocr_pipeline import run_ocr  # noqa: E402


def main():
    df = pd.read_csv(DATA_DIR / "claims.csv")
    tampered = df[df["image_tampered"] == 1].head(60)
    clean = df[df["image_tampered"] == 0].head(60)

    model, img_size = ff.load_tamper_model()

    def scores(rows):
        return [ff.score_tamper(model, img_size, DATA_DIR / p)
                for p in rows["image_path"]]

    from sklearn.metrics import roc_auc_score

    t, c = scores(tampered), scores(clean)
    auc = roc_auc_score([1] * len(t) + [0] * len(c), t + c)
    print(f"Tamper score  tampered bills: mean {np.mean(t):.3f}")
    print(f"Tamper score  clean bills   : mean {np.mean(c):.3f}")
    print(f"Tamper AUC on this sample   : {auc:.3f}")
    print("  -> AUC above ~0.90 means usable even if the scores look small\n")

    print("OCR amount check on 20 tampered bills (claimed vs read):")
    found = 0
    failures = []
    for _, row in tampered.head(20).iterrows():
        res = run_ocr(str(DATA_DIR / row["image_path"]),
                      use_preprocessing=True, use_gpu=True)
        read = ff.parse_amount(res["text"])
        found += read is not None
        if read is None:
            failures.append((row["claimed_amount"], res["text"]))
        print(f"  claimed {row['claimed_amount']:>7}  |  OCR read {read}")
    print(f"\nAmount found on {found}/20 bills")
    print("  -> want most found, and tampered bills reading HIGHER than claimed")

    for claimed, text in failures[:3]:
        print(f"\n--- raw OCR text where amount was NOT found (claimed {claimed}) ---")
        print(repr(text))


if __name__ == "__main__":
    main()
