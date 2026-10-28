import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fuse_features import run_ocr_upscaled, parse_amount, DATA_DIR, OCR_SCALE  # noqa: E402

FUSED = DATA_DIR / "fused_features.csv"
BACKUP = DATA_DIR / "fused_features_ocr1x.csv"
PARTIAL = DATA_DIR / "ocr_refresh_partial.csv"
SAVE_EVERY = 25


def main():
    df = pd.read_csv(FUSED)
    if not BACKUP.exists():
        shutil.copy(FUSED, BACKUP)
        print(f"Backed up old table to {BACKUP.name}")

    done = {}
    if PARTIAL.exists():
        part = pd.read_csv(PARTIAL)
        done = {int(r.idx): (r.conf, r.mismatch, int(r.found)) for r in part.itertuples()}
        print(f"Resuming: {len(done)}/{len(df)} bills already done")

    todo = [i for i in range(len(df)) if i not in done]
    print(f"OCR at {OCR_SCALE}x for {len(todo)} bills")
    t0 = time.time()
    buffer = []
    for n, i in enumerate(todo, 1):
        row = df.iloc[i]
        res = run_ocr_upscaled(DATA_DIR / row["image_path"])
        amount = parse_amount(res["text"])
        if amount is None:
            mismatch, found = np.nan, 0
        else:
            mismatch = abs(amount - row["claimed_amount"]) / max(row["claimed_amount"], 1)
            found = 1
        done[i] = (res["mean_confidence"], mismatch, found)
        buffer.append((i, res["mean_confidence"], mismatch, found))

        if n % SAVE_EVERY == 0 or n == len(todo):
            pd.DataFrame(buffer, columns=["idx", "conf", "mismatch", "found"]).to_csv(
                PARTIAL, mode="a", header=not PARTIAL.exists(), index=False)
            buffer = []
            rate = (time.time() - t0) / n
            left = rate * (len(todo) - n) / 60
            print(f"  {len(done)}/{len(df)}   {rate:.1f}s per bill   ~{left:.0f} min left")

    df["ocr_confidence"] = [done[i][0] for i in range(len(df))]
    df["ocr_amount_mismatch_pct"] = [done[i][1] for i in range(len(df))]
    df["ocr_amount_found"] = [done[i][2] for i in range(len(df))]
    df.to_csv(FUSED, index=False)
    PARTIAL.unlink()
    print(f"\nDone. Updated OCR columns in {FUSED.name}")
    found = df.ocr_amount_found.mean()
    print(f"Amount found on {found:.1%} of bills")


if __name__ == "__main__":
    main()
