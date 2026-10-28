import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fuse_features import run_ocr, parse_amount, DEVICE, DATA_DIR  # noqa: E402

SCALES = [2, 3]
REPORT = HERE / "upscale_report.md"


def parse_last(text):
    cleaned = re.sub(r"\bCL\S*", " ", text or "", flags=re.IGNORECASE)
    vals = []
    for tok in re.findall(r"\d[\d,\.]{2,}\d|\d{4,}", cleaned):
        digits = re.sub(r"[^\d]", "", tok)
        if digits and 1000 <= int(digits) <= 1_000_000:
            vals.append(int(digits))
    return vals[-1] if vals else None


def score(values, claimed):
    n = len(values)
    exact = sum(v is not None and abs(v - c) / max(c, 1) < 0.01 for v, c in zip(values, claimed))
    wild = sum(v is not None and abs(v - c) / max(c, 1) >= 0.5 for v, c in zip(values, claimed))
    none = sum(v is None for v in values)
    return exact / n, wild / n, none / n


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    df = pd.read_csv(HERE / "data" / "fused_features.csv")
    gen = df[df.is_fraud == 0].sample(n, random_state=7).reset_index(drop=True)
    claimed = gen["claimed_amount"].tolist()

    # 1x baseline from the stored fuse_features result (current parser)
    base = [np.nan if pd.isna(m) else m for m in gen["ocr_amount_mismatch_pct"]]
    base_vals = [None if pd.isna(m) else c * (1 + m) for m, c in zip(base, claimed)]
    # stored mismatch is absolute, so the sign is unknown; only |diff| matters
    # for exact/wild/none, which is all we report for the 1x row
    rows = []
    n_exact = sum(v is not None and m < 0.01 for v, m in zip(base_vals, base) if not pd.isna(m))
    n_wild = sum((not pd.isna(m)) and m >= 0.5 for m in base)
    n_none = sum(pd.isna(m) for m in base)
    rows.append(("1x  (stored)", "current", n_exact / n, n_wild / n, n_none / n))

    texts = {s: [] for s in SCALES}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in gen.iterrows():
            img = Image.open(DATA_DIR / row["image_path"]).convert("RGB")
            for s in SCALES:
                big = img.resize((img.width * s, img.height * s), Image.LANCZOS)
                p = Path(tmp) / f"{i}_{s}x.png"
                big.save(p)
                res = run_ocr(str(p), use_preprocessing=True,
                              use_gpu=(DEVICE.type == "cuda"))
                texts[s].append(res["text"])
            if i == 0:
                print(f"first bill took {time.time() - t0:.0f}s -> "
                      f"estimated total {(time.time() - t0) * n / 60:.0f} min")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n}")

    for s in SCALES:
        for pname, fn in [("current", parse_amount), ("last", parse_last)]:
            vals = [fn(t) for t in texts[s]]
            rows.append((f"{s}x", pname, *score(vals, claimed)))

    print(f"\n{n} genuine bills (claimed amount = truth)")
    print(f"{'variant':<14}{'parser':<10}{'exact':>8}{'wild>=50%':>11}{'none':>8}")
    for v, p, e, w, z in rows:
        print(f"{v:<14}{p:<10}{e:>8.1%}{w:>11.1%}{z:>8.1%}")

    lines = [f"# Upscale test — {n} genuine bills\n",
             "| variant | parser | exact (<1%) | wild (>=50%) | none |",
             "|---|---|---|---|---|"]
    lines += [f"| {v} | {p} | {e:.1%} | {w:.1%} | {z:.1%} |" for v, p, e, w, z in rows]
    lines.append(f"\nSample of {n} bills is small: differences under ~8 points are noise.")
    REPORT.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT.name}")


if __name__ == "__main__":
    main()
