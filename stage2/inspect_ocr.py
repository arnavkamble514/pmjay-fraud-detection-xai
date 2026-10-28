import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fuse_features import run_ocr, parse_amount, DEVICE, DATA_DIR  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    df = pd.read_csv(HERE / "data" / "fused_features.csv")
    bad = df[(df.is_fraud == 0) & (df.ocr_amount_found == 1)
             & (df.ocr_amount_mismatch_pct >= 0.5)]
    print(f"{len(bad)} genuine bills with mismatch >= 50% "
          f"(of {int(((df.is_fraud == 0) & (df.ocr_amount_found == 1)).sum())} read)\n")

    for _, row in bad.sample(n, random_state=1).iterrows():
        res = run_ocr(str(DATA_DIR / row["image_path"]), use_preprocessing=True,
                      use_gpu=(DEVICE.type == "cuda"))
        text = res["text"]
        print("=" * 70)
        print(f"{row['image_path']}   claimed = {row['claimed_amount']:,.0f}   "
              f"parse_amount picked = {parse_amount(text)}   "
              f"(stored mismatch {row['ocr_amount_mismatch_pct']:.2f})")
        lines = text.splitlines()
        hits = [i for i, l in enumerate(lines) if "amount" in l.lower()]
        show = sorted({j for i in hits for j in range(max(0, i - 1), min(len(lines), i + 3))})
        print("OCR lines near 'Amount':" if show else "No 'Amount' line found; full text:")
        for j in (show or range(len(lines))):
            print(f"   | {lines[j]}")
    print()


if __name__ == "__main__":
    main()
