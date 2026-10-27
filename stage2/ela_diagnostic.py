"""
Stage 2, step 7 — ELA diagnostic: is the tamper signal in the images at all?

subgroup_analysis.py showed the tamper CNN barely separates tampered bills
from genuine ones (AUC ~0.65). That has two possible causes, and they need
opposite fixes:

  1. The signal is in the image, but the CNN pipeline loses it (it was
     trained on CASIA photos, and resizing an 800x600 bill to CNN input
     size can wash out a one-line edit).          -> fix the detector
  2. The generator's edits leave almost no detectable trace.
                                                   -> fix the generator

This script skips the CNN entirely. It computes simple full-resolution
Error Level Analysis (ELA) statistics per image — "is any block of text
compressed differently from the rest?" — and measures how well each one
separates amount-edit and field-edit bills from genuine bills.

It also runs a SHORTCUT CHECK: the JPEG save quality of each file. If
quality alone separates tampered from genuine bills, the generator has a
tell that any detector could be scoring instead of the actual edit.

Decision rule, fixed before looking at results (field-edit bills):
  best local ELA feature AUC >= 0.85  -> signal exists; the CNN pipeline
                                         is losing it
  0.65 - 0.85                         -> a weak trace exists
  below 0.65                          -> the edits leave almost no trace;
                                         fix the generator first

Writes data/ela_features.csv (one row per claim, so these features can be
tested as model inputs later) and ela_report.md.

Run:  python ela_diagnostic.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from forensics import forensic_features

HERE = Path(__file__).parent
DEFAULT_CSV = HERE / "data" / "fused_features.csv"
REPORT = HERE / "ela_report.md"

ELA_QUALITY = 90       # re-save quality used to expose compression differences
BLOCK = 16             # block size in pixels (full resolution, no resizing)
INK_THRESHOLD = 160    # grayscale below this counts as text ("ink")
MIN_INK_FRAC = 0.02    # a block needs at least this much ink to count

LOCAL_FEATURES = ["ela_block_max_z", "ela_block_min_z", "ela_block_spread", "ela_ink_std",
                  "noise_block_min_z", "noise_low_frac"]
GLOBAL_FEATURES = ["ela_mean", "ela_ink_mean"]
SHORTCUT_FEATURE = "jpeg_luma_q"


def assign_group(row):
    if row["is_fraud"] == 0:
        return "genuine"
    if row["image_tampered"] == 1:
        return "amount_tamper" if row["tamper_field"] == "Amount Claimed" else "field_tamper"
    return "other_fraud"


def ela_features(path):
    img = Image.open(path)
    tables = getattr(img, "quantization", None) or {}
    # mean of the luminance quantization table: lower = higher save quality
    luma_q = float(np.mean(tables[0])) if 0 in tables else np.nan

    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    a = np.asarray(rgb, dtype=np.int16)
    b = np.asarray(resaved, dtype=np.int16)
    ela = np.abs(a - b).max(axis=2).astype(np.float32)
    gray = np.asarray(rgb.convert("L"))

    h, w = ela.shape
    hb, wb = h // BLOCK, w // BLOCK
    crop = (slice(0, hb * BLOCK), slice(0, wb * BLOCK))
    block_ela = ela[crop].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    block_ink = (gray[crop] < INK_THRESHOLD).reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    ink_blocks = block_ela[block_ink >= MIN_INK_FRAC]

    out = {"ela_mean": float(ela.mean()), SHORTCUT_FEATURE: luma_q}

    # noise consistency — same function the model pipeline uses
    out.update(forensic_features(rgb))
    if ink_blocks.size < 5:
        ela_keys = [k for k in LOCAL_FEATURES if k.startswith("ela_")] + ["ela_ink_mean"]
        out.update({k: np.nan for k in ela_keys})
        return out
    med = np.median(ink_blocks)
    mad = np.median(np.abs(ink_blocks - med)) * 1.4826 + 1e-6
    # the edited line could be compressed MORE or LESS than the rest of
    # the page, so measure outliers in both directions
    out.update({
        "ela_ink_mean": float(ink_blocks.mean()),
        "ela_ink_std": float(ink_blocks.std()),
        "ela_block_max_z": float((ink_blocks.max() - med) / mad),
        "ela_block_min_z": float((med - ink_blocks.min()) / mad),
        "ela_block_spread": float(np.percentile(ink_blocks, 95) - np.percentile(ink_blocks, 5)),
    })
    return out


def separation(df, feature, positives):
    """AUC of one feature for a set of positive rows vs genuine bills,
    direction-agnostic."""
    keep = (df.group == "genuine") | positives
    sub = df.loc[keep, [feature]].assign(label=positives[keep].astype(int)).dropna()
    if sub.label.nunique() < 2:
        return np.nan, ""
    auc = roc_auc_score(sub.label, sub[feature])
    return max(auc, 1 - auc), ("higher" if auc >= 0.5 else "lower")


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    base = csv.parent
    df = pd.read_csv(csv)
    df["tamper_field"] = df["tamper_field"].fillna("")
    df["group"] = df.apply(assign_group, axis=1)
    print(f"Loaded {len(df)} claims; computing ELA features at full resolution...")

    feats = []
    for i, p in enumerate(df["image_path"], 1):
        feats.append(ela_features(base / p))
        if i % 500 == 0:
            print(f"  processed {i}/{len(df)}")
    F = pd.DataFrame(feats)
    F.insert(0, "claim_id", df["claim_id"].values)
    F.to_csv(base / "ela_features.csv", index=False)
    df = pd.concat([df.reset_index(drop=True), F.drop(columns="claim_id")], axis=1)

    features = LOCAL_FEATURES + GLOBAL_FEATURES + [SHORTCUT_FEATURE]
    if "tamper_score" in df.columns:
        features.append("tamper_score")

    has_style = "tamper_style" in df.columns and df["tamper_style"].fillna("").ne("").any()
    if has_style:
        df["tamper_style"] = df["tamper_style"].fillna("")
    tampered = df.image_tampered == 1
    groups = [("amount_tamper", df.group == "amount_tamper"),
              ("field_tamper", df.group == "field_tamper")]
    if has_style:
        groups += [("naive", tampered & (df.tamper_style == "naive")),
                   ("careful", tampered & (df.tamper_style == "careful"))]

    print(f"\nSeparation from genuine bills (AUC, direction-agnostic; 0.5 = none, 1.0 = perfect)")
    header = f"{'feature':<20}" + "".join(f"{g:>15}" for g, _ in groups) + "  tampered is"
    print(header)
    rows = []
    for f in features:
        seps = [separation(df, f, mask) for _, mask in groups]
        direction = seps[1][1]
        kind = ("local" if f in LOCAL_FEATURES else "global ELA" if f in GLOBAL_FEATURES
                else "SHORTCUT CHECK" if f == SHORTCUT_FEATURE else "current CNN")
        rows.append((f, kind, [s for s, _ in seps], direction))
        print(f"{f:<20}" + "".join(f"{s:>15.3f}" for s, _ in seps) + f"  {direction:<7} ({kind})")

    local_rows = [r for r in rows if r[1] == "local"]
    best_local = max(local_rows, key=lambda r: r[2][1])          # field edits
    shortcut = next(r for r in rows if r[0] == SHORTCUT_FEATURE)
    shortcut_auc = max(shortcut[2][0], shortcut[2][1])

    best_auc = best_local[2][1]
    print("\nVerdict")
    if best_auc >= 0.85:
        verdict = (f"The edit signal IS in the images: {best_local[0]} separates field "
                   f"edits at AUC {best_auc:.2f} with no CNN at all. If the CNN scores "
                   f"far lower, the CNN pipeline is losing it — fix the detector "
                   f"(full-resolution input, crops, or fine-tuning on bills), or use "
                   f"these features directly.")
    elif best_auc >= 0.65:
        verdict = (f"Only a weak trace exists: best local feature {best_local[0]} reaches "
                   f"AUC {best_auc:.2f} on field edits. Some signal, but neither the "
                   f"CNN nor simple features will catch these edits reliably.")
    else:
        verdict = (f"The edits leave almost no detectable trace: best local feature "
                   f"reaches only AUC {best_auc:.2f} on field edits. Fix the generator "
                   f"before blaming the detector.")
    print(" ", verdict)
    shortcut_note = ""
    if shortcut_auc >= 0.70:
        shortcut_note = (f"WARNING — shortcut: JPEG save quality alone separates tampered "
                         f"bills (AUC {shortcut_auc:.2f}). Tampered bills are saved in a "
                         f"different quality range, so a detector can score the file's "
                         f"quality instead of the edit. Fix this in the generator.")
        print(" ", shortcut_note)
    else:
        print(f"  Shortcut check passed: JPEG save quality alone gives AUC {shortcut_auc:.2f}.")
    style_note = ""
    if has_style:
        best_naive = max(r[2][2] for r in local_rows)
        best_careful = max(r[2][3] for r in local_rows)
        style_note = (f"Best local feature by forger style: naive {best_naive:.2f}, "
                      f"careful {best_careful:.2f}.")
        print(f"  {style_note}")

    lines = [
        "# ELA Diagnostic — is the tamper signal in the images?\n",
        f"{len(df)} claims. Full-resolution ELA (re-save quality {ELA_QUALITY}, "
        f"{BLOCK}x{BLOCK} blocks) and noise-consistency statistics. No CNN involved.\n",
        "Separation = AUC vs genuine bills, direction-agnostic (0.5 = none, 1.0 = perfect).\n",
        "| Feature | Type | " + " | ".join(g for g, _ in groups) + " | Tampered is |",
        "|---|---|" + "---|" * len(groups) + "---|",
    ]
    for f, kind, seps, d in rows:
        lines.append(f"| `{f}` | {kind} | " + " | ".join(f"{s:.3f}" for s in seps) + f" | {d} |")
    lines += [
        "\nDecision rule (fixed before results): best local feature on field "
        "edits >= 0.85 means the signal exists; 0.65-0.85 means a weak trace; "
        "below 0.65 means the generator's edits leave almost no trace. "
        "JPEG quality alone must stay below 0.70 (shortcut check).\n",
        f"**Verdict:** {verdict}\n",
    ]
    if shortcut_note:
        lines.append(f"**{shortcut_note}**\n")
    else:
        lines.append(f"Shortcut check passed: JPEG save quality alone gives AUC {shortcut_auc:.2f}.\n")
    if style_note:
        lines.append(f"{style_note}\n")
    lines += ["## Caveat\n",
              "Synthetic bills from generate_synthetic_claims.py. A feature that works "
              "here is evidence about this generator, not about real-world bill forgery."]
    REPORT.write_text("\n".join(lines))
    print(f"\nWrote {base / 'ela_features.csv'} and {REPORT.name}")


if __name__ == "__main__":
    main()
