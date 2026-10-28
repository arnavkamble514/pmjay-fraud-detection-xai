import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from ocr_pipeline import run_ocr, extract_claim_fields

DATASET = "devp1866/noisy-medical-document-images-ocr"
DATA_DIR = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "benchmark_results.json"
SAMPLE_SIZE = 40  # raise once you've confirmed it runs end to end

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*")):
        print(f"Dataset already present at {DATA_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    print("Download complete.")


def find_images(root):
    return sorted(
        p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def benchmark(images, use_preprocessing):
    label = "with preprocessing" if use_preprocessing else "raw"
    print(f"\n--- Running {label} ({len(images)} images) ---")

    confs, low_ratios, field_hits, times = [], [], [], []

    for i, img in enumerate(images, 1):
        try:
            t0 = time.time()
            out = run_ocr(str(img), use_preprocessing=use_preprocessing)
            times.append(time.time() - t0)

            confs.append(out["mean_confidence"])
            low_ratios.append(out["low_conf_ratio"])

            fields = extract_claim_fields(out["text"])
            # how many of the 4 target field types we recovered
            hits = sum([
                bool(fields["amounts"]),
                bool(fields["dates"]),
                bool(fields["icd_codes"]),
                bool(fields["patient_name_guess"]),
            ])
            field_hits.append(hits / 4.0)

            if i % 10 == 0:
                print(f"  {i}/{len(images)} done")
        except Exception as e:
            print(f"  skipped {img.name}: {e}")

    n = len(confs) or 1
    return {
        "config": label,
        "images_processed": len(confs),
        "mean_confidence": sum(confs) / n,
        "mean_low_conf_ratio": sum(low_ratios) / n,
        "mean_field_recovery": sum(field_hits) / n,
        "mean_seconds_per_image": sum(times) / (len(times) or 1),
    }


def main():
    download_dataset()

    images = find_images(DATA_DIR)
    if not images:
        print(f"No images found under {DATA_DIR}. Check the download.")
        sys.exit(1)

    print(f"Found {len(images)} images; sampling {SAMPLE_SIZE}.")
    sample = images[:SAMPLE_SIZE]

    results = [
        benchmark(sample, use_preprocessing=False),
        benchmark(sample, use_preprocessing=True),
    ]

    RESULTS.write_text(json.dumps(results, indent=2))

    print("\n=== Benchmark summary ===")
    for r in results:
        print(f"\n{r['config']}:")
        print(f"  images processed      : {r['images_processed']}")
        print(f"  mean OCR confidence   : {r['mean_confidence']:.3f}")
        print(f"  mean low-conf ratio   : {r['mean_low_conf_ratio']:.3f}")
        print(f"  mean field recovery   : {r['mean_field_recovery']:.3f}")
        print(f"  seconds per image     : {r['mean_seconds_per_image']:.2f}")

    print(f"\nSaved to {RESULTS}")
    print("\nUse the config with higher field recovery in Stage 2.")
    print("Report both numbers in your project report.")


if __name__ == "__main__":
    main()
