import re
import cv2
import easyocr
import numpy as np

_reader = None


def get_reader(use_gpu=True):
    """EasyOCR model loads once and is reused — loading is slow."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=use_gpu)
    return _reader


def preprocess(image_path):
    """
    Clean up a scanned/photographed document before OCR.

    Medical documents are often photographed at an angle under bad
    light. Grayscale + adaptive threshold + denoise recovers a lot of
    accuracy on exactly that kind of input.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=11,
    )
    return thresh


def run_ocr(image_path, use_preprocessing=True, use_gpu=True):
    """
    Extract all text from a document image.

    Returns:
        {
          "text": full extracted text as one string,
          "lines": [(text, confidence), ...],
          "mean_confidence": float 0-1,
          "low_conf_ratio": fraction of lines below 0.5 confidence
        }

    mean_confidence and low_conf_ratio become features in Stage 2 —
    a re-saved or digitally edited document often OCRs with
    noticeably lower confidence than a clean original.
    """
    reader = get_reader(use_gpu=use_gpu)
    source = preprocess(image_path) if use_preprocessing else image_path
    results = reader.readtext(source, detail=1)

    lines = [(text, float(conf)) for _, text, conf in results]
    confs = [c for _, c in lines]

    return {
        "text": " ".join(t for t, _ in lines),
        "lines": lines,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "low_conf_ratio": (
            float(sum(c < 0.5 for c in confs) / len(confs)) if confs else 1.0
        ),
    }


# --- Field extraction -------------------------------------------------
# Stage 2 needs structured fields, not raw text, so it can cross-check
# what the document SAYS against what the claim row CLAIMS.

AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹)\s?([\d,]+(?:\.\d{1,2})?)", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
ICD_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b")


def extract_claim_fields(ocr_text):
    """
    Pull structured fields out of raw OCR text.

    Returns dict with amounts, dates, icd_codes, patient_name_guess.
    All are best-effort — Stage 2 treats a miss as its own signal.
    """
    amounts = [
        float(a.replace(",", "")) for a in AMOUNT_RE.findall(ocr_text)
    ]
    dates = DATE_RE.findall(ocr_text)
    icd_codes = ICD_RE.findall(ocr_text)

    name = None
    m = re.search(
        r"(?:patient\s*name|name\s*of\s*patient|patient)\s*[:\-]\s*([A-Za-z .]{3,40})",
        ocr_text, re.I,
    )
    if m:
        name = m.group(1).strip()

    return {
        "amounts": amounts,
        "max_amount": max(amounts) if amounts else None,
        "dates": dates,
        "icd_codes": icd_codes,
        "patient_name_guess": name,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)

    out = run_ocr(sys.argv[1])
    print(f"Mean confidence : {out['mean_confidence']:.3f}")
    print(f"Low-conf ratio  : {out['low_conf_ratio']:.3f}")
    print(f"\nText:\n{out['text'][:800]}")
    print(f"\nFields: {extract_claim_fields(out['text'])}")
