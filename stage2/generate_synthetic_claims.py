import io
import random
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42)

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMG_DIR = DATA_DIR / "images"
CLAIMS_CSV = DATA_DIR / "claims.csv"

N_CLAIMS = 3000
FRAUD_RATE = 0.18

# compression ranges — identical for genuine and tampered bills (v3 fix)
SCAN_QUALITY = (75, 95)     # first compression, when the bill is scanned
RESAVE_QUALITY = (65, 92)   # second compression, upload / forward / re-save

HOSPITALS = [f"{name} Multispeciality Hospital" for name in
             ["Sunrise", "Shivneri", "Jeevan", "Sahyadri", "Ashirwad",
              "Krishna", "Om", "Sanjeevani", "Lifeline", "Vishwas"]]
STATES = ["Maharashtra", "Gujarat", "Karnataka", "Uttar Pradesh", "Bihar",
          "Madhya Pradesh", "Rajasthan", "Tamil Nadu"]

# (diagnosis, procedure, ICD-10-like code, typical PM-JAY package rate INR, typical LOS days)
PROCEDURES = [
    ("Acute appendicitis", "Appendectomy", "K35.8", 18000, 3),
    ("Cataract", "Phacoemulsification with IOL", "H25.9", 10000, 1),
    ("Gallstones", "Laparoscopic cholecystectomy", "K80.2", 28000, 3),
    ("Fracture femur", "Internal fixation", "S72.3", 45000, 6),
    ("Hernia", "Hernioplasty", "K40.9", 20000, 2),
    ("Coronary artery disease", "Angioplasty", "I25.1", 90000, 3),
    ("Normal delivery", "Vaginal delivery", "O80", 9000, 2),
    ("C-section", "Caesarean delivery", "O82", 22000, 4),
    ("Tonsillitis", "Tonsillectomy", "J35.0", 12000, 2),
    ("Kidney stones", "PCNL", "N20.0", 35000, 3),
]

FIRST_NAMES = ["Amit", "Priya", "Ravi", "Sneha", "Vikram", "Anjali", "Suresh",
               "Pooja", "Rahul", "Kavita", "Manoj", "Deepa", "Arjun", "Meena"]
LAST_NAMES = ["Sharma", "Patil", "Kumar", "Singh", "Deshmukh", "Yadav",
              "Iyer", "Reddy", "Joshi", "Mishra"]

# fields a "field" tamper can target, besides the amount
TAMPERABLE_FIELDS = ["Patient Name", "Diagnosis", "Diagnosis Code"]


def make_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_bill(fields):
    """Draw a clean hospital-bill-style image. Returns (image,
    field_positions): every printed label -> (y, value_x)."""
    W, H = 800, 600
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title_font = make_font(26)
    font = make_font(18)

    draw.text((40, 30), fields["hospital"], font=title_font, fill="black")
    draw.line((40, 70, 760, 70), fill="black", width=2)

    rows = [
        ("Claim ID", fields["claim_id"]),
        ("Patient Name", fields["patient_name"]),
        ("Age", str(fields["age"])),
        ("Diagnosis", fields["diagnosis"]),
        ("Procedure", fields["procedure"]),
        ("Diagnosis Code", fields["diagnosis_code"]),
        ("Admission Date", fields["admission_date"]),
        ("Discharge Date", fields["discharge_date"]),
    ]
    field_positions = {}
    y = 100
    for label, value in rows:
        draw.text((40, y), f"{label}:", font=font, fill="black")
        draw.text((280, y), value, font=font, fill="black")
        field_positions[label] = (y, 280)
        y += 34

    amount_y = y + 20
    draw.line((40, amount_y - 10, 760, amount_y - 10), fill="black", width=1)
    draw.text((40, amount_y), "Amount Claimed (INR):", font=font, fill="black")
    draw.text((320, amount_y), f"{fields['claimed_amount']:,}", font=font, fill="black")
    field_positions["Amount Claimed"] = (amount_y, 320)

    return img, field_positions


def random_scan_params():
    base = random.randint(236, 250)
    return {
        "tint": tuple(min(255, base + random.randint(-2, 4)) for _ in range(3)),
        "blur": random.uniform(0.3, 0.8),
        "noise": random.uniform(2.0, 5.0),
        "seed": random.randrange(2 ** 32),
    }


def add_noise(img, sigma, np_rng):
    arr = np.asarray(img, dtype=np.float32)
    noise = np_rng.normal(0.0, sigma, arr.shape[:2] + (1,))
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_scan(img, scan, np_rng):
    """Make a clean render look scanned: paper tint, slight blur, noise."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tinted = Image.fromarray((arr * np.array(scan["tint"], dtype=np.float32)).astype(np.uint8))
    blurred = tinted.filter(ImageFilter.GaussianBlur(scan["blur"]))
    return add_noise(blurred, scan["noise"], np_rng)


def apply_tamper(img, field_positions, field_label, new_value_text, scan, careful, np_rng):
    """A real edit on the already-compressed scan: cover one printed value
    with a patch and draw a different value on it.

    naive   — patch is flat paper colour with crisp text (no scan noise).
    careful — patch gets blur and noise close to the scan's own, so only
              the compression-history difference is left."""
    y, value_x = field_positions[field_label]
    x0, y0, x1, y1 = value_x - 5, y - 4, 755, y + 24
    patch = Image.new("RGB", (x1 - x0, y1 - y0), scan["tint"])
    ImageDraw.Draw(patch).text((value_x - x0, y - y0), new_value_text,
                               font=make_font(18), fill="black")
    if careful:
        patch = patch.filter(ImageFilter.GaussianBlur(scan["blur"] * random.uniform(0.8, 1.2)))
        patch = add_noise(patch, scan["noise"] * random.uniform(0.8, 1.2), np_rng)
    img = img.copy()
    img.paste(patch, (x0, y0))
    return img


def random_dates():
    return random.randint(1, 300)


def pick_different(current, options):
    choices = [o for o in options if o != current]
    return random.choice(choices) if choices else current


def generate_claim(claim_id, fraud_pool_state):
    diagnosis, procedure, code, package_rate, typical_los = random.choice(PROCEDURES)
    hospital = random.choice(HOSPITALS)
    state = random.choice(STATES)
    patient_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    age = random.randint(1, 85)
    provider_id = f"H{HOSPITALS.index(hospital):03d}"

    admission_day = random_dates()
    los = max(1, int(random.gauss(typical_los, 1)))
    discharge_day = admission_day + los

    fraud_type = "none"
    is_fraud = 0
    is_tampered = False
    tamper_field = None
    tamper_new_value = None
    tamper_style = None

    # Some genuine claims legitimately cost more / stay longer
    # (complications) — deliberately overlapping mild fraud ranges.
    if random.random() < 0.08:
        claimed_amount = int(package_rate * random.uniform(1.25, 1.7))
        los = max(los, int(typical_los * random.uniform(1.3, 2.2)))
        discharge_day = admission_day + los
    else:
        claimed_amount = int(package_rate * random.uniform(0.85, 1.25))

    if random.random() < FRAUD_RATE:
        is_fraud = 1
        fraud_type = random.choice(
            ["upcoding", "phantom_billing", "impossible_los", "document_tamper"]
        )
        if fraud_type == "upcoding":
            claimed_amount = int(package_rate * random.uniform(1.15, 2.5))
        elif fraud_type == "phantom_billing":
            # amount inflated while the documented stay is shorter than typical
            claimed_amount = int(package_rate * random.uniform(1.05, 1.6))
            los = max(1, typical_los - random.randint(0, 2))
            discharge_day = admission_day + los
        elif fraud_type == "impossible_los":
            los = max(typical_los + 1, int(typical_los * random.uniform(1.8, 4.0)))
            discharge_day = admission_day + los
            claimed_amount = int(package_rate * random.uniform(1.1, 2.0))
        elif fraud_type == "document_tamper":
            # the claim RECORD stays correct — only the bill image is forged
            is_tampered = True
            tamper_style = "careful" if random.random() < 0.5 else "naive"
            if random.random() < 0.5:
                tamper_field = "Amount Claimed"
                tamper_new_value = f"{int(claimed_amount * random.uniform(1.05, 1.7)):,}"
            else:
                tamper_field = random.choice(TAMPERABLE_FIELDS)
                if tamper_field == "Patient Name":
                    forged = patient_name
                    while forged == patient_name:
                        forged = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
                    tamper_new_value = forged
                elif tamper_field == "Diagnosis":
                    tamper_new_value = pick_different(diagnosis, [p[0] for p in PROCEDURES])
                else:
                    tamper_new_value = pick_different(code, [p[2] for p in PROCEDURES])

    fields = {
        "claim_id": f"CLM{claim_id:06d}",
        "patient_name": patient_name,
        "age": age,
        "hospital": hospital,
        "diagnosis": diagnosis,
        "procedure": procedure,
        "diagnosis_code": code,
        "admission_date": f"Day {admission_day}",
        "discharge_date": f"Day {discharge_day}",
        "claimed_amount": claimed_amount,
    }

    row = {
        "claim_id": fields["claim_id"],
        "provider_id": provider_id,
        "hospital": hospital,
        "state": state,
        "patient_age": age,
        "diagnosis_code": code,
        "procedure": procedure,
        "package_rate": package_rate,
        "claimed_amount": claimed_amount,
        "length_of_stay": los,
        "typical_los": typical_los,
        "amount_to_package_ratio": round(claimed_amount / package_rate, 3),
        "los_ratio": round(los / typical_los, 3),
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
        "image_tampered": 1 if is_tampered else 0,
        "tamper_field": tamper_field or "",
        "tamper_style": tamper_style or "",
        "image_path": f"images/{fields['claim_id']}.jpg",
    }

    # Every bill: render -> scan (tint, blur, noise) -> first JPEG
    # compression -> [edit, if tampered] -> second compression on save.
    # Both qualities come from the same ranges for every bill, so the
    # compression history itself never reveals the label.
    scan = random_scan_params()
    np_rng = np.random.default_rng(scan["seed"])
    img, field_positions = render_bill(fields)
    img = apply_scan(img, scan, np_rng)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(*SCAN_QUALITY))
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if is_tampered:
        img = apply_tamper(img, field_positions, tamper_field, tamper_new_value,
                           scan, tamper_style == "careful", np_rng)
    save_quality = random.randint(*RESAVE_QUALITY)
    return row, img, save_quality


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, N_CLAIMS + 1):
        row, img, quality = generate_claim(i, None)
        img.save(IMG_DIR / f"{row['claim_id']}.jpg", "JPEG", quality=quality)
        rows.append(row)
        if i % 500 == 0:
            print(f"  generated {i}/{N_CLAIMS}")

    with open(CLAIMS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_fraud = sum(r["is_fraud"] for r in rows)
    tampered = [r for r in rows if r["image_tampered"]]
    n_amount = sum(1 for r in tampered if r["tamper_field"] == "Amount Claimed")
    n_careful = sum(1 for r in tampered if r["tamper_style"] == "careful")
    print(f"\nWrote {len(rows)} claims to {CLAIMS_CSV}")
    print(f"Fraud rate: {n_fraud}/{len(rows)} ({n_fraud/len(rows):.1%})")
    print(f"Tampered bills: {len(tampered)} "
          f"({n_amount} amount edits, OCR-catchable; "
          f"{len(tampered) - n_amount} other-field edits, image-forensics-only)")
    print(f"Forger style: {len(tampered) - n_careful} naive, {n_careful} careful")
    print(f"Images saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
