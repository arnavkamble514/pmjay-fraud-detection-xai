"""
Stage 2, step 1 — synthetic PM-JAY claim generator.

No public row-level PM-JAY dataset exists, so this creates a structurally
realistic one: N claims with patient/hospital/procedure/amount fields,
each with a matching generated "hospital bill" image. A subset of claims
is deliberately made fraudulent using patterns real fraud detection
literature actually documents (upcoding, phantom/duplicate billing,
impossible length-of-stay, document tampering). Ground truth is known
because we generated it.

Outputs:
  stage2/data/claims.csv        one row per claim, structured fields + label
  stage2/data/images/<id>.png   one bill image per claim

Run:  python generate_synthetic_claims.py
"""

import io
import random
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMG_DIR = DATA_DIR / "images"
CLAIMS_CSV = DATA_DIR / "claims.csv"

N_CLAIMS = 3000
FRAUD_RATE = 0.18

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


def make_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_bill(fields):
    """Draw a simple hospital-bill-style image with the genuine amount.
    Returns (image, amount_y) so a tamper edit can target the amount."""
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
    y = 100
    for label, value in rows:
        draw.text((40, y), f"{label}:", font=font, fill="black")
        draw.text((280, y), value, font=font, fill="black")
        y += 34

    amount_y = y + 20
    draw.line((40, amount_y - 10, 760, amount_y - 10), fill="black", width=1)
    draw.text((40, amount_y), "Amount Claimed (INR):", font=font, fill="black")
    amount_text = f"{fields['claimed_amount']:,}"
    draw.text((320, amount_y), amount_text, font=font, fill="black")

    return img, amount_y


def apply_tamper(img, amount_y, tampered_amount):
    """A real edit: white out the true amount and redraw a different
    figure over it, on the already-compressed image."""
    draw = ImageDraw.Draw(img)
    font = make_font(18)
    draw.rectangle((315, amount_y - 4, 500, amount_y + 24), fill="white")
    draw.text((320, amount_y), f"{tampered_amount:,}", font=font, fill="black")
    return img


def random_dates():
    start_day = random.randint(1, 300)
    return start_day


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
    tampered_amount = None

    # Genuine claims aren't all identical: some legitimately have
    # complications (higher cost, longer stay) without being fraud.
    # These deliberately reach into the same numeric territory as mild
    # fraud below, so the model can't just threshold on the ratio.
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
            # overlaps the genuine "complicated case" range above —
            # a ratio alone can't separate them
            claimed_amount = int(package_rate * random.uniform(1.15, 2.5))
        elif fraud_type == "phantom_billing":
            claimed_amount = int(package_rate * random.uniform(1.05, 1.6))
            # shorter than typical, but not always a fixed giveaway of 1
            los = max(1, typical_los - random.randint(0, 2))
            discharge_day = admission_day + los
        elif fraud_type == "impossible_los":
            los = max(typical_los + 1, int(typical_los * random.uniform(1.8, 4.0)))
            discharge_day = admission_day + los
            claimed_amount = int(package_rate * random.uniform(1.1, 2.0))
        elif fraud_type == "document_tamper":
            # the claim record looks fine; the bill IMAGE is altered —
            # sometimes by a lot, sometimes just a subtle nudge
            tampered_amount = int(claimed_amount * random.uniform(1.05, 1.7))

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
        "image_tampered": 1 if tampered_amount is not None else 0,
        "image_path": f"images/{fields['claim_id']}.jpg",
    }

    # Give every bill a JPEG history, like real scanned/photographed
    # documents (and like CASIA). Genuine: compressed at q90, saved at
    # q90. Tampered: edited after the first compression, then re-saved
    # at a different quality — the double-compression trace ELA reads.
    img, amount_y = render_bill(fields)
    first_pass_quality = random.randint(78, 95)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=first_pass_quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if tampered_amount is not None:
        img = apply_tamper(img, amount_y, tampered_amount)
        # re-saved after editing, at an independently random quality —
        # overlaps the genuine range, so quality alone isn't a tell
        save_quality = random.randint(65, 88)
    else:
        save_quality = first_pass_quality
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
    print(f"\nWrote {len(rows)} claims to {CLAIMS_CSV}")
    print(f"Fraud rate: {n_fraud}/{len(rows)} ({n_fraud/len(rows):.1%})")
    print(f"Images saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
