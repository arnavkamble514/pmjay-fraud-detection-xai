"""
Stage 3 — minimal Streamlit demo.

One page: fill in a claim, optionally upload a bill image, see the
fraud score, risk band, and the top reasons behind it. This is the
smallest possible thing that proves the pipeline end to end in a UI,
not the final app — Purva builds the review queue / batch upload /
overview pages around this.

Run (from stage2/, with claimguard-env active):
    pip install streamlit   # if not already installed
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from predict import score_claim, ClaimScorer

st.set_page_config(page_title="ClaimGuard", page_icon="🩺", layout="centered")

st.title("ClaimGuard")
st.caption("PM-JAY fraud risk scoring — demo on synthetic claims")

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Patient age", min_value=0, max_value=120, value=45)
        procedure = st.text_input("Procedure", value="Laparoscopic cholecystectomy")
        diagnosis_code = st.text_input("Diagnosis code", value="K80.2")
        state = st.text_input("State", value="Maharashtra")
    with col2:
        provider_id = st.text_input("Provider ID", value="H003")
        claimed_amount = st.number_input("Claimed amount (INR)", min_value=0, value=52000)
        length_of_stay = st.number_input("Length of stay (days)", min_value=1, value=9)

    bill_image = st.file_uploader("Bill image (optional)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Score this claim")

if submitted:
    claim = {
        "patient_age": patient_age,
        "procedure": procedure,
        "diagnosis_code": diagnosis_code,
        "state": state,
        "provider_id": provider_id,
        "claimed_amount": claimed_amount,
        "length_of_stay": length_of_stay,
    }

    image_path = None
    if bill_image is not None:
        suffix = Path(bill_image.name).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(bill_image.getvalue())
        tmp.close()
        image_path = tmp.name

    try:
        with st.spinner("Scoring..." if image_path is None else "Running OCR + tamper check..."):
            result = score_claim(claim, image_path=image_path)
    except FileNotFoundError as e:
        st.error(
            f"Model files not found: {e}\n\n"
            f"Run `python train_final_model.py` in this folder first."
        )
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    score = result["score"]
    band = result["risk_band"]
    band_color = {"low": "green", "medium": "orange", "high": "red"}[band]

    st.markdown(f"### Risk: :{band_color}[{band.upper()}]  —  score {score:.2f}")
    st.progress(min(score, 1.0))

    if not result["image_checked"]:
        st.info("No bill image was checked for this claim. Upload one above for the full score.")

    st.markdown("**Why:**")
    for reason in result["top_reasons"]:
        arrow = "🔺" if reason["direction"] == "raises risk" else "🔻"
        st.write(f"{arrow} {reason['detail']}")

    with st.expander("Raw output"):
        st.json(result)

st.divider()
st.caption(
    "Trained and evaluated on synthetic PM-JAY-style claims — no real claims data "
    "was used. Scores are a pipeline demonstration, not a validated prediction."
)
