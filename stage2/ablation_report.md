# Ablation Study — Does Multimodal Fusion Help?

5-fold cross-validated AUC, 3000 claims, fraud rate 18.4%.

| Feature set | AUC | Std dev | # numeric features |
|---|---|---|---|
| A. Structured only | 0.8045 | 0.0274 | 9 |
| B. Structured + OCR | 0.9399 | 0.0153 | 12 |
| C. Structured + OCR + tamper (full model) | 0.9375 | 0.0147 | 13 |

**Adding OCR features: +0.1354 AUC — a real, meaningful improvement.**

**Adding the tamper CNN score: -0.0024 AUC — essentially no difference — within noise.**

**Total gain from structured-only to the full model: +0.1330 AUC.**

## What this means

The multimodal fusion claim holds up on this synthetic dataset: combining OCR and tamper-detection signal with structured claim data measurably outperforms structured data alone.

## Caveat

This test is on synthetic data from generate_synthetic_claims.py. The synthetic document-tamper fraud type is specifically designed to be caught by the tamper CNN, so any apparent benefit here may not transfer to real-world documents with more varied tampering patterns.


![Ablation chart](ablation_chart.png)
