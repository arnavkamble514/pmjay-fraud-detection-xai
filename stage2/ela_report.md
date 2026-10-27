# ELA Diagnostic — is the tamper signal in the images?

3000 claims. Full-resolution ELA (re-save quality 90, 16x16 blocks) and noise-consistency statistics. No CNN involved.

Separation = AUC vs genuine bills, direction-agnostic (0.5 = none, 1.0 = perfect).

| Feature | Type | amount_tamper | field_tamper | naive | careful | Tampered is |
|---|---|---|---|---|---|---|
| `ela_block_max_z` | local | 0.525 | 0.533 | 0.521 | 0.538 | lower |
| `ela_block_min_z` | local | 0.544 | 0.570 | 0.506 | 0.512 | lower |
| `ela_block_spread` | local | 0.513 | 0.591 | 0.511 | 0.565 | higher |
| `ela_ink_std` | local | 0.513 | 0.594 | 0.516 | 0.563 | higher |
| `noise_block_min_z` | local | 0.833 | 0.790 | 0.953 | 0.642 | higher |
| `noise_low_frac` | local | 0.829 | 0.748 | 0.940 | 0.610 | higher |
| `ela_mean` | global ELA | 0.533 | 0.541 | 0.515 | 0.563 | higher |
| `ela_ink_mean` | global ELA | 0.508 | 0.577 | 0.512 | 0.574 | higher |
| `jpeg_luma_q` | SHORTCUT CHECK | 0.541 | 0.528 | 0.542 | 0.527 | lower |

Decision rule (fixed before results): best local feature on field edits >= 0.85 means the signal exists; 0.65-0.85 means a weak trace; below 0.65 means the generator's edits leave almost no trace. JPEG quality alone must stay below 0.70 (shortcut check).

**Verdict:** Only a weak trace exists: best local feature noise_block_min_z reaches AUC 0.79 on field edits. Some signal, but neither the CNN nor simple features will catch these edits reliably.

Shortcut check passed: JPEG save quality alone gives AUC 0.54.

Best local feature by forger style: naive 0.95, careful 0.64.

## Caveat

Synthetic bills from generate_synthetic_claims.py. A feature that works here is evidence about this generator, not about real-world bill forgery.