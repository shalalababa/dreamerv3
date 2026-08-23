# B4 (aleatoric repair, Amendment-1 share form) — REGISTERED READ,
23 Aug 2026

**Prereg:** `PREREG_trackB_alea_20260822.md` + Amendment 1 +
Amendment 2 (the cross-instrument tolerance recalibration after a
registered refusal — see below). **Reader:**
`uncfield/se_b4_read.py`; bundle `b4_alea_20260823_035808` verified
byte-exact; 4/4 runs, both instruments per run, ckpt-identity green.

## Execution history (fully disclosed)

1. **First invocation REFUSED (execution preserved by the M13
   side-file rule):** the R-A1-M7a cross-instrument gate fired at
   its 1e-4 tolerance on all 4 runs. Diagnosis (all-keys table):
   mixed-sign ~1–2 % divergence on the independent channels
   (distractor, velocity) with the position-correlated dups at
   ±0.1–0.3 % — bf16 graph-order noise between the two compiled
   instruments, not form drift. Amendment 2 recalibrated the
   tolerance to 5e-2 (the gate's purpose — ≥50 % form drift —
   intact); the within-instrument float64 share provenance passed
   4/4 throughout.
2. **Second invocation EXECUTED and CONSUMED the one execution:
   outcome cell = REPAIR-BY-DEATH** (collapsed: s120, s121, s122
   on the intrinsic leg).

## CORRECTION (same day): the REPAIR-BY-DEATH cell is
INSTRUMENT-MISCALIBRATED — its physical claim is NOT supported

The collapse leg compares the mask's `base_raw` (ANCHOR-point
var-of-μ: 1.52–1.96e-4) against the Stage-1 reference
`intrinsic_min` (1.885e-3), which is the se_probe pse2 intrinsic —
an IMAGINED-ROLLOUT statistic. The R-A1-B4 revision made the leg
unit-commensurable (both raw variances) but NOT operating-point
commensurable: imagined-rollout disagreement runs ~10× anchor-point
disagreement for a healthy det WM too. The selfcheck could not
catch this because its fixtures set both scales by construction
(lesson banked). The substrate is in fact MANIFESTLY HEALTHY by
the read's own rows: θ₁(distractor) = 4.21/5.74/5.47/5.50 —
squarely in the Stage-1 det band (4.72–5.64) — real levels
3.6–5.4e-3 (reference floor 2.9e-4), raw distractor shares
0.40–0.43 (det: 0.43–0.49), clip fractions 0.04–0.07 (in-support).
The adjudicated cell STANDS as the registered outcome of this
execution; its INTERPRETATION ("the gauss arm did not preserve the
substrate") is withdrawn by this dated correction. The registered
primary (the share-form repair ratio) was NEVER adjudicated (the
gate precedes it) and, with the execution consumed, will not be.

## The descriptive answer (pre-branch rows, R-A1-m17 — reported,
never adjudicated; value-aware status disclosed)

| run | share_raw(d) | share_norm(d) | ratio |
|---|---|---|---|
| s120 | 0.401 | 0.427 | 1.07 |
| s121 | 0.432 | 0.456 | 1.06 |
| s122 | 0.433 | 0.461 | 1.06 |
| s123 | 0.432 | 0.454 | 1.05 |

**The aleatoric whitening does NOT repair in the deployed world
model — the distractor's share of decoder-projected ensemble
variance is unchanged-to-slightly-higher (≈1.06×) in 4/4.** The
toy verified the mechanism 448× in isolation; at the real
representation it does nothing. Reading (descriptive): σ's learned
per-latent-dim floor does not align with the distractor's latent
footprint — the misprice is REPRESENTATIONALLY ENTANGLED (the
distractor's variance is carried by latent dims shared with real
features, so per-dim aleatoric normalization cannot isolate it).
This composes cleanly with the program's ledger: the misprice is a
level/allocation property (det baseline), proxy-borne
(APT/coupling rows), dose-onset-shaped (B1/onset), behaviorally
consequential (A1), and now — descriptively — NOT removable by
per-dim aleatoric normalization at the deployed representation.
Masked rows: ≈0 as predicted (the construction-inert
exchangeability control behaved).

## Consequences

- The constructive column reports as: mechanism-verified repair
  design; deployment-level failure; the failure itself is the
  taxonomy finding (descriptive, so worded as such — the
  registered primary terminated at a mis-calibrated validity gate,
  disclosed).
- No further B4 compute. The gauss-arm checkpoints remain
  available for any future registered follow-up (e.g., a
  subspace-aware normalization), none planned.
- LESSON (banked): a cross-statistic reference leg must match the
  OPERATING POINT, not just the units; selfcheck fixtures must
  draw reference and measured values from their REAL distributions
  rather than matching scales by construction.

Artifacts: `se_b4_read.json` (the consumed execution) +
`se_b4_read_NOTADJ_20260823T090742.json` (the first refusal).
