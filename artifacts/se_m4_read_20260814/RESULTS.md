# SE M4 PERSISTENCE READ (registered secondary) — 14 Aug 2026

**Governance:** parent prereg v2.2 §5 (M4) operationalized in
`PREREG_nfi_scale_exhibit_amend2_20260814.md` §3; frozen collator
`uncfield/se_m4_read.py` built + selfchecked + reviewed (#23, all
findings applied, selfcheck re-PASS) BEFORE this single execution.
Inputs: bundle `uncfield_se_m4_20260814_174131` (64/64 sha OK,
committed manifest; 16 snapshot probe passes executed on the Midway3-
synced run dirs per the pinned protocol) + the Stage-1 bundle for the
100% byte-identity gate.

## VERDICT

**WEAK FORM (registered decision content) HOLDS: 8/8** — the
distractor's share is above its per-run permutation-null median at
100% in every seed (restated from inputs byte-identical to the
Stage-1 read's).

**Trajectory (reporting): the misprice is FULLY FORMED BY 25% OF
TRAINING and statistically FLAT thereafter — training does not remove
it at any observed stage.**

| fraction | distractor share (null median) | indicator | θ₁ (vs source) |
|---|---|---|---|
| 25% (realized 24.7–25.5%) | 0.622 (0.320) | **8/8** | 5.24× |
| 50% (realized 50.1–50.6%) | 0.644 (0.321) | **8/8** | 5.81× |
| 100% | 0.611 (0.320) | **8/8** | 5.26× |

Non-decay descriptive (registered labels): share₁₀₀ − share₂₅ =
−0.011, seed-level BCa [−0.033, +0.009], n=8 → **FLAT-OR-MIXED** —
no rise, no decay.

D-family: 0/8 at every fraction (shares 0.23–0.24, below the ~0.32
null median throughout) — duplicates are never farmed at any training
stage, extending the Stage-1 informative null across the training
trajectory.

## Gates (all green)

Zero invalid cells (n_eval 512 ×24; calibration within acceptance at
every fraction incl. the 25% checkpoints); probe seeds 0 ×24;
snapshot selection VERIFIED-NEAREST ×16 against the synced
ckpt_snapshots manifests; realized fractions within ±0.006 of
targets; cross-fraction raw_norms/norm_floor bit-identity (the
fixed-eval-distribution witness, review #23 M4) ×8; 100% inputs
BYTE-IDENTICAL to Stage-1 ×8 (which carries the final-ckpt gate and
the fit counters: 99.5–99.9% ×8, flag OK). Probe stdout not bundled
(same disclosure as Stage-1).

## Licensed wording (NFI paper, external-validity section)

"The misprice is not a transient of early training: it is fully
formed by 125k of 500k steps and statistically flat thereafter (8/8
seeds above the permutation null at 25%, 50%, and 100% of training;
share 0.62 → 0.64 → 0.61, non-decay CI [−0.033, +0.009]; overpricing
5.2× → 5.8× → 5.3× the source key) — the deployed objective's own
training does not remove the fictitious value it assigns."
This is the parent §2 "finite-time non-decay" claim in its strongest
observable form. NOT licensed: any claim about formation BEFORE 125k
steps (no earlier checkpoint retained).
