# TM2 bridge — registered ONE read (16 Aug 2026)

Registration: PREREG_tm2_bridge_20260812 (+ amend1) — freeze 7b51ad58 /
21f4ac88, both pre-wave. Reader tm2_bridge_read.py (same freeze, clean),
selfcheck PASS pre-execution. Bundles: fits tm2_bridge_20260815_200027
(200 files sha-OK, 48/48 fits) + adapts tm2_bridge_20260816_173625
(488 files sha-OK, 48/48 adapts scores=112-equivalent, collate 48/48 QC
pass; this bundle is self-contained and supplied both inputs). Fit
witnesses 48/48 update==total + TM2_FIT_DONE.

## VERDICTS

- **P-F2 [free − rec] = INDETERMINATE**: +16.04 [−8.12, +35.43],
  perm p=.175, d_z 0.35, n=16; MDE80 33.9 score units; closure fraction
  of the aware−free gap by the rec arm = 14.8%. Decoder-parity
  reconstruction added to the reward-free objective does NOT
  detectably recover adaptation value (observed effect less than half
  the MDE; direction if anything free ≥ rec).
- **P-F1 = POSITIVE-NOT-RESOLVED**: aware − free = +115.86
  [+84.4, +157.3], perm p=3.1e-05, d_z 1.50 — the family gap REPLICATES
  at full size; ratio vs the pinned anchor gap (108.3385) = 1.069
  [0.779, 1.452] (CI spans 1 — attenuation neither confirmed nor
  excluded).
- aware − rec = +131.9 [+98.2, +171.8], p=3.1e-05. Cell means:
  aware 196.0/241.0, free 87.9/117.4, rec 86.6/86.7.

Consequence: the reward-aware objective carries the family's adaptation
value; bolting a reconstruction pathway onto the reward-free objective
does not re-open the boundary at this dose/power. Input-production
conformance: collate + counters produced locally from the sha-verified
bundle (registered forms; house precedent).
