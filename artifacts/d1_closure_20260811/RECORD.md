# D1 closure — proprio leak-excluded recompute + X2 anchor — 2026-08-11

Executes the remaining INSTANCE items of review finding **D1** (probe
holdout leak; `Review_Resolution_20260808` — the registered
leak-exclusion sensitivity rule). LABELED SENSITIVITY: no verdict
re-adjudicated anywhere. Script archived at `scripts/d1_closure.py`;
inputs: `paper1_d1_closure_payload_20260810_232932` (sha-manifest
verified, 48 errors.npz + proprio probeset npz), the regen bundle's
proprio errors, the committed `d1_overlap.json` census, local
`fingerpx_v1`/`finger_v1` probeset manifests (finger_v1 manifest sha
a78fe58ca526… ≡ the E4 summaries' pin — chain closed).

## Leg A — proprio finger_v1, q1 TASK fits (restored ORIGINAL weights)

Leak = the 7 side-1 fit-buffer episodes that are also probe episodes
(committed census; side-0 overlap = 0). h0 in-regime rew-NLL,
frame-weighted, per-side means over 8 fits:

- s0: 1.8073 → 1.8073 (**delta 0.0 exactly** — the registered
  overlap-0 identity holds).
- s1: 1.0226 → 1.0606 (**delta +0.038 nats**).

Every full value reproduces the archived E4 summaries bitwise
(max reproduction error 0.0). Consequence: the proprio member-band
facts (≤1.5 bar, 0.827 anchor) are leak-insensitive — the D1 objection
is discharged on the proprio side with a +0.04-nat worst case.

## Leg B — X2 anchor recompute (pxpx, the missing piece)

- s0: 41.155 → 41.337 (+0.18); s1: 5.523 → 7.376 (+1.85).
- **Pooled X2 baseline: 23.339 (≡ the executed rde/pe pin
  23.338872993169502 exactly) → leak-excluded 24.356** (+1.02, ~4.4%,
  well inside the member band [19.41, 27.29]).
- Verdict sensitivity (disclosed, not re-adjudicated): rde
  PARTIAL-RELIEF (8.26 leak-excluded vs baseline 24.36 — relief stands,
  inclusion <2.0 still not reached) and pe NO-RELIEF (24.07 vs 24.36 —
  still at baseline) are BOTH insensitive to the leak exclusion.

## Leg C — canonical cross-check

The payload's rde/pe errors reproduce the 08-08 sensitivity numbers to
**0.0 exactly** (full and leak-excluded, both sides) — payload/local
chain integrity confirmed, and the archived "rde 7.47→8.26,
pe 23.03→24.07" disclosures now rest on canonically-bundled inputs.

## Status

**D1 is fully closed**: pixel sensitivity (08-08) + X2 anchor (this) +
proprio recompute (this) all under the registered rule, all
verdict-preserving. The `--holdout` guard remains in force for every
future search.
