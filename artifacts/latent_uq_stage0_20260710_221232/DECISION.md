# Latent-UQ Stage 0 — aggregate decision (2026-07-11)

Inputs: the six cell analyses in this directory (generated cluster-side at
commit d102ac4, 10 Jul). **Verified locally**: the `cup_p2e` cell was re-run
from the raw dumps (`local_results/evpi_stage0_20260710_221924/raw/`) with
`probing/latent_uq_analysis.py` — all 900 numeric fields and every verdict
bit-identical; the analysis dirs in the snapshot and this artifact are
byte-identical throughout.

## Aggregate verdicts (final checkpoint, anchor h=5)

| cell | probe distribution | verdict |
|---|---|---|
| cup_p2e | held-out, same policy | **TRACKS_DENSITY (5/5 seeds)** |
| finger_p2e | held-out, same policy | **TRACKS_DENSITY (5/5 seeds)** |
| cup_apt_cross | cross-policy | TRACKS_ERROR |
| cup_random_cross | cross-policy | TRACKS_ERROR |
| finger_apt_cross | cross-policy | TRACKS_ERROR |
| finger_random_cross | cross-policy | AMBIGUOUS |

Pre-registered one-step calibration read (EVPI note App. A.4 item 5):
**FAIL** in both held-out cells (all 10 seeds).

Temporal structure (visible in every held-out cell): at the earliest
checkpoint (~100K) disagreement TRACKS_ERROR; from ~200K on it flips to
TRACKS_DENSITY and the bias *strengthens* with training — the attractor
pulls harder as the model converges.

## Decision

1. **Biased Dreams (arXiv 2604.25416) replicates on our RSSM ensembles** —
   robustly, 10/10 held-out seed-cells, both domains. On-distribution,
   ensemble disagreement is a *density* meter, not an error meter.
2. **D0 stays frozen.** The registered routing ("TRACKS_DENSITY ⇒ Stage-1
   fixes become the priority and the D0 ensemble read is suspect") applies:
   do not launch the 24+24 `D0_FROZEN` runs on the raw-disagreement
   instrument — they would measure the attractor artifact.
3. **The cross-policy TRACKS_ERROR result is the constructive lead**: the
   instrument is informative exactly where probes leave the training
   distribution. Stage-1 candidates should exploit or correct for this
   (density-partialed disagreement; early-checkpoint calibration; OOD-probe
   designs), and any fix must re-pass this same six-cell battery before D0
   unfreezes.
4. Paper-2 (EVPI) remains off the ICLR-2027 critical path; no calendar
   impact. Side benefit for Paper 1: a clean in-house replication of the
   attractor bias to cite where plan v3 motivates Gate D0.

## Addendum (2026-07-12): unfreeze criterion superseded

Per the 11-Jul EVPI editorial review and
`research_notes/other research/EVPI_Plan_Revision_20260712.md`: re-passing
this six-cell battery is **necessary but no longer sufficient** to unfreeze
the D0 24+24 runs. The unfreeze criterion is now **Gate D1**: an
operation-specific `ÊVSI_o` must show monotone calibration and useful
ranking against randomized oracle value-of-computation labels `Δ_o(s)`
(restorable-state interventions), with positive net decision value after
gate overhead. Item 3's "must re-pass the six-cell battery" stands as the
precondition; Gate D1 is the decision test. The Stage-0 phase change
(error-tracking → density-tracking with training) is promoted to the lead
asset of either Paper-2 route.
