# Gate 0 decision — 2 Jul 2026 (deadline 10 Jul)

**Decision: GO. Phase 4 launches with the full domain set:
`dmc_cup_catch` + `dmc_finger_turn_hard` (primary), `dmc_reacher_hard`
(sanity), `dmc_walker_walk` (collinear control).**

## Evidence

Two-stage evaluation, both stages pre-registered (plan v3 §3.6; runbook v2
Phase 3), both computed before any transfer outcome was observed.

### Stage 1 — whole-buffer screen (`probing/gate0.py`)

| domain | verdict | note |
|---|---|---|
| reacher_hard | **GO** | random-vs-goal: Δcov=0.0011, Δocc=0.4275 (~150x noise); p2e-vs-apt: Δcov=0.0287, Δocc=0.0018 |
| cup_catch | NO-GO | pilots lie on an anti-correlated Cov–Occ frontier; no raw pair aligns |
| finger_turn_hard | NO-GO | double near-miss (apt-vs-goal Δcov=0.011 vs tol 0.005 at 5.5x occ difference) |

The window-pool criterion passed on all three domains (e.g. cup: 650 high-occ
+ 2000 zero-occ windows vs 200 required each): the quadrants are *not empty*,
the raw pilot buffers just do not happen to align pairwise.

### Stage 2 — segment-level constructability (`probing/gate0_compose.py`)

Plan §3.6 item 3 defines feasibility at segment granularity ("enough segments
at matched coverage with distinct occupancy"), which is what Phase 6 (Axis 1)
actually uses; the whole-buffer screen is sufficient but not necessary.
Composed candidates: 400 buffers x 200 windows, sampled without replacement
from (pilot x occupancy-bin) strata; pairs require <=20% window overlap.

| domain | verdict | Q1 (cov-matched / occ-different) | Q2 (cov-different / occ-matched) |
|---|---|---|---|
| cup_catch | **GO-compose** | Δocc=0.575 (0.064 vs 0.638) at Δcov=0.005 (tol 0.019); 7151 pairs | Δcov=0.389 (>raw pilot range) at Δocc=0.021; 15573 pairs |
| finger_turn_hard | **GO-compose** | Δocc=0.668 (0.015 vs 0.683) at Δcov=0.002 (tol 0.007); 6213 pairs | Δcov=0.184 (>raw pilot range) at Δocc=0.015; 14175 pairs |

The constructable region is a broad 2-D cloud on both primary domains
(gate0_compose.png), i.e. coverage and occupancy dissociate under
segment-level composition. The pilots' raw anti-correlation (cup) is the
coverage<->occupancy confound the intervention design exists to break, and is
itself a useful motivating figure for the paper.

## Caveats carried into Phase 4/6

1. **Occ-matching tolerance is size-dependent.** occ_sep_min here reflects
   200-window candidates (noise ~0.02). Phase 6 buffers are larger; re-verify
   occupancy matching at the actual buffer size (noise shrinks, tolerance
   tightens).
2. **High-occ strata are thin in the pilots** (cup: 650 high-occ windows
   pooled, mostly from random+goal; finger: 555, mostly goal+random+p2e).
   Scaling composed buffers to WM-training size needs proportionally more raw
   data — provided by Phase 4 (500K steps x 5 seeds per mode) plus the
   reward-on goal runs; top up with extra random/goal collection if a stratum
   runs short (cheap).
3. **Fallbacks remain pre-registered** if Phase 6 construction surprises:
   (a) auxiliary noisy-goal / coverage-max collection, (b) domain swap
   (manipulator_bring_ball; reacher promotion — reacher already passed the
   strict screen), (c) rescope to Axis 2.

## Files

Per domain: `gate0.json`/`gate0.png` (whole-buffer screen),
`gate0_compose.json`/`gate0_compose.png` (segment-level check; cup+finger).
Reproduce: `python -m probing.gate0 ...` and `python -m probing.gate0_compose
...` on the four pilot replays (`pilot_{p2e,apt,random,goal}_<dom>_seed1`).
