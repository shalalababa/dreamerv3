# A2 REWARD-INVISIBILITY CERTIFICATE — PRE-STATED ANALYSIS PLAN

Written 2026-08-30 13:18 CST (UTC+08:00), **before any test statistic
was computed** on the A2 score data. Engineering-mode / descriptive:
this certifies an architectural fact (the scarecrow fence does not
touch the reward channel), it is NOT a registered primary and carries
no prereg.

## Claim to certify

The A2 scarecrow intervention — planting a position-modulated
misprice via the distractor wrapper's `mod_key`/`mod_lo`/`mod_hi`
triple — displaces occupancy (measured: −0.0632, p = .0143,
`artifacts/a2_scarecrow_read_20260824/RESULTS.md`) **without changing
the reward stream the agent experiences**. Upgrade from architectural
assertion to measured statement.

## Data (frozen before analysis)

`local_results/a2_scarecrow_20260824_184000_light/runroot/sc_*_s*/scores.jsonl`
— 20 runs × 496 episodes (`step`, `episode/score`). Verified against
the committed manifest `manifests/a2_scarecrow_20260824_184000_light.sha256`
(40/40 scores.jsonl + config.yaml files OK, 0 mismatches) before any
number was read.

Arms: ctrl s76–79, scare s80–83, scare2 s84–87, pen1 s88–91,
pen2 s92–95 (n = 4 each).

## Pre-stated summary statistics (per run)

- **T1 — final-window mean return.** Mean of `episode/score` over the
  **last 100 logged episodes** of the run (episodes 397–496).
- **T2 — full-trajectory AUC.** Trapezoidal area of `episode/score`
  against `step`, divided by the step span — a time-normalized mean
  return over the whole training trajectory.
- **T3 — final-window episode-score distribution.** The pooled
  final-window episodes of an arm (4 runs × 100 = 400 scores);
  statistic = two-sample Kolmogorov–Smirnov D against the comparison
  arm's pooled 400.

## Pre-stated tests

Exact permutation, **run-level exchangeability** (runs are the
independent units; episodes within a run are not).

- **Primary contrasts (4 vs 4):** all C(8,4) = 70 assignments.
  Statistic = difference of arm means for T1 and T2; D for T3.
  Two-sided p = fraction of |permuted stat| ≥ |observed|. **The
  two-sided floor on 70 assignments is 2/70 = 0.0286**; one-sided
  floor 1/70 = 0.0143. Both reported.
- **Supplementary pooled contrast (8 vs 4):** {scare ∪ scare2} vs
  ctrl, all C(12,4) = 495 assignments, two-sided floor 2/495 =
  0.0040 — finer resolution than the 4v4 floor allows.

Contrast set, with the direction expected **before** computing:

| contrast | expectation |
|---|---|
| scare − ctrl | INDISTINGUISHABLE (the certificate) |
| scare2 − ctrl | INDISTINGUISHABLE (the certificate) |
| {scare ∪ scare2} − ctrl | INDISTINGUISHABLE (pooled, finer floor) |
| pen1 − ctrl | DISTINGUISHABLE (positive control) |
| pen2 − ctrl | DISTINGUISHABLE (positive control) |
| scare2 − scare | descriptive only |

**The certificate is void if the positive control fails**: if pen1/pen2
do not separate from ctrl on T1/T2, the instrument cannot detect a
reward-channel change and the scare nulls mean nothing.

## Pre-stated effect-size bounds

- 95% CI on Δ(scare − ctrl) for T1 and T2 by **inversion of the
  two-sided exact permutation test** (the set of shifts δ not rejected
  at the achievable level), reported alongside a Welch-t 95% CI and a
  10,000-draw run-level bootstrap percentile CI as parametric /
  resampling complements.
- **Footprint ratio:** |CI bound on Δ(scare − ctrl)| divided by
  |Δ(pen − ctrl)|, expressed as a percentage — "the fence moves
  occupancy −6.3 pp while moving return by [CI], a reward-channel
  footprint bounded at X% of the penalty arms' footprint."

## Pre-stated caveat (stated before computing, not after)

The penalty wrapper **replaces** the env reward rather than adding to
it (`embodied/envs/regionpenalty.py:63` — "task reward fully REPLACED,
never added"), so the pen arms' `episode/score` is a penalty return on
a different scale from the ctrl/scare task return. The positive
control is therefore **separation-by-construction at the measurement
level**: it proves the instrument reads the reward channel and would
fire on an arm whose reward channel is touched, but it is not an
independent effect. The footprint ratio inherits this
scale-incommensurability and is reported as an order-of-magnitude
bound, not a like-for-like ratio. No post-hoc statistic will be added
to this plan.
