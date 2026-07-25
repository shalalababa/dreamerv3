# Collusion pilot stages 2 + 3 — measurement calibration + Design-B viability (2026-07-25)

Local CPU sweeps on the v2 harness (`collusion_pilot_v2_20260725`,
selfcheck PASS) run same-day as the stage-1 GO
(`artifacts/collusion_stage1_20260725/`). Pilot analyses per the design
of record (`Design_Collusion_Pilot_20260724.md` stages 2–3); nothing
here is confirmatory — these numbers CALIBRATE the confirmatory prereg.

## Instrument v2 (additions only; v1 semantics untouched)

`collusion/pilot.py` v2 adds: distance-from-collusive-path punishment
label (`punish_occ_dist`: periods with min price ≥ 1 grid step below
the terminal greedy path's minimum; offline label), the two-label
Jaccard, `coverage_late` (unique joint states in the last 20% of the
log), and `--alpha/--beta` knobs. `collusion/designb.py` (stage 3,
selfcheck PASS): recorded sessions + offline stream-replay learner +
punishment-drop and size-matched random-drop interventions.

**v1 validation**: local v2 rerun of seeds 0–99 at the baseline
reproduces the cluster stage-1 csv on every shared column, every
session — the only differences are last-digit float variation in
`delta_profit` on 11/100 sessions (cross-machine summation; all
integer/boolean/occupancy columns bit-identical). Logic-identical.

## Stage 2a — punishment-regime definition (baseline, n=100)

| label | mean | sd | min–max | frac > 0.9 | assoc. with outcome |
|---|---|---|---|---|---|
| undercut+W5 occupancy | 0.727 | 0.170 | 0.46–0.99 | **0.26** | corr Δ −0.10; occ HIGHER in no-fingerprint sessions (0.89 vs 0.65) |
| distance occupancy | 0.495 | 0.135 | 0.00–0.91 | 0.01 | corr Δ +0.70 |

The undercut-window label saturates in a quarter of sessions and
INVERTS against the fingerprint (noisy non-collusive sessions look
like perpetual punishment); under fast ε-decay (knob sweep) it
degenerates further (sd 0.30–0.37, sessions at ≈1.0). The distance
label is well-behaved everywhere. **Calibration recommendation for the
confirmatory prereg: distance form as the primary regime definition,
undercut form demoted to descriptive.** Registered caveat to resolve at
prereg time: the distance reference is the session's own terminal path
(mildly outcome-anchored); an exogenous-reference variant (distance
below the monopoly path) should be evaluated as the alternative when
the confirmatory prereg freezes.

## Stage 2b — coverage axis (RQ1 feasibility RESOLVED)

Whole-log coverage is dead (1.000 everywhere at baseline; 0.997 even
at β=5e-4 — ε-greedy visits all 225 states early regardless).
**`coverage_late` is the viable axis**: baseline mean 0.512, sd 0.190,
range 0.22–0.95, and nearly orthogonal to the distance occupancy
(corr −0.12) and to Δ_profit (−0.02). Matched (Cov_late, Occ_dist)
pairs ARE constructible from baseline logs — the stage-1 "coverage
knob needed" concern is resolved by the measurement-window choice, not
by new knobs. β sweep (seeds 200–219, n=20/arm) as comparative-statics
validation: faster decay lowers Δ (0.78→0.70) and fingerprint
(0.60→0.40) and collapses coverage_late (0.50→0.006), matching
Calvano's exploration statics; β stays at 4e-6 for the confirmatory
baseline.

## Stage 3 — Design-B viability probe (n=20, seeds 0–19)

- **Exactness anchor: 20/20** — offline replay of the untouched
  (s,a,r,s′) stream reproduces the online agent's Q float-exactly
  (tabular Q-learning is stream-deterministic). Design B's machinery
  is sound: interventions are stream edits with everything else fixed.
- Punishment-drop intervention (delete undercut-phase transitions,
  mean 72% of the stream): Δ 0.825 → 0.508, fingerprint 0.70 → 0.20.
- **Size-matched random-drop control: Δ 0.825 → 0.615, fingerprint
  0.70 → 0.40** — most of the naive effect is DATA VOLUME. The
  composition-specific increment is −0.107 (sd 0.246; 10/20 sessions
  in the predicted direction) — directionally punishment-specific on
  the fingerprint (0.20 vs 0.40) but not separable from volume at
  n=20 with deletion this blunt.

**Verdict: Design B VIABLE** (exact anchor + evaluable interventions +
the control machinery already built), with two binding lessons for the
confirmatory prereg: (1) every deletion arm needs its size-matched
random control (now a probe column, not an afterthought); (2)
wholesale deletion at ~72% dose is too blunt — register graded
interventions (cap punishment occupancy at quantiles / partial
down-weighting) so composition dose varies while volume stays near
constant. Staged-both default stands: Design A (stream interventions
online) runs regardless; Design B joins with these controls.

## Next (confirmatory prereg inputs now complete)

Regime = distance form (with exogenous-reference variant evaluated at
freeze); axes = coverage_late × punish_occ_dist (matchable); baseline
β=4e-6, 2M iters, conv window 100K; interventions = graded,
volume-controlled; measurement fingerprint-conditional (stage-1
failure-to-learn minority). Remaining before the prereg: pick the
exogenous vs terminal-path reference on these logs, and fix the graded
intervention ladder.

## Provenance

- `baseline_v2_sessions100.csv` — v2 rerun, seeds 0–99 (validation +
  stage-2 source).
- `knob_beta_{4e-6,2e-5,1e-4,5e-4}.csv` — β sweep, seeds 200–219.
- `designb_probe20.csv` — stage-3 probe (online/replay/nopunish/
  random-drop per session).
- Commands: `python -m collusion.pilot run --sessions .. --beta ..`;
  `python -m collusion.designb probe --sessions 20 ...` (local CPU,
  ~1.5 h total).
