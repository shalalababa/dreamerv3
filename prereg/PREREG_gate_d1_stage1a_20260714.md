# Gate D1 + Stage-1A registration — density-residual instrument and 24+24 unfreeze criteria (frozen 2026-07-14)

Quantifies the `[reg.]` placeholders of the 12-Jul Paper-2 revision
(`EVPI_Plan_Revision_20260712.md` §Stage-1A/§Gate-D1) and its theory
addendum (§6 ranked density-residualized disagreement as the
mechanism-endorsed Stage-1A primary). Frozen **before any
density-residual analysis has run on real dumps** (the instrument was
implemented today in `probing/latent_uq_analysis.py --instrument
density_residual` and validated on the synthetic selfcheck only).

## A. Stage-1A registered read (runs locally on the existing Stage-0 dumps)

**Instrument (primary candidate):** disagreement D replaced by its
out-of-fold isotonic residual on the kNN-distance density proxy —
5 folds by stream cluster (seeded hash, seed 0), isotonic direction by
SSE, constant extension off-range. Everything else identical to the
frozen Stage-0 protocol: same dumps, horizons {1, 5}, headline = anchor
row at h=5 of the final checkpoint, stream-clustered bootstrap n=300,
verdict threshold 2 SE on Δ = pcorr(D,E|ρ) − pcorr(D,ρ|E).

**Known baseline (disclosed):** raw-instrument Stage-0 verdicts — within
cells: 10/10 sub-cells TRACKS_DENSITY at the final checkpoint (cup_p2e ×5
seeds, finger_p2e ×5); cross cells: cup_apt, finger_apt, cup_random
TRACKS_ERROR; finger_random AMBIGUOUS. The residual instrument's
verdicts are unknown.

**Registered decision rule** (final-checkpoint verdicts under
density_residual):

- **PASS:** ≥ 8/10 within sub-cells TRACKS_ERROR and 0/10
  TRACKS_DENSITY, and no cross cell that was TRACKS_ERROR becomes
  TRACKS_DENSITY. ⇒ density-residual disagreement is the Stage-1A
  primary instrument; it feeds Gate-D1 EVSI features and is Route B's
  "calibration fix" asset.
- **MARGINAL:** ≥ 5/10 within sub-cells TRACKS_ERROR, 0 TRACKS_DENSITY.
  ⇒ advances as a candidate alongside independent-refit ensembles
  (menu item 4); Route B reports it with the mixed table.
- **FAIL:** any other outcome (in particular, majority AMBIGUOUS =
  disagreement carries no error signal beyond density at the final
  checkpoint). ⇒ the fix pivots to refit-ensemble instruments (the
  cross cells already pass raw); density-residual is reported as a
  negative result.

**Registered secondary:** the early-checkpoint error-tracking regime
(known at ~100K under raw) must be preserved (not destroyed) under
residualization — same verdict machinery at the earliest retained
checkpoint, reported per cell, no decision role.

## B. Gate D1 — quantified unfreeze criteria for the 24+24 D0 runs

Prerequisite: Stage-1B oracle labels (restorable-state interventions,
§C). Gate D1 PASSES only if ALL of:

1. **Monotone calibration:** Spearman(ÊVSI_o, realized Δ_o) ≥ **0.3**
   with run-clustered bootstrap 95% CI excluding 0, evaluated with
   cross-fitting by model/run seed (labels never share RNG or rollouts
   with feature computation — addendum §4 binding condition).
2. **Useful ranking:** states in the top ÊVSI quintile realize mean
   Δ_o ≥ **2×** the population mean (lift CI excluding 1).
3. **Positive net decision value:** the gated policy (execute o iff
   V̂oC > 0) beats BOTH always-o and never-o on return net of metered
   cost (wall-clock + env steps + gate overhead), CI excluding 0
   against the better of the two.

Thresholds 0.3 / 2× are provisional-but-registered; any change requires
a dated amendment before the D1 read. The six-cell battery remains
necessary-but-not-sufficient (per the 12-Jul revision).

## C. Stage-1B infrastructure (spec; build follows this registration)

`d0/oracle_labels.py` (next build): MuJoCo restorable-state snapshot at
sampled decision states; paired execute/skip of the operation with
common random numbers; high-budget evaluation of
Δ_o(s) = V(decision after o) − V(decision now); amortized ÊVSI_o
trained with cross-fitting by run. Operation pair for the first paper
(one-pair scope per the revision): **real-data purchase vs
imagined-rollout purchase** at matched metered cost.

## D. Ordering statement

At freeze: no `--instrument density_residual` invocation has touched any
real dump (grep over artifacts/ and local_results/ analysis dirs: every
existing analysis.json lacks the `instrument` field = raw-era outputs);
Stage-1B code does not exist; no D0 24+24 run has been submitted.
