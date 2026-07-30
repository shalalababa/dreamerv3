# Collusion confirmatory-prereg calibration (2026-07-30)

Local CPU calibration for the Paper-4 confirmatory registration
(`prereg/PREREG_collusion_confirm_20260730.md`), executing the two
freeze-time calls left open by stages 2+3
(`artifacts/collusion_stage23_20260725/`): the distance-reference
choice and the graded intervention ladder. Instruments:
`collusion_pilot_v3_20260730` + `collusion_designb_v2_20260730`, both
selfcheck PASS. All runs on PILOT seed ranges (0–99 baseline, 0–19
graded, 0–9 smokes) — disjoint from every confirmatory cohort. Nothing
here is confirmatory; these numbers calibrate the prereg.

## 1. v3 validation (additions-only)

Local v3 rerun of baseline seeds 0–99 vs the stage-2 record
(`baseline_v2_sessions100.csv`): **all 15 shared columns identical on
all 100 sessions** (bit-exact; same machine as the v2 run). The v3
uniform path consumes the identical RNG stream as v1/v2.

## 2. Distance-reference choice → **TERMINAL** (frozen)

| label (n=100 baseline) | mean | sd | frac>0.9 | corr Δ_profit | mean fp / no-fp |
|---|---|---|---|---|---|
| terminal (`punish_occ_dist`) | 0.495 | 0.136 | 0.01 | +0.70 | 0.522 / 0.438 |
| exogenous (`punish_occ_dist_exo`) | 0.981 | 0.058 | **0.96** | −0.24 | 0.980 / 0.984 |

The exogenous (monopoly-reference) label is degenerate: it covers
~98% of the stream in nearly every session (max 0.998, none at
exactly 1.0), carries no regime information (fp vs no-fp
indistinguishable), and its graded d=1 arm deletes ~everything
(mask_frac ≈ 0.996 ⇒ Q collapses toward init: I_dp −0.348 measures
LEVEL destruction, not regime composition). The terminal label is
well-behaved everywhere; its mild outcome-anchoring (corr +0.70 with
Δ_profit; reference derived from the session's own converged path) is
disclosed as a limitation in the prereg. Exogenous is retained as a
registered descriptive column only.

## 3. Graded ladder (terminal, seeds 0–19, doses .25/.5/1.0)

exact_replay 20/20; mask_frac 0.50 (sd 0.07); fp_online 0.70.

| dose | I_dp mean (sd) | fp-cond I_dp | I_fp mean (sd) |
|---|---|---|---|
| 0.25 | +0.000 (0.000) | +0.000 | +0.000 (0.324) |
| 0.50 | −0.034 (0.102) | −0.023 (0.083) | −0.050 (0.224) |
| 1.00 | −0.185 (0.286) | −0.134 (0.287) | −0.300 (0.657) |

- **d=0.25 is behaviorally inert on Δ_profit** (target ≡ control ≡
  online in 20/20 sessions — w=0.75 flips no greedy action on the
  evaluated path); d=0.50 mostly inert (17/20 sessions unchanged).
  Tabular argmax readout gives threshold-like dose response; the
  registered primary sits at the top dose, the ladder is descriptive.
- Composition-specific increment at d=1: fp-cond −0.134 (sd 0.287) —
  same direction as, and larger than, the stage-3 undercut-label
  deletion increment (−0.107).

## 4. Design-A smoke (baseline seeds 0–9; λ=1 arms n=10, λ=0.5 arms n=5) — DISSOCIATION

All numbers recomputed from the archived `designa_smoke.csv` (pooled
over both 5-seed batches for the λ=1 arms; an earlier draft of this
table quoted per-batch subsets — corrected pre-freeze, second-review
finding).

| arm | n | Δ_profit (sd) | fingerprint | occ_dist | cov_late | undercut |
|---|---|---|---|---|---|---|
| baseline (0–9) | 10 | +0.844 | 0.80 | 0.512 | 0.481 | 0.298 |
| forbid λ=1 | 10 | **+0.987 (0.0000)** | **0.00** | 0.000 | 0.004 | 0.000 |
| forbid λ=0.5 | 5 | +0.868 (0.048) | 1.00 | 0.530 | 0.482 | 0.262 |
| force λ=1 | 10 | +0.637 (0.204) | 0.40 | 0.417 | 0.283 | 0.360 |
| force λ=0.5 | 5 | +0.804 (0.061) | 0.40 | 0.494 | 0.508 | 0.442 |

Forbidding exploration-generated undercuts RAISES the price level to
near-monopoly while the punishment fingerprint COLLAPSES to zero — an
experimentally induced Abada–Lambin failure-to-learn regime (prices
lock high because deviation is never explored; no punishment strategy
is ever learned). **Consequence, disclosed: the Design-A registered
primaries were re-specified PRE-FREEZE from this smoke** — P-CLA1 =
fingerprint collapse (CI < 0), P-CLA2 = price level maintained/raised
(CI > 0), a dissociation prediction, replacing the draft's naive
"profit drops" direction. Two further disclosures: (a) the forbid λ=1
dynamics are DEGENERATE in the smoke — all 10 sessions reach the
identical Δ_profit (0.9868543033, sd 0) and converge at the 100K
window minimum (~100,036 iters vs baseline ~1.8M): the restricted
exploratory support creates a deterministic upward price ratchet to a
single attractor. If this replicates at n=100 the forbid side of the
two-sample CIs contributes ~zero variance (CIs then driven by
baseline spread — valid, disclosed). (b) The forbid arm collapses
coverage_late (0.004) and occupancy (0.000) jointly — a blunt
manipulation; registered as the coarse generation-time claim with the
coverage confound audited as a secondary. λ=0.5 forbid is
indistinguishable from baseline (threshold-like λ response).

## 5. Power (for the prereg's disclosed sizing)

- P-CLB1 (fp-cond, top dose): sd 0.287 ⇒ expected n_fp≈68 detects
  ≈0.097; calibration point −0.134. n=100 unconditional detects 0.080.
- P-CLA1/2: two-sample 100v100 at baseline Δ sd 0.122 detects ≈0.048
  (fingerprint-rate contrasts of the smoke's magnitude are far larger).

## Provenance + disclosures

- csvs: `calib_v3_baseline100.csv` (v3 rerun, seeds 0–99),
  `graded_terminal20.csv` / `graded_exo20.csv` (seeds 0–19, doses
  .25/.5/1.0), `designa_smoke.csv` (forbid/force × λ∈{1, .5}, 30
  sessions). Raw parts + logs:
  `local_results/collusion_calib_20260730/`.
- Batch: 56 jobs, 13-way parallel, ~10 min wall. One job
  (graded_terminal seed 19) died silently (empty log, no csv); rerun
  to completion same session BEFORE any prereg number was finalized —
  sessions are seed-deterministic (selfcheck-validated), so the rerun
  is exact.
- Ordering note: the `dpr_*` (Δ_price) columns were added to
  `designb.graded_session` AFTER the calibration graded runs and
  BEFORE the freeze (additions-only; no computed value changed —
  selfcheck identity checks cover the weighted-replay path). The
  calibration graded csvs therefore lack `dpr_*`; all confirmatory
  csvs will carry them and the frozen reader requires them.
- Analysis method: means/sds/correlations + the d-inertness counts
  computed by a one-off script over these csvs (numbers reproduced in
  this file); no frozen reader consumed calibration data.
