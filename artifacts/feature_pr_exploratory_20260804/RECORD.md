# EXPLORATORY — pixel trunk-feature breadth probe (2026-08-04)

**Status: labeled post-read exploratory analysis** under the
value-blindness policy of `PREREG_rde_pixel_20260802` (post-read
analyses allowed, labeled, VERDICT-UNTOUCHABLE — the rde ONE read's
PARTIAL-RELIEF verdict and every registered quantity stand exactly as
recorded in `artifacts/rde_pixel_read_20260804/`). Not a frozen
instrument; no registered decision consumes this. Intended consumer:
theory-paper discussion (the narrowing account of the missing P-RD2
behavioral lift).

## Prediction (stated in-chat BEFORE any value existed, 4 Aug)

"rde trunks — especially s1 — show materially lower feature-space
participation ratio than the X2 task arm at similar-or-better reward
legibility."

## Result: CONFIRMED, with a sharper structure than predicted

Instrument: `probing/feature_pr_probe.py` — h=0 filtering-posterior
features of the frozen `fingerpx_v1` probeset (verbatim
stratified_error extraction path), covariance eigenspectrum per block
(deter / stoch / head_input = `feat2tensor` — the exact reward-head
input space), matrix-Chan float64 moments. 48 runs (16 rde + 16 X2 +
16 fpxpx).

Cell means (pr_deter / pr_head / h0 in-regime rew-NLL):

| arm | side | pr_deter | pr_head | rew_nll_in |
|---|---|---|---|---|
| fpxpx (recon only) | s0 | 14.24 | 46.05 | — (no reward head) |
| fpxpx | s1 | 13.33 | 44.39 | — |
| X2 (recon+reward) | s0 | 11.22 | 30.29 | 41.16 |
| X2 | s1 | 14.71 | 44.18 | 5.52 |
| rde (reward only) | s0 | 3.02 | 35.06 | 11.54 |
| rde | s1 | 6.66 | 9.23 | 3.41 |

Seed-paired [rde − X2] contrasts (same buffer + seed; 8-seed cluster
bootstrap B=10K rng 0):

- **pr_deter: s0 −8.21 [−9.40, −6.89]; s1 −8.05 [−8.83, −7.21]** —
  the witness-scope trunk state is less than half as broad in rde,
  BOTH sides, CI entirely negative — at better reward legibility on
  both sides.
- **pr_head: s1 −34.94 [−36.10, −33.50]** — the reward head's input
  geometry collapses from ~44 to ~9 effective dimensions on side 1;
  **s0 +4.77 [−6.72, +15.17]** — not narrowed (stoch-dominated, see
  mechanism).
- pr_stoch broad everywhere (38–53, all arms).

**Co-localization with the behavioral deficit (the narrowing story's
key exhibit):** the side whose head-input space collapsed (s1, 44→9)
is exactly the side carrying the rde read's behavioral deficit (AUC
60.7 vs X2 82.6); the side with preserved breadth (s0) sits at the X2
level (79.9 vs 77.2). Breadth loss and competence loss co-vary across
sides while legibility IMPROVES on both — transfer needs breadth, not
just legibility.

## Mechanism of the two side patterns (descriptive)

head_input PR is trace-weighted, which is the point — it is the
geometry the head actually faces:

- **rde s1**: deter carries large variance (deter_std ≈ 0.46) at low
  PR (6.7), so deter's trace dominates the concatenation → the head
  faces a ~9-dimensional signal. Legibility best (≈2.7–3.2 nats,
  one seed at 8.3).
- **rde s0**: 6/8 seeds have near-dead deter (PR 1.5–2.3, deter_std
  ≈ 0.03 — near-rank-2 recurrent state, barely above the alive
  threshold) → stoch dominates the trace and pr_head stays broad-ish;
  these six converge to an almost identical 8.33-nat solution. The
  two seeds that DID recruit deter variance (3, 4) got WORSE NLL
  (19.9 / 22.5 — swamping-baseline level).
- New baseline descriptive: X2's pooled 23.34 is strongly bimodal by
  side (s0 41.2 / s1 5.5) — the registered pooled estimator was
  fair (registered pre-outcome) but the side split is large.
- corr(deter_std, pr_deter) = −0.08 across 48 runs — the scalar
  witness and breadth are near-orthogonal (variance magnitude ≠
  spread), which is why this probe was needed at all.

## Theory reading (discussion-level, non-causal, n=8/cell)

The three arms trace a breadth–legibility frontier in the head-input
space: fpxpx (recon only) broad-and-illegible → X2 (recon+reward)
broad-and-swamped → rde (reward only) narrow-and-legible. Removing
the dominant generic bidder (reconstruction) reallocates the trunk's
included support toward what the surviving reward-bearing objectives
bid for — the spectral-competition picture of inclusion — and the
policy, which needs breadth, pays for the reallocation exactly where
it happens (s1). Supports the narrowing account of P-RD2's null;
figure: `fig_feature_pr.png` (also
`research_notes/figures_theory_20260804/` with data json + script).

## Provenance

- Results bundle `local_results/feature_pr_fingerpx_20260804_203436/`
  (manifest 54/54 OK); probe sha `1f4b99fa…` bit-identical local vs
  bundle code record. 48/48 runs measured.
- **witness_match TRUE on all 32 gated runs** (16 rde + 16 fpxpx:
  recomputed deter_std ≡ the frozen witness value in each run's e4
  summary at rtol 1e-4) — the extraction path is validated against
  the frozen instrument on real data; X2's July summaries predate
  the column (gate null there, covered by the 32).
- Legibility values joined from committed csvs only
  (`artifacts/pixel_swamping_20260725/e4_fingerpx_v1.csv` h0 rows;
  rde bundle `e4_fingerpx_v1_rde.csv` h0 rows).
- Files here: `feature_pr_fingerpx.csv` (collated), `stats.json`
  (cell means + paired bootstraps), `fig_feature_pr.png`.
