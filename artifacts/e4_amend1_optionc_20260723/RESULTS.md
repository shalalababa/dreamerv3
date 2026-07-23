# E4 measure passes — Amendment-1 fits + Option-C volume fits (2026-07-23)

Registered-descriptive membership panels (E4 machinery =
`probing/stratified_error.py`, PREREG_e4_stratified_error_20260714;
LEVELS are the discriminating quantity, per the 17-Jul panel note).
Results: `local_results/e4_amend1_optionc_20260722_094933/` (cluster
pass 22 Jul); csvs copied here. Conventions: rew_nll_in = reward-head
NLL on in-regime frames at horizon h; err_diff = dynamics-NLL
out − in (positive = in-regime modeled better); s1 = hi-occupancy
side for the q1 pairs.

## A. Amendment-1 rgo — the fresh batch REPLICATES the mechanism panel

rew_nll_in at h=0 (mean ± sd over 8 fits):

| arm×side | seeds 1–8 (original) | seeds 9–16 (Amendment-1) |
|----------|----------------------|--------------------------|
| rgo s1 (hi) | **1.21 ± 0.20** | **1.07 ± 0.10** |
| rgo s0 (lo) | 1.70 ± 0.20 | 1.92 ± 0.47 |
| sgb s1 (hi) | 2.30 ± 0.13 | **NOT MEASURED (gap)** |
| sgb s0 (lo) | 4.29 ± 0.14 | **NOT MEASURED (gap)** |

- The fresh rgo fits (whose adapt runs drove the +51.5* Amendment-1
  behavioral confirmation) carry the SAME representation-level
  signature as the original batch — hi-side reward-NLL ≈ 1.1 vs sgb's
  2.3, the rgo/sgb separation that anchors the "reward-predictability
  of transferred features" mechanism. Mechanism and behavior now
  replicate on the same fresh seeds.
- err_diff is batch-invariant to the third decimal (s0 h0: −0.060 vs
  −0.055; s1 h0: +0.011 both; h20 ≈ +0.60 everywhere) — regime
  modeling stays arm- and batch-invariant; still not the carrier.
- Side effect on reward-NLL is large in both arms (lo side ≫ hi side:
  rgo 1.70–1.92 vs 1.07–1.21; sgb 4.29 vs 2.30) — consistent with the
  17-Jul note that more reward frames train the head better
  regardless of wiring; the LEVEL at matched side is what separates
  arms.
- **COVERAGE GAP (recorded): the `ax1wm_finger_sgb*` half of the pass
  returned only seeds 1–8** — the Amendment-1 sgb fits (seeds 9–16)
  are absent from `e4_finger_v1_p3amend1_sgb.csv`. The sgb fresh-batch
  panel column stays open; if those fits are still on the runroot, a
  repeat pass with the same glob (or the fits' actual naming) closes
  it. No conclusion above depends on it (the rgo replication is the
  load-bearing half; sgb seeds 1–8 remain the reference level).

## B. Option-C volume fits — volume helps the rep a little, and the
## behavioral anomaly is NOT rep-level

Task arms (reward-aware), rew_nll_in at h=0 over 6 fits:

| fit cell | occ | episodes | rew_nll_in (h0) |
|----------|-----|----------|-----------------|
| v200 s1  | .323 | 200 | 0.83 ± 0.16 |
| v400 s0 (corrective) | .290 | 400 | **0.73 ± 0.02** |
| v400 s1  | .094 | 400 | 0.94 ± 0.29 |

- **Occupancy-matched volume contrast** (.323/200ep vs .290/400ep):
  doubling replay volume at ~matched fraction slightly improves
  rep-level reward predictability (0.83 → 0.73) and collapses its
  seed variance (sd 0.16 → 0.02) — volume buys STABILITY of the
  reward-relevant representation more than level. Feeds the P-B4
  volume-term note (theory addendum): consistent with a
  representation-not-compression capacity story, small effect.
- **The 18-Jul behavioral anomaly** (low-occ .094/400ep beating
  hi-occ .323/200ep, 433 vs 204) **has no rep-level counterpart**:
  the anomalous cell's reward-NLL is the WORST and noisiest of the
  three (0.94 ± 0.29), and rew_nll_all is flat (0.10–0.13) across
  cells. Whatever drove the anomaly, it is not superior reward-head
  representation at h=0 — recorded as still-unexplained, now with the
  rep level excluded as its mechanism.
- Apt arms (fv*, reward_free): no reward head (reward_aware = 0,
  empty rew_nll columns — expected); their err_diff profile matches
  the task arms (h0 ≈ 0 ±0.02, h20 +0.46..+0.64) — regime-modeling
  arm-invariance extends to the volume fits.

## Provenance

Cluster pass 22 Jul (`logs/e4_optc_volume_52500114.out` + amend1
logs); fit sets: rgo 32 (16 seeds × 2 sides), sgb 16 (8 seeds × 2
sides — gap above), volume 36 (task+apt × {v200s1, v400s0, v400s1} ×
6 seeds); 4 horizons (0/1/5/20) per fit; per-fit rows in the csvs
here. Descriptive panels only — no registered decision consumed them.
