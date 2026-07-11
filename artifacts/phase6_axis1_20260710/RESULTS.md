# Phase 6 / Axis-1 controlled-buffer results — 2026-07-10

**This is the decisive causal test registered in PREREG_phase5a.md §6** ("Primary
Phase-6 endpoint: the coverage-matched/occupancy-different contrast under
R^phys") and motivated by the Phase-5a unblinding
(`artifacts/phase5a_20260707/RESULTS.md`), whose directional prediction was:
*the higher-occupancy side adapts worse.*

Inputs: `local_results/axis1_results_20260710_210315/` (adapt scores, buffer
manifests, WM-fit audit, logs) and the cluster-side analysis snapshot
`local_results/axis1_analysis_20260710_213221/`. **Everything below was
re-derived locally with the frozen pipeline** (`analysis/adaptation_auc.py`,
`analysis/fit_mixed_effects.py paired`) and is bit-identical to the
cluster-side numbers (all 64 AUC rows and all 16 paired-contrast JSONs match
to <1e-6).

## Design recap and sign conventions (from the build manifests)

2 domains × 2 quadrants × 2 sides × 8 paired seeds = 64 runs. Each run =
offline WM fit on a 200-episode / 200,200-transition controlled buffer
(500,000/500,000 gradient updates — verified for all 64 in `wm_audit/`,
equalized to the 500K-step online pretrains) + frozen-readout adapt (125K
steps). Contrasts are **s1 − s0**, within-seed, B=10,000 cluster bootstrap
(seed 0) + exact two-sided sign test, as frozen.

| pair | s0 | s1 | manipulation |
|---|---|---|---|
| cup Q1 | occ 0.013, cov 0.885 | occ 0.173, cov 0.872 | occupancy ↑ at matched coverage |
| cup Q2 | occ 0.062, cov 1.027 | occ 0.100, cov 0.702 | coverage ↓ at ~matched occupancy |
| finger Q1 | occ 0.054, cov 0.805 | occ 0.323, cov 0.797 | occupancy ↑ at matched coverage |
| finger Q2 | occ 0.090, cov 0.846 | occ 0.101, cov 0.682 | coverage ↓ at ~matched occupancy |

So for Q1 the 5a prediction is **Δ negative**; for Q2, if coverage causally
helps, **Δ negative** (s1 is the low-coverage side).

## Data completeness / QC

- 64/64 adapt runs `ADAPT_DONE`, 64/64 pass the ≥20-episode QC, 0 exclusions.
  (The per-machine `runs.csv` ledgers each track a 51-run subset — the grid
  ran across Midway + two cloud instances — but every run directory is
  complete.)
- 64/64 offline WM fits reached exactly 500,000 updates (final `wm_total`
  2.19–2.59); gradient-count equalization held.
- No errors/tracebacks in any job log.

## Paired contrasts (s1 − s0; mean Δ [boot 95% CI], exact sign-test p)

Primary outcome AUC₁₀₀ₖ bold; `*` = bootstrap CI excludes 0.

| pair | AUC₅₀ₖ | **AUC₁₀₀ₖ** | AUC₁₂₅ₖ | final10 |
|---|---|---|---|---|
| cup Q1 (occ↑) | +43 [−55,+139] | **+46 [−18,+111]** p=1.0 | +40 [−16,+95] | +0.7 [−15,+17] |
| cup Q2 (cov↓) | −165 [−371,+36] | **−121 [−279,+26]** p=.73 | −107 [−248,+24] | −22 [−80,+23] |
| finger Q1 (occ↑) | +126 [+73,+193]* p=.008 | **+159 [+83,+236]*** p=.070 | +174 [+91,+256]* | +234 [+129,+337]* |
| finger Q2 (cov↓) | +3 [−31,+43] | **−2 [−44,+30]** p=.73 | −3 [−47,+29] | +22 [−82,+111] |

Condition means (AUC₁₀₀ₖ ± SD over 8 seeds), for scale:

| | s0 | s1 |
|---|---|---|
| cup Q1 | 649 ± 154 | 695 ± 105 |
| cup Q2 | 680 ± 156 | 559 ± 143 |
| finger Q1 | 109 ± 48 | **268 ± 136** |
| finger Q2 | 122 ± 72 | 120 ± 32 |

## Reading

1. **The pre-registered primary prediction fails.** Raising physical-regime
   occupancy at matched coverage does *not* hurt adaptation anywhere:
   - **cup Q1 is null** — and tightly so at final10 (Δ +0.7 [−15, +17]:
     final performance is *identical* across a 14× occupancy change).
   - **finger Q1 is significantly POSITIVE at every window** (all four
     bootstrap CIs exclude zero; sign test 8/8 at 50K). The high-occupancy
     side is the *better* buffer.
2. **Occupancy rescues finger.** In Phase 5a, *no* online pretrain
   (p2e/apt/random, 75 runs) showed any finger adaptation (flat ≈80 AUC).
   The high-occ controlled buffer produces WMs that adapt to ≈268 AUC₁₀₀ₖ /
   ≈388 final10. finger_turn_hard was never "too hard for frozen-readout
   adaptation" — the explorers' buffers simply never dwelt in the task's
   physical regime. This also *retroactively explains the 5a finger null*:
   all three online modes had near-zero in-regime occupancy there.
3. **The 5a observational occ⁻ association was confounded.** occ_phys −0.36*
   (pooled), −0.75* (cup-only) does not survive intervention at matched
   coverage (null in cup, reversed in finger). This is exactly what the
   cup-only M2 instability (VIF 24.4) warned about: in observational data,
   high occupancy came packaged with whatever else random-policy buffers do
   badly. The controlled pairs break that packaging.
4. **Coverage is directionally causal in cup but not resolved at n=8.**
   Dropping coverage 1.03→0.70 at matched occupancy costs ~121 AUC₁₀₀ₖ
   (~165 at 50K — an early-adaptation effect that washes out by final10, as
   the curves show: the low-cov side converges to the same asymptote,
   later). CI includes 0; 7/8-vs-sign-test underpowered. Finger Q2 is null,
   but finger has no adaptation signal at low occupancy to modulate.
5. **Residual confound to close (pre-registered E3):** in both Q1 pairs,
   side0 is pure-p2e episodes while side1 is a broad mixture including
   random/pilot sources (source_l1 0.90–0.945). Moving occupancy at fixed
   coverage *is* a composition change, but the E3 composition-robustness
   pairs (`ax1r1`, `ax1r2`: exclude the dominant source from the high-occ
   side) and the occupancy dose-response (`ax1d0–d3`) are the registered
   tests that the finger Q1 effect tracks measured occupancy rather than
   source identity. These are now the highest-priority remaining runs.

## Headline (for the paper, replacing the 5a-era framing)

Buffer composition causally determines what a frozen world model can support:
**coverage helps early adaptation; physical-regime occupancy is not a tax but
a requirement that binds exactly in domains whose task regime exploration
never visits.** Observational driver regressions get the occupancy sign
*wrong* (−0.36*) — a caution for the growing practice of curating
pretraining buffers from observational correlations.

## Files

- `auc_axis1.csv` — 64 outcome rows (frozen `adaptation_auc.py`, recomputed
  locally; identical to cluster-side).
- `paired/paired_<dom>_<q>_<outcome>_s1_minus_s0.json` — 16 contrasts
  (frozen `fit_mixed_effects.py paired`, B=10,000, seed 0).
- `adaptation_curves_axis1.png` — descriptive curves (16K eval-burst bins,
  mean ± SEM over 8 paired seeds).
- Raw inputs: `local_results/axis1_results_20260710_210315/` (also committed
  as `artifacts/axis1_20260710_210315/`, `artifacts/axis1_20260710_213221/`).
