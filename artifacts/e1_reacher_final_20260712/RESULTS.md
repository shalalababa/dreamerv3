# E1 reacher replication — registered driver-level read (2026-07-12)

The registered E1 analysis (7-Jul addendum §E1): the frozen Phase-5a
pipeline with reacher as its own population, reported separately, never
pooled into the registered cup+finger primary. Inputs:
`local_results/e1_reacher_final_20260712_001736/` (measure grid + latent
dumps + critic) and the adapt outcomes verified 11 Jul
(`local_results/e1_reacher_adapt_20260711_095948/`, 75/75 QC).

**Everything below is the frozen local pipeline** (`collate_drivers.py` →
`fit_mixed_effects.py --primary_domains reacher --control_domain none`,
cluster bootstrap B=1000 seed 0). Driver rows re-derived locally are
**identical (56/56)** to the cluster-side `drivers.csv`. NOTE: the
cluster-side fit (`reacher_primary_measured_only/`) pre-filtered the AUC
table to the 56 measured cells before fitting, which changes the
within-domain z-scoring and M0 population relative to the frozen spec
(e.g. cov β 0.296 vs frozen 0.318) — **the numbers below, from the frozen
spec (M0 on all 75 QC rows; M1/M2 on the 56 merged rows), are canonical.**
Same qualitative reads in both variants.

## Gates and QC

- **Critic gate: PASS.** Held-out R² = 0.3696 ≥ 0.2
  (`critics/critic_reacher_v1.npz.meta.json`) → VSA is primary for
  reacher, retdec reported alongside.
- Driver cells: 56/75 (19 pre-registered snapshot-distance exclusions,
  enumerated in `reacher_driver_exclusions.csv` — same upstream
  `measure.sbatch` rule as Phase 5a).
- Driver degeneracies to report: reacher occ_phys is range-compressed
  (0.005–0.012 across ALL modes — explorers never dwell in-regime, as in
  cup/finger); occ_rew ≈ occ_phys (sparse reward ⇒ regime ≈ reward);
  fwd↔geom near-perfectly collinear (M2 VIFs 147); retdec spans −181.6
  to 1.0 (severely degenerate critics at some checkpoints).

## Registered models (outcome AUC₁₀₀ₖ; boot 95% CI; `*` = excludes 0)

| model | term | β | boot 95% CI | |
|---|---|---|---|---|
| M0 dose (n=75) | milestone_c | −0.042 | [−0.153, +0.062] | |
| **M1 (n=56)** | **cov** | **+0.318** | **[+0.132, +0.512]** | * |
| M1 | occ_phys | −0.071 | [−0.366, +0.249] | |
| M1 | occ_rew | −0.071 | [−0.365, +0.240] | |
| M1 (panel) | fwd | +0.129 | [−1.772, +1.590] | |
| M1 (panel) | geom | +0.126 | [−1.666, +1.916] | |
| M1 (panel) | vsa | −0.134 | [−0.321, +0.592] | |
| M1 (secondary) | retdec | +0.082 | [+0.024, +2.969] | *(degenerate; see below) |
| M2 joint | cov | +0.287 | [+0.109, +0.547] | * |
| M2 joint | occ_phys | −0.010 | [−0.358, +0.309] | |
| M2 joint | fwd | +0.421 | [−3.028, +3.462] | |
| M2 joint | geom | −0.352 | [−3.820, +3.361] | |
| M2 joint | vsa | −0.057 | [−0.209, +0.912] | |

M2 max VIF = 147 (fwd/geom) > 10 ⇒ **M2 is reported unstable per the
frozen rule**; note the instability is confined to the fwd/geom pair
(their VIFs 147/147) while cov/occ_phys/vsa VIFs are 1.2–1.4 — cov's
joint estimate is nonetheless quoted only as corroboration, M1 carries.
retdec's nominally significant CI is an artifact of its −181..1 range
(single-outlier leverage under z-scoring); reported as degenerate, not as
a panel result.

## Reading

1. **Coverage replicates in a third decoupled domain**: cov +0.318*
   observationally (cup +0.71*, pooled 5a +0.28*), M0 dose null again —
   *what* the buffer covers matters, more pretraining alone does not.
2. **No negative occupancy association in reacher** (−0.07, CI spans 0).
   With occupancy range-compressed to ~0.01 across modes this is largely
   by construction — but that itself supports the post-Axis-1
   interpretation: the 5a pooled occ⁻ (−0.36*) was carried by cup's
   composition confound, not by a general law. Three domains now show
   three different observational occupancy stories (cup −0.75*, finger
   ~0, reacher ~0 at no variation) while coverage is positive in all
   signal-bearing domains.
3. **Model-side panel null/degenerate again** (fwd/geom collinear, vsa
   null) — transfer tracks buffer content, not standalone WM quality, in
   every domain measured so far.
4. Reacher's moderate transfer (AUC₁₀₀ₖ ≈ 95–98 for p2e/apt) at
   near-zero regime occupancy contrasts with finger (no transfer at
   occ 0.05, strong at 0.32): regime occupancy binds in finger but not
   reacher — domain heterogeneity in *whether* the occupancy bottleneck
   is active, exactly the kind of scoping the v4 re-review demanded
   (report per-domain, no pooled consistency claim).

## Files

- `results.json` — frozen fit output (this read).
- `drivers_reacher.csv` — 56 verified driver rows.
- Cluster variant (measured-only z-scoring; non-canonical):
  `local_results/e1_reacher_final_20260712_001736/analysis/`.
- Outcome-side descriptives: `artifacts/e1_reacher_20260711_095952/`.
