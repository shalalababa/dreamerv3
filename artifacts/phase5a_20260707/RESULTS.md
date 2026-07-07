# Phase 5a unblinded results — 2026-07-07

**This is the first read of adaptation outcomes.** Pipeline exactly as frozen in
`analysis/PREREG_phase5a.md` (2026-07-06); no deviations (see
`analysis/DEVIATIONS.md`). Inputs: adapt snapshot
`local_results/runroot_snapshot_20260707_133357/` (225 runs), measure snapshot
`local_results/runroot_snapshot_20260707_132942/` (45 pretrain runs × retained
milestones).

## Data completeness / QC

- **Adapt grid 225/225 complete** (3 modes × 3 domains × 5 seeds × 5
  milestones), all with `ADAPT_DONE`; **all 225 pass the ≥20-episodes QC**
  (scores arrive as ~16-episode eval bursts every ~16K steps, so the primary
  100K window holds ~96 episodes). `exclusions.csv`: 0 outcome exclusions.
- **Driver rows: 214/225.** The 11 missing cells are the pre-registered
  snapshot-distance exclusion (PREREG §1: no retained checkpoint within 5,000
  steps of the milestone; `measure.sbatch` skips those milestones upstream).
  Cells enumerated in `exclusions.csv`; 9 of 11 are `random_*` at 300K/500K
  (sparse late snapshot retention), plus `p2e_finger_seed1@500K`.
  NOTE (pre-registered tension to report): the *adapt* runs for those same
  cells used the nearest snapshot even when >5,000 steps off (e.g.
  `random_cup_seed3` ckpt500000 ran from step 489,472). Those rows enter M0
  (dose-response, milestone label) but have no driver row, so they drop out of
  M1/M2 automatically.
- Walker has no `occ_phys`/`vsa` by design (PREREG §4 domain notes); walker
  `occ_rew` is degenerate (≈0.999 everywhere — dense reward), reported as
  uninformative.

## Descriptive (PREREG §8 allows; not headline)

Mean AUC₁₀₀ₖ over seeds × milestones (±SD):

| domain | p2e | apt | random |
|---|---|---|---|
| cup | **661.8 ± 197.7** | **537.7 ± 244.2** | 170.3 ± 92.2 |
| finger | 78.8 ± 22.0 | 86.7 ± 28.7 | 84.8 ± 21.5 |
| walker (control) | 197.2 ± 102.4 | 189.3 ± 105.3 | 131.0 ± 43.7 |

- **Cup: large exploration-pretraining transfer.** p2e/apt separate from
  random at every milestone (see `adaptation_curves.png`); final10 reaches
  ~830–930 vs ~220 for random.
- **Finger: no signal.** Flat curves, no visible readout learning within the
  125K adapt budget in any condition; mode means within noise of each other.
  finger_turn_hard appears too hard for frozen-readout adaptation at this
  scale.
- **Walker (collinear control): moderate transfer**, growing with milestone
  (pooled AUC₁₀₀ₖ 101 → 273 from 100K → 500K), p2e/apt above random from
  300K on.

Exploratory Pearson r with AUC₁₀₀ₖ (within domain, n≈69–73):

| driver | cup | finger | walker |
|---|---|---|---|
| cov | **+0.71** | −0.12 | +0.69 |
| occ_phys | **−0.75** | −0.01 | (undefined) |
| occ_rew | +0.08 | −0.01 | +0.46 |
| fwd | −0.40 | +0.19 | +0.41 |
| geom | −0.29 | +0.13 | +0.13 |
| vsa | −0.59 | +0.24 | (undefined) |
| retdec | −0.68 | +0.08 | −0.02 |

Pattern matches the study hypothesis in cup: coverage tracks transfer,
physical-regime occupancy *anti*-tracks it (cov↔occ_phys r = −0.63 in the
pooled primary domains, so the M1/M2 partial effects below are the formal
read). Model-side drivers are heavily intercorrelated (fwd↔geom 0.97,
fwd↔vsa 0.82) — M2 VIFs expected high, as pre-registered.

## Pre-registered models (M0/M1/M2, PREREG §6) — primary outcome AUC₁₀₀ₖ

Effect sizes are std-AUC per std-driver (within-domain z-scored). Registered
inference = cluster bootstrap over pretrain_run, B=1000, seed 0, percentile
95% CI. `*` = CI excludes 0. Full numbers in `results.json`.

### Primary population: cup+finger pooled (150 rows, 30 runs; M1/M2 n=141)

| model | term | β | boot 95% CI | |
|---|---|---|---|---|
| M0 dose | milestone_c | +0.077 | [−0.014, +0.168] | |
| **M1 (primary)** | **cov** | **+0.277** | **[+0.045, +0.488]** | * |
| **M1 (primary)** | **occ_phys** | **−0.363** | **[−0.603, −0.206]** | * |
| M1 | occ_rew | −0.040 | [−0.196, +0.094] | |
| M1 (panel) | fwd | −0.061 | [−0.243, +0.094] | |
| M1 (panel) | geom | −0.061 | [−0.228, +0.087] | |
| M1 (panel) | vsa | −0.113 | [−0.364, +0.075] | |
| M1 (secondary) | retdec | −0.253 | [−0.696, +0.007] | |
| M2 joint | cov | +0.255 | [+0.002, +0.501] | * |
| M2 joint | occ_phys | −0.313 | [−0.646, −0.121] | * |
| M2 joint | fwd | −0.004 | [−0.299, +0.346] | |
| M2 joint | geom | −0.039 | [−0.320, +0.198] | |
| M2 joint | vsa | +0.208 | [−0.058, +0.500] | |

M2 max VIF = 5.3 (fwd) < 10 → M2 is stable per prereg and reportable.
Zero bootstrap failures in any model.

**Both pre-registered primary endpoints land, in the hypothesized
directions:** buffer coverage positively predicts early adaptation;
physical-regime occupancy *negatively* predicts it (not merely null), and
both partial effects survive jointly in M2. The model-side panel
(fwd/geom/vsa) is null throughout — transfer tracks what the buffer covered,
not how accurate/decodable the frozen WM is on its own rollouts.

### Control population: walker (75 rows, 15 runs; M1 n=73)

| model | term | β | boot 95% CI | |
|---|---|---|---|---|
| M0 dose | milestone_c | +0.507 | [+0.363, +0.652] | * |
| M1 | cov | +0.697 | [+0.571, +0.921] | * |
| M1 | occ_rew | +0.544 | [+0.433, +0.659] | * |
| M1 | fwd | +0.665 | [+0.443, +0.972] | * |
| M1 | geom | +0.133 | [−0.038, +0.416] | |
| M1 | retdec | −0.020 | [−0.236, +0.182] | |

occ_phys/vsa undefined on walker (PREREG §4); M2 not fit (requires all five
drivers). The control behaves as designed: in the collinear domain
essentially everything (dose, coverage, occ_rew, fwd) "predicts" adaptation,
which is exactly why the observational estimate cannot separate drivers
there and the decoupled domains + Phase-6 interventions carry the claim.

### Caveats to carry into the paper

- The pooled primary effect is carried by cup; finger contributes ~no signal
  (flat adaptation — finger_turn_hard appears too hard for frozen-readout
  adaptation in 125K steps). Per-domain exploratory fits should be reported
  in the supplement, labeled exploratory.
- M0 (dose alone) is null in the primary population — more pretraining is
  not by itself better; *what* the buffer contains is.
- occ_phys's negative sign exceeds the pre-registered hypothesis (which was
  "coverage, not occupancy" — i.e. occupancy null). The Phase-6 Axis-1
  intervention (occupancy moved at matched coverage) is the causal test of
  whether this negative association is real or confounded.

## Secondary/robustness outcome windows (PREREG §2)

Same frozen pipeline with W = 50K (secondary; `results_auc50k/`) and 125K
(robustness; `results_auc125k/`). Primary-population betas (boot 95% CI;
`*` = excludes 0):

| term | AUC₅₀ₖ | AUC₁₀₀ₖ (primary) | AUC₁₂₅ₖ |
|---|---|---|---|
| M0 milestone_c | +0.024 | +0.077 | +0.111 [+0.015, +0.206]* |
| M1 cov | +0.210 [−0.023, +0.424] | **+0.277*** | **+0.315 [+0.098, +0.525]*** |
| M1 occ_phys | **−0.281 [−0.514, −0.128]*** | **−0.363*** | **−0.404 [−0.640, −0.258]*** |
| M1 fwd | −0.111 | −0.061 | −0.089 |
| M1 geom | −0.082 | −0.061 | −0.097 |
| M1 vsa | −0.158 | −0.113 | −0.097 |
| M1 retdec | −0.225 [−0.621, −0.002]* | −0.253 | −0.234 [−0.676, −0.026]* |
| M2 cov | +0.102 | +0.255* | +0.297 [+0.052, +0.510]* |
| M2 occ_phys | −0.240 [−0.584, −0.035]* | −0.313* | −0.348 [−0.683, −0.176]* |

- **occ_phys is significantly negative at every window, in M1 and M2.**
- **cov is significant at 100K/125K, attenuated (same sign) at 50K** —
  coverage's benefit needs adaptation steps to express; the early window is
  noisier (fewer eval bursts).
- retdec's negative association reaches significance at 50K/125K and is
  borderline at 100K — report as the secondary panel result it is.
- Model-side panel (fwd/geom/vsa) null at all windows.
- Walker control unchanged at all windows (dose/cov/fwd/occ_rew all
  "significant"; geom joins at 50K).

## Exploratory per-domain fits (supplement; NOT pre-registered)

`exploratory_cup_only/`, `exploratory_finger_only/` — same frozen code with
`--primary_domains <dom> --control_domain none`, outcome AUC₁₀₀ₖ.

**cup only (75 rows, 15 runs):** M1 cov **+0.715 [+0.618, +0.816]***,
M1 occ_phys **−0.749 [−0.834, −0.678]***; vsa −0.540* and retdec −0.682*
also negative (value decodability is highest exactly in the high-occupancy
random runs — retdec means: random 0.93 vs p2e 0.17 — so it proxies
occupancy here). **Cup-only M2 is unstable (max VIF 24.4)**: within cup,
cov and occ_phys are nearly perfectly anti-correlated across runs, so the
joint model can't split them — cov collapses to ~0 wide-CI while occ_phys
holds at −0.755*. This is the pre-registered VIF>10 situation: M1 + the
Phase-6 interventions carry the claim; the *pooled* primary M2 stayed
stable (VIF 5.3) because finger decorrelates the drivers.

**finger only (75 rows, 15 runs):** buffer-side null everywhere (cov −0.13,
occ_phys −0.01); weak positive fwd +0.190* and vsa +0.246* — interpret
cautiously given the domain shows no adaptation signal to begin with
(AUC spread ≈ noise); M2 unstable (VIF 23.9).

**Reading across the two:** the pooled primary result is cup's signal
diluted by finger, and within cup the observational data cannot separate
coverage from occupancy — which is precisely the identification gap the
Axis-1 controlled buffers (coverage matched, occupancy split) were built to
close. Phase-5a's directional prediction for that contrast: **the
higher-occupancy side adapts worse.**

## Files

- `auc.csv` — 225 outcome rows (frozen `analysis/adaptation_auc.py`).
- `drivers.csv` — 214 driver rows (frozen `analysis/collate_drivers.py`).
- `exclusions.csv` — 0 outcome + 11 driver-cell exclusions.
- `results.json` — frozen `analysis/fit_mixed_effects.py` output
  (M0/M1/M2, cluster bootstrap B=1000, seed 0).
- `adaptation_curves.png` — descriptive adaptation curves.
