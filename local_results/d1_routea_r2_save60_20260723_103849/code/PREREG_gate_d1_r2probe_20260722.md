# PREREG: Route-A R2 decision probe (frozen 2026-07-22)

The one registered Route-A move left after the Gate-D1 fail
(`artifacts/gate_d1_20260718/RESULTS.md` §"One registered Route-A move
remains"; options analysis
`research_notes/D1_RouteA_Revision_Options_20260718.md`, recommendation
= R2 + R1 riding along + R3's cheap additions). Paper-2 track, off the
ICLR-2027 critical path. This file + both instruments are committed
BEFORE any R2 pilot or label exists.

**This probe is a RESOURCE GATE** (does Route A get a full wave?), not a
paper-confirmatory claim. Its registered rule is deliberately liberal
and its multiplicity is disclosed below; any full Route-A wave that it
authorizes carries its own confirmatory registration on fresh runs.

## Registered prediction (the estimand relocation)

From Stage-0's phase change (signals track error at ~100K, flip to
density-tracking at ~200K; cross-policy cells stay error-tracking) and
the attractor mechanism (theory addendum §6): **per-state EVSI
calibration against oracle Δ exists where epistemic signals still track
error — early in training and/or under distractor pressure — and
degenerates at converged dose-zero checkpoints.** The Gate-D1 fail is
then the boundary of a law, not a dead end.

## Design: 12 fresh local pilots × 2 checkpoints (single GPU)

- Pilots: {e1 = dose zero, e4 = `d0_dose3` (dim 32, scale 3.0)} ×
  {cup, finger} × seeds {11, 12, 13}, run ids
  `d1r2_{dom}_{dose}_seed{s}`. Config identical to the cluster D1
  pilots (`scripts/d0.sbatch`): `dmc_proprio d0_probe [d0_dose3]`,
  `--env.dmc.render False`, 1e5 steps. `checkpoint_watcher` retains
  every save during training.
- **Fresh LOCAL pilots for BOTH doses (registered rationale):** all
  cells come from the same machine/build, so the dose and checkpoint
  contrasts are never platform-confounded; fresh seeds 11–13 keep every
  outcome in this probe prospective. The cluster Gate-D1 labels stay
  the frozen record of that read and are NOT merged into this one.
- Checkpoints: **late** = final (`ckpt`, step ≥ 90K asserted); **early**
  = retained snapshot nearest 25K, valid iff its exact step ∈
  [10K, 40K] (asserted by the read; protects against watcher gaps).
- Cells: {early, late} × {e1, e4} × {real, imag}; **domains POOLED**
  within cell (6 runs/cell; registered rationale: probe power — the
  Gate-D1 LORO at n=2 sibling runs showed heterogeneity inversion;
  here LORO trains on 5 runs). Per-domain numbers reported
  descriptively.

## Instruments (committed with this file)

- `d0/oracle_labels.py` **R2 extension** (labeler_version
  `r2_ext_20260722`; selfcheck PASS incl. new asserts): emits udyn +
  the belief latent (RSSM deter, f16) per labeled state, a strided
  base-trajectory latent reference (`ref_deter/episode/step`), the
  distractor dose in meta, and value-bootstrapped G columns
  (`g_*_boot = G_h + V̂(s_h)`; R3 descriptive companion — the plain
  undiscounted G stays the primary label; Gate-D1-era columns
  unchanged in semantics).
- `analysis/gate_d1_r2_read.py` (frozen read; selfcheck PASS: density
  ordering, F0-cell fires, F1-only cell fires under F1 not F0,
  boundary stays clean, all-null does not fire, missing-cell trips).
  Criteria C1/C2/C3 are IMPORTED from the frozen 14-Jul
  `analysis/gate_d1_read.py` — thresholds and the run-clustered
  bootstrap are unchanged.
- `scripts/d1r2_local.sh` (driver; stages pilots → smoke → labels with
  ordering enforced).

## Pinned dials (the Gate-D1 read disclosed these post-hoc; this time
they are registered pre-outcome)

states 200, horizon 100, label_every 25, actions 8, rollouts 16,
labeler seed 0, ref_stride 5; kNN density k = 10 with same-episode
exclusion window ±10 steps; EVSI-hat = LORO-cross-fitted OLS on
standardized features (unchanged); feature sets F0 = {uq, aflip,
evpi_plugin, evpi_split, gap}, F1 = F0 + {udyn, logdens, udyn_resid}
(udyn_resid = within-run OLS residual of udyn on logdens — the
Stage-1A density-residual asset as an ÊVSI feature; the R1 test).

## Ordering and smoke

1. Freeze-commit this file + both scripts (+ the labeler diff).
2. `./scripts/d1r2_local.sh pilots` (12 runs, sequential).
3. `./scripts/d1r2_local.sh smoke` — 5-state extended-labeler pass on
   `d1r2_cup_e1_seed11` (real MuJoCo path: determinism assert, CRN
   identity, new fields). Output `*_smoke.npz` is excluded from the
   read by filename. Any instrument fix ⇒ dated amendment BEFORE the
   labels stage.
4. `./scripts/d1r2_local.sh labels` (24 passes) →
   `python -m analysis.gate_d1_r2_read --labels <dir> --output
   artifacts/gate_d1_r2_<date>`.

## Registered decision rule

- **R2 FIRES iff registered criterion C1 (Spearman(ÊVSI, Δ) ≥ 0.3 with
  run-clustered bootstrap CI > 0) passes in ≥ 1 of the three PREDICTED
  cells {early×e1, early×e4, late×e4}, for either op, under either
  feature set.**
- {late×e1} is the registered **boundary-replication cell**: predicted
  FAIL (it reproduces the failed Gate-D1 cell on the local platform —
  the port-integrity check). A C1 pass there is flagged as an anomaly
  and is NOT evidence for firing.
- Multiplicity: 3 predicted cells × 2 ops × 2 feature sets = 12 looks,
  uncorrected — disclosed; acceptable for a resource gate whose
  positive outcome only buys a registered replication wave.
- C2 (ranking lift), C3 (net decision value), per-feature Spearmans
  (incl. against the `_boot` labels), and per-domain splits: registered
  DESCRIPTORS, never decisional.

## Consequence map (frozen)

- **FIRES** ⇒ Route A revived as the boundary-law paper ("the value of
  computation is identifiable exactly where epistemic uncertainty still
  tracks error, and provably degenerates at convergence"). A full
  Route-A wave (its own prereg, fresh seeds, both domains, held-out
  validation per the 12-Jul revision) is required before anything
  further; **the 24+24 D0 runs stay frozen either way — this probe
  cannot unfreeze them.**
- **Does not fire** ⇒ Route A CLOSED; Route B absorbs the package
  (attractor + phase change + cross-policy boundary + Stage-1A fix +
  failed Gate D1 + this probe as a registered negative + the R5
  real-vs-imagined exchange-rate section); the 24+24 stay frozen
  permanently.
- Either way, the per-cell per-feature Spearman panel is the empirical
  phase-change exhibit for whichever paper survives.

## Disclosure

Known at freeze: the Gate-D1 fail (0/4 cells, all numbers), Stage-0
(10/10 attractor, phase change, cross-policy), Stage-1A (density-
residual passes 14/14), D1 descriptives (mean Δ_real +0.083 cup /
+0.222 finger; Δ_imag ≈ 0). Unknown: every early-checkpoint and dosed-
cell quantity; every extended-feature (udyn/logdens/udyn_resid)
calibration number anywhere; the local platform has produced only a
2-state CPU micro-smoke of the pre-extension labeler (18 Jul), never
sweep-grade labels (the boundary cell doubles as its validation). Compute [prov.
estimates]: pilots ~1–2 h each, label passes ~20–40 min each ⇒ ~1.5–2.5
days sequential on the 5090.
