# Deviations from PREREG_phase5a.md

Any change to the frozen spec gets an entry here with date, what changed,
and why. (Registration record for new stages: tracked `prereg/` directory,
immutable dated files; see plan v4.)

## Deviations

- **2026-07-11 — DISCOVERED IMPLEMENTATION DEVIATION: Axis-1 offline WM
  fits were reward-aware.** Found by the 11-Jul editorial code audit,
  verified same day. All 64 Axis-1 fits ran `offline_fit --configs
  dmc_proprio` with default `expl.mode: task` (scripts/axis1.sbatch; every
  saved `wm_audit/*/config.yaml`), which trains a reward head on logged
  task rewards with representation gradients (`reward_grad: true`) plus
  the replay-value loss (`repval_loss/repval_grad: true`) — whereas the
  Phase-4 online pretrains (`expl.mode` p2e/apt/random) skip both losses
  (`agent.py`: `reward_free`). In finger the regime is definitionally the
  reward condition (regimes.py), so reward supervision was differential by
  intervention side. **Consequence:** the 10-Jul Axis-1 read
  (`artifacts/phase6_axis1_20260710/`) is reclassified as the
  *reward-aware arm*; no reward-free causal claim rests on it. Corrective
  protocol frozen the same day in
  `prereg/PREREG_axis1_corrective_20260711.md` (reward-free `expl.mode
  apt` refit of the same buffers, seeds 1–8, run ids `adapt_ax1f*`;
  2×2 occupancy × reward-supervision read). E3 submissions and the
  seed-9–16 extension are paused behind the corrective read; queued
  reward-aware jobs cancelled. `axis1.sbatch` and `submit_all.sh` now
  require an explicit `AXIS1_EXPL_MODE` and derive distinct run-id
  prefixes, so the arm is always visible in run names and saved configs.
  Buffers, searches, and all Phase-5a observational results are unaffected
  (the deviation is in the fitting objective only). Plan v4
  (`research_notes/Research_Plan_v4_20260711.tex`) reorganizes the
  program around this; disclosure category: *corrective replication*.
  **Same-day operational follow-ups (v4 re-review, pre-outcome):**
  (1) bundle-path defect confirmed and fixed — `submit_axis1_bundle`'s
  explicit `--export` list lacked `AXIS1_EXPL_MODE` (bundle children would
  have died at the guard); added + value-validated (`apt|task`) + threaded
  through the bundle child; (2) flag-path defect caught by the required
  smoke test — `--expl.mode` is not a config key; corrected to
  `--agent.expl.mode`; (3) smoke test PASS on a real 400-step debug replay
  (apt: losses {con,dyn,position,rep,velocity}, no `rew` head; task: same
  + `rew`; saved configs record the mode; evidence in
  `artifacts/smoke_axis1_expl_20260711/`); per-run audit rule added.
  (4) `prereg/PREREG_axis1_corrective_amendment1_20260711.md` freezes the
  inferential hierarchy (finger Q1 sole confirmatory, cup Q2 secondary,
  bootstrap CI is the only decision criterion) and redefines the E3v2
  replication unit as the collector run (within-run cov-matched high/low
  pairs; feasibility measured from the frozen indices: finger 15/15 runs
  Δocc 0.20–0.36, cup 10/15 at 0.11–0.37; exact sign-flip permutation
  across collector runs is the primary small-cluster inference; `ax1d*`/
  `ax1r*` demoted to buffer resamples; mode strings `ax1w<collector>s<side>`
  reserved).

## Code notes and registrations (non-deviations)

- **2026-07-16 — W1/W2/W3 reads executed as registered (Amendment 2
  §B/§C/§D; `artifacts/w123_e3v2_20260716/RESULTS.md`).** Audit PASS
  248/248 (arm loss sets, saved configs, registered replay paths, QC;
  bit-consistency with the P0/10-Jul/aware-1–16 records on all
  overlapping rows). **W1 PRIMARY (finger E3v2 within-collector):
  interaction REPLICATES — Δ̂ mean +84.0, 14/14 collectors positive,
  exact sign-flip perm p = 0.00012 (two-sided floor at n=14); hier boot
  CI [+48.2, +124.0]** on buffers with source_l1 ≡ 0 (diversity confound
  eliminated by design). Apt simple effect −5.1 (perm p=0.57) = third
  independent reward-free null. Registered interpretation fires:
  buffer-level causal support for the objective–data-alignment headline;
  **scaling pilot authorized**. W1 SECONDARY (cup): does not replicate
  (−140, 3/9, p=0.31); labeled descriptive mirror pattern (apt simple
  +213, task null) consistent with the battery/E4 mechanism. W2 (cup-Q1
  apt pooled 1–16): +93.1 [−33.0, +217.1] ⇒ CI includes 0 ⇒
  directional-only, not claimed (fresh 9–16 +35, attenuation again). W3
  (finger-Q2 completion, descriptive): −5.4 [−41.9, +28.3] at n=16 —
  the Q2 null stands. Feasibility disclosures: finger p2e3 and cup apt2
  pairs SHORT at build scale (pre-outcome, frozen criteria) ⇒ 14 + 9
  collector units instead of 15 + 10.

- **2026-07-14 — E4 + Goodhart reads executed as registered
  (`artifacts/e4_goodhart_20260714/RESULTS.md`).** E4 (descriptive):
  relative in-regime advantage of hi-occ buffers holds 8/8 in both arms
  and both domains, but is **arm-invariant** (apt = task at modeling the
  regime better, yet apt transfers nothing) ⇒ regime-modeling accuracy
  is NOT the interaction's carrier; **reward-head NLL tracks the
  behavioral interaction across cells** (finger Q1 improves on hi, cup
  Q1 flat, cup Q2 follows reward density) ⇒ mechanism = reward
  predictability of transferred features; registered "higher err_out"
  leg fails (hi side better everywhere, differentially in-regime);
  obs-decoder version of signature 3 fails (task's obs advantage is on
  unrewarded frames). Cup-apt measure pass outstanding. Goodhart
  (registered rule): pool n=63, Spearman +0.15, top-1/3/5 regret = 0 —
  neither advance trigger fires ⇒ continue-but-weaker (stronger
  evaluator next); descriptive fragility: M's rank-2 policy has real
  return 0.0, LOO regret 100%.

- **2026-07-14 — Four registrations frozen + Stage-1A read PASSES.** (1)
  `prereg/PREREG_e4_stratified_error_20260714.md` — E4 estimator/probe
  sets/signatures (descriptive only; `probing/stratified_error.py`,
  selfcheck PASS, measure path e2e-validated on a local W0 checkpoint;
  `submit_all.sh e4-measure`). (2)
  `prereg/PREREG_p3_gradient_path_20260714.md` — gradient-path factorial
  (arms sgb/rgo/vgo via `AXIS1_ARM`, flags derived in axis1.sbatch,
  audit = saved config) + reward-label transforms sh/rl
  (`probing/relabel_replay.py`, selfcheck PASS); primary = B_rgo − B_sgb
  fully prospective; 80 jobs reserved. (3)
  `prereg/PREREG_gate_d1_stage1a_20260714.md` — quantified Gate-D1
  unfreeze criteria + Stage-1A decision rule. (4) **Stage-1A registered
  read executed same day
  (`artifacts/latent_uq_stage1a_20260714/RESULTS.md`): density-residual
  instrument (out-of-fold isotonic residual on the density proxy,
  `latent_uq_analysis --instrument density_residual`) PASSES — 10/10
  within sub-cells flip TRACKS_DENSITY→TRACKS_ERROR at 500K, all four
  cross cells TRACKS_ERROR (finger_random improves from AMBIGUOUS),
  early-checkpoint regime preserved.** Route B's calibration fix is now
  a registered, passing asset; Gate D1 (oracle-label criteria) still
  gates the 24+24 runs.

- **2026-07-13 — W0 read executed as registered (Amendment 2 §A); n=16
  interaction CONFIRMED.** Read of the 16 apt finger-Q1 seed-9–16 runs
  (`artifacts/w0_n16_interaction_20260713/RESULTS.md`). Audit PASS 16/16
  (saved `expl.mode: apt`; loss-line component set byte-identical to the
  audited P0 finger runs, no `rew`/`value` term); QC 16/16; seeds-1–8
  deltas bit-checked against the frozen paired JSONs. Primary (decision on
  CI alone): fully-paired n=16 interaction **+98.72 [+45.81, +158.83]
  excludes 0**, 13/16 seeds, exact 2¹⁶ permutation p=0.00272, d_z 0.83,
  LOO [+81.0, +110.4]. Registered subsidiary fresh-batch 9–16: +40.17
  [−12.77, +90.68] ns — attenuation entirely in the aware arm; honest
  effect size = pooled +98.7. Secondary: reward-free simple effect pooled
  1–16 **−3.71 [−16.97, +9.76]** (tight null; all apt cells on the ~80
  floor). Buffer battery (descriptive, hi sides only; lo re-run
  requested): finger occ↔reward episode-corr = **+1.00** (occupancy ≡
  reward density) vs cup **−0.51/−0.70** (sign flip explains domain
  heterogeneity); lo sides are mono-source vs multi-source hi ⇒ source
  diversity confounded with occupancy in the original Axis-1 buffers —
  E3v2 within-collector (Amendment 2 §B) now carries de-confounded
  identification, not just replication. Correction note added to the P0
  RESULTS.md audit prose (finger/cup loss-key lists were swapped;
  verification unaffected).

- **2026-07-13 — P0 corrective read executed as registered; decision-tree
  branch 2 fires.** Read of the 64 reward-free (`ax1f*`) runs per
  `prereg/PREREG_axis1_corrective_20260711.md` + Amendment 1
  (`artifacts/p0_axis1_corrective_20260713/RESULTS.md`). Per-run audit PASS
  64/64 (saved `expl.mode: apt`; loss lines contain no `rew` term); QC
  64/64. Headline confirmatory contrast (finger Q1, AUC₁₀₀ₖ, B=10K seed-0
  bootstrap CI): **+2.04 [−22.04, +24.16] = NULL** ⇒ the registered pivot
  branch fires (occupancy-rescue unsupported as reward-free; headline moves
  to observational reversal + occupancy × reward-supervision interaction;
  E3 as designed deprioritized). Registered 2×2 interaction (same seeds,
  same buffers): finger Q1 **+157.28 [+76.51, +240.68]**, 8/8 seeds, exact
  permutation p=0.0078, d_z 1.24; other three cells null (specific).
  **Supplementary, NOT registered for this read (labeled in RESULTS.md):**
  the reward-aware seed-9–16 extension (launched before the 11-Jul pause,
  snapshot `local_results/axis1_seed_extension_20260711_201510/`) was read
  as a fresh-seed robustness check of the reward-aware simple effect:
  finger Q1 +30.71 [−15.51, +75.59] ns — strong attenuation vs seeds 1–8
  (+159.32); pooled 1–16 +95.02 [+44.20, +152.91] still excludes zero.
  Winner's-curse caveat carried on the +157 interaction magnitude; fix =
  the protocol's registered re-pointing of the extension: reward-free apt
  finger-Q1 fits, seeds 9–16 (16 jobs), for a fully-paired n=16
  interaction. Disclosure category: corrective replication (confirmatory
  part) + labeled supplementary (extension read).

- **2026-07-12 — E1 registered read; canonical-variant note; search-within
  built.** (a) Reacher critic gate PASS (held-out R² 0.3696 ≥ 0.2 ⇒ VSA
  primary). (b) The registered E1 driver-level analysis was run with the
  frozen pipeline locally (`artifacts/e1_reacher_final_20260712/RESULTS.md`):
  cov +0.318* [0.132, 0.512], occ_phys null, M0 null, M2 unstable
  (fwd/geom VIF 147), retdec degenerate (range −181..1). The cluster-side
  fit (`reacher_primary_measured_only/`) pre-filtered AUC rows to measured
  cells before z-scoring — a deviation from the frozen spec input contract;
  qualitatively identical, but the frozen local run is canonical. Driver
  rows verified identical (56/56). (c) `search-within` mode +
  `selfcheck-within` added to `build_controlled_replay.py` implementing
  Amendment 1's collector-run units (per-source cov-matched high/low
  pairs, corner + rank-weighted candidates, cov_tol semantics unchanged,
  source_l1 ≡ 0 by construction; all three prior selfchecks re-PASS);
  submit section to follow before E3v2 launch.

Changes to the analysis *code* that enforce the frozen spec rather than
alter it, logged for transparency. All entries below were made before any
adaptation outcome was read locally.

- **2026-07-06 — mode filter in `fit_mixed_effects.py`.** PREREG §1 defines
  the dose-response unit of analysis with mode ∈ {p2e, apt, random}, but the
  frozen code read every `adapt_*` row in auc.csv. Once Axis-1 intervention
  adapts (run ids `adapt_ax1<q>s<side>_<dom>_seed<k>_ckpt<updates>`, mode
  strings `ax1q1s0` etc.) land in the same runroot, they would have leaked
  into M0/M1/M2 and the within-domain z-scoring. Added `--modes` (default
  `p2e apt random`) applied before z-scoring. No formula, window, driver, or
  inference change.
- **2026-07-06 — `paired` CLI convenience.** `paired_contrast()` itself is
  unchanged (PREREG §6). The CLI now accepts auc.csv directly: optional
  `--domain` filter, `mode` used as `cond` when no `cond` column exists,
  qc_pass filter applied. Pairing remains within-seed.
- **2026-07-06 — Axis-1 run naming registered.** Phase-6 adapt runs follow
  the frozen id pattern with mode = `ax1<quadrant>s<side>`, milestone =
  offline gradient-update count (500000 = equalized to the 500K-step online
  pretrains at train_ratio 1024 ⇒ 1 update/env step). AUC extraction
  (PREREG §2) applies to them unchanged; they enter only the paired
  contrasts, never the dose-response models.
- **2026-07-07 — driver-side exclusions enumerated post-hoc into
  exclusions.csv (unblinding day).** PREREG §1/§7 exclude rows whose
  milestone has no retained checkpoint within 5,000 steps. That exclusion is
  applied *upstream* by `scripts/measure.sbatch` (skips the milestone), so
  the frozen local pipeline never sees those cells and `adaptation_auc.py`
  cannot list them. The 11 affected cells (10 `random_*` at 300K/500K plus
  `p2e_finger_seed1@500K`; enumerated in
  `artifacts/phase5a_20260707/exclusions.csv`) were appended to
  exclusions.csv by a one-off script after the frozen collation ran.
  Exclusion *rule* unchanged; this only makes the pre-registered log
  complete. Note the corresponding *adapt* runs used the nearest snapshot
  even when >5,000 steps off, so those 11 rows appear in M0 (milestone is
  the label) but never in M1/M2 (no driver row to merge).
- **2026-07-11 — Gate F applied; E1 outcome verification; E3 pre-submit
  fixes registered; latent-UQ stage0 read.** (a) Gate F (frozen rule, 7-Jul
  addendum): oracle mean AUC₁₀₀ₖ 68.34 < μ+2σ = 131.83 of the 5a finger rows
  ⇒ rule fires — finger stays reported-null *in the observational
  population*; interpretation per the 10-Jul amendment (registered before
  any Gate-F outcome existed): buffer deficiency, not readout ceiling
  (Axis-1 high-occ buffer adapts; p2elong 496K-step budget probe stays
  flat). `artifacts/e2_gatef_20260711_092902/DECISION.md`. (b) E1 reacher:
  75/75 adapts verified bit-identical; mode-level direction replicates
  (p2e/apt ≈95–98 vs random 42 AUC₁₀₀ₖ); registered population fit awaits
  the measure grid. (c) E3 pre-submit check
  (`artifacts/e3_precheck_20260711/PRECHECK.md`): finger dose + both finger
  rpairs + cup r1 GO; registered pre-outcome fixes — cup dose re-search
  with `--n_candidates 1200 --beam 300` adopted iff total_abs_dev improves;
  cup r2 exclusion widened (all random labels; fallback original+r1-hi
  dominants) until the pair differs from r1; control-reuse rule where an
  r-pair side is member-identical to an already-run buffer (cup r1 lo ≡
  q1 s0 ⇒ contrast ax1r1s1 vs existing ax1q1s0, no duplicate runs). These
  are all search/efficiency changes made before any E3 outcome exists; no
  frozen criterion (cov tol, occ-sep, overlap, seed) changed.
  **Redo check (same day, still pre-outcome):** cup dose re-search ADOPTED
  (dev 0.2258→0.1567; levels 0.061/0.117/0.230/0.232 — top two duplicate at
  the cup frontier, kept as a pure-error replicate). Cup r2 all-random
  exclusion returned OK but docc 0.048 = 0.30× target (1.18× the separation
  floor) — **new registered minimum-dose criterion: an r-pair is adopted
  only if docc ≥ 0.5 × target_docc**; the all-random pair fails ⇒ fallback
  exclusion (random4 apt2 random2 apt5 random3) to be searched/rebuilt; if
  that also fails the criterion, cup keeps r1 as its only r-pair and the
  all-random pair is descriptive-only (no adapt runs). Code note:
  search-dose/search-rpair now record `n_candidates` in their output
  criteria (metadata only; the frozen `search` output is untouched).
  (d) Latent-UQ
  stage0 (other study): Biased Dreams attractor bias replicates 10/10
  held-out seed-cells ⇒ D0 stays frozen
  (`artifacts/latent_uq_stage0_20260710_221232/DECISION.md`).
- **2026-07-10 — Phase-6/Axis-1 unblinded.** The 64-run paired grid was
  analyzed on the cluster with the frozen scripts (copied alongside the
  data; `local_results/axis1_analysis_20260710_213221/`) and re-derived
  locally from raw `scores.jsonl` — all 64 AUC rows and all 16 paired
  contrasts bit-identical (<1e-6). No deviation; write-up in
  `artifacts/phase6_axis1_20260710/RESULTS.md`. Ordering note: the E1–E7
  addendum (7 Jul) was frozen **before** this read, so its E3 directional
  prediction ("dose-response slope negative") predates and now conflicts
  with the Axis-1 outcome (occupancy null in cup, positive in finger).
  Per the addendum's own freeze rule the E3 prediction is NOT revised;
  E3 outcomes will be reported against the frozen 7-Jul prediction, with
  the Axis-1-informed expectation noted as post-hoc. A dated post-Axis-1
  amendment in the addendum (registered before any E3/E5/extension run
  exists) adds: finger dose-response (+20 jobs, same `ax1d*` modes),
  Axis-1 seed extension 9–16 (same run-id scheme; seeds-1–8 read remains
  the registered primary, pooled estimates labeled post-unblinding
  confirmatory extension), and the occupancy-floored finger demo — new
  mode strings `rrwc` (natural finger draw) / `rrwd` (floored draw),
  parsing under the frozen `RUN_RE` and excluded from dose-response
  models by the `--modes` filter.
- **2026-07-07 — post-5a enhancement modules pre-registered; run-naming
  registry extended.** `research_notes/Addendum_Post5a_Enhancements_20260707.tex`
  (frozen before any Axis-1 outcome was read) registers modules E1–E7 and
  the following new mode strings, all parsing under the frozen `RUN_RE` and
  all excluded from the dose-response population models by the `--modes`
  filter: `goalwm` (Gate-F finger oracle probe), `p2elong` (finger budget
  probe), `ax1d0`–`ax1d3` (occupancy dose levels, coverage-matched),
  `ax1r1s0/1`, `ax1r2s0/1` (composition-robustness pairs), `rrwa`/`rrwb`
  (occupancy-capped replay demo). Reacher joins as a replication
  population: its rows use the ordinary dose-response modes
  (p2e/apt/random) and ARE in scope of the population models when the
  pipeline is run with reacher in `--primary_domains` — reported as a
  separate population, never silently pooled into the registered
  cup+finger primary. Reacher `regimes.py` threshold 0.025 confirmed
  against `artifacts/gate0_20260702/gate0_reacher/gate0.json`
  (goal 0.433 vs reward-free 0.005–0.010).
