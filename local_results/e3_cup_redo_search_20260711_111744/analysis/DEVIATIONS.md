# Deviations from PREREG_phase5a.md

None so far. Any change to the frozen spec gets an entry here with date,
what changed, and why.

## Code notes (non-deviations)

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
