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
