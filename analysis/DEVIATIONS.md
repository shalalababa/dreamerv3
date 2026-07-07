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
