# Realized-training disclosure + sensitivity (dated addendum, 2026-08-07)

Labeled post-read integrity addendum to the 25m record (and, by
extension, the 12m Option-B record it contrasts with). **No registered
verdict is re-adjudicated**; this note discloses realized training
values the prereg required in the artifact ("realized upd/s, walltime,
and any update-count fallback recorded … no blank slots",
`PREREG_compcapacity_confirm_20260726.md`) that were never recorded,
plus labeled sensitivity recomputes. Trigger: the 07-Aug adversarial
review of the Paper-3 design doc (finding M5) demanded recovery of the
realized values; recovery then surfaced the truncation facts below.

## Facts (recovered from the bundles' OFFLINE_FIT_PROGRESS)

- **25m wave (`compcapacity_25m_full_20260801_214345/wm_meta`): 22/32
  fits reached 500000 updates; 10 truncated heterogeneously**
  (111,900–347,250; walltime kills on the per-lane queue). By cell:
  task-s0 7/8 full (seed7 141,750), task-s1 6/8 (seed3 242,450, seed7
  170,775), apt-s0 5/8, apt-s1 4/8.
- **12m wave (`scaling_option_b_20260726_130956/runroot_light`): 15/32
  full; 17 truncated** (104,550–431,425). By cell: task-s0 6/8,
  **task-s1 1/8** (range 133,800–500,000), apt-s0 5/8, apt-s1 3/8.
- Mechanism (structural, verified in `scripts/axis1.sbatch`): the
  adapt stage loads `latest_ckpt()` = newest completed checkpoint at
  ANY step, and the fit stage is idempotent-skipped whenever any done
  checkpoint exists — so a walltime-killed fit flowed silently into
  its adapt and the E4 pass. Neither read had a counter gate; the
  registered uniform-reduced-count fallback is NOT what happened.
- The registered verdicts were executed exactly as frozen (the frozen
  readers consume auc.csv/E4 csv only); this note changes no verdict.

## Sensitivity (labeled exploratory; seed-cluster bootstrap B=10K rng 0)

Full table + per-run rows: `realized_training_sensitivity_20260807.json`.

- **Dose–outcome coupling is absent to NEGATIVE**: pooled within-cell
  corr(update-counter, AUC100k) = **+0.019 at 12m, −0.309 at 25m**;
  truncated fits' cell means sit at-or-ABOVE full fits' in 6/8 cells
  (deltas +2.8 to +57.9). Truncation did not depress the affected
  cells — the direction of any bias is against, not for, the
  registered conclusions' pattern.
- **Interaction@25m, full-fits-only** (n = 6/7/4/5): **+85.8
  [+24.7, +145.4]** vs +113.8 [+62.1, +164.1] all — attenuated,
  still entirely positive. C1's 25m rung survives the exclusion.
- **Primary (task-lo rise 25m−12m), full-only** (n = 7 vs 6): **−16.3
  [−54.5, +16.4]** vs −12.2 all — the bounded null is unchanged.
- 12m interaction cannot be recomputed full-only (task-s1 has one
  full fit); its robustness rests on the dose-decoupling result
  above (+0.02 pooled at 12m) and is disclosed as such.
- Note (descriptive curiosity, no claim): at 25m, more fit updates
  weakly associate with WORSE downstream frozen-readout transfer
  (−0.31) — consistent with early plateau + mild late overfit of the
  WM on the static buffer.

## Task-hi 25m membership (review finding B1; recorded here as a labeled descriptive)

h=0 in-regime rew-NLL, `ax1wm_finger_s25q1s1_seed1..8` (from this
artifact's own `e4_25m_finger_v1.csv`): 2.82, 1.93, 2.22, 1.34, 7.35,
2.03, 1.23, 5.85 — **mean 3.10; 6/8 above the 1.5 member bar** (12m
hi reference: 1.304, in-band). The hi side drifts OUT of the
membership band at 25m. Not dose-driven: the two extremes (7.35,
5.85) are FULL-trained fits; truncated seed7 has the lowest value
(1.23). Consequences: the "metric-consistent regime-A" wording is
re-scoped to what the records license (AUC flat + LO membership not
approaching the band; hi-side ordering hi<lo preserved at every size
but absolute hi levels drift up at 25m); the Paper-3 F2 figure must
draw this honestly. Partial-unblinding disclosure: these hi-side
values were first computed during the 07-Aug design review (outside
any registered read); they are descriptive only and touch no frozen
decision rule.

## Standing consequences

1. Any future wave read MUST verify fit counters against
   total_updates before consuming adapt/E4 outputs (provenance
   checklist addition; the axis1 idempotent-skip + latest_ckpt pair
   is a standing trap).
2. Paper-3 claim wording (C1/C3/C4) carries the realized-training
   disclosure; F1/F2 captions disclose per-cell full-fit counts.
3. No top-up re-runs: the sensitivity shows the exclusions do not
   move any conclusion, and re-running would spend the expensive
   wave class to answer a question this note already answers
   (decision D4 in `research_notes/Paper3_Design_20260807.md`).
4. Realized parameter counts for size12m/size25m remain to be
   computed (config blocks recovered here; a CPU param-count pass
   can pin the true multipliers — the "12×/25×" labels are config
   names, and the 12m record already shows 15.4× realized).
