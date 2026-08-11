# AMENDMENT 2 to the three refit registrations — 2026-08-10 (substrate correction)

Amends `PREREG_refit_reads_amend1_20260810.md`. Written AFTER the finger and
cup read executions: the finger replication rider returned BIT-IDENTICAL
NLL values vs the archived 14-Jul panel, which triggered a weight-hash
census (sha256 of `agent.pkl` payloads + per-run NLL bit-identity). This
amendment corrects Amendment 1's substrate classification, which was based
on checkpoint-directory DATES; dates were misleading because the refit jobs
re-saved restored original weights under fresh timestamps. **No decision
rule, no verdict, and no primary quantity changes** — the corrections
affect only provenance labeling and the (non-verdict-bearing) sensitivity
split.

## Corrected substrate census (weight-hash ground truth)

- **finger (32)**: **26 measured on ORIGINAL July weights** (all 16 task +
  10 apt; 24 as bitwise re-saves under Aug names + 2 measured on the July
  checkpoint directly) and **6 genuinely fresh apt fits** (fq1s0 seeds
  5/7/8, fq1s1 seeds 6/7/8 — their original payloads were the actual loss
  casualties, e.g. emptied ckpt dirs). Amendment 1's "30 fresh + 2
  original" was wrong.
- **cup (32)**: task s0 = **8/8 original weights** (proven by 8/8 bitwise
  NLL identity vs the archived panel); task s1 = 8/8 genuinely fresh
  (population-consistent, not bit-identical); apt (16) = no archived
  comparator, fresh-named, no old payloads present.
- **rl/sh (32)**: genuinely fresh (no original weights ever archived;
  clean fresh counters). Weight comparison structurally impossible and
  unnecessary.

## Consequences (disclosed, none verdict-bearing)

1. **The 2026-08-08 loss inventory was WRONG for finger q1 and cup q1s0
   task fits** — they were recoverable and were in fact restored. The
   "regeneration = fresh draws" premise holds only for: 6 finger apt fits,
   8 cup task s1 fits, cup apt, and all rl/sh fits.
2. For #9 / P-C3 this substrate is STRONGER than registered: the measures
   ran mostly on the exact original fits behind the archived verdicts.
3. The executed finger `read.json`'s `sensitivity_fresh_only` block is a
   ckpt-name-date split, not a true substrate split (output not edited —
   executed-reader discipline). The corrected substrate sensitivity is in
   the RECORD, labeled POST-READ: apt pooled AUROC 0.4017 [0.3816, 0.4383]
   (original-weights n=10) vs 0.4000 [0.3771, 0.4116] (fresh n=6) —
   substrate-stable; primary verdict unchanged in both classes.
4. **Instrument fact worth keeping**: `stratified_error` h0 evaluation is
   bitwise job-to-job reproducible given identical weights (16/16 finger
   task + 8/8 cup s0 task bit-identical across a 27-day gap, different
   jobs). Cross-job drift previously measured (~7e-4) is specific to the
   ridge-probe recompute path (different batching/code path), not to the
   E4 evaluator. The replication rider caught the provenance story exactly
   as designed.
