# Goodhart fixed-code partial re-test read — 2026-08-11

Registration: `prereg/PREREG_goodhart_partial_20260811.md` +
`PREREG_goodhart_partial_amend1_20260811.md` (E1 dropped — p2elong
evaluator missing; pool floor 88; within-family scope). Reader:
`analysis/goodhart_partial_read.py` (frozen, committed; bundled copy
byte-identical; selfcheck PASS incl. the amendment floor legs). Bundle:
`goodhart_partial_20260811_090931` (sha-manifest verified; 88/88 E2
score jsons). The registered collate step (not run cluster-side) was
executed locally, mechanically, with the registered command against the
sha-verified swave bundles' scores.jsonl
(`E2_pool_metrics.json` archived here); the reader then verified every
`real` against the pinned archived final10 (1e-6 provenance gate — all
88 pass). ONE execution.

## Verdict (E2, sole evaluator): **LOTTERY-REPLICATES / SELECTION-NOT-USEFUL**

- Spearman(m_return, real) = **+0.165**, perm p = .124 (B=10K) — the
  registered p ≥ .05 branch.
- **Top-1 regret fraction 87.4%** vs random-pick mean regret ~47% —
  trusting the evaluator's top pick is WORSE than picking at random
  from this pool; top-k and inversion descriptives in `read.json`.
- Forward wording (weaker-of rule, single evaluator, disclosed):
  **the evaluator cannot rank policies even with the fixed code** —
  now cleanly attributable to WM evaluation itself, not the
  double-symlog defect.

## Interpretation boundary (frozen in the registration)

- The archived sprint/p2elong conclusions remain PERMANENTLY WITHDRAWN
  (destroyed pools) — this read does not and cannot close them.
- Scope: within-family policy pool (88 s×w_r task-arm policies; full
  final10 spread 0–904.5 but one policy class) — cross-class ranking
  wording unlicensed per Amendment 1.
- Striking continuity, descriptive only: the fixed-code ρ (+0.165)
  sits exactly at the withdrawn-era levels (+0.153 / +0.209) — the
  bug-compression hypothesis for the lottery finding is not supported;
  evaluator-based policy selection fails at this correlation level with
  or without the bug. For #28 (prospective curation), the instrument
  enters as a documented negative: WM imagined-return ranking of
  policies ≈ lottery, two eras, two code versions.
