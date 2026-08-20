# AMENDMENT 1 — PREREG_u1a_pooled_20260820 (20 Aug 2026, pre-outcome)

**Trigger**: registration-integrity review post-freeze (d2247e96); applied
pre-outcome, no job submitted. Frozen body unedited; superseded
clause-by-clause below.

## B4 — power disclosure + underpowered branch (the review's central finding)
On the look-1 per-seed basis (sd = 197.55, from `unfrozen_u1.json`
per_seed), the pooled n=16 test at α=.0294 has **MDE80 = 149.1 — larger
than the look-1 point estimate of +145.0**; power ≈ .78 at the full observed
effect and ≈ .44 at a shrunk +100. Registered consequence: a third branch is
added. **NO-CALL-UNDERPOWERED**: if the primary does not fire AND the
realized MDE80 exceeds the look-1 point estimate, the outcome is recorded as
underpowered-non-replication, NOT as evidence against the interaction; the
SUGGESTIVE label persists on power grounds and this is stated wherever the
retirement would have been stated. The permanent-retirement clause applies
only to a powered non-fire (realized MDE80 ≤ 145.0) or a significant
reversal (reported verbatim as U1A-REVERSED).

## B5 — α honesty (copied form of the dose_task_amend1 block)
Look 1 was adjudicated on the U1 registration's percentile-CI rule
(cluster bootstrap, decision-on-CI-alone) and **FIRED** ([+22.9, +276.5]);
its perm p (.0859) was never a decision input. This wave is therefore a
post-hoc-multiplicity RE-TEST graded on a different statistic, not a
near-miss extension. Realized family-wise error across the two looks:
≤ .05 + .0294 = .0794 (union bound), ≈ .06 under approximate independence —
conservative relative to no correction, NOT a clean overall-.05 design.

## M7 — fresh-only same-sign conjunct
The primary becomes: pooled perm p < .0294 AND pooled BCa CI lower > 0 AND
the fresh-only (seeds 9–16) point estimate is positive. Rationale: without
the sign conjunct the pooled primary can pass on look-1 rows alone — the
winner's-curse structure the carrier registration exists to kill. The
fresh-only estimate remains otherwise non-decisional (no α claim).

## M8 — estimator pinned
`analysis/domains_read.one_sample` (exact 2^16 sign-flip + BCa) on the 16
per-seed interaction deltas. Look-1's estimator differed (cluster bootstrap
percentile); disclosed. Look-1 per-seed deltas come from the committed
`unfrozen_u1.json` per_seed array (order = seeds 1–8 by the U1 submit loop;
the array is unlabeled in the artifact — disclosed as PARTIAL attestation).

## M9 + M6 — config reference and era disclosure
"Byte-identical to the U1 wave's arms" is superseded: the U1-era fits were
lost 08-08 and regenerated 08-10. The reference for config identity is the
surviving `finger_refit_20260810` configs (paths + sha256s recorded in the
witness `_meta` by the registered preflight; reader refuses if absent);
excepted keys as in the carrier Amendment. Disclosed: look-1 rows came from
pre-regeneration fits (July era); fresh rows are Aug-era — era enters as a
block within the paired contrast, and the read reports the by-batch split.

**Ride-along commit**: this file + analysis/u1a_pooled_read.py +
STUDY_LEDGER.md.
