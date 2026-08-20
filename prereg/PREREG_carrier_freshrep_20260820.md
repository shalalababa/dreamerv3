# PREREG — Carrier fresh replication (BH re-entry), 20 Aug 2026

**Status: FROZEN at user commit. ONE read execution. LAST look — declared.**
Motivated by `reviews/PreSubmission_Synthesis_20260819.md` §4.1 (rescue lens);
precedent for α-paid seed extension: `PREREG_dose_task_amend1_20260811`
(near-miss extended, Pocock paid, look-1 disclosed, LAST look declared,
passed .0293 vs .0294). This registration follows that template.

## Look-1 disclosure (all numbers pre-existing, artifacts cited)

- Registered Amendment-1 decision rule (percentile CI alone) FIRED:
  pooled rgo−sgb **+51.46 [+8.95, +94.63]**, n=16 seed-pairs
  (`artifacts/p3_amend1_20260717/`); fresh batch seeds 9–16 +52.7
  (no winner's curse at look 1); sgb placebo +5.2 tight null.
- What demoted it: the 8-Aug post-hoc 21-primary BH family — carrier at
  rank 9, perm p = .040894 vs criterion .021429 — plus Pocock exposure
  from the one prior extension. Estimand split registered 8 Aug
  (LEVEL claim retained; interaction-carrier form demoted).
- Re-entry target (recomputed in the synthesis, verified): **perm
  p ≤ .016667** (= 7·.05/21); at that p the BH step-up rejects through
  rank 7, which also re-admits `Scaling12_interaction` (rank 6,
  p=.015625) — the registered free-rider of this wave. sd of paired
  deltas 91.26 ⇒ N≈33 at +51.5, N≈43 at +45, **N≈55 at conservative +40**.

## Design

- Arms: **rgo, sgb** × sides {0,1} × **seeds 17–56** (40 fresh seeds) on
  the frozen finger q1 substrate (KEEP_SUBSTRATE; inventory verified
  20 Aug: axis1_finger/q1 present with manifests; no rebuild). 160 fit+
  adapt cells, same configs as Amendment 1 byte-for-byte except seed.
- **vgo EXCLUDED — stated up front**: it appears in no estimand here; its
  TOST rider was marginal at achievable N (±25 needs the pooled 56 and a
  near-zero point estimate) and the wiring audit already re-scoped vgo to
  the attenuation form. No new vgo claim will be made from this wave.
  Exclusion is a cost decision made before any new data exist.

## Estimands and decision rules

- **PRIMARY (fresh-only)**: [B_rgo − B_sgb] pooled over sides, seeds
  17–56 (n=40 pairs), permutation p primary + BCa CI, **α=.05
  two-sided**. Carries no accumulated α (disjoint seeds, single look).
  FIRE ⇒ the carrier replicates out-of-sample; the winner's-curse
  objection is dead.
- **SECONDARY (pooled re-entry)**: seeds 1–56 (n=56), **perm p ≤ .016667**
  ⇒ the carrier re-enters the 21-primary BH family at the α it was
  demoted under, and Scaling12 re-admits by step-up arithmetic (no new
  Scaling12 data; its re-entry is a family fact, reported as such).
- Seeds 1–16 rows come from the bit-checked Amendment-1 artifact —
  never re-measured.
- **Pre-stated failure consequence** (copied from the synthesis): if the
  fresh-only primary does not fire, the interaction-carrier estimand is
  **permanently retired** — no third look, ever. The LEVEL claim
  (estimand split, 8 Aug) is unaffected either way.

## Gates (any failure = refusal, read not consumed)

Fit counters == registered updates for all 160 (07-Aug standing rule);
kwargs/config byte-identity vs the Amendment-1 arm configs (seed field
excepted); STRICT modal n_ep; replay-linkage witness per side; no
cross-invocation pairing (arms compared within-seed on the same
invocation's adapts only).

## Reader

`analysis/carrier_freshrep_read.py` — to be frozen (selfcheck PASS)
**before any job is submitted**; reuses the frozen p3_factorial_read
contrast machinery with the seed ranges above; literal pins block:
{.016667, 91.26, seeds 17–56, artifacts/p3_amend1_20260717/auc_pooled_1_16.csv sha}.

## Ops

Substrate check → 160 fits+adapts (Vast lanes, 3 h/cell ≈ 480 GPU-h;
re-run until drained) → adaptation_auc collate + witness block → bundle +
sha manifest → **[ME] ONE read**.
