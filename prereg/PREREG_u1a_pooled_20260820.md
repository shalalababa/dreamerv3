# PREREG — U1a pooled re-test (headline interaction under unfrozen adaptation), 20 Aug 2026

**Status: FROZEN at user commit. ONE read execution. Second and LAST look
— declared.** Motivation: `reviews/PreSubmission_Synthesis_20260819.md`
§4.2 — after the 8-Aug downgrade of U1a to SUGGESTIVE, the record cannot
currently say the headline occupancy×supervision interaction survives
unfrozen (full fine-tuning) adaptation: the program's own top-ranked
objection. Precedent: `PREREG_dose_task_amend1_20260811` (α-paid
extension, look-1 disclosed, LAST look).

## Look-1 disclosure

U1a (n=8, `artifacts/unfrozen_u1_20260801/unfrozen_u1.json`):
interaction under unfrozen adaptation **+145.0 [+22.9, +276.5]** —
downgraded to SUGGESTIVE in the 8-Aug review resolution. That number and
its seeds (1–8) are look 1. This wave is look 2 of exactly 2.

## Design — REBUILD branch (inventory 20 Aug: seeds 9–16 fits 0/16 present)

- Cells: task/apt (reward-aware vs reward-free fit) × sides {hi, lo} ×
  **seeds 9–16** on the frozen finger q1 substrate = **32 fit+adapt jobs**
  (~160 GPU-h at 8 h/RCC or 3 h/5060Ti), configs byte-identical to the
  U1 wave's arms except seed.
- Adaptation: **unfrozen** (full fine-tuning), U1 protocol verbatim.

## Estimand and decision rule

- **PRIMARY**: W0-form interaction [task_hi − task_lo] − [apt_hi −
  apt_lo] under unfrozen adaptation, **pooled seeds 1–16** (look-1 rows
  from the frozen U1 artifact, never re-measured), permutation p primary
  + BCa CI, **Pocock K=2 boundary α=.0294** two-sided.
- SUBSIDIARY (labeled, non-decisional): fresh-only seeds 9–16 estimate —
  the winner's-curse check, reported with CI, no α claim.
- FIRE ⇒ the headline interaction is licensed under realistic
  (unfrozen) adaptation; the SUGGESTIVE label lifts.
- **Failure consequence, pre-stated**: no fire ⇒ U1a is retired to
  SUGGESTIVE permanently; the paper carries "frozen-protocol result;
  unfrozen replication did not confirm" wherever the headline is stated.
  No third look.

## Gates

Fit counters == registered updates ×32; config byte-identity vs U1 arms
(seed excepted); STRICT modal n_ep; within-invocation comparisons only;
witness block; refusal on any failure (read not consumed).

## Reader

`analysis/u1a_pooled_read.py` — frozen (selfcheck PASS) **before any job
is submitted**; W0 interaction machinery; literal pins {.0294, seeds
9–16, the U1 artifact path + sha}.

## Ops

Substrate check → 32 fits+adapts (re-run until drained) → collate +
witness → bundle + sha manifest → **[ME] ONE read**.
