# AMENDMENT 1 — PREREG_carrier_freshrep_20260820 (20 Aug 2026, pre-outcome)

**Trigger**: registration-integrity review (ONE Opus reviewer, user-approved)
returned after the freeze commit d2247e96; all findings below are applied
pre-outcome — no job submitted, no outcome exists. The frozen prereg body is
NOT edited; this amendment supersedes it clause-by-clause. The frozen reader
`analysis/carrier_freshrep_read.py` (+ `rescue_slate_common.py`) is edited in
the same commit as this file (FB-Amendment-1 precedent: instrument fix,
dated, pre-outcome).

## B1 — FIRE is a conjunction; a reversal is not a replication
FIRE ≡ permutation p < α AND BCa CI lower bound > 0 (the Amendment-1
conjunction form), for BOTH the fresh primary and the pooled re-entry.
A significant negative (perm p < .05 AND CI upper < 0) is the new branch
**CARRIER-REVERSED**: reported verbatim, the estimand is retired, and no BH
re-entry or Scaling12 re-admission is claimed. Reader edited accordingly.

## B2 — missing gates restored to the reader
(1) Replay-linkage gate (`--linkage`): the rescue_load literal block adapted
to `axis1_finger/q1/side{0,1}` — every fit's replay source must resolve to
the registered side path; refusal if absent or short. (2) Within-invocation
pairing gate: the witness records `invocation_id` per run; the reader
refuses if the rgo and sgb members of any (seed, side) pair differ.

## B3 — modal n_ep pinned to look-1
`MODAL_NEP = 96` (the look-1 csv is uniformly 96, 64/64). The reader refuses
if the fresh modal ≠ 96 (dose_task_amend1 precedent), not merely on internal
non-uniformity.

## M1 — the BH family is frozen by membership, not only threshold
The family is the 21-row `bh` table in
`artifacts/review_response_20260808/stats_readjudication.json` (committed),
frozen as of the 8-Aug assembly. Concurrent registrations (u1a included) do
NOT retroactively alter this read's threshold; any later family
re-declaration is reported separately and cannot loosen or tighten .016667
for this read.

## M2 — N-ladder basis stated
N 33/43/55 at +51.5/+45/+40 is at 80% power against the .016667 re-entry
threshold; the registered pooled n=56 targets the conservative +40 rung. The
fresh-only primary (n=40, α=.05) has power ≈.95/.88/.79 at the same effects.

## M3 — α accounting completed
.016667 also clears the Pocock K=3 two-sided boundary (.0221). "Carries no
accumulated α" is a statement about this DATASET (disjoint seeds, single
look); it is the third test of the carrier HYPOTHESIS, and the paper says so.

## M4 — machinery named correctly
The reader uses `analysis/rescue_slate_common.one_sample_auto`: exact 2^n
sign-flip for n ≤ 20; Monte-Carlo sign-flip at n > 20 with N_MC=200000,
MC_SEED=20260820, p = (count+1)/(N_MC+1) — deterministic given the pinned
seed; MC se at the .016667 boundary ≈ 2.9e-4 (~1.7% relative). The frozen
body's "p3_factorial_read contrast machinery" phrase is superseded (that
module's exact enumeration is infeasible at n=40 and its CI is percentile,
not BCa).

## M5 — config-identity reference made auditable
The ops preflight byte-compares each fresh arm config against the surviving
seeds 1–8 archived rgo/sgb configs; the reference file paths + sha256s are
recorded IN THE WITNESS `_meta` (fields `config_ref_paths`,
`config_ref_shas`) by the registered preflight, alongside
`amend1_config_match`. Excepted keys, enumerated: seed, run_id/logdir,
replay/data paths, platform/device fields. The reader refuses if the ref
fields are absent.

## M6 — cross-era pooling disclosed
Look-1 rows (July, Midway3-era) pool with fresh rows (Aug, Vast/RCC).
The arm contrast is within-era per seed, so era enters as a block, not a
confound; the read reports a by-batch descriptive split (look-1 vs fresh
means), mirroring `stats_readjudication.batch_tests` (look-1 batch test
p = .958). The sd 91.26 planning constant is single-era and is planning-only.

## m2, m7, m8 — nits
Reader constant becomes `ALPHA_POOLED = 7 * 0.05 / 21` (exact). Job count
clarified: 160 fits + 160 adapts = 320 jobs ≈ 480 GPU-h at 3 h per
fit+adapt cell on a 5060 Ti lane. The frozen body's "reader to be frozen"
status line is superseded: the reader was frozen at d2247e96 and is edited
by this amendment pre-outcome.

**Ride-along commit**: this file + analysis/carrier_freshrep_read.py +
analysis/rescue_slate_common.py + STUDY_LEDGER.md.
