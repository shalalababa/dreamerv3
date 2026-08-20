# AMENDMENT 1 — PREREG_crowding_behavioral_20260820 (20 Aug 2026, pre-outcome)

**Trigger**: registration-integrity review post-freeze (d2247e96); applied
pre-outcome, no adapt submitted, no behavioral quantity inspected. Frozen
body unedited; superseded clause-by-clause.

## B6 — design corrected; pairing unit defined
The rescue wave is **w_r ∈ {1,100} × sides {0,1} × seeds 1–4 = 16 fits**
(PREREG_rescue_load_20260815 L61; rescue read per_fit keys). The frozen
body's "8 seeds" is wrong. The pairing unit is **(side, seed)**: n = 8
pairs. A missing fit drops its (side, seed) pair only — both w_r members —
and the realized n is recorded by dated edit before outcomes exist.

## B7 — honesty block (known at freeze / unknown)
KNOWN AT FREEZE: 16 FROZEN-readout adapts from these same fits exist
(`local_results/ruw_20260816_142429/runroot/`,
`adapt_ax1ruw{1,100}q1s{0,1}_finger_seed{1..4}`, frozen_wm: true) and their
w100−w1 AUC delta is computable at zero compute. **It has NOT been
inspected** (the reviewer deliberately declined to compute it; so did the
registrant), and it will NOT be computed before the ONE read. The reader
reports it as a labeled FROZEN-PROTOCOL SECONDARY (look-0 descriptive, no α)
inside the same read — so it becomes visible exactly once, alongside the
primary. UNKNOWN: every unfrozen-adapt quantity.

## B8 — adapt protocol and namespace registered
Adaptation: `AXIS1_ADAPT_CONFIG=unfrozen_readout`, U1 protocol verbatim
("standard adaptation config" in the frozen body is superseded — house
default is frozen_readout, which is not this wave). RUN_ID namespace:
`adapt_ax1uzruw{1,100}q1s{side}_finger_seed{k}_ckpt500000` — fresh,
disjoint from the executed frozen adapts; the read REFUSES if any target
dir pre-existed at submission (recorded by the preflight; the axis1
latest_ckpt idempotent-skip hole is the documented hazard).

## B9 — NO-CALL rule made unit-coherent
The frozen body's rule (MDE80 vs the +0.2144 AUROC probe gap) compared
incompatible units and is superseded: **NO-CALL-UNDERPOWERED iff the
primary does not fire AND realized MDE80 > 33.22 AUC** (the scratch-anchor
sd, `zero_compute.json` — the smallest behavioral scale the house treats as
meaningful). Otherwise a non-fire is a powered null: the mechanism claim
stays probe-level and the three downstream algorithm candidates gate CLOSED.

## M10 — the non-override argument, corrected and quoting the clause
The frozen clause is PREREG_rescue_load_20260815 item (v): "the adapts run
at rew=1.0 in BOTH cells (AXIS1_WR reaches the FIT invocation only —
verified) under the default frozen_readout adapt config — adapt scores must
never be read as 'behavior under w_r=100'." Its hazard is ADAPT-TIME w_r
misattribution. This wave satisfies the clause's premise rather than
circumventing it: AXIS1_WR is fit-time-only and this wave's adapts likewise
run at rew=1.0 in both arms. The estimand here is the behavioral consequence
of the w_r=100 PRETRAINING, which the frozen text neither adjudicated nor
forbade. That text stands unedited. (The frozen body's "no behavioral
estimand and no behavioral controls" reasoning is superseded by this
paragraph.)

## M11 — σ=8 exclusion restated as unavailability
No σ=8 rescue fits exist (the rescue wave ran q1_nzs2 only); σ=2 is the
only available substrate, so the "trim" is not a costed choice. The σ=8
ladder merge (task 0.6786 vs apt 0.6751) is the task-vs-apt merge, noted
only as an independent reason not to build a σ=8 extension.

**Ride-along commit**: this file + analysis/crowding_behavioral_read.py +
STUDY_LEDGER.md.
