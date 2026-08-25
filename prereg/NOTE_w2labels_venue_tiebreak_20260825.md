# PRE-DECLARATION — Wave-2 labeling produced at TWO venues; tie-break rule
**Written 2026-08-25, BEFORE either cohort has completed a single cell.**
(Instance 30: 0/32 npz at time of writing, ~39 h elapsed. RCC: not yet
submitted beyond a smoke.) This is what makes the rule below
outcome-independent rather than a selection.

## Why two venues exist
`PREREG_p2_wave2_freshcohort_20260821` pins no venue. The 32 fresh-cohort label
passes were started on vast instance 30 at 8 passes/GPU. That box is a measured
pathology, not merely slow: BLAS/torch thread pools were left unbounded and
sized themselves to the whole machine (162 threads/proc, **5184 threads on 64
hardware threads**, **1,073,346 involuntary vs 3,581 voluntary context
switches**). It has burned 1173 core-hours against ~530 implied by the
identical-dial predecessor wave, which finished all 32 in 20.6 h wall-clock. No
progress reading is obtainable (vast containers run `ptrace_scope=1` without
`CAP_SYS_PTRACE`, so py-spy cannot attach; the labeler writes its npz only at
completion), so the remaining time is genuinely unknown.

A second cohort is therefore being produced on RCC with thread pools capped, as
a hedge. Both compute the SAME registered quantity from the SAME inputs
(identical `ckpt_late.pt`, identical dials: states 200, horizon 100,
label_every 25, actions 8, ref_stride 5, `--w1_repeats 8`,
`--env_seed 20260809`).

## The rule
1. **The FIRST cohort to write all 32 registered npz is the one that is read.**
   Completion time is not the estimand and is not a function of the labels'
   values, so this selects on something orthogonal to the outcome.
2. The other cohort is **retained, never deleted, and NEVER read** — not as a
   robustness check, not pooled, not reported. Retaining it is an audit
   convenience only. Reading both and choosing would be selection; this rule
   exists precisely to forbid that.
3. Outputs are written to **separate directories** so the two can never mix:
   instance 30 -> `$RUNROOT/w1_labels/`, RCC -> `$RUNROOT/w1_labels_rcc/`.
   The bundle carries exactly one cohort's 32 cells plus the 2 dup passes.
4. The winning cohort's venue, GPU model and thread settings are recorded in
   the bundle NOTES and the ledger, whichever venue wins.

## Why venue cannot bias the primary
The registered primaries are **within-pass estimands on the fresh cohort**; the
prereg states the archived W1 cells are "not re-read, not re-labeled, and not
pooled into any primary". No primary compares a Wave-2 number against a number
computed on other hardware, so a device difference cannot enter it. The dup gate
is likewise a within-pass CRN check (branches of identical actions inside one
pass). Only the **pooled secondary** mixes cohorts, and it is pre-registered as
carrying "no decision weight", reported "as a DESCRIPTIVE pooled point with its
provenance split" — a venue difference is one more entry in a provenance split
it already reports.

Independently: this labeling stack is **not** reproducible across invocations
even on one machine (2 Aug measurement: `g_all` max|Δ| = 95.0, `m_now` flips
32% on back-to-back invocations of the same pass on the same box). Cross-venue
variation is therefore a draw from a distribution the design already tolerates,
not a new class of error.

## Dup passes
The 2 dup-gate npz already exist from instance 28, are on RCC, sha-verified, and
PASSED the reader's own `dup_gate` (exact zero on every estimand). They are
independent of this tie-break and are used regardless of which cohort wins.
