# AMENDMENT 1 to PREREG_gate_d1_r2probe_20260722 (2026-07-23)

Written 2026-07-23, BEFORE the frozen read
(`analysis/gate_d1_r2_read.py`) was executed on any R2 label. The 24
label files exist (generated 22 Jul, pulled 23 Jul under
`local_results/d1_routea_r2_save60_20260723_103849/`); no label
content, Spearman, or criterion value has been computed or seen at the
time of writing. Every statement below rests only on code facts, git
history, and checkpoint STEP metadata (nearest.json).

## A. Registered rule change: imagined-op cells demoted (code defect)

External review (`research_notes/D1_GPT_Analysis_20260723.md`) claimed
the imagined purchase is ill-defined. VERIFIED against code before any
outcome was read:

- Each `d0_eval` samples candidate actions as {policy mode} ∪ {M−1
  FRESH policy samples} (`dreamerv3/agent.py` "Candidate actions:
  policy mode plus M - 1 samples"; agent RNG advances between calls).
- `op_imag` (`d0/oracle_labels.py`) averages the Q matrices of
  `budget` such evaluations COLUMN-BY-COLUMN and applies the argmax
  index to the FIRST evaluation's candidate set. Only column 0 (mode)
  is action-aligned across evaluations; columns 1..M−1 average
  Q-values of DIFFERENT actions.

The imagined-op Δ therefore does not measure "additional imagination
refining the same candidates"; it mixes a variance-reduced mode column
with mixture columns. Consequence, registered here pre-read:

- **The decision rule is RESTRICTED to real-op cells: R2 FIRES iff C1
  passes in ≥ 1 of {early×e1, early×e4, late×e4} × {real} × {F0, F1}
  — 6 looks (was 12).** This is a strict tightening.
- All imag-op numbers are reported as DESCRIPTIVE-DEFECTIVE (labeled
  as such), never decisional.
- Retroactive note on Gate D1 (18 Jul): its two imag cells carry the
  same defect. This does NOT alter the Gate-D1 verdict (its real cells
  also failed 0-for-2, and the gate was already failed), but the
  imag ≈ 0 result is reclassified from "evidence imagination has no
  value" to "uninterpretable under the defective estimand".

## B. Disclosure: instrument patches between smoke and labels

The registered smoke (stage 3) failed on the known JAX transfer-guard
gotcha; two patches were committed before the labels stage
(`fd66addb`, `f2798309`, 22 Jul 11:04/11:13; labels started 16:17):
net effect = `jax.config.update('jax_transfer_guard', 'allow')` after
agent load in `main_real`. Diff-verified plumbing only — no label
semantics, dials, or meta touched. The prereg required a dated
amendment BEFORE the labels stage; this disclosure is one day late
(recorded as a process deviation), but the patched labeler is
bit-identical to the code copy shipped with the results.

## C. Disclosure: attempt 1 discarded outcome-blind; save60 rerun

Attempt 1's retained "early" snapshots landed at ~41.4K steps —
outside the registered validity window [10K, 40K] (the frozen read
would assert-fail). The attempt was discarded on step METADATA alone
(no labels read) and the full registered grid rerun with denser
checkpoint saving (runroot `d1r2_r2probe_save60`), same seeds/configs/
dials. Rerun early snapshots: ~25.6K (|error| ≤ ~700). Same platform
(single CUDA instance) for all 12 pilots + smoke + 24 label passes.

## D. Interpretive caveat on dose cells (NOT a rule change)

The same external review observes that the D0 distractor doses are
reward-irrelevant nuisance BY DESIGN (`EVPI_theory_note_20260702.tex`:
"uncontrollable (no action coupling) and reward-irrelevant"), i.e. the
original design treats high dose as a dissociation control, not a
source of decision-relevant uncertainty. The registered dose-cell
prediction stands as frozen (its defensible channel is indirect: dose
slows task-dynamics learning, so a dosed 1e5-step run is effectively
earlier in training). Registered here pre-read: **if firing comes ONLY
from a dose cell (early×e4 or late×e4) and not early×e1, the
resource-gate consequence ("Route A revived") must weigh this
ambiguity** — a dose-only fire supports the phase/support-boundary
reading (Route B mechanism) at least as much as Route A.

## E. Everything else unchanged

Cells, criteria, pinned dials, boundary-replication cell, consequence
map (24+24 frozen either way), and the frozen read itself are
unchanged; the read is executed as frozen and its original 12-look
`fires` field is reported alongside the amended 6-look verdict.
