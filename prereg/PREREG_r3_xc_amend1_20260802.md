# PREREG R3 xc Amendment 1: within-pass dual-chooser pairing (frozen 2026-08-02)

Amends `PREREG_r3_amend2_20260730.md` (freeze commit `6271c770`,
verified against `git log`). This file + the dual-extended
`d0/oracle_labels.py` (`--dual_chooser`; full selfcheck PASS incl. the
new xc8 dual legs and every pre-existing leg) + the amended
`analysis/r3_xconsumer_read.py` (selfcheck PASS) + the amended
`scripts/r3_xc_local.sh` are committed BEFORE any dual label exists:
`--dual_chooser` has never been executed on any real checkpoint, and
no cross-checkpoint statistic of any kind exists anywhere.

## Why the parent design is unexecutable (probe, 2026-08-01)

The parent paired a control pass against a swapped pass across SEPARATE
labeler invocations and QUARANTINED unless `g_all` matched at atol
1e-5. The pre-wave determinism probe
(`artifacts/r3_xc_detprobe_20260801/`, diagnostic — no registered gate
ran) executed the registered control configuration twice with
mutually identical dials (probe-only labeler seed 0, not the wave's
registered seed 1 — see Disclosure), same instance, same session,
back-to-back: `episode`/`step`
selection EXACT, but **g_all max |Δ| = 95.0** and `m_now` flips on
64/200 states. The drift is invocation-level (no job/node/session
pinning can remove it — the same substrate fact that invalidated the
repair parent, `AUDIT.md` in `artifacts/r3_repair_quarantine_20260801/`).
The parent's pairgate could therefore only ever QUARANTINE; the
128-pass wave was never submitted and the parent reader was NEVER
EXECUTED. The parent's re-entry logic (a dated amendment before any
label) is what this file executes.

## Amended mechanism: both choosers inside ONE invocation

`--dual_chooser` (labeler version **`d1fix_20260724_xc2`**, exact pin;
requires `--consumer_checkpoint` + `--oracle_all`; the labeler itself
enforces only that A consumer checkpoint is given — the wave passes
the OTHER maturity, and other-maturity identity is pinned by the
READER from the meta checkpoint suffixes):

- The pass runs under the eval agent's OWN heads (control). The
  consumer heads are loaded once via the existing overlay path (all
  its guards: counter snapshot/restore, no-op/partial-coverage
  refusal, config-safety comparer), snapshotted, and swapped OUT again
  before labeling starts (`DualHeads`).
- Per labeled state, control choosers are computed exactly as the
  parent registered (`m_now` plug-in, `m_real` op_real). Then the
  SHADOW choosers: the labeler resets the policy-RNG counter to the
  control `d0_eval`'s window, swaps in the consumer heads, re-runs
  `d0_eval` (⇒ `m_now_x`), re-runs op_real with the control's
  `rng_mark` and the CONTROL candidate set (⇒ `m_real_x`,
  `real_scores_x`), and swaps back. `g_all` is computed ONCE, from the
  control pass, for the control candidates, under the same rng_mark —
  the RNG schedule seen by the base trajectory and the rollouts is
  identical to a non-dual pass.
- **Machine-checked guards (frozen in the labeler, per state):**
  candidate identity `max|cands_shadow − cands| ≤ DUAL_CANDS_ATOL =
  1e-4` (the policy net is outside the overlay regex, so the replayed
  draws repeat up to float jitter; beyond the bound the shadow would
  rank actions `g_all` was never realized for — the pass REFUSES);
  per-candidate env-reward identity across the two op_real sweeps
  (atol 1e-6; CRN witness); shadow RNG-consumption identity (counter
  equality assert); head-swap read-back on a witness key in the LIVE
  policy store (a silent swap failure would fabricate chooser
  agreement — refused); byte-identical consumer/control head sets
  refused (`--dual_allow_identical` exists for an identical-heads
  smoke only and is NOT part of this wave).
- `op_imag` is not shadowed (imag is retired; unchanged from parent).
- Sample-noise class: each chooser evaluation retains the same
  per-evaluation sampling noise the parent's per-pass choosers had.
  The amendment removes ONLY the cross-invocation `g_all` comparison;
  it does not assume within-process float determinism anywhere — the
  gates above bound the only quantities that must agree.
- Default labeler paths are byte-identical: without `--dual_chooser`
  nothing changes for any existing registered caller (the `_xc1`,
  `_cm1`, behavior-shift, and base paths are untouched; the full
  labeler selfcheck including all pre-existing legs passes; stdout
  estimand redaction extends to `_xc2`).

## Estimands (per labeled state, one `g_all`, `--oracle_all`)

- achieved(ctrl) = `g_all[m_real] − g_all[m_now]`
- achieved(shadow) = `g_all[m_real_x] − g_all[m_now_x]`
- `d_pair` = achieved(shadow) − achieved(ctrl)

These are the SAME registered quantities as the parent (each pass of
the parent pair computed achieved under its own heads); they are now
realized within one invocation, which also removes the parent's
residual cross-pass chooser-noise asymmetry.

## Amended design

- **64 passes** (was 128): per run, one dual pass per eval side —
  `xc_{dom}_{dose}_seed{s}_{evalmat}_dual.npz` in `$R3_ROOT/xc_labels`,
  consumer = the OTHER maturity (enforced by the reader from meta
  checkpoint suffixes; a control-configured dual file cannot enter).
- Dials identical to the parent registration (states 200, horizon 100,
  label_every 25, actions 8, rollouts 16, ref_stride 5, max_steps
  1000, **labeler seed 1**, `--oracle_all`) + `dual_chooser: true` +
  `dual_cands_atol: 1e-4` stamped and pinned.
- **Ordering: freeze-commit → existence gate (unchanged, all-or-nothing
  32 runs, SUBSTRATE_GONE marker) → smoke (REGISTERED GATE: 2 dosed
  cup-e4-seed31 5-state DUAL smokes, one per eval side — each dual
  pass smokes BOTH choosers, so two files cover both swap directions
  and both controls; reader-asserted authenticity) → dualgate
  (REGISTERED GATE: ONE full-dial dual pass, cup e1 seed31 eval=late —
  one of the 64 — checked by the amended reader's `--dualgate`:
  meta pins, finiteness, `g_now = g_all[m_now]`, chooser bounds,
  candidate-drift bound; VALUE-BLIND — `d_pair`/achieved and their
  means are never computed) → labels (remaining 63) → ONE read.**
- The parent's 4-smoke set and `--pairgate` are retired with the
  two-pass design (unexecuted). The amended reader rejects `_xc1`
  files by exact version pin and `_h{mat}` filenames by regex.

## Registered decision rules (amended `analysis/r3_xconsumer_read.py`)

Unchanged in form from the parent — run-clustered percentile bootstrap
(cluster = training run, 32 clusters, B = 10K, `default_rng(0)`,
machinery imported from frozen `analysis/r3_read.py` with import-pin
asserts):

- **P-XC1 (late eval):** per-run mean of per-state `d_pair` from the
  eval=late dual passes (consumer = early); CI entirely < 0 ⇒ immature
  heads HURT.
- **P-XC2 (early eval):** per-run mean of per-state `d_pair` from the
  eval=early dual passes (consumer = late); CI entirely > 0 ⇒ mature
  heads RESCUE.
- **Verdict map unchanged:** HEADS-IRRELEVANT / HEAD-DEFICIT /
  HEAD-HARM / MIXED (both fire, or any unregistered-direction CI
  excursion) / SUBSTRATE-GONE. **QUARANTINE re-scoped** to the
  within-pass integrity class: `g_now ≠ g_all[m_now]`, candidate drift
  beyond 1e-4, chooser index out of range, non-finite shadow content —
  any violation ⇒ no adjudication, audit first.
- **Guards (machine-checked per file):** exact `d1fix_20260724_xc2`
  version; full dial identity incl. labeler seed 1, `consumer_regex`,
  `dual_chooser`, `dual_cands_atol`, and `dual_allow_identical=False`
  (the labeler stamps the flag, so a smoke-mode pass cannot enter the
  wave); checkpoint suffix = the filename cell AND consumer suffix =
  the OTHER maturity; train_seed vs filename; parent Amendment-1 array
  guards + all four chooser-index ranges + shadow array shapes;
  all-or-nothing 64-file grid; unregistered seeds trip; the reader's
  candidate-drift bound is import-pinned to the labeler's constant.
- **Honesty valve (non-decisional):** exact-zero chooser disagreement
  across every state of every pass appends a SUSPECT-NO-OP disclosure
  to the verdict (audit before publishing) — byte-level agreement
  everywhere is stronger than a genuine null needs, and the labeler's
  identical-heads refusal + swap read-back should make it unreachable.
- Secondaries (descriptive, never decisional): per-domain splits;
  within-pass chooser disagreement rates per side; opportunity level
  per eval side; max candidate drift.

## Registered prediction, interpretive limit, consequence map

Inherited VERBATIM from the parent: prediction = both primaries
straddle 0 (HEADS-IRRELEVANT, per the committed P-R3c null); the
feature-mismatch lower-bound caveat on any HEAD-DEFICIT fire; the full
consequence map (including the regardless-clauses: committed
R3/reacher/doubling reads stand, committed R3 labels never re-read,
24+24 frozen, imag retired, Route A closed). The power logic also
carries over unchanged — per-state pairing on shared states and a
shared `g_all` cancels trajectory and state-sampling variance; the
amendment strengthens it (one fewer noise source: no cross-pass
chooser asymmetry).

## Costs

64 full passes + 2 five-state smokes (the dualgate pass is one of the
64). A dual pass adds one d0_eval + one M-step op_real sweep + head
swaps on top of a parent pass — the 8×100-step oracle rollouts
dominate, so ≈ 1.1–1.2× a parent pass ⇒ **≈ 0.55–0.6× the parent
wave's compute (~70–150 GPU·h total)**; sequential ≈ 3–6 days, lane
parallel ≈ 1–2 days.

## Disclosure and ordering

Known at freeze: everything through 2026-08-02 — the full 1-Aug read
set (U1, doubling, second-k, 25m + E4 counterpart), the repair
Amendment 1 build (same within-pass pattern; its wave is NOT yet run
or read — no outcome flows either way), and the determinism probe
itself. The probe touched INTEGRITY quantities only
(episode/step identity, `g_all` closeness, `m_now` agreement; json
fields machine-checkable in the artifact); no achieved/`d_pair`
statistic has ever been computed from any xc file.
`--consumer_checkpoint` has executed exactly twice, as the seed-0
CONTROL-config probe on one cell — those files are excluded from this
wave three ways (labeler seed 0 ≠ registered seed 1; `_a/_b` naming
never matches the dual regex; `_xc1` version ≠ `_xc2` pin). No
swapped-heads pass has ever run. Unknown: every dual-chooser quantity;
whether the 32 run dirs still exist (the existence gate adjudicates).
Value-blindness in flight: dual passes REDACT the per-pass estimand
summary on stdout (`_xc2` in the frozen redaction set); smoke/dualgate
stdout is consultable for crash/dial sanity with no estimand exposure;
the amended reader is the single consumer, executed ONCE.
