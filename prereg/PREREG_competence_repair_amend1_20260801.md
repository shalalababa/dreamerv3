# PREREG Amendment 1: Competence-repair — within-pass shadow pairing (2026-08-01)

Amends `PREREG_competence_repair_20260730.md` (freeze `6271c770`)
AFTER the parent's registered **QUARANTINE** branch fired at the smoke
stage (detgate: `episode/step ok=True, m_now ok=False, g_all ok=False`)
and AFTER the quarantine-mandated instrument audit completed
(`artifacts/r3_repair_quarantine_20260801/AUDIT.md`). This file + the
amended `analysis/repair_read.py` (selfcheck PASS) are committed BEFORE
the detgate re-run, before any of the remaining 31 passes, and before
any repair estimand exists anywhere. The parent's ONE-read rule is
unchanged: no read has executed and exactly one will.

## Why the parent design is unexecutable (audit finding, registered)

The parent paired each repaired pass against the COMMITTED R3 late
label of the same run, gated on bitwise reproduction (`m_now` exact,
`g_all` atol 1e-5). The audit showed that assumption is false on the
substrate: **two same-day jobs of identical code disagree with each
other as much as either disagrees with the committed pass** (deter max
≈ 1.4; `g_all` per-state rank ρ ≈ 0.02–0.06; `m_now` flips ≈ 50/200;
same-code pair drift = cross-era drift on every gate quantity). The
labeling stack is not job-to-job deterministic on Midway3; the chooser
and the post-freeze code changes were exonerated (the no-consumer
audit pass drifts identically). `episode/step` identity HELD in all
comparisons — state selection is job-invariant. Consequence: **the
parent primary (paired vs the committed labels) is RETIRED as
unmeasurable-on-substrate — an instrument fact, not an outcome.** No
environment reconstruction can pass the parent gate as registered
(even same-day jobs do not reproduce each other), so unlike second-k
there is no env-matched-retry path.

## Amended estimand (per labeled state, WITHIN each repaired pass)

Every repaired pass already stores BOTH choosers evaluated on the same
states, the same trajectories, and the same `g_all` (parent
§Intervention, unchanged labeler): `m_real` (consumer argmax of
`ghat`) and `m_real_probe` + `real_scores` (the unchanged `op_real`
probe, whose pick is `argmax(real_scores)` in the frozen labeler).

- achieved_rep   = `g_all[arange, m_real] − g_now`
- achieved_probe = `g_all[arange, m_real_probe] − g_now`
- **primary target: per-state paired [achieved_rep − achieved_probe]
  = `g_all[arange, m_real] − g_all[arange, m_real_probe]`** (the
  `g_now` terms cancel algebraically) → per-run mean → run-clustered
  percentile bootstrap (32 clusters, B=10K, rng 0 — parent machinery,
  unchanged).

This is the SAME functional the parent wanted — consumer choice vs
probe choice on identical states under identical `g_all` — with the
control chooser evaluated in-pass instead of read from the committed
file. It is immune to cross-job nondeterminism by construction: both
sides of every per-state pair come from the same process, same
rollouts, same RNG stream. **Honest baseline change, disclosed:** the
control is the current-era probe choice, not the committed realized
choice itself (the committed `m_real` came from the same `op_real`
operator, but in a different job whose per-state picks drifted ≈ 50% —
audit: `m_real` flips committed↔audit_nocm 101/200; the ≈ 25% figure
in the audit table is the plugin `m_now` flip rate, a different
quantity); the bridge from the amended primary to the committed pooled
gap point is therefore approximate, and the committed achieved level
is reported descriptively, never as a paired quantity.

## Amended gates (replacing the parent determinism gate)

1. **State-identity gate (cross-file, value-blind, retained):** per
   cell, `episode`/`step` must match the committed npz EXACTLY — this
   is the component that held in the audit and it licenses "the
   registered committed states". ANY violation ⇒ **QUARANTINE** (same
   branch, same consequence).
2. **Within-pass identity gates (new/retained, machine-checked per
   repaired file):** `g_now == g_all[arange, m_now]` (atol 1e-5,
   oracle_all identity, parent guard retained); `m_real` is a
   maximizer of the stored `ghat` (tie-tolerant, parent guard
   retained); **NEW:** `m_real_probe` is a maximizer of the stored
   `real_scores` (tie-tolerant, same construction) — the pass provably
   contains a coherent shadow pair.
3. **REMOVED (registered-unpassable per the audit):** the cross-pass
   `m_now` and `g_all` comparisons and the `PAIR_ATOL` tolerance
   machinery. No cross-job float comparison GATES anything in this
   read; the one cross-era float quantity that remains — the
   descriptive `opportunity_max_cell_absdiff` secondary — is
   registered as drift context and gates nothing.
3b. **Within-pass identity failure = non-read (registered):** the
   gates in item 2 are load-time assertions; a violation aborts the
   read before any estimand is computed or printed. Per the
   R3-Amendment-1 precedent, such an abort is a NON-read: a
   value-blind instrument repair + a single re-run preserves the
   ONE-read rule (the read that counts is the one that adjudicates).
4. **Detgate re-specification (smoke stage):** `--detgate` now checks,
   on the ONE registered smoke cell (cup/e1/seed31/late, the EXISTING
   pass file — it is not re-labeled): meta version pins both sides,
   `episode/step` identity vs the committed label, `g_all` finiteness,
   and the within-pass identity gates above. Value-blind: no estimand
   functional is computed; the only value arrays touched are used
   solely inside identity/argmax assertions. The frozen driver
   `scripts/r3_rep_local.sh` is UNCHANGED byte-wise: its `smoke` stage
   skips the existing pass file and re-runs `--detgate`, which mints
   `REP_DETGATE_OK` under the amended semantics; `labels` then runs
   the remaining 31 passes exactly as registered. The driver's
   COMMENTS still describe the parent design (committed labels as
   paired control; episode/step/g_all detgate) — stale BY DESIGN as a
   freeze decision; this amendment, not those comments, governs.

Everything else is inherited unchanged from the parent: the full
weights-provenance protocol (four-way sha pinning incl. the frozen
`COMMITTED_LABELS_DIGEST`), all dial/version/checkpoint/LORO guards,
the existence-gate and SUBSTRATE-GONE mechanics, all-or-nothing 32-cell
grid both sides, the stray-file trip, the MATERIALITY source-tie leg,
and the labeler itself (no labeler change of any kind).

## Amended decision rules (frozen in the amended `analysis/repair_read.py`)

- **P-REP1 (repair helps at all):** pooled per-run mean of per-state
  [achieved_rep − achieved_probe]; CI entirely > 0.
- **P-REP2 (materiality; adjudicated only when P-REP1 fires):** point
  ≥ **0.6465**, the parent's registered constant (50% of the committed
  R3 pooled gap point; truncation-direction note in the parent).
  **Bridge disclosure:** the bar was derived against the committed
  realized-choice baseline; under the amended in-pass probe baseline
  the "half the committed gap" reading is approximate (the probe is
  the same operator that produced the committed choices, so the
  baseline achieved level is expected to be distribution-equal; this
  is an interpretation note, not a threshold change).
- **Verdict map unchanged:** REPAIRED / PARTIAL /
  NOT-REPAIRABLE-FROM-OBSERVABLES / HARMFUL / QUARANTINE (now =
  state-identity violation) / SUBSTRATE-GONE.
- **Sensitivity leg (new, disclosure-only):** the primary recomputed
  excluding the smoke cell (cup/e1/31 — its label file predates this
  amendment). Adjudication is on ALL 32 by registration. If any fire
  status (P-REP1 / materiality / harmful) differs between the 32-cell
  and 31-cell computations, the read appends a registered **FRAGILITY
  DISCLOSURE** to the verdict text; it never changes the verdict.
- **Secondaries (descriptive, never decisional, amended):** per-domain
  / per-dose splits of the paired diff; pooled achieved_rep,
  achieved_probe (in-pass), and achieved_orig (committed,
  cross-era descriptive context); oracle-agreement rates for
  `m_real`, `m_real_probe` (in-pass), and the committed realized
  choice; in-pass chooser change rate (`m_real != m_real_probe`);
  residual gap fraction with the IN-PASS opportunity
  (`max(g_all) − g_now` from the repaired pass) as denominator; a
  `cross_era_drift` block (probe-vs-committed-realized agreement rate
  and per-state opportunity max-absdiff — under the audit these are
  expected to show ≈ 50% probe-choice drift (`m_real` flips 101/200
  committed↔audit_nocm; the plugin `m_now` flip rate was ≈ 25%) and
  O(1) float drift; they are drift context, explicitly NOT sanity
  expectations and NOT paired estimands).

## Power

Parent logic carries over strengthened: the pairing is now within a
single process on shared states AND shared rollout draws, so residual
variance is pure chooser-difference variance; the detectable per-run
effect remains bounded by the parent's ≈ 0.42 and realistically far
below. Registered as logic, not as a fake precision number.

## Costs

Zero additional compute relative to the parent: the trainer ran
(model sha `be802f1e…` recorded before any labeling; reusable as-is
under the unchanged provenance protocol), the smoke pass exists, the
31 remaining passes were already budgeted, and the detgate re-run is
CPU-seconds. Ordering: **this amendment + the amended reader are
committed → `smoke` stage re-run on RCC (detgate re-run mints the
marker; no re-labeling) → `labels` stage (31 passes) → bundle sync →
ONE read.** **Model-restore rule (registered):** if
`$R3_ROOT/rep_model` is missing on scratch at resume, RESTORE
`consumer_model.npz` (sha `be802f1e…`) from the synced quarantine
bundle — do NOT re-run the trainer: a byte-different retrained model
would mismatch the existing smoke pass's `consumer_model_sha256` stamp
and crash the read with no registered outcome.

## Consequence map

Inherited verbatim from the parent (all six verdicts and the
regardless-clause: the committed R3 read and every prior registration
stand; committed labels never modified; 24+24 frozen; imag retired;
Route A closed). QUARANTINE's trigger is now the state-identity gate.

## Disclosure (known at amendment time)

Known: the full parent record; the quarantine audit
(`artifacts/r3_repair_quarantine_20260801/AUDIT.md`) — the three
audit scripts are ARCHIVED VERBATIM in
`artifacts/r3_repair_quarantine_20260801/audit_scripts/` (01 from the
session scratchpad; 02/03 recovered verbatim from the session
transcript's tool-call log) so what they loaded is machine-checkable,
not testimony: **no script ever loads `m_real`, `m_real_probe`,
`real_scores`, or `ghat` from the rep smoke file** (01 never opens
the rep file — its `m_real` flip count is committed↔audit_nocm only;
02/03 load base-path arrays + `m_now` only), and no achieved value,
no per-state paired difference, and no functional of [`m_real` vs
`m_real_probe`] has been computed anywhere for any pass — the amended
primary is fully unknown, including on the smoke cell. The smoke
file's bytes as of the audit are git-pinned by the committed bundle
manifest (commit dd3138e7, sha256 `0d549b6b…`), matching the on-disk
file at this amendment's commit. `rep_labels/` on scratch and in the
bundle also contains the audit's no-consumer pass
`audit_nocm_cup_e1_seed31_late.npz` (which carries an unredacted
probe-arm summary of an already-adjudicated committed cell): the
frozen reader neither loads nor trips on it by construction —
`FILE_REP` requires the `rep_` prefix and the stray-file trip globs
`rep_*.npz`, so the audit file is skipped before `np.load`. The no-consumer audit pass printed the labeler's standard
summary line (`mean delta_real +0.100, mean delta_imag +0.065`) — a
drifted re-measurement of an already-adjudicated committed cell from
the PROBE arm of a pass with no consumer model; the smoke pass's
consumer-arm summary was REDACTED by the frozen labeler. The consumer
model's lambda choices were printed at training (parent-registered
stdout). Unknown: every quantity of the amended estimand, on every
cell. The smoke cell's label file predates this amendment — that is
exactly why the sensitivity leg above is registered. One registered
resolution: the parent required the smoke pass to be "a wave pass";
this amendment keeps it in the primary (all 32) WITH the sensitivity
leg and fragility disclosure, rather than discarding a
provenance-clean pass whose estimand no one has seen.
