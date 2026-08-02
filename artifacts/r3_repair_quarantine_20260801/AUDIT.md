# Competence-repair — registered QUARANTINE + instrument audit (2026-08-01)

Registered branch of `PREREG_competence_repair_20260730.md` (freeze
commit `6271c770`): the smoke-stage **determinism gate failed**
(`episode/step ok=True, m_now ok=False, g_all ok=False`) ⇒ the frozen
driver stopped the wave; **the remaining 31 passes were never run and
no read executed**. Per the frozen consequence map, QUARANTINE is an
**instrument outcome**: "instrument audit before ANY use of repaired
labels; the wave is not interpretable evidence in any direction." The
committed R3 read, the reacher/doubling/ladder outcomes, and every
prior registration stand untouched.

Bundle: `local_results/r3_repair_quarantine_20260801_210414/`
(sha-verified at sync per the results-sync policy). Stage record:
existence gate PASSED (32/32 runs + 64/64 committed labels); trainer
PASSED (deterministic CPU, 32 LORO models, F=522, sha `be802f1e…`
recorded to `CONSUMER_MODEL_SHA256` before any labeling — the full
weights-provenance protocol held); smoke repaired pass wrote its file
and then tripped the detgate; a **no-consumer audit pass**
(`--consumer_model` absent, all committed dials) was run per the
quarantine clause to isolate the chooser.

## Audit findings (value-blind: gate quantities only — episode/step,
## m_now/m_real flips, float drift on cands/qfull/g_all/udyn/deter;
## no achieved/delta functional of the repair estimand was computed)

Triangle comparison on cup/e1/seed31/late (200 states):

| pair | deter max | udyn max | cands max | qfull max | g_all max | g_all per-state rank ρ (mean) | m_now flips |
|---|---|---|---|---|---|---|---|
| committed ↔ audit_nocm | 1.391 | 5.7e-4 | 2.801 | 41.7 | 39 | 0.049 | 57/200 |
| committed ↔ rep (smoke) | 1.238 | 6.3e-4 | 2.781 | 43.8 | 46 | 0.055 | 53/200 |
| **audit_nocm ↔ rep (same code, same day)** | 1.355 | 6.7e-4 | 2.891 | 29.0 | 31 | 0.019 | 48/200 |

m_real flips committed↔audit_nocm: 101/200. episode/step identical in
all three pairs (metas identical incl. checkpoint path and all dials;
both current passes stamp the registered versions).

## Root cause: the labeling stack is not job-to-job deterministic on
## this cluster — NOT era drift, NOT the chooser, NOT the code changes

The decisive row is the third: **two same-day jobs of the identical
current code disagree with each other exactly as much as either
disagrees with the committed pass** (SLURM 52874313 vs 52886747, both
midway3). One mechanism explains all three legs:

- Single-forward, RNG-free quantities barely move (udyn ≤ 7e-4 —
  bf16-scale numeric drift). The model and pipeline are faithful.
- Quantities that compound through sampled recurrences diverge:
  burn-in categorical draws flip on numeric near-ties across
  jobs/nodes, so `deter` walks apart (max ≈ 1.4 on a 512-d latent),
  RNG/rollout streams desynchronize (occasional whole-draw candidate
  changes, max 2.8 on a [−1,1] action axis), and the oracle-rollout
  returns `g_all` end up with ≈ zero per-state rank correlation.
- Exonerated: the chooser (the no-consumer pass drifts identically;
  base-path inertness holds); the post-wave code changes
  (frozen_enc backfill + cm extension — a same-code pair drifts the
  same amount, so code delta is not required and not sufficient).
- Likely proximate source: node/GPU heterogeneity across SLURM
  allocations and/or nondeterministic GPU kernels. Job logs do not
  record hostname/GPU (same instrumentation gap as the second-k
  audit); adding that to the dumper/labeler remains open.

**Design-level conclusion:** the registered detgate assumed a re-pass
reproduces the committed pass bitwise (g_all atol 1e-5). That
assumption is FALSE on this substrate and was never previously
testable — every committed cell was labeled exactly once; this gate is
the first cross-job bitwise reproduction ever attempted on the
cluster. The gate did its job: it refused before any estimand was
touched.

## Paths forward

1. **RECOMMENDED — Amendment: within-pass shadow pairing.** The rep
   passes already store BOTH choosers evaluated on the same states and
   the same `g_all`: `m_real` (consumer argmax) and `m_real_probe` +
   `real_scores` (the unchanged op_real probe). Re-register the
   primary as the per-state paired
   `g_all[arange, m_real] − g_all[arange, m_real_probe]` **within each
   rep pass** — algebraically the same functional the original design
   wanted (the `g_now` terms cancel), with the control chooser
   evaluated in-pass instead of read from the committed file. Immune
   to cross-job nondeterminism by construction; zero extra compute;
   the trained consumer model is provenance-intact and reusable.
   Requires a dated amendment (estimand pairing + gate replacement:
   within-pass identities `g_now == g_all[m_now]`,
   `m_real == argmax ghat`, `m_real_probe == argmax real_scores`) +
   reader/selfcheck changes. Honest costs to disclose: the paired
   control is no longer the committed artifact itself (baseline =
   current-era probe choice, so the bridge to the committed gap point
   0.6465 bar is approximate); the smoke rep file exists at amendment
   time (estimand never computed anywhere — the frozen labeler
   redacted the consumer-arm stdout and this audit touched gate
   quantities only) ⇒ register the primary on all 32 with that
   disclosure + a sensitivity leg excluding the smoke cell.
2. **Terminal quarantine** — record and drop the interventional arm;
   the repair question (NOT-REPAIRABLE-FROM-OBSERVABLES vs PARTIAL vs
   REPAIRED) stays unanswered. Wasteful relative to path 1 (one
   reader amendment + the already-budgeted 31 passes).
3. **NOT viable: environment-matched retry** (the second-k remedy).
   The committed side's node/env is unrecorded and the audit shows
   even same-day jobs don't reproduce each other — no environment
   reconstruction can pass the frozen gate as registered.

## Cross-registration exposure (checked this audit)

- **xc cross-checkpoint wave (`PREREG_r3_amend2_20260730`, not yet
  submitted): EXPOSED.** Its pairing gate requires a control pass and
  a swapped pass — separate invocations — to label identical states
  with identical `g_all` (atol 1e-5). Measured cross-job drift is
  ~10⁶ × that tolerance. Before submitting the 128-pass wave, run a
  cheap reproducibility probe: label ONE cell twice within a single
  SLURM job on one node and compare gate arrays. If same-node/
  same-job determinism holds, pin each run's 4 passes to one
  job/node (execution detail, no prereg change: the registration
  does not constrain placement). If it fails, xc needs its own
  within-pass rethink before any submission.
- **TM2-R3 wave: NOT exposed.** Its determinism assert is
  within-pass (env snapshot/restore reproduction inside one
  process), which this failure mode does not touch.
- **Second-k quarantine (31 Jul) reinterpreted slightly:** the
  cross-machine story stands, but this audit shows job-to-job
  nondeterminism exists even within one cluster — the Vast-matched
  retry recommended there should include the same one-cell twice-run
  probe before committing to the 102 passes.

## Disclosure

The no-consumer audit pass printed the labeler's standard per-pass
summary line (`mean delta_real +0.100, mean delta_imag +0.065`) — a
drifted re-measurement of an already-adjudicated committed cell,
pre-existing disclosed instrument behavior. The rep smoke pass's
consumer-arm summary was REDACTED by the frozen labeler ("estimand
summary redacted: consumer arm") — no repair estimand has been
printed, computed, or read anywhere as of this audit. Guard-threshold
or decision-rule changes require a dated amendment; none is made
here — path 1 is a proposal, not a registration.
