# PREREG: adaptation-horizon wave (late-crossing test, 4× budget) — 2026-08-11 (post-review REVISED form)

User GO 2026-08-11. Purpose: the one protocol attack whose answer can
change headline WORDING — "the reward-free payoff arrives later." Tests
whether the reward-free-≈-untrained facts are horizon-bounded at 4× the
registered adaptation budget (1.25e5 → 5e5 steps) under the UNFROZEN
protocol (`unfrozen_readout`, U-calib/U1 lineage). **This file replaces
the pre-review draft in full** (batch review findings 6–11, 21, 23–24,
28 applied).

## Honesty block

Known at freeze: all executed reads, incl. the 1.25e5-budget unfrozen
facts (apt-unfrozen 0.54×/0.61× of scratch AUC, U1) and every frozen
verdict. Unknown: every trajectory beyond 1.25e5 steps. **Substrate
disclosure (review #28)**: per the 10-Aug weight census
(`PREREG_refit_reads_amend2_20260810`) the fresh apt refits are fq1s0
seeds 5/7/8 + fq1s1 seeds 6/7/8 — so ALL EIGHT hzf donors below
(seeds 1–4, both sides) are restored ORIGINAL July weights;
disclosed, favourable (no refit-substrate mixing inside the wave).

## Design — 24 fresh 5e5-step runs (3 arms × 8)

hzf = apt-unfrozen (WM `ax1wm_finger_fq1s{side}_seed{f}`), hzt =
task-unfrozen positive control (WM `ax1wm_finger_q1s{side}_seed{f}`),
sides {0,1} × seeds {1..4}; hscratch = fresh MODE=goal runs, seeds
{1..8}. NEW run ids; never resume or extend an existing run.

**Pre-submit gates ([YOU], registered — review #8/#10):**
1. Donor dual witness on the 16 WMs (NOT bare ls):

```
python - <<'PY'
import json, glob, os
root = os.environ['RUNROOT']
wms = sorted(glob.glob(root + '/ax1wm_finger_fq1s[01]_seed[1-4]')) \
    + sorted(glob.glob(root + '/ax1wm_finger_q1s[01]_seed[1-4]'))
assert len(wms) == 16, wms
for w in wms:
    p = json.load(open(w + '/OFFLINE_FIT_PROGRESS'))
    assert p['update'] == p['total'], (w, p)
    ck = sorted(glob.glob(w + '/ckpt/*'))
    assert any('500000' in os.path.basename(c) for c in ck), (w, ck[-3:])
    print('OK', w)
PY
```

   (the literal python block is the gate; refuse on any assert).
2. Record the realized elapsed walltimes of one U1 adapt and one
   scratch-anchor run (sacct or log timestamps); the `--time` values
   below must be ≥1.5× the linear 4× extrapolation — if not, RAISE
   them (or the partition cap, whichever is lower) before submitting
   (review #10: 10/32 + 17/32 walltime-kill history).

**Registered submissions (literal; review #21).** hz arms — for side
in 0 1; for f in 1 2 3 4:

```
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
  --gres=$SLURM_GRES --time=30:00:00 \
  --job-name=adapt_ax1hztq1s${side}_finger_seed${f}_ckpt500000 \
  --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1hztq1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_q1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=5e5,AXIS1_EXPL_MODE=task,AXIS1_ADAPT_CONFIG=unfrozen_readout \
  $REPO/scripts/axis1.sbatch
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
  --gres=$SLURM_GRES --time=30:00:00 \
  --job-name=adapt_ax1hzfq1s${side}_finger_seed${f}_ckpt500000 \
  --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1hzfq1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_fq1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=5e5,AXIS1_EXPL_MODE=apt,AXIS1_ADAPT_CONFIG=unfrozen_readout \
  $REPO/scripts/axis1.sbatch
```

hscratch — for k in 1 2 3 4 5 6 7 8:

```
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
  --gres=$SLURM_GRES --time=20:00:00 \
  --job-name=adapt_hscratch_finger_seed${k}_ckpt0 \
  --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,RUN_ID=adapt_hscratch_finger_seed${k}_ckpt0,TASK=dmc_finger_turn_hard,MODE=goal,SEED=${k},STEPS=5e5,SAVE_EVERY=900 \
  $REPO/scripts/pilot.sbatch
```

(Submit from a clean shell — the `--export=ALL` module-leakage
gotcha.) **Truncation-recovery policy (registered; review #9/#10)**: a
walltime-truncated run's dir is DELETED (after archiving its logs
off-runroot), and the same id is resubmitted fresh with raised
`--time`; the truncation is disclosed in the bundle notes. No resumes.
Cleanup: add `adapt_ax1hz* adapt_hscratch_*` to KEEP_PENDING in the
freeze-commit. Bundle: `runroot_light/<run_id>/{scores.jsonl,config.yaml}`
× 24 (+ logs) — **config.yaml is REQUIRED per run** (provenance gate
input).

## Registered decision rules

Reader gates (fail-closed; review #8/#9): per-run `config.yaml`
provenance — hz runs must show `run.from_checkpoint` naming the
registered WM for that (arm, side, seed), `from_checkpoint_regex
'^(enc|dyn|dec)/'`, `agent.frozen_wm: False`, `agent.expl.mode` =
task (hzt) / apt (hzf); scratch runs must NOT be WM-initialized. Any
unregistered `adapt_*` dir in the bundle ⇒ REFUSE. Per-run steps must
be strictly increasing (appended/resumed file ⇒ REFUSE). The
registered id set must be complete (a missing run ⇒ REFUSE); short
runs (max step < 4.9e5) are EXCLUDED-disclosed; an arm below n=6 ⇒
**INSTRUMENT-LIMITED**.

Late window = steps ∈ [4.0e5, 5.0e5]; per-run statistic = mean episode
score in the window. **Registered margin δ = 0.15 × mean(scratch
late-window)** (relative, computed in-read; review #6).

- **P-H1 (primary; house BCa 95% B=10K rng 0 + two-sample label
  permutation, cluster = run)** on hzf − hscratch, graded in this
  order:
  1. CI > 0 ∧ p < .05 ⇒ **CROSSES-LATE** — the reward-free null is
     horizon-bounded; headline wording MUST carry the budget
     qualifier (pre-registered consequence).
  2. else if CI upper < δ: **CI < 0 ∧ p < .05 ⇒
     APT-INIT-BELOW-SCRATCH-AT-4X** (strict inferiority — reported as
     its own branch, NOT as "≈ untrained"; also bounded); otherwise ⇒
     **BOUNDED-NULL-4X** — no meaningful late advantage; the
     late-crossing attack is answered with data.
  3. else **INDETERMINATE** (CI upper ≥ δ; genuinely uninformative
     width; licenses nothing).
- **P-H2 (descriptive, gated runs only — review #11)**: per-arm
  half-open 5e4-step bin curves; first sustained-positive bin of
  (hzf − hscratch) if any; early-window [0.75e5, 1.25e5] mean-score
  ratio — a DIRECTIONAL, non-matched comparison vs the U-wave
  AUC-based 0.5–0.6× anchors (review #24: the uzf runs that produced
  those are deleted; like-for-like recomputation impossible).
- **P-H3 (positive control)**: hzt − hscratch late-window; expected
  CI > 0 ∧ p < .05; otherwise the whole read is flagged
  **INSTRUMENT-SUSPECT** (branch still reported, carries the flag).
- MDE disclosure: n=8v8 late-window means; between-run sd basis
  recorded in-read; INDETERMINATE expected to be common at this n.

Reader: `analysis/horizon_read.py` (REVISED; selfcheck PASS incl.
true-crossing-at-3.5e5 bin localization, early-only-advantage
rejection, and wrong-WM / unregistered-dir / appended-file refusals).
ONE read execution. Reviewer: 2026-08-11 batch review (ONE, Opus 5) —
findings 6–11/21/23–24/28 applied in this revised form.
