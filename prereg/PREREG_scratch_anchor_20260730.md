# PREREG — from-scratch baseline anchor (descriptive; frozen 2026-07-30)

8 jobs. Purpose: an ABSOLUTE-SCALE ANCHOR for Paper 1's figures (the
30-Jul adversarial review's one new-compute item): per-arm learning
curves and headline AUC effects need a from-scratch (no-pretraining)
DreamerV3 reference so effect sizes are interpretable outside the
project. GO'd by the owner 30 Jul. Committed BEFORE any scratch run
exists.

## Class: DESCRIPTIVE — no decision rules

- These runs anchor figures and effect-size normalization ONLY. No
  hypothesis, no fire/no-fire rule, no registered contrast. Any
  claim-level comparison (e.g. "pretrained arm X beats / does not
  beat from-scratch") would require its own future registration —
  this file explicitly does NOT license such claims.
- Interpretive note registered up front: these are standard online
  DreamerV3 runs with reward present from step 0 (config-default
  `agent.expl.mode: task`, nothing frozen, no checkpoint) — the
  "train from scratch at the adaptation budget" reference, NOT a
  reward-free arm.

## Design: 8 runs, zero code change

- Grid: dmc_finger_turn_hard × seeds 1–8, `size1m` via `dmc_proprio`,
  `run.steps 1.25e5`, 16 envs — byte-compatible scoring cadence with
  the adapt protocol (48/96/112 episodes at 50k/100k/125k), so
  `analysis/adaptation_auc.py` collates rows into the canonical csv
  with mode `scratch`, milestone `0`, standard qc (≥20 episodes by
  100k).
- Driver: `scripts/pilot.sbatch` with `MODE=goal` (yields plain
  `--configs dmc_proprio`, the same `main.py` train path the adapt
  stage uses; no checkpoint loaded — `run.from_checkpoint` stays at
  its `''` default; nothing frozen). The saved per-run `config.yaml`
  (`from_checkpoint: ''`, `frozen_wm: false`) is the audit trail.
- RUN_ID: `adapt_scratch_finger_seed<k>_ckpt0` (parses under the
  collector's frozen RUN_RE: mode `scratch`, milestone `0`; placed in
  the same `$RUNROOT` so the standard collation finds it).
- After rc=0: `touch $RUNROOT/adapt_scratch_finger_seed<k>_ckpt0/ADAPT_DONE`
  (status-tool visibility only; the collector does not require it).
  pilot.sbatch never flips its `runs.csv` row to DONE — either accept
  the stale-RUNNING rows (harmless, phase=`pilot` labels distinguish
  them) or fix with
  `sed -i "s#^adapt_scratch_finger_seed<k>_ckpt0,\(.*\),RUNNING,#adapt_scratch_finger_seed<k>_ckpt0,\1,DONE,#" $MANIFEST`.

Submit (clean shell; the --export=ALL module-leakage gotcha applies.
Source `scripts/env.sh` first — it sets variables only, loads no
modules, so it is compatible with the clean-shell rule. Staging:
submit seed 1 first; confirm `scores.jsonl` appears, a `mode=scratch`
row collates, and the elapsed time fits the 4 h envelope; then the
remaining 7):

```
source $REPO/scripts/env.sh
for k in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=04:00:00 \
    --job-name=adapt_scratch_finger_seed${k}_ckpt0 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,RUN_ID=adapt_scratch_finger_seed${k}_ckpt0,TASK=dmc_finger_turn_hard,MODE=goal,SEED=${k},STEPS=1.25e5,SAVE_EVERY=900 \
    $REPO/scripts/pilot.sbatch
done
```

## Consumption

- Fig 2a (distillation doc figure plan): per-arm return-vs-steps
  learning curves from `scores.jsonl` + the scratch anchor curve,
  AUC100k window shaded; headline effects additionally reported as %
  of anchors — raw AUC stays PRIMARY in every caption (the n=8 anchor
  mean is a presentation denominator, never load-bearing for a claim). Descriptive summaries (means/sds) enter
  `Paper1_ResponseCurves` when rows land — no frozen reader (nothing
  decisional to protect); the canonical-csv rows + this file are the
  record.
- Frozen-reader isolation: no registered reader consumes mode
  `scratch` — verified by naming (every frozen reader pins explicit
  mode strings; `scratch` appears in none).

## Disclosure

Known at freeze: all reads through 30 Jul. No scratch run, log, or
score exists. The 25m and U1 reads are pending and unaffected (mode
`scratch` collides with no registered mode; the collator regenerates
the csv deterministically from the run dirs on each invocation — same
shared-csv discipline as the U-wave note, and all registered readers
consume frozen artifact csv baselines, never the live csv's history). Compute: 8 × ≤4 h single-GPU.
