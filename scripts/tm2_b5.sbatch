#!/bin/bash
#SBATCH --job-name=tm2b5
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# B5 TD-MPC2 column — PREREG_trackB_tm2_20260822 (rev per review R4).
# Stages via TM2B5_STAGE: apismoke | smoke | train | qmask
#   apismoke: random-init agent on cheetah+se+planted (rc-gated)
#   smoke:    5k-step training + a REDUCED-S tm2_qmask mechanics
#             pass (--n_eval 128; the 512 pin is the reader's gate)
#   train:    one seed (TM2B5_SEED in 132..139), 100k steps
#             (walltime 24h — cheetah 53-dim, no resume: a kill
#             discards the run, R4-m13)
#   qmask:    the instrument pass on ckpt_late.pt (skip-if-recorded;
#             the instrument itself refuses to overwrite, R4-M2)
# Env: the tdmpc2 conda env (TM2_CONDA_ENV) + TDMPC2_ROOT pinned at
# e9f59321 (asserted). PYTHONPATH carries the repo for embodied/.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
CONDA_ENV="${TM2_CONDA_ENV:?set TM2_CONDA_ENV}" activate_env
export TDMPC2_ROOT="${TDMPC2_CHECKOUT:?set TDMPC2_CHECKOUT}/tdmpc2"
cd "$TDMPC2_ROOT/.." && [ "$(git rev-parse HEAD)" = \
  "e9f59321933cbc8e11a002b842adc7d4ffae8ff1" ] || {
  echo "FATAL: tdmpc2 checkout not at the pinned commit"; exit 1; }
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
STAGE="${TM2B5_STAGE:?set TM2B5_STAGE (apismoke|smoke|train|qmask)}"
RUNROOT_B5="${RUNROOT_B5:-$RUNROOT/tm2_b5}"
mkdir -p "$RUNROOT_B5"
case "$STAGE" in
  apismoke)
    python - <<'PY'
from probing.tdmpc2_compat import Dv3TaskEnv, build_cfg
import os
env = Dv3TaskEnv('dmc_cheetah_run', seed=9, dose='se', planted=True)
assert env._extra_keys == ('distractor', 'planted_dup0',
                           'planted_dup1', 'planted_dup2',
                           'planted_const'), env._extra_keys
d = int(env.observation_space.shape[0])
assert d == 53, d
cfg = build_cfg(os.environ['TDMPC2_ROOT'], 'dmc_cheetah_run', d,
                int(env.action_space.shape[0]),
                env.max_episode_steps)
from tdmpc2 import TDMPC2
agent = TDMPC2(cfg)
import probing.tm2_qmask   # import-time deps (scipy etc.) checked here
o = env.reset()
for t in range(20):
    a = agent.act(o, t0=(t == 0), eval_mode=True)
    o, r, done, info = env.step(a)
print('B5 APISMOKE PASS: 53-dim layout, planner acts, env steps')
PY
    ;;
  smoke)
    python -m probing.tdmpc2_r3_train --logdir "$RUNROOT_B5/b5_smoke" \
      --task dmc_cheetah_run --dose se --planted --seed 9 \
      --steps 5000 --early_steps 2500 --eval_freq 5000
    python -m probing.tm2_qmask --run_logdir "$RUNROOT_B5/b5_smoke" \
      --ckpt ckpt_late.pt --n_eval 128
    echo "B5 SMOKE PASS"
    ;;
  train)
    SEED="${TM2B5_SEED:?set TM2B5_SEED (132..139)}"
    case "$SEED" in 13[2-9]) ;; *) echo "seed $SEED outside 132-139"; exit 1;; esac
    python -m probing.tdmpc2_r3_train \
      --logdir "$RUNROOT_B5/tm2se_s$SEED" \
      --task dmc_cheetah_run --dose se --planted --seed "$SEED" \
      --steps 100000
    ;;
  qmask)
    SEED="${TM2B5_SEED:?set TM2B5_SEED}"
    case "$SEED" in 13[2-9]) ;; *) echo "seed $SEED outside 132-139"; exit 1;; esac
    D="$RUNROOT_B5/tm2se_s$SEED"
    [ -f "$D/TM2R3_TRAIN_DONE" ] || { echo "no TM2R3_TRAIN_DONE in $D"; exit 1; }
    if [ -f "$D/tm2_qmask/tm2_qmask.json" ]; then
      echo "SKIP: tm2_qmask.json already recorded for s$SEED"; exit 0
    fi
    python -m probing.tm2_qmask --run_logdir "$D" --ckpt ckpt_late.pt
    ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date)] DONE $STAGE"
