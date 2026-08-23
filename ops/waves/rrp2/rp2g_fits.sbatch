#!/bin/bash
#SBATCH --job-name=rrp2_fits
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# rrp2 path-B rca'-source fits (PREREG_rrp2_20260823 step 0b; review B1:
# the path that previously had no vehicle). Runs ONLY when the surviving
# ax1wm_finger_rgoq1s1_seed{1..8} carrier fits are NOT 8/8 complete:
# 8 fresh rgo-from-random-init fits ax1wm_finger_rp2gq1s1_seed{1..8}.
#
# The rgo arm's flags are transcribed from scripts/axis1.sbatch:139
# (AXIS1_ARM=rgo -> --agent.repval_grad False) with the task expl mode;
# NO from_checkpoint is ever passed (random init IS the arm — the reader
# gates the saved config's run.from_checkpoint empty).
#
# The echoed "offline_fit WM_RUN=... REPLAY=..." line below is the
# realized-replay witness the bundle builder parses (its rrp2 scan
# covers the rp2g prefix); job name rrp2_fits -> log rrp2_fits_*.out
# matches the builder's rrp2*.out glob. Keep these logs with the wave.
#
# Env: CELLS = space-separated seeds (e.g. "1 2 3 4"), ~2h/fit.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${RUNROOT:?set RUNROOT}"
: "${CELLS:?set CELLS (seeds)}"
source "$REPO/scripts/env.sh"
cd "$REPO"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" PYTHONUNBUFFERED=1
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python || true)"

REPLAY="$RUNROOT/axis1_finger/q1/side1"
[ -f "$(dirname "$REPLAY")/manifest.json" ] || { echo "ERROR: no manifest.json beside $REPLAY"; exit 2; }

n_ok=0; n_fail=0
for k in $CELLS; do
  case "$k" in [1-8]) ;; *) echo "FAIL seed '$k' not in 1..8"; n_fail=$((n_fail+1)); continue ;; esac
  WM="ax1wm_finger_rp2gq1s1_seed${k}"
  WMDIR="$RUNROOT/$WM"
  done_ck="$(find "$WMDIR/ckpt" -mindepth 2 -maxdepth 2 -name done -printf '%h\n' 2>/dev/null | sort | tail -1)"
  if [ -n "$done_ck" ] && [ "$(basename "$done_ck" | sed 's/.*-//')" = "000000500000" ]; then
    echo "skip $WM (done at 500000)"; n_ok=$((n_ok+1)); continue
  fi
  echo "[$(date)] offline_fit WM_RUN=$WM REPLAY=$REPLAY UPDATES=500000 SEED=$k"
  ( unset FORCE_WM AXIS1_WR AXIS1_SIZE AXIS1_BASE_CONFIG \
        AXIS1_INIT_WM AXIS1_INIT_REGEX AXIS1_INIT_FROZEN_ENC 2>/dev/null || true
    "$PYTHON" -u -m probing.offline_fit \
      --static_replay "$REPLAY" --updates 500000 \
      --save_every_updates 50000 \
      --logdir "$WMDIR" --configs dmc_proprio \
      --task dmc_finger_turn_hard --seed "$k" \
      --agent.expl.mode task --agent.repval_grad False \
      --env.dmc.render False )
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK     fit $WM"; n_ok=$((n_ok+1))
  else
    echo "FAIL   fit $WM rc=$rc"; n_fail=$((n_fail+1))
  fi
done
echo "[$(date)] rp2g fits: ok=$n_ok failed=$n_fail"
[ "$n_fail" -eq 0 ] || exit 1
exit 0
