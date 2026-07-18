#!/bin/bash
#SBATCH --job-name=dv3_tdmpc2
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# TD-MPC2 cross-family arm (PREREG_tdmpc2_crossfamily_20260716.md):
# bridge (once per side) -> offline WM fit (arm aware|free) -> frozen-rep
# online adapt. Resumable at every stage (marker files).
#
# Required env: TM2_ARM (aware|free), TM2_SIDE (0|1), TM2_SEED
# Optional: TM2_DOMAIN (finger), TM2_QUAD (q1), TM2_UPDATES (500000),
#           TM2_STEPS (125000), TM2_CONDA_ENV, TDMPC2_CHECKOUT
# Run ids: tm2wm_<dom>_<arm><quad>s<side>_seed<k> and
#          adapt_tm2<arm><quad>s<side>_<dom>_seed<k>_ckpt<updates>
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"

TM2_CONDA_ENV="${TM2_CONDA_ENV:-/scratch/midway3/$USER/conda_envs/tdmpc2}"
TDMPC2_CHECKOUT="${TDMPC2_CHECKOUT:-/scratch/midway3/$USER/tdmpc2}"
export TDMPC2_ROOT="$TDMPC2_CHECKOUT/tdmpc2"
CONDA_ENV="$TM2_CONDA_ENV" activate_env
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

: "${TM2_ARM:?aware|free}"; : "${TM2_SIDE:?0|1}"; : "${TM2_SEED:?}"
TM2_DOMAIN="${TM2_DOMAIN:-finger}"
TM2_QUAD="${TM2_QUAD:-q1}"
TM2_UPDATES="${TM2_UPDATES:-500000}"
TM2_STEPS="${TM2_STEPS:-125000}"
case "$TM2_DOMAIN" in
  finger) TASK=dmc_finger_turn_hard ;;
  cup)    TASK=dmc_cup_catch ;;
  *) echo "ERROR: unknown TM2_DOMAIN=$TM2_DOMAIN"; exit 2 ;;
esac
case "$TM2_ARM" in aware|free) ;; *) echo "ERROR: TM2_ARM=$TM2_ARM"; exit 2 ;; esac

SIDE_DIR="$RUNROOT/axis1_${TM2_DOMAIN}/${TM2_QUAD}/side${TM2_SIDE}"
DATA="$RUNROOT/tm2_data/${TM2_DOMAIN}_${TM2_QUAD}_side${TM2_SIDE}.pt"
WM_RUN="$RUNROOT/tm2wm_${TM2_DOMAIN}_${TM2_ARM}${TM2_QUAD}s${TM2_SIDE}_seed${TM2_SEED}"
ADAPT_RUN="$RUNROOT/adapt_tm2${TM2_ARM}${TM2_QUAD}s${TM2_SIDE}_${TM2_DOMAIN}_seed${TM2_SEED}_ckpt${TM2_UPDATES}"

# --- stage 1: bridge (once per side; guarded against concurrent writers) ---
if [ ! -f "$DATA" ]; then
  mkdir -p "$(dirname "$DATA")"
  LOCK="$DATA.lock"
  if mkdir "$LOCK" 2>/dev/null; then
    bridge_lock_owner=1
    cleanup_bridge_lock() {
      if [ "${bridge_lock_owner:-0}" = "1" ]; then
        rmdir "$LOCK" 2>/dev/null || true
      fi
    }
    trap cleanup_bridge_lock EXIT
    echo "[$(date)] bridge $SIDE_DIR -> $DATA"
    python -m probing.tdmpc2_bridge convert \
      --side_dir "$SIDE_DIR" --task "$TASK" --output "$DATA"
    rmdir "$LOCK"
    bridge_lock_owner=0
    trap - EXIT
  else
    echo "[$(date)] waiting for concurrent bridge of $DATA"
    while [ -d "$LOCK" ]; do sleep 30; done
    [ -f "$DATA" ] || { echo "bridge failed elsewhere"; exit 1; }
  fi
fi

# --- stage 2: offline WM fit ---
if [ ! -f "$WM_RUN/TM2_FIT_DONE" ]; then
  echo "[$(date)] offline_fit WM_RUN=$(basename "$WM_RUN") ARM=$TM2_ARM DATA=$DATA UPDATES=$TM2_UPDATES SEED=$TM2_SEED"
  python -m probing.tdmpc2_offline_fit \
    --data "$DATA" --logdir "$WM_RUN" --arm "$TM2_ARM" \
    --task "$TASK" --updates "$TM2_UPDATES" --seed "$TM2_SEED"
fi

# --- stage 3: frozen-representation adapt ---
if [ ! -f "$ADAPT_RUN/ADAPT_DONE" ]; then
  echo "[$(date)] adapt RUN_ID=$(basename "$ADAPT_RUN") TASK=$TASK STEPS=$TM2_STEPS SEED=$TM2_SEED"
  python -m probing.tdmpc2_adapt \
    --pretrained "$WM_RUN" --logdir "$ADAPT_RUN" \
    --task "$TASK" --seed "$TM2_SEED" --steps "$TM2_STEPS"
fi
echo "[$(date)] tdmpc2 $(basename "$ADAPT_RUN") rc=0 status=DONE"
