#!/bin/bash
# Run one deterministic shard of the Axis-1 grid on a single local GPU.
#
# Intended for non-Slurm cloud boxes. Launch this with CUDA_VISIBLE_DEVICES set
# to one GPU and CLOUD_SHARD/CLOUD_SHARDS selecting its share of the 64 cells.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO/scripts/env.sh"

if [ ! -x "${CONDA_ENV:-}/bin/python" ]; then
  CONDA_ENV="$(python - <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.executable).resolve().parents[1])
PY
)"
  export CONDA_ENV
fi

export PATH="$CONDA_ENV/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export CONDA_PREFIX="$CONDA_ENV"
export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

: "${RUNROOT:?set RUNROOT}"
: "${CONDA_ENV:?set CONDA_ENV}"
: "${CLOUD_SHARD:?set CLOUD_SHARD, zero-indexed}"
: "${CLOUD_SHARDS:?set CLOUD_SHARDS}"

cd "$REPO"
mkdir -p "$RUNROOT/_cloud_logs"

read -ra seeds <<< "${AXIS1_SEEDS:-1 2 3 4 5 6 7 8}"
read -ra quads <<< "${AXIS1_QUADS:-q1 q2}"
read -ra doms <<< "${AXIS1_DOMAINS:-cup finger}"
updates="${AXIS1_UPDATES:-500000}"
steps="${STEPS:-1.25e5}"
: "${AXIS1_EXPL_MODE:?set AXIS1_EXPL_MODE=apt (reward-free) or task (reward-aware arm)}"
case "$AXIS1_EXPL_MODE" in
  apt|task) ;;
  *) echo "AXIS1_EXPL_MODE must be 'apt' or 'task', got: $AXIS1_EXPL_MODE" >&2; exit 2 ;;
esac
transform="${AXIS1_TRANSFORM:-}"
case "$transform" in
  ''|sh|rl|srd0|srd1|sid) ;;
  *) echo "AXIS1_TRANSFORM must be sh|rl|srd0|srd1|sid or empty, got: $transform" >&2; exit 2 ;;
esac
if [ -n "$transform" ]; then
  [ "$AXIS1_EXPL_MODE" = task ] || {
    echo "AXIS1_TRANSFORM=$transform requires AXIS1_EXPL_MODE=task" >&2
    exit 2
  }
  id_prefix="${AXIS1_ID_PREFIX:-ax1${transform}}"
  quad_suffix="_${transform}"
  export AXIS1_ARM="${AXIS1_ARM:-full}"
else
  id_prefix="${AXIS1_ID_PREFIX:-$([ "$AXIS1_EXPL_MODE" = task ] && echo ax1 || echo ax1f)}"
  quad_suffix=""
fi
wm_infix="${id_prefix#ax1}"

idx=0
ran=0
for dom in "${doms[@]}"; do
  case "$dom" in
    cup) task=dmc_cup_catch ;;
    finger) task=dmc_finger_turn_hard ;;
    synth) task=synth_reach ;;
    *) echo "unknown axis1 domain: $dom" >&2; exit 2 ;;
  esac
  for q in "${quads[@]}"; do
    root="$RUNROOT/axis1_${dom}/${q}${quad_suffix}"
    [ -f "$root/manifest.json" ] || {
      echo "missing built Axis-1 buffer manifest: $root/manifest.json" >&2
      exit 2
    }
    for side in 0 1; do
      for s in "${seeds[@]}"; do
        if [ $((idx % CLOUD_SHARDS)) -ne "$CLOUD_SHARD" ]; then
          idx=$((idx + 1))
          continue
        fi

        run_id="adapt_${id_prefix}${q}s${side}_${dom}_seed${s}_ckpt${updates}"
        wm_run="ax1wm_${dom}_${wm_infix}${q}s${side}_seed${s}"
        log="$RUNROOT/_cloud_logs/${run_id}.out"
        if [ "${FORCE:-0}" != "1" ] && [ -f "$RUNROOT/$run_id/ADAPT_DONE" ]; then
          echo "[$(date)] SKIP done: $run_id"
          idx=$((idx + 1))
          continue
        fi

        echo "[$(date)] START shard=$CLOUD_SHARD/$CLOUD_SHARDS gpu=${CUDA_VISIBLE_DEVICES:-all} run=$run_id"
        set +e
        RUN_ID="$run_id" WM_RUN="$wm_run" TASK="$task" SEED="$s" \
          REPLAY="$root/side${side}" UPDATES="$updates" STEPS="$steps" \
          AXIS=axis1 PAIRED_SEED_SET="${id_prefix}_${dom}_${q}" \
          AXIS1_EXPL_MODE="$AXIS1_EXPL_MODE" AXIS1_SIZE="${AXIS1_SIZE:-}" \
          AXIS1_ARM="${AXIS1_ARM:-full}" RENDER="${RENDER:-False}" \
          bash "$REPO/scripts/axis1.sbatch" > "$log" 2>&1
        rc=$?
        set -e
        if [ "$rc" -ne 0 ]; then
          echo "[$(date)] FAILED run=$run_id rc=$rc log=$log" >&2
          exit "$rc"
        fi
        echo "[$(date)] DONE run=$run_id log=$log"
        ran=$((ran + 1))
        idx=$((idx + 1))
      done
    done
  done
done

echo "[$(date)] shard=$CLOUD_SHARD finished ran=$ran"
