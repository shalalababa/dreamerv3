#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One adaptation run.  (dv3ops P1, 2026-08-14)
#
# This is the script that used to be written into /tmp by a heredoc at the top
# of every adapt wave. Being regenerated per wave meant it was re-derived, and
# re-derived means it could drift between waves that were supposed to be
# comparable. It is now versioned with the repo and shipped by push-repo.
#
#   adapt_one.sh <run_id> <seed> <cfg> <task> <steps> <ckpt_root|NONE>
#
# Idempotent: an existing ADAPT_DONE short-circuits. NOTE that this is the
# resume path implicated in the axis1 truncation -- a run that resumes and
# immediately declares itself done leaves ADAPT_DONE with too few scores, so
# always verify realized counters (wavecheck does this) rather than markers.
# ---------------------------------------------------------------------------
set -euo pipefail

run_id="${1:?run_id}"
seed="${2:?seed}"
cfg="${3:?cfg}"
task="${4:?task}"
steps="${5:?steps}"
ckpt_root="${6:-NONE}"

source /root/.dreamer_vast_env 2>/dev/null || true
source "$REPO/scripts/env.sh"
cd "$REPO"

PYTHON="${CONDA_ENV:-/venv/main}/bin/python"
[ -x "$PYTHON" ] || PYTHON=python

latest_done_ckpt () {
  local root="$1"
  if [ -f "$root/done" ]; then
    printf '%s\n' "$root"
  else
    find "$root" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/done' ';' -print |
      sort | tail -1
  fi
}

logdir="$RUNROOT/$run_id"
mkdir -p "$logdir"

if [ -f "$logdir/ADAPT_DONE" ]; then
  echo "SKIP done $run_id"
  exit 0
fi

ckpt=""
if [ -n "$ckpt_root" ] && [ "$ckpt_root" != NONE ]; then
  ckpt="$(latest_done_ckpt "$ckpt_root")"
  [ -n "$ckpt" ] || { echo "ERROR: no done checkpoint under $ckpt_root" >&2; exit 1; }
fi

echo "[$(date)] adapt START $run_id seed=$seed cfg=$cfg task=$task ckpt=${ckpt:-EMPTY}"
manifest_append "$run_id" adapt "$task" "$cfg" "$seed" "$steps" "${ckpt:-EMPTY}" "$logdir" RUNNING

if [ -n "$ckpt" ]; then
  "$PYTHON" -u dreamerv3/main.py \
    --logdir "$logdir" --configs dmc_proprio "$cfg" \
    --task "$task" --seed "$seed" \
    --env.dmc.render False \
    --run.from_checkpoint "$ckpt" --run.steps "$steps"
else
  "$PYTHON" -u dreamerv3/main.py \
    --logdir "$logdir" --configs dmc_proprio "$cfg" \
    --task "$task" --seed "$seed" \
    --env.dmc.render False \
    --run.steps "$steps"
fi

touch "$logdir/ADAPT_DONE"
manifest_append "$run_id" adapt "$task" "$cfg" "$seed" "$steps" "${ckpt:-EMPTY}" "$logdir" DONE
echo "[$(date)] adapt DONE $run_id"
