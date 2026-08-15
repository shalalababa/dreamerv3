#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One LeWM training run, with its checkpoint relocated.  (dv3ops P1, 2026-08-14)
#
#   lewm_train_one.sh <side> <seed>
#
# THE PROBLEM THIS FIXES
# The external trainer writes through stable-pretraining, which saves into its
# own cache rather than the run dir:
#     /root/.cache/stable-pretraining/runs/<date>/<time>/<hash>/checkpoints/last.ckpt
# while every registered downstream (distill_encoder, the graft wave, the
# bundle) requires a self-contained run dir:
#     $RUNROOT/lewm_finger_s<side>_seed<seed>/config.yaml
#     $RUNROOT/lewm_finger_s<side>_seed<seed>/lewm_weights.ckpt
# Relocating by hand is a step that gets skipped, and it is skipped silently:
# the training succeeded, so nothing looks wrong until distill fails 16 times.
#
# WHY CACHE ISOLATION RATHER THAN "NEWEST last.ckpt"
# Lanes now run two trainings concurrently, and all of them share one cache
# root, so "most recent checkpoint" is a race with no error path -- it can
# quietly attach the wrong seed's weights to a run dir, which is unrecoverable
# by inspection afterwards. Giving each run a private HOME makes the cache
# per-run, so the lookup is exact. The post-hoc assertions stay anyway: if the
# isolation ever stops working, this must fail loudly, not guess.
#
# COPY, NOT SYMLINK
# The run dir is rsynced to RCC and hard-linked into bundles. A symlink into
# /root/.cache resolves to nothing on the other side, and the archive would
# record a dangling link as if it were weights.
# ---------------------------------------------------------------------------
set -euo pipefail

side="${1:?side}"
seed="${2:?seed}"

source /root/.dreamer_vast_env 2>/dev/null || true
: "${RUNROOT:?RUNROOT unset}"

run="lewm_finger_s${side}_seed${seed}"
outdir="$RUNROOT/$run"

if [ -f "$outdir/config.yaml" ] && [ -s "$outdir/lewm_weights.ckpt" ]; then
  echo "SKIP complete $run"
  exit 0
fi

data="$RUNROOT/lewm_data/finger_pxq1m_side${side}.h5"
[ -f "$data" ] || { echo "ERROR: missing LeWM dataset $data" >&2; exit 2; }

LEWM_CHECKOUT="${LEWM_CHECKOUT:-/workspace/le-wm}"
PY="${LEWM_CONDA_ENV:-/workspace/conda_envs/lewm}/bin/python"
[ -x "$PY" ] || { echo "ERROR: no LeWM python at $PY (run scripts/lewm_env_setup.sh)" >&2; exit 2; }

mkdir -p "$outdir"

# Private cache root for this run only.
cache_home="$outdir/.sp_cache"
rm -rf "$cache_home"
mkdir -p "$cache_home"
started="$(date +%s)"

echo "[$(date)] lewm train START $run (cache=$cache_home)"
(
  cd "$LEWM_CHECKOUT"
  export PYTHONPATH="${REPO:-/workspace/dreamerv3}:$LEWM_CHECKOUT:${PYTHONPATH:-}"
  export HOME="$cache_home" XDG_CACHE_HOME="$cache_home/.cache"
  "$PY" train.py data=pusht \
    data.dataset.name="$data" \
    data.dataset.frameskip=1 \
    "data.dataset.keys_to_load=[pixels,action]" \
    "data.dataset.keys_to_cache=[action]" \
    img_size=64 seed="$seed" \
    output_model_name=lewm \
    subdir="$outdir" \
    trainer.devices=1 \
    wandb.enabled=false
)

# --- relocate ---------------------------------------------------------------
# Search the private cache first; fall back to the shared one only for
# checkpoints written after this run started, and refuse to guess if that is
# ambiguous.
find_ckpt () {
  local root="$1" newer="$2"
  [ -d "$root" ] || return 1
  if [ "$newer" = strict ]; then
    find "$root" -name 'last.ckpt' -newermt "@$started" -print 2>/dev/null | sort
  else
    find "$root" -name 'last.ckpt' -print 2>/dev/null | sort
  fi
}

mapfile -t hits < <(find_ckpt "$cache_home" any)
scope="private cache"
if [ "${#hits[@]}" -eq 0 ]; then
  mapfile -t hits < <(find_ckpt "/root/.cache/stable-pretraining" strict)
  scope="shared cache (written after this run started)"
fi

if [ "${#hits[@]}" -eq 0 ]; then
  echo "ERROR: $run trained but no last.ckpt found in $cache_home" >&2
  echo "       or in /root/.cache/stable-pretraining newer than the run start." >&2
  echo "       Refusing to invent weights; inspect the trainer log." >&2
  exit 3
fi
if [ "${#hits[@]}" -gt 1 ]; then
  echo "ERROR: $run matched ${#hits[@]} checkpoints in $scope:" >&2
  printf '       %s\n' "${hits[@]}" >&2
  echo "       Ambiguous: attaching the wrong seed's weights would be undetectable" >&2
  echo "       downstream, so this fails instead of picking one." >&2
  exit 3
fi

ckpt="${hits[0]}"
cp -f "$ckpt" "$outdir/lewm_weights.ckpt"
echo "[$(date)] relocated from $scope: $ckpt -> $outdir/lewm_weights.ckpt"

# The Hydra config lands beside the checkpoint; downstream reads it from the run dir.
if [ ! -f "$outdir/config.yaml" ]; then
  cfg="$(find "$(dirname "$ckpt")/.." -maxdepth 2 -name 'config.yaml' -print 2>/dev/null | head -1)"
  [ -n "$cfg" ] && cp -f "$cfg" "$outdir/config.yaml"
fi

[ -s "$outdir/lewm_weights.ckpt" ] || { echo "ERROR: relocated checkpoint is empty" >&2; exit 3; }
[ -f "$outdir/config.yaml" ] || {
  echo "ERROR: no config.yaml in $outdir; the downstream contract needs both" >&2; exit 3; }

# The cache has served its purpose and is large; the run dir keeps the copy.
rm -rf "$cache_home"

echo "[$(date)] lewm train DONE $run"
