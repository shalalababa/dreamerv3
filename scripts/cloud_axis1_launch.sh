#!/bin/bash
# Launch one Axis-1 worker per local GPU on a non-Slurm cloud instance.
#
# Example:
#   CLOUD_SHARDS=8 CLOUD_SHARD_OFFSET=0 ./scripts/cloud_axis1_launch.sh
#   CLOUD_SHARDS=8 CLOUD_SHARD_OFFSET=4 ./scripts/cloud_axis1_launch.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO/scripts/env.sh"

: "${RUNROOT:?set RUNROOT}"
if [ ! -x "${CONDA_ENV:-}/bin/python" ]; then
  CONDA_ENV="$(python - <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.executable).resolve().parents[1])
PY
)"
  export CONDA_ENV
fi
: "${CONDA_ENV:?set CONDA_ENV}"

mkdir -p "$RUNROOT/_cloud_logs"

local_gpus="${LOCAL_GPUS:-}"
if [ -z "$local_gpus" ]; then
  local_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
[ "$local_gpus" -gt 0 ] || { echo "No GPUs visible."; exit 2; }

cloud_shards="${CLOUD_SHARDS:-$local_gpus}"
offset="${CLOUD_SHARD_OFFSET:-0}"

echo "Launching $local_gpus local workers with CLOUD_SHARDS=$cloud_shards offset=$offset"
for ((gpu=0; gpu<local_gpus; gpu++)); do
  shard=$((offset + gpu))
  log="$RUNROOT/_cloud_logs/worker_shard${shard}.out"
  echo "  GPU $gpu -> shard $shard/$cloud_shards log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" CLOUD_SHARD="$shard" CLOUD_SHARDS="$cloud_shards" \
    bash "$REPO/scripts/cloud_axis1_worker.sh" > "$log" 2>&1 &
done

wait
echo "All local cloud workers finished."
