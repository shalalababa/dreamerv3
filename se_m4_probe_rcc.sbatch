#!/bin/bash
#SBATCH --job-name=se_m4_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=10-17
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
cd "$REPO"

export PATH="$CONDA_ENV/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export CONDA_PREFIX="$CONDA_ENV"
export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || { echo "ERROR: no python at $PYTHON"; exit 2; }

seed="$SLURM_ARRAY_TASK_ID"
run="$RUNROOT/se_cheetah_seed${seed}"
snaproot="$run/ckpt_snapshots"

[ -f "$run/TRAINING_DONE" ] || { echo "ERROR: missing TRAINING_DONE for $run"; exit 2; }
[ -d "$run/replay" ] || { echo "ERROR: missing replay for $run"; exit 2; }
[ -d "$snaproot" ] || { echo "ERROR: missing ckpt_snapshots for $run"; exit 2; }
[ -f "$snaproot/manifest.json" ] || { echo "ERROR: missing $snaproot/manifest.json"; exit 2; }

"$PYTHON" - "$run" <<'PY'
import glob, json, os, sys

run = sys.argv[1]
snaproot = os.path.join(run, "ckpt_snapshots")
targets = {"snap25": 125000, "snap50": 250000}
tol = 5000
nearest = {}

for label, target in targets.items():
    candidates = []
    for p in glob.glob(os.path.join(snaproot, "step*")):
        base = os.path.basename(p)
        try:
            step = int(base.replace("step", ""))
        except ValueError:
            continue
        payload = (
            os.path.exists(os.path.join(p, "agent.pkl")) or
            bool(glob.glob(os.path.join(p, "**", "agent.pkl"), recursive=True))
        )
        candidates.append((abs(step - target), step, p, payload))
    if not candidates:
        raise SystemExit(f"no snapshots for {run}")
    err, step, path, payload = min(candidates, key=lambda x: x[0])
    if err > tol or not payload:
        raise SystemExit(
            f"bad {run} {label}: target={target} step={step} "
            f"err={err} payload={payload} path={path}"
        )
    nearest[label] = {
        "target_step": target,
        "selected_step": step,
        "abs_error": err,
        "checkpoint": path,
        "output": os.path.join(run, f"se_probe_{label}"),
    }

out = os.path.join(snaproot, "nearest.json")
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(nearest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, out)

print(json.dumps(nearest, indent=2, sort_keys=True))
PY

snap25="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snap25"]["checkpoint"])' "$snaproot/nearest.json")"
snap50="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snap50"]["checkpoint"])' "$snaproot/nearest.json")"

echo "seed=$seed snap25=$snap25"
"$PYTHON" -m uncfield.se_probe \
  --run_logdir "$run" \
  --ckpt "$snap25" \
  --output "$run/se_probe_snap25" \
  --platform cuda

echo "seed=$seed snap50=$snap50"
"$PYTHON" -m uncfield.se_probe \
  --run_logdir "$run" \
  --ckpt "$snap50" \
  --output "$run/se_probe_snap50" \
  --platform cuda

echo "DONE se_cheetah_seed${seed}"
