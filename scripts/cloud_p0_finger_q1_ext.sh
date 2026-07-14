#!/bin/bash
# Manage the reward-free corrective Axis-1 finger-Q1 seed 9-16 cloud extension.
#
# Required local env:
#   IP1 PORT1 ... IP4 PORT4   instance SSH coordinates, or set INSTANCES.
# Optional:
#   REPO=/home/rickybao/projects/dreamerv3
#   RUNROOT=/scratch/midway3/$USER/dreamerv3_runs
#   INSTANCES="1 2 3 4"
#   LOCAL_GPUS=4 CLOUD_SHARDS=16 SHARD_OFFSET_BASE=0
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNROOT="${RUNROOT:-/scratch/midway3/$USER/dreamerv3_runs}"
INSTANCES="${INSTANCES:-1 2 3 4}"
LOCAL_GPUS="${LOCAL_GPUS:-4}"
CLOUD_SHARDS="${CLOUD_SHARDS:-16}"
SHARD_OFFSET_BASE="${SHARD_OFFSET_BASE:-0}"
SEEDS="${AXIS1_SEEDS:-9 10 11 12 13 14 15 16}"

CLOUD_REPO=/workspace/dreamerv3
CLOUD_RUNROOT=/workspace/dreamerv3_runs

usage() {
  cat <<'EOF'
Usage: scripts/cloud_p0_finger_q1_ext.sh <command>

Commands:
  check-local   Verify local RCC repo/runroot and axis1_finger/q1 buffers.
  push-repo     Rsync repo to all instances.
  push-buffer   Rsync only axis1_finger/q1 to all instances.
  verify        Check env, guard strings, GPU, and finger-Q1 buffers remotely.
  launch        Launch only AXIS1_EXPL_MODE=apt finger-Q1 seeds 9-16.
  status        Summarize expected corrective run IDs on each instance.
  pull          Alias for pull-light.
  pull-light    Pull auditable outputs, configs, logs, and per-instance runs.csv.
  pull-full     Pull full adapt_ax1f*, ax1wm_*_fq*, _cloud_logs, and runs.csv.
  audit-local   Audit local RUNROOT after pulling.
  all-prep      check-local, push-repo, push-buffer, verify.

This script intentionally does not launch E3/dose/rpair jobs and does not use
AXIS1_EXPL_MODE=task. Corrective run IDs are:
  adapt_ax1fq1s{0,1}_finger_seed{9..16}_ckpt500000
  ax1wm_finger_fq1s{0,1}_seed{9..16}
EOF
}

ssh_base() {
  local port="$1"
  printf 'ssh -p %s -o ServerAliveInterval=30 -o ServerAliveCountMax=120' "$port"
}

instance_vars() {
  local id="$1"
  local ip_var="IP${id}"
  local port_var="PORT${id}"
  ip="${!ip_var:-}"
  port="${!port_var:-}"
  if [ -z "$ip" ] || [ -z "$port" ]; then
    echo "Missing $ip_var/$port_var for INSTANCES=\"$INSTANCES\"" >&2
    exit 2
  fi
}

for_instances() {
  local fn="$1"
  local slot=0 id ip port
  for id in $INSTANCES; do
    instance_vars "$id"
    "$fn" "$id" "$ip" "$port" "$slot"
    slot=$((slot + 1))
  done
}

check_local() {
  echo "repo=$REPO"
  echo "runroot=$RUNROOT"
  test -d "$REPO"
  test -f "$REPO/scripts/cloud_axis1_launch.sh"
  test -f "$REPO/scripts/cloud_axis1_worker.sh"
  test -f "$REPO/scripts/axis1.sbatch"

  local root="$RUNROOT/axis1_finger/q1"
  test -f "$root/manifest.json"
  for side in side0 side1; do
    local count
    count="$(find "$root/$side" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
    echo "$side npz=$count"
    [ "$count" -gt 0 ] || {
      echo "No npz files under $root/$side" >&2
      exit 2
    }
  done

  rg -n "AXIS1_EXPL_MODE|agent.expl.mode|adapt_ax1f|ax1wm_.*fq" \
    "$REPO/scripts/cloud_axis1_worker.sh" "$REPO/scripts/axis1.sbatch"
}

push_repo_one() {
  local id="$1" ip="$2" port="$3"
  echo "== push repo inst$id =="
  rsync -az --delete --info=progress2 \
    -e "$(ssh_base "$port")" \
    --exclude='.git/' \
    --exclude='local_results/' \
    --exclude='artifacts/' \
    --exclude='research_notes/' \
    --exclude='__pycache__/' \
    "$REPO/" root@"$ip":"$CLOUD_REPO"/
}

push_buffer_one() {
  local id="$1" ip="$2" port="$3"
  echo "== push axis1_finger/q1 inst$id =="
  ssh -p "$port" -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
    root@"$ip" "mkdir -p $CLOUD_RUNROOT/axis1_finger"
  rsync -az --delete --partial --append-verify --info=progress2 --stats \
    -e "$(ssh_base "$port")" \
    "$RUNROOT/axis1_finger/q1/" \
    root@"$ip":"$CLOUD_RUNROOT"/axis1_finger/q1/
}

verify_one() {
  local id="$1" ip="$2" port="$3"
  echo "== verify inst$id =="
  ssh -p "$port" -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
    root@"$ip" 'bash -s' <<'REMOTE'
set -euo pipefail
source /root/.dreamer_vast_env 2>/dev/null || source ~/.dreamer_vast_env
cd /workspace/dreamerv3

echo "host=$(hostname)"
echo "repo=$PWD"
echo "runroot=$RUNROOT"
nvidia-smi -L
python - <<'PY'
import jax
print("jax", jax.__version__)
print(jax.devices())
PY

echo "== guard strings =="
rg -n "AXIS1_EXPL_MODE|agent.expl.mode|adapt_ax1f|ax1wm_.*fq" \
  scripts/cloud_axis1_worker.sh scripts/axis1.sbatch

echo "== buffer =="
test -f "$RUNROOT/axis1_finger/q1/manifest.json"
for side in side0 side1; do
  printf "%s npz: " "$side"
  find "$RUNROOT/axis1_finger/q1/$side" -name '*.npz' | wc -l
done
REMOTE
}

launch_one() {
  local id="$1" ip="$2" port="$3" slot="$4"
  local offset=$((SHARD_OFFSET_BASE + slot * LOCAL_GPUS))
  local label="inst${id}"
  echo "== launch $label offset=$offset shards=$CLOUD_SHARDS local_gpus=$LOCAL_GPUS =="
  ssh -p "$port" -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
    root@"$ip" 'bash -s' -- \
    "$offset" "$label" "$CLOUD_SHARDS" "$LOCAL_GPUS" "$SEEDS" "${FORCE_LAUNCH:-0}" <<'REMOTE'
set -euo pipefail
offset="$1"
label="$2"
cloud_shards="$3"
local_gpus="$4"
seeds="$5"
force_launch="$6"

source /root/.dreamer_vast_env 2>/dev/null || source ~/.dreamer_vast_env
cd "$REPO"
mkdir -p "$RUNROOT/_cloud_logs"

if [ "$force_launch" != "1" ]; then
  if ps -eo pid,cmd | grep -E 'cloud_axis1_worker|probing.offline_fit|dreamerv3/main.py' | grep -v grep; then
    echo "Refusing to launch because Dreamer/Axis-1 processes are already running. Set FORCE_LAUNCH=1 to override." >&2
    exit 2
  fi
fi

AXIS1_EXPL_MODE=apt \
AXIS1_DOMAINS="finger" \
AXIS1_QUADS="q1" \
AXIS1_SEEDS="$seeds" \
LOCAL_GPUS="$local_gpus" \
CLOUD_SHARDS="$cloud_shards" \
CLOUD_SHARD_OFFSET="$offset" \
nohup ./scripts/cloud_axis1_launch.sh \
  > "$RUNROOT/_cloud_logs/p0_finger_q1_s9_16_${label}_launch.out" 2>&1 &

echo "launched $label offset=$offset pid=$!"
REMOTE
}

status_python() {
  cat <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
logs = root / "_cloud_logs"
seeds = range(9, 17)
rows = []

def read_text(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""

def expl_mode(config):
    text = read_text(config)
    if not text:
        return "missing"
    try:
        import yaml
        data = yaml.safe_load(text) or {}
        mode = (((data.get("agent") or {}).get("expl") or {}).get("mode"))
        if mode:
            return str(mode)
    except Exception:
        pass
    match = re.search(r"expl:\s*\{[^}]*\bmode:\s*([^,}\s]+)", text)
    if match:
        return match.group(1)
    return "unknown"

def score_lines(path):
    try:
        with path.open() as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0

for side in (0, 1):
    for seed in seeds:
        run = f"adapt_ax1fq1s{side}_finger_seed{seed}_ckpt500000"
        wm = f"ax1wm_finger_fq1s{side}_seed{seed}"
        run_dir = root / run
        wm_dir = root / wm
        log = logs / f"{run}.out"
        log_text = read_text(log)
        rows.append({
            "run": run,
            "wm": wm,
            "adapt_started": run_dir.is_dir(),
            "adapt_done": (run_dir / "ADAPT_DONE").exists() and score_lines(run_dir / "scores.jsonl") > 0,
            "scores": score_lines(run_dir / "scores.jsonl"),
            "wm_started": wm_dir.is_dir(),
            "wm_mode": expl_mode(wm_dir / "config.yaml"),
            "wm_progress": read_text(wm_dir / "OFFLINE_FIT_PROGRESS").strip().replace("\n", ";"),
            "failed": (run_dir / "FAILED").exists() or ("FAILED" in log_text),
        })

total = len(rows)
adapt_started = sum(r["adapt_started"] for r in rows)
adapt_done = sum(r["adapt_done"] for r in rows)
wm_started = sum(r["wm_started"] for r in rows)
wm_apt = sum(r["wm_mode"] == "apt" for r in rows)
failed = sum(r["failed"] for r in rows)

print(
    f"corrective finger-Q1 s9-16: adapt_done={adapt_done}/{total} "
    f"adapt_started={adapt_started}/{total} wm_started={wm_started}/{total} "
    f"wm_expl_apt={wm_apt}/{total} failed={failed}"
)

missing = [r["run"] for r in rows if not r["adapt_started"]]
bad_modes = [f'{r["wm"]}:{r["wm_mode"]}' for r in rows if r["wm_started"] and r["wm_mode"] != "apt"]
not_done = [r["run"] for r in rows if r["adapt_started"] and not r["adapt_done"]]

if missing:
    print("missing_adapt_dirs:", " ".join(missing))
if not_done:
    print("not_done:", " ".join(not_done))
if bad_modes:
    print("BAD_WM_EXPL_MODE:", " ".join(bad_modes))

for r in rows:
    if r["failed"] or (r["wm_started"] and r["wm_mode"] != "apt"):
        print(f'problem run={r["run"]} wm={r["wm"]} mode={r["wm_mode"]} failed={r["failed"]}')
PY
}

status_one() {
  local id="$1" ip="$2" port="$3"
  echo "== status inst$id =="
  ssh -p "$port" -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
    root@"$ip" 'bash -s' <<REMOTE
set -euo pipefail
source /root/.dreamer_vast_env 2>/dev/null || source ~/.dreamer_vast_env
cd /workspace/dreamerv3
python - "\$RUNROOT" <<'PY'
$(status_python)
PY
echo
echo "running tasks:"
ps -eo pid,etime,cmd | grep -E 'cloud_axis1_worker|probing.offline_fit|dreamerv3/main.py' | grep -v grep || true
echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv || true
REMOTE
}

pull_light_one() {
  local id="$1" ip="$2" port="$3"
  local tag="inst${id}"
  echo "== pull-light $tag =="
  mkdir -p "$RUNROOT"
  rsync -az --partial --append-verify --info=progress2 --stats \
    -e "$(ssh_base "$port")" \
    --include='/adapt_ax1f*/' \
    --include='/adapt_ax1f*/scores.jsonl' \
    --include='/adapt_ax1f*/ADAPT_DONE' \
    --include='/adapt_ax1f*/AXIS1_STAGE' \
    --include='/adapt_ax1f*/config.yaml' \
    --include='/ax1wm_*_fq*/' \
    --include='/ax1wm_*_fq*/config.yaml' \
    --include='/ax1wm_*_fq*/OFFLINE_FIT_PROGRESS' \
    --include='/ax1wm_*_fq*/TRAINING_DONE' \
    --include='/_cloud_logs/' \
    --include='/_cloud_logs/***' \
    --exclude='*' \
    root@"$ip":"$CLOUD_RUNROOT"/ \
    "$RUNROOT"/

  rsync -az --partial --append-verify \
    -e "$(ssh_base "$port")" \
    root@"$ip":"$CLOUD_RUNROOT"/runs.csv \
    "$RUNROOT/runs_p0_finger_q1_s9_16_${tag}.csv" || true
}

pull_full_one() {
  local id="$1" ip="$2" port="$3"
  local tag="inst${id}"
  echo "== pull-full $tag =="
  mkdir -p "$RUNROOT"
  rsync -az --partial --append-verify --info=progress2 --stats \
    -e "$(ssh_base "$port")" \
    --include='/adapt_ax1f*/***' \
    --include='/ax1wm_*_fq*/***' \
    --include='/_cloud_logs/***' \
    --exclude='*' \
    root@"$ip":"$CLOUD_RUNROOT"/ \
    "$RUNROOT"/

  rsync -az --partial --append-verify \
    -e "$(ssh_base "$port")" \
    root@"$ip":"$CLOUD_RUNROOT"/runs.csv \
    "$RUNROOT/runs_p0_finger_q1_s9_16_${tag}.csv" || true
}

audit_local() {
  echo "== audit local $RUNROOT =="
  python - "$RUNROOT" <<PY
$(status_python)
PY
}

cmd="${1:-}"
case "$cmd" in
  check-local) check_local ;;
  push-repo) for_instances push_repo_one ;;
  push-buffer) check_local; for_instances push_buffer_one ;;
  verify) for_instances verify_one ;;
  launch) for_instances launch_one ;;
  status) for_instances status_one ;;
  pull|pull-light) for_instances pull_light_one ;;
  pull-full) for_instances pull_full_one ;;
  audit-local) audit_local ;;
  all-prep)
    check_local
    for_instances push_repo_one
    for_instances push_buffer_one
    for_instances verify_one
    ;;
  -h|--help|help|"") usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
