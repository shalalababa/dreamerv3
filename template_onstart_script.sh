#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Vast onstart template  (dv3ops P0, 2026-08-14)
#
# v1 (2026-07-26, archived at temp_files/ops/template_onstart_script_v1_20260726.sh)
# carried the whole helper library inline and sat 135 bytes under Vast's 16384
# char onstart cap. That cap was shaping the architecture: features were being
# dropped to fit it. The helpers now live in the repo at
# scripts/ops/instance_helpers.sh and arrive with `dv3ops push-repo`; what
# remains here is only what must exist BEFORE the first sync.
#
# Per-instance edits before pasting into the Vast template box:
#   DV3_INSTANCE_LABEL   short name used in alert titles, e.g. inst4
#   DV3_NTFY_TOPIC       shared across instances; long random string
#   DV3_HC_URL           per-instance healthchecks.io ping URL (dead-man switch)
# Leaving the last two empty is safe: notification silently no-ops.
# ---------------------------------------------------------------------------
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/startup.log) 2>&1
echo "[$(date)] startup begin"

mkdir -p /workspace/dreamerv3 /workspace/dreamerv3_runs/_cloud_logs \
  /workspace/dreamerv3_runs/_queue_control /root/.ssh
chmod 700 /workspace/dreamerv3_runs/_queue_control 2>/dev/null || true

[ -x /venv/main/bin/python ] || { echo "ERROR: /venv/main/bin/python missing"; exit 1; }

chown root:root /root /root/.ssh
chmod 700 /root /root/.ssh
k=/root/.ssh/authorized_keys
[ -f "$k" ] && { chown root:root "$k"; chmod 600 "$k"; } || true

cat > /root/.dreamer_vast_env <<'EOF'
export REPO=/workspace/dreamerv3
export RUNROOT=/workspace/dreamerv3_runs
export CONDA_ENV=/venv/main
export MANIFEST=$RUNROOT/runs.csv

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PATH=$CONDA_ENV/bin:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# --- dv3ops notification ---
# DV3_INSTANCE_LABEL is the one worth editing per instance: it prefixes every
# alert title, so "[inst4] lane 2 ABORTED" tells you where to look without
# opening anything. Defaults to the container hostname if left empty.
export DV3_INSTANCE_LABEL=
export DV3_NTFY_TOPIC=dreamerv3_ops_notification
# Optional dead-man's switch; empty = disabled, and the watchdog no-ops
# silently rather than erroring. Fill in a per-instance healthchecks.io ping
# URL to get alerted when an instance dies without saying anything.
export DV3_HC_URL=

source "$CONDA_ENV/bin/activate" >/dev/null 2>&1 || true
EOF

# Shim only. The real helpers ship with the repo, so an instance always runs
# the helper version of the fork last pushed to it -- v1 froze helpers at
# rental time with no way to tell which vintage an instance was on.
cat > /root/dreamer_instance_helpers.sh <<'EOF'
DV3_HELPERS_REAL="${REPO:-/workspace/dreamerv3}/scripts/ops/instance_helpers.sh"
if [ -r "$DV3_HELPERS_REAL" ]; then
  source "$DV3_HELPERS_REAL"
else
  echo "dv3: helpers not installed yet -- run 'dv3ops push-repo <N>' from RCC" >&2
  echo "dv3: (expected at $DV3_HELPERS_REAL)" >&2
fi
EOF

chmod 644 /root/dreamer_instance_helpers.sh /root/.dreamer_vast_env

grep -q dreamer_vast_env /root/.bashrc 2>/dev/null ||
  printf 'source /root/.dreamer_vast_env\nsource /root/dreamer_instance_helpers.sh\n' >> /root/.bashrc

echo "[$(date)] startup done -- next: dv3ops push-repo <N> from RCC"
