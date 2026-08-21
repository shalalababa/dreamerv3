#!/usr/bin/env bash
# Queue ONE noboot run onto one instance lane. Four runs, four machines: the
# amendment asks for the Stage-1 "same wave/site discipline", and one run per
# box keeps each 5e5-step run on its own GPU with nothing co-resident.
#   usage: submit_one.sh <lane_file_index 0-3> <instance> <lane>
set -euo pipefail
I="${1:?index}"; N="${2:?instance}"; L="${3:?lane}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; GEN="$HERE/generated"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"
f="lane_train_${I}.cmds"
[ -f "$GEN/$f" ] || { echo "missing $GEN/$f" >&2; exit 2; }
REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
REMOTE="$REMOTE_ROOT/_waves/se_noboot"
dv3_ssh_dv3 "$N" "mkdir -p \"$REMOTE\""
rsync -az --checksum -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" "$GEN/" "root@$(dv3_ip "$N"):$REMOTE/"
echo "-- $(grep -o 'RUN_ID=se_noboot_s[0-9]*' "$GEN/$f" | head -1) -> inst $N lane $L"
dv3_ssh_dv3 "$N" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/$f\" $L"
