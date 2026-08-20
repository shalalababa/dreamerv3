#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Queue ONE seed-block of carrier_freshrep onto ONE instance.
#
# Why this exists: generated/submit.sh queues EVERY stage's lanes onto the
# instance it is given, which for this wave means all 160 cells on one box.
# The spec deliberately carves the wave into six per-instance stages (a seed's
# four cells must stay on one machine -- see the spec header), so the submit
# step has to be block-scoped. Same shape as fb_fits/submit_zeroshot_only.sh.
#
#   usage: submit_block.sh <block 1-6> <instance>
#
# Mechanics are copied verbatim from the generated submit.sh (rsync the wave
# dir, then dv3_queue_or_add per lane) -- do not "improve" them here.
# ---------------------------------------------------------------------------
set -euo pipefail
B="${1:?usage: submit_block.sh <block 1-6> <instance>}"
N="${2:?usage: submit_block.sh <block 1-6> <instance>}"
case "$B" in 1|2|3|4|5|6) ;; *) echo "block must be 1-6, got: $B" >&2; exit 2 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$HERE/generated"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

for l in 0 1; do
  [ -f "$GEN/lane_blk${B}_${l}.cmds" ] || { echo "MISSING $GEN/lane_blk${B}_${l}.cmds -- run dv3ops gen --lanes 0,1" >&2; exit 2; }
done

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: cannot read RUNROOT on instance $N" >&2; exit 1; }
REMOTE="$REMOTE_ROOT/_waves/carrier_freshrep"

echo "== block $B -> instance $N  (seeds: $(sed -n '2p' "$GEN/lane_blk${B}_0.cmds" >/dev/null 2>&1; grep -oE 'seed[0-9]+' "$GEN/lane_blk${B}_0.cmds" "$GEN/lane_blk${B}_1.cmds" | sed 's/.*seed//' | sort -n -u | tr '\n' ' '))"
dv3_ssh_dv3 "$N" "mkdir -p \"$REMOTE\""
rsync -az --checksum \
  -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" \
  "$GEN/" "root@$(dv3_ip "$N"):$REMOTE/"

for l in 0 1; do
  n_tasks="$(grep -c . "$GEN/lane_blk${B}_${l}.cmds")"
  echo "-- stage blk$B lane $l: $((n_tasks - 1)) task(s) --"
  dv3_ssh_dv3 "$N" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/lane_blk${B}_${l}.cmds\" $l"
done
echo "queued block $B on instance $N. verify: dv3ops status $N"
