#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Append ONLY the zeroshot lanes to a live instance.
#
# WHY THIS EXISTS: generated/submit.sh appends EVERY stage's lane files,
# including lane_fit_*.cmds. The 16 fits are already queued and running, so
# running it now would queue a SECOND copy of all 16 fits -- ~8 h of GPU
# re-running training over its own finished checkpoints. There is no
# per-stage flag on the generated script, so the zeroshot append is done
# here instead.
#
# Safe to run while the fits are still going: dv3_queue_or_add appends to
# the live queue and the supervisor picks up late adds ("QUEUE_RESUME ...
# late adds found"). Lanes are sequential and the lane map is per-seed
# (lane k = seeds {k+1, k+5}, verified identical across the fit and
# zeroshot stages), so each lane's zeroshots sit behind that lane's own
# fits -- no barrier needed.
#
# Each queued command also guards itself on FB_FIT_DONE and exits 3 if the
# fit did not complete, so a mis-ordering or a dead fit REFUSES rather than
# evaluating a half-written checkpoint.
#
#   usage: submit_zeroshot_only.sh <instance>
# ---------------------------------------------------------------------------
set -euo pipefail
N="${1:?usage: submit_zeroshot_only.sh <instance>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: cannot read RUNROOT on instance $N" >&2; exit 1; }
REMOTE="$REMOTE_ROOT/_waves/fb_fits"

# Refuse if the fits are not actually the thing already queued -- this
# script's whole safety argument is "the fits are already in these lanes".
for i in 0 1 2 3; do
  [ -f "$HERE/generated/lane_zeroshot_$i.cmds" ] || {
    echo "ERROR: generated/lane_zeroshot_$i.cmds missing -- run 'dv3ops gen --wave fb_fits'" >&2
    exit 2; }
done

dv3_ssh_dv3 "$N" "mkdir -p \"$REMOTE\""
rsync -az --checksum \
  -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" \
  "$HERE/generated/" "root@$(dv3_ip "$N"):$REMOTE/"

for i in 0 1 2 3; do
  n=$(grep -cv '^#' "$HERE/generated/lane_zeroshot_$i.cmds")
  echo "-- stage zeroshot lane $i: $n task(s) --"
  dv3_ssh_dv3 "$N" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/lane_zeroshot_$i.cmds\" $i"
done

echo
echo "appended. fit lanes were NOT re-queued."
