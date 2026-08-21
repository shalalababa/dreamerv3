#!/usr/bin/env bash
# Queue ONE min-dose block (4 dose + 4 scratch cells) onto lanes 2,3 of one
# instance. Both arms travel together so the two-sample contrast is never
# split across machines; lanes 2-3 are the short lanes on the 4-GPU boxes
# (2 U1a cells vs 4 on lanes 0-1), so this wave costs no extra wall-clock.
#   usage: submit_block.sh <A|B> <instance>
set -euo pipefail
B="${1:?block A or B}"; N="${2:?instance}"
case "$B" in A|B) ;; *) echo "block must be A or B" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; GEN="$HERE/generated"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

# the fit substrate must be complete on this box before anything is queued
ok="$(dv3_ssh_dv3 "$N" 'R=/workspace/dreamerv3_runs/axis1_reacher/dose; n=$(ls $R/level1/*.npz 2>/dev/null | wc -l); [ "$n" -ge 200 ] && [ -f $R/manifest.json ] && echo yes || echo "no($n)"')"
[ "$ok" = yes ] || { echo "REFUSE: instance $N level-1 buffer incomplete: $ok" >&2; exit 2; }

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
REMOTE="$REMOTE_ROOT/_waves/mindose_reacher"
dv3_ssh_dv3 "$N" "mkdir -p \"$REMOTE\""
rsync -az --checksum -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" "$GEN/" "root@$(dv3_ip "$N"):$REMOTE/"
echo "== min-dose block $B -> instance $N (lanes 2,3)"
# dose first: it is the long cell (fit + unfrozen adapt) and the scratch
# cells are short, so leading with the long pole keeps the two lanes level.
for arm in dose scratch; do for l in 2 3; do
  f="lane_blk${B}_${arm}_${l}.cmds"
  [ -f "$GEN/$f" ] || { echo "missing $GEN/$f" >&2; exit 2; }
  echo "-- blk$B $arm lane $l: $(( $(grep -c . "$GEN/$f") - 1 )) cell(s)"
  dv3_ssh_dv3 "$N" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/$f\" $l"
done; done
echo "queued min-dose block $B on instance $N"
