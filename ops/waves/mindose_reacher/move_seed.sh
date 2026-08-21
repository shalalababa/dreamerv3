#!/usr/bin/env bash
# Move one min-dose seed (its dose cell + its scratch cell) to another
# instance lane. Both arms travel together: each machine must carry equal
# dose and scratch cells so device stays a BALANCED nuisance factor inside
# the two-sample contrast rather than a confound (se_grad_arms precedent).
#
# Destination is gated before the source is touched, and the source task is
# only removed if it is still pending.
#
#   usage: move_seed.sh <seed> <src_inst> <dst_inst> <dst_lane>
set -euo pipefail
S="${1:?seed}"; SRC="${2:?src instance}"; DST="${3:?dst instance}"; L="${4:?dst lane}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; GEN="$HERE/generated"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$SRC"; dv3_require_instance "$DST"

tmp="$(mktemp)"
grep -h "_seed${S}_ckpt" "$GEN"/lane_blk*_dose_*.cmds "$GEN"/lane_blk*_scratch_*.cmds > "$tmp" || true
n=$(grep -c . "$tmp")
[ "$n" -eq 2 ] || { echo "REFUSE: seed $S resolved $n command(s), expected 2 (dose + scratch)" >&2; rm -f "$tmp"; exit 2; }

# the fit substrate must be complete on the destination
ok="$(dv3_ssh_dv3 "$DST" 'R=/workspace/dreamerv3_runs/axis1_reacher/dose; n=$(ls $R/level1/*.npz 2>/dev/null | wc -l); [ "$n" -ge 200 ] && [ -f $R/manifest.json ] && echo yes || echo "no($n)"')"
[ "$ok" = yes ] || { echo "REFUSE: instance $DST level-1 buffer incomplete: $ok" >&2; rm -f "$tmp"; exit 2; }

REMOTE_ROOT="$(dv3_ssh_dv3 "$DST" 'printf %s "$RUNROOT"')"
REMOTE="$REMOTE_ROOT/_waves/mindose_reacher"
dv3_ssh_dv3 "$DST" "mkdir -p \"$REMOTE\""
f="moved_seed${S}.cmds"
{ echo "# min-dose seed $S (dose + scratch) moved from instance $SRC"; cat "$tmp"; } > "/tmp/$f"
scp -q -P "$(dv3_port "$DST")" "${DV3_SSH_OPTS[@]}" "/tmp/$f" "root@$(dv3_ip "$DST"):$REMOTE/"
dv3_ssh_dv3 "$DST" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/$f\" $L" | tail -1

got="$(dv3_ssh_dv3 "$DST" "cat /workspace/dreamerv3_runs/_queue_control/lane_*/queue_*/tasks.txt /workspace/dreamerv3_runs/_queue_control/lane_*/queue_*/add_*.txt 2>/dev/null | grep -c '_seed${S}_ckpt' || true")"
[ "${got:-0}" -ge 2 ] || { echo "REFUSE to remove from source: destination holds ${got:-0} cell(s) for seed $S" >&2; rm -f "$tmp"; exit 1; }
echo "  destination holds $got cell(s) for seed $S"

# ---- source-side occupancy -------------------------------------------------
# Gating only the DESTINATION is what left instance 8 lane 3 idle for 7h on
# 2026-08-21: the cells arrived safely, nothing noticed the source lane had
# nothing behind them, and the box kept billing 4 GPUs to run 2. A box costs
# the same whether one lane or all of them are busy, so the thing to protect is
# not "no cell is lost" but "no lane goes dark while its siblings still work".
# Pending means NOT YET STARTED: while a task runs, next_index == running_index,
# so pending is strictly idx > next_index.
# Override with DV3_ALLOW_SOURCE_DRAIN=1 when the source is about to be
# destroyed anyway -- draining it deliberately is the whole point then.
occ="$(dv3_ssh_dv3 "$SRC" "
  for l in 0 1 2 3 4 5 6 7; do
    qc=/workspace/dreamerv3_runs/_queue_control/lane_\$l
    [ -f \"\$qc/active_queue\" ] || continue
    qd=\"\$qc/\$(cat \$qc/active_queue)\"
    nx=\$(cat \$qd/next_index 2>/dev/null || echo 1)
    pend=\$(awk -v n=\"\$nx\" 'NR>n && \$0 !~ /^#removed#/ && \$0 !~ /_seed${S}_ckpt/ && NF' \"\$qd/tasks.txt\" | wc -l)
    echo \"\$l \$pend\"
  done")"
[ -n "$occ" ] || { echo "REFUSE: instance $SRC reported no active lanes" >&2; rm -f "$tmp"; exit 2; }
echo "  source instance $SRC, pending per lane AFTER this move:"
echo "$occ" | while read -r l pend; do echo "    lane $l: $pend"; done
dark=$(echo "$occ" | awk '$2==0' | wc -l)
busy=$(echo "$occ" | awk '$2>0'  | wc -l)
if [ "$dark" -gt 0 ] && [ "$busy" -gt 0 ] && [ "${DV3_ALLOW_SOURCE_DRAIN:-0}" != 1 ]; then
  echo "REFUSE: this move leaves $dark lane(s) with nothing queued on instance $SRC" >&2
  echo "        while $busy lane(s) keep running -- that is a half-empty billed box." >&2
  echo "        Move the rest of $SRC's work out too, or set DV3_ALLOW_SOURCE_DRAIN=1" >&2
  echo "        if $SRC is being destroyed once its current tasks finish." >&2
  rm -f "$tmp"; exit 3
fi

# only now remove from the source, and only pending indices
dv3_ssh_dv3 "$SRC" "
  source /root/dreamer_instance_helpers.sh
  for l in 0 1 2 3 4 5 6 7; do
    qc=/workspace/dreamerv3_runs/_queue_control/lane_\$l
    [ -f \"\$qc/active_queue\" ] || continue
    qd=\"\$qc/\$(cat \$qc/active_queue)\"
    nx=\$(cat \$qd/next_index)
    idx=\$(grep -n '_seed${S}_ckpt' \"\$qd/tasks.txt\" | grep -v '#removed#' | cut -d: -f1 | tr '\n' ' ')
    [ -n \"\$idx\" ] || continue
    for i in \$idx; do
      if [ \"\$i\" -lt \"\$nx\" ]; then echo \"  lane \$l task \$i already dispatched -- NOT removed\"; continue; fi
      dv3_remove_tasks \$l \$i | sed 's/^/  lane '\$l': /'
    done
  done"
rm -f "$tmp" "/tmp/$f"
echo "seed $S: $SRC -> $DST lane $L done"
