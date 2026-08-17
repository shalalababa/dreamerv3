#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Move queued se_grad_arms runs off instance 13 onto a spare instance.
#
#   usage: move_tasks.sh <dest-instance> <run_id>:<dest-lane> [<run_id>:<lane> ...]
#   e.g.:  move_tasks.sh 16 se_flat_s36:0 se_het_s29:1
#
# KEEP THE PRIMARY BALANCED. The registered primary is occupancy(hetero) -
# occupancy(flat), exact one-sided over C(8,4)=70. Splitting those eight runs
# across machines is only acceptable if every machine carries EQUAL numbers of
# hetero and flat: then the device is a nuisance factor orthogonal to the
# contrast rather than confounded with it, and label permutation stays
# meaningful. An instance holding 2 hetero and 0 flat would put a device
# difference directly inside the primary, which a 4-vs-4 permutation test
# cannot separate from the effect. This script REFUSES an unbalanced move.
#
# Order is deliberate: the destination is gated BEFORE anything is removed
# from 13, so a failed gate leaves the tasks queued where they were.
# Commands are extracted from generated/lane_train_*.cmds, never retyped.
# ---------------------------------------------------------------------------
set -uo pipefail
DEST="${1:?usage: move_tasks.sh <dest-instance> <run_id>:<lane> ...}"; shift
[ $# -gt 0 ] || { echo "usage: move_tasks.sh <dest-instance> <run_id>:<lane> ..." >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
GEN="$HERE/generated"
SRC=13

# ---- balance check on what is being moved --------------------------------
nh=0; nf=0
for spec in "$@"; do
  case "${spec%%:*}" in
    se_het_*)  nh=$((nh+1)) ;;
    se_flat_*) nf=$((nf+1)) ;;
    *) echo "REFUSE: $spec is not a hetero/flat run; this script is for the primary pair" >&2; exit 2 ;;
  esac
done
if [ "$nh" -ne "$nf" ]; then
  echo "REFUSE: unbalanced move -- $nh hetero vs $nf flat onto instance $DEST." >&2
  echo "        Every machine must carry equal hetero and flat, or the device" >&2
  echo "        difference lands inside the registered primary contrast." >&2
  exit 1
fi
echo "balance OK: $nh hetero + $nf flat -> instance $DEST"

# ---- gate the destination BEFORE removing anything -----------------------
echo
echo "== gating destination $DEST =="
if ! bash "$HERE/submit_gate.sh" "$DEST" >/tmp/movegate.$$ 2>&1; then
  sed 's/^/  /' /tmp/movegate.$$; rm -f /tmp/movegate.$$
  echo "REFUSE: destination gate FAILED -- nothing removed from instance $SRC" >&2
  exit 1
fi
grep -E "GATE PASS|PASS   --distractor" /tmp/movegate.$$ | sed 's/^/  /'; rm -f /tmp/movegate.$$

# ---- extract the exact generated commands --------------------------------
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
for spec in "$@"; do
  rid="${spec%%:*}"
  line="$(grep -h "DV3_TASK_NAME=$rid " "$GEN"/lane_train_*.cmds)"
  [ "$(grep -c . <<<"$line")" -eq 1 ] || {
    echo "REFUSE: expected exactly 1 generated command for $rid" >&2; exit 1; }
  printf '%s\n' "$line" > "$STAGE/$rid.cmds"
done

# ---- remove from 13 (only tasks not yet started) --------------------------
echo
echo "== removing from instance $SRC =="
for spec in "$@"; do
  rid="${spec%%:*}"
  out="$(dv3_ssh_dv3 "$SRC" "source /root/.dreamer_vast_env; source /root/dreamer_instance_helpers.sh
    for l in 0 1 2 3; do
      q=\$(ls -d \"\$RUNROOT/_queue_control/lane_\$l\"/queue_* 2>/dev/null | head -1)
      [ -n \"\$q\" ] || continue
      idx=\$(grep -n \"DV3_TASK_NAME=$rid \" \"\$q/tasks.txt\" | cut -d: -f1 | head -1)
      [ -n \"\$idx\" ] || continue
      run=\$(cat \"\$q/running_index\" 2>/dev/null || echo 1)
      if [ \"\$idx\" -le \"\$run\" ]; then echo \"BUSY $rid lane \$l idx \$idx (running_index \$run)\"; exit 0; fi
      dv3_remove_tasks \$l \$idx >/dev/null && echo \"REMOVED $rid from lane \$l idx \$idx\"
      exit 0
    done
    echo \"NOTFOUND $rid\"" 2>/dev/null | grep -E '^(REMOVED|BUSY|NOTFOUND)')"
  echo "  ${out:-NO RESPONSE for $rid}"
  case "$out" in REMOVED*) ;; *) echo "REFUSE: $rid was not safely removed -- not submitting it" >&2; exit 1;; esac
done

# ---- queue on the destination --------------------------------------------
echo
echo "== queueing on instance $DEST =="
dv3_require_instance "$DEST"
REMOTE_ROOT="$(dv3_ssh_dv3 "$DEST" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: no RUNROOT on $DEST" >&2; exit 1; }
REMOTE="$REMOTE_ROOT/_waves/se_grad_arms"
dv3_ssh_dv3 "$DEST" "mkdir -p \"$REMOTE\""
for spec in "$@"; do
  rid="${spec%%:*}"; lane="${spec##*:}"
  rsync -az --checksum -e "ssh -p $(dv3_port "$DEST") ${DV3_SSH_OPTS[*]}" \
    "$STAGE/$rid.cmds" "root@$(dv3_ip "$DEST"):$REMOTE/$rid.cmds"
  echo "-- $rid -> instance $DEST lane $lane --"
  dv3_ssh_dv3 "$DEST" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/$rid.cmds\" $lane"
done
echo
echo "moved: $* -> instance $DEST"
