#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the four ungated2 runs on the two 2-lane instances, one per lane.
#
#   instance 14: lane 0 se_ungate2_s32   lane 1 se_ungate2_s33
#   instance 15: lane 0 se_ungate2_s34   lane 1 se_ungate2_s35
#
# WHY THESE FOUR, AND ONLY THESE FOUR. The registered primary is
# occupancy(hetero) - occupancy(flat), exact one-sided over C(8,4) = 70. Those
# eight runs stay together on instance 13. Moving any hetero or flat run to a
# different machine would put a device difference INSIDE the primary contrast,
# where a 4-vs-4 permutation test cannot tell it apart from the effect -- the
# 1-Aug nondeterminism rule is exactly about this. ungated2 is the dose
# REFERENCE: it appears only in secondaries the prereg labels DESCRIPTIVE
# (hetero - ungated2, flat - ungated2, and the cross-wave check against the
# prior ungated arm's 0.560, which is already cross-wave). So this split moves
# the device boundary onto the descriptive axis and keeps the primary clean.
#
# REGISTRATION NOTE, for the owning chat rather than ops: prereg §3 says all
# three arms run "on the SAME site". These four now run on different machines
# from hetero/flat -- same GPU model (RTX 5060 Ti, verified), different
# physical hosts. The primary is untouched; the two dose secondaries become
# cross-instance and should be disclosed as such, or ungated2 re-run on 13 if
# same-site is wanted for them.
#
# Commands are EXTRACTED from generated/lane_train_*.cmds, never retyped, so
# they are byte-identical to what `dv3ops gen` produced from the spec and the
# spec stays the single source of truth for all 12 runs. `dv3ops check --wave
# se_grad_arms` and the bundle continue to cover all 12 regardless of where
# each executed; DV3_TASK_NAME is the run_id, so logs land as
# se_ungate2_s*.out on whichever instance ran them.
#
#   usage: submit_ungated2_split.sh          (does 14 and 15)
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"

GEN="$HERE/generated"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# run_id -> "instance lane"
place () {
  case "$1" in
    se_ungate2_s32) echo "14 0" ;;
    se_ungate2_s33) echo "14 1" ;;
    se_ungate2_s34) echo "15 0" ;;
    se_ungate2_s35) echo "15 1" ;;
    *) return 1 ;;
  esac
}

fail=0
for rid in se_ungate2_s32 se_ungate2_s33 se_ungate2_s34 se_ungate2_s35; do
  line="$(grep -h "DV3_TASK_NAME=$rid " "$GEN"/lane_train_*.cmds)"
  n=$(grep -c . <<<"$line")
  if [ "$n" -ne 1 ]; then
    echo "ERROR: expected exactly 1 generated command for $rid, found $n" >&2
    fail=1; continue
  fi
  read -r inst lane <<<"$(place "$rid")"
  printf '%s\n' "$line" > "$STAGE/$rid.cmds"
  echo "$rid -> instance $inst lane $lane"
done
[ "$fail" -eq 0 ] || { echo "ABORT: command extraction failed" >&2; exit 1; }

for rid in se_ungate2_s32 se_ungate2_s33 se_ungate2_s34 se_ungate2_s35; do
  read -r inst lane <<<"$(place "$rid")"
  dv3_require_instance "$inst"
  REMOTE_ROOT="$(dv3_ssh_dv3 "$inst" 'printf %s "$RUNROOT"')"
  [ -n "$REMOTE_ROOT" ] || { echo "ERROR: no RUNROOT on instance $inst" >&2; exit 1; }
  REMOTE="$REMOTE_ROOT/_waves/se_grad_arms"
  dv3_ssh_dv3 "$inst" "mkdir -p \"$REMOTE\""
  rsync -az --checksum -e "ssh -p $(dv3_port "$inst") ${DV3_SSH_OPTS[*]}" \
    "$STAGE/$rid.cmds" "root@$(dv3_ip "$inst"):$REMOTE/$rid.cmds"
  echo "-- $rid -> instance $inst lane $lane --"
  dv3_ssh_dv3 "$inst" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/$rid.cmds\" $lane"
done

echo
echo "ungated2 placed: 14 = s32/s33, 15 = s34/s35. hetero+flat remain on 13."
