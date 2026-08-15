#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Push the seven COMPLETED graft WM fits to an instance so the adapt half can
# be retried there.  (2026-08-15)
#
# These fits are stage OUTPUTS, not declared inputs, so push_donors.sh does not
# carry them -- but a retry of the adapt half needs them present, or
# axis1.sbatch would re-run a 500k-update fit that already succeeded elsewhere.
# With them in place `offline_fit_complete` is satisfied (ckpt 500000 AND
# progress 500000) and stage 1 is skipped.
#
#   push_fits_i8.sh <instance>
#
# Written as a FILE rather than an inline ssh string on purpose: building this
# by nested quoting through ssh mangled the rsync option array into a literal
# and rsync exited "syntax or usage error" (§0.9, hit again 2026-08-15).
# ---------------------------------------------------------------------------
set -uo pipefail

N="${1:?usage: push_fits_i8.sh <instance>}"
DV3OPS_ROOT="${DV3OPS_ROOT:-$HOME/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: cannot read RUNROOT on instance $N" >&2; exit 1; }

FITS=(
  ax1wm_finger_jppxq1ms0_seed4 ax1wm_finger_jppxq1ms0_seed5
  ax1wm_finger_jppxq1ms0_seed6 ax1wm_finger_jppxq1ms0_seed8
  ax1wm_finger_jppxq1ms1_seed4 ax1wm_finger_jppxq1ms1_seed5
  ax1wm_finger_jppxq1ms1_seed8
)

# Fail before transferring rather than half way through.
src=()
for f in "${FITS[@]}"; do
  [ -d "$RUNROOT/$f" ] || { echo "MISSING fit: $RUNROOT/$f" >&2; exit 1; }
  src+=("$RUNROOT/$f")
done

echo "[fits] $RUNROOT -> instance $N:$REMOTE_ROOT/  (${#src[@]} fits)"
rsync "${DV3_RSYNC_OPTS[@]}" \
  -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" \
  "${src[@]}" "root@$(dv3_ip "$N"):$REMOTE_ROOT/"
rc=$?
[ "$rc" -eq 0 ] || { echo "[fits] rsync FAILED rc=$rc" >&2; exit "$rc"; }
echo "[fits] done"
