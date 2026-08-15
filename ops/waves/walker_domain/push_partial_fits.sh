#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Push PARTIAL walker WM fits to the instance that will finish them. (2026-08-15)
#
#   push_partial_fits.sh <instance> <wm_run> [wm_run...]
#
# The six fq1s0 seeds 7-12 stopped mid-fit when instance 6's GPU1 died, at
# 33-62% of 500k updates. Their checkpoints survive on RCC, so shipping them to
# whichever box picks the cell up lets axis1.sbatch RESUME
# ("resuming partial offline WM from $CKPT") instead of refitting from zero --
# roughly 12 GPU-hours saved across the six.
#
# Safe by construction: offline_fit_complete requires ckpt >= UPDATES OR
# progress >= UPDATES, and every one of these is far below 500k, so none can be
# mistaken for finished. The wave's done_when still gates on the fit's own
# counters plus update==total_updates, so a resume that stops short cannot pass
# as complete.
#
# A file, not an inline ssh string: building the rsync option array through
# nested ssh quoting mangles it (§0.9, hit 2026-08-15).
# ---------------------------------------------------------------------------
set -uo pipefail

N="${1:?usage: push_partial_fits.sh <instance> <wm_run>...}"; shift
[ $# -gt 0 ] || { echo "ERROR: name at least one wm_run" >&2; exit 2; }

DV3OPS_ROOT="${DV3OPS_ROOT:-$HOME/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: cannot read RUNROOT on instance $N" >&2; exit 1; }

src=()
for w in "$@"; do
  d="$RUNROOT/$w"
  [ -d "$d" ] || { echo "MISSING fit: $d" >&2; exit 1; }
  # Refuse to ship anything that already claims completion: that would be a
  # finished fit being re-shipped, which is a different operation entirely.
  prog="$(awk -F= '$1=="update"{print $2+0}' "$d/OFFLINE_FIT_PROGRESS" 2>/dev/null || echo 0)"
  if [ "${prog:-0}" -ge 500000 ]; then
    echo "REFUSING $w: progress=$prog is already complete, not a partial" >&2
    exit 1
  fi
  echo "  $w  progress=${prog:-0}/500000"
  src+=("$d")
done

echo "[partial-fits] -> instance $N:$REMOTE_ROOT/  (${#src[@]} fits)"
rsync "${DV3_RSYNC_OPTS[@]}" \
  -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" \
  "${src[@]}" "root@$(dv3_ip "$N"):$REMOTE_ROOT/"
rc=$?
[ "$rc" -eq 0 ] || { echo "[partial-fits] rsync FAILED rc=$rc" >&2; exit "$rc"; }
echo "[partial-fits] done"
