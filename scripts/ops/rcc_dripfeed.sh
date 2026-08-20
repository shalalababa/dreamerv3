#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Drip-feed a wave's cells onto RCC under the gpu QOS submit cap.
#
# WHY: `sacctmgr show qos gpu` carries MaxSubmitJobsPU = 12 -- a user may hold
# at most twelve gpu jobs QUEUED+RUNNING at once. A 16- or 32-cell wave cannot
# be submitted in one pass: the surplus is REJECTED, not deferred, and sbatch
# says so on stderr while the loop around it happily reports success. (That is
# how the first crowding pass "submitted 16" while 6 were refused.)
#
# This tops the queue up to CAP-1 every INTERVAL seconds until every cell in
# the submitter script has landed, then exits. Individual jobs are kept
# deliberately -- one cell per job -- rather than bundling several cells into
# one long job: the 08-07 realized-training discovery was a bundle silently
# truncating cells at the walltime boundary with nothing recorded.
#
# IDEMPOTENT by construction: the wave's own submit script already skips a
# cell whose run dir exists, and this adds a squeue name check so a cell that
# is merely PENDING is never submitted twice.
#
#   usage: rcc_dripfeed.sh <submit_script> [cap] [interval_s] [max_rounds]
# ---------------------------------------------------------------------------
set -uo pipefail
SUB="${1:?usage: rcc_dripfeed.sh <submit_script> [cap] [interval] [rounds]}"
CAP="${2:-12}"; INTERVAL="${3:-300}"; ROUNDS="${4:-288}"
export PATH=/software/slurm-current-el8-x86_64/bin:$PATH
[ -f "$SUB" ] || { echo "no such submit script: $SUB" >&2; exit 2; }

for r in $(seq 1 "$ROUNDS"); do
  inq=$(squeue -h -u "$USER" | wc -l)
  free=$(( CAP - 1 - inq ))
  if [ "$free" -le 0 ]; then
    echo "[$(date +%H:%M)] queue full ($inq/$CAP), waiting"; sleep "$INTERVAL"; continue
  fi
  # DV3_MAX_SUBMIT is read by the wave submit script: it stops after that many
  # successful sbatch calls, so we never overrun the cap mid-pass.
  out="$(DV3_MAX_SUBMIT="$free" bash "$SUB" 2>&1)"
  echo "$out" | grep -vE "^sbatch: (error: )?(Verify|Using|Partition|QOS-Flag|Account|Verification)" | tail -6
  if echo "$out" | grep -q "ALL CELLS ACCOUNTED FOR"; then
    echo "[$(date +%H:%M)] every cell submitted or already present -- drip-feed done"
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "[$(date +%H:%M)] drip-feed hit its round limit with cells still pending" >&2
exit 1
