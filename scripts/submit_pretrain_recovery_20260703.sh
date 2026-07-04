#!/bin/bash
# ---------------------------------------------------------------------------
# One-off Phase 4 recovery submissions for the Jul 3 partial pretrain state.
#
# This script intentionally does NOT change the main sweep in submit_all.sh.
# It writes explicit runlists under $RUNROOT/_submit_runlists/ and submits them
# through pretrain_bundle.sbatch with per-bundle walltimes.
#
# Usage:
#   ./scripts/submit_pretrain_recovery_20260703.sh cancel-bundle12
#   DRYRUN=1 ./scripts/submit_pretrain_recovery_20260703.sh immediate
#   ./scripts/submit_pretrain_recovery_20260703.sh immediate
#   DRYRUN=1 ./scripts/submit_pretrain_recovery_20260703.sh later-after-inspect
#
# Immediate plan assumes bundle_012 (apt_finger_seed1-3) is canceled before
# submitting apt_finger_seed1/2 as single-child recovery jobs.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
source "$REPO/scripts/env.sh"

cmd="${1:-}"

slurm_user() {
  local user="${USER:-${LOGNAME:-}}"
  [ -n "$user" ] || return 1
  printf '%s\n' "$user"
}

queued_job_id_by_name() {  # queued_job_id_by_name <job_name>
  command -v squeue >/dev/null 2>&1 || return 1
  local user
  user="$(slurm_user)" || return 1
  squeue -h -u "$user" -o "%i %j" |
    awk -v name="$1" '$2 == name {print $1}'
}

require_bundle12_not_queued() {
  if queued_job_id_by_name pretrain_bundle_012 >/tmp/pretrain_bundle_012_jobs.$$ &&
     [ -s /tmp/pretrain_bundle_012_jobs.$$ ]; then
    echo "ERROR: pretrain_bundle_012 is still queued/running:"
    sed 's/^/  jobid /' /tmp/pretrain_bundle_012_jobs.$$
    rm -f /tmp/pretrain_bundle_012_jobs.$$
    echo "Cancel it first: $0 cancel-bundle12"
    exit 2
  fi
  rm -f /tmp/pretrain_bundle_012_jobs.$$
}

runlist_dir=""
make_runlist_dir() {
  runlist_dir="$RUNROOT/_submit_runlists/pretrain_recovery_20260703_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$runlist_dir"
  echo "Runlist dir: $runlist_dir"
}

write_runlist() {  # write_runlist <filename> <row>...
  local file="$1"; shift
  : > "$file"
  for row in "$@"; do
    printf '%s\n' "$row" >> "$file"
  done
}

submit_runlist() {  # submit_runlist <job_name> <time> <runlist>
  local job_name="$1"; local walltime="$2"; local runlist="$3"
  local cmd_args=(
    sbatch
    --account="$SLURM_ACCOUNT"
    --partition="$SLURM_PARTITION"
    --gres="$SLURM_GRES"
    --time="$walltime"
    --job-name="$job_name"
    --export="ALL,REPO=$REPO,RUN_ID=$job_name,RUNLIST=$runlist,STEPS=${STEPS:-5e5}"
    "$REPO/scripts/pretrain_bundle.sbatch"
  )

  echo
  echo "== $job_name ($walltime) =="
  sed 's/^/  /' "$runlist"
  if [ "${DRYRUN:-0}" = "1" ]; then
    printf '%q ' "${cmd_args[@]}"
    echo
  else
    "${cmd_args[@]}"
  fi
}

cancel_bundle12() {
  mapfile -t ids < <(queued_job_id_by_name pretrain_bundle_012 || true)
  if [ "${#ids[@]}" -eq 0 ]; then
    echo "No queued/running pretrain_bundle_012 job found."
    exit 0
  fi
  printf 'Canceling pretrain_bundle_012 job IDs: %s\n' "${ids[*]}"
  if [ "${DRYRUN:-0}" = "1" ]; then
    printf 'scancel %s\n' "${ids[*]}"
  else
    scancel "${ids[@]}"
  fi
}

submit_immediate() {
  require_bundle12_not_queued
  make_runlist_dir

  # Keep the almost-finished P2E cleanup first so it cannot be starved behind
  # slow APT resumes.
  write_runlist "$runlist_dir/immediate_001_p2e5_aptw23.tsv" \
    $'pretrain_p2e_walker_seed5\tdmc_walker_walk\texpl_p2e\t5\t5e5' \
    $'pretrain_apt_walker_seed2\tdmc_walker_walk\texpl_apt\t2\t5e5' \
    $'pretrain_apt_walker_seed3\tdmc_walker_walk\texpl_apt\t3\t5e5'

  write_runlist "$runlist_dir/immediate_002_aptw45.tsv" \
    $'pretrain_apt_walker_seed4\tdmc_walker_walk\texpl_apt\t4\t5e5' \
    $'pretrain_apt_walker_seed5\tdmc_walker_walk\texpl_apt\t5\t5e5'

  write_runlist "$runlist_dir/immediate_003_aptw1.tsv" \
    $'pretrain_apt_walker_seed1\tdmc_walker_walk\texpl_apt\t1\t5e5'

  write_runlist "$runlist_dir/immediate_004_aptf1.tsv" \
    $'pretrain_apt_finger_seed1\tdmc_finger_turn_hard\texpl_apt\t1\t5e5'

  write_runlist "$runlist_dir/immediate_005_aptf2.tsv" \
    $'pretrain_apt_finger_seed2\tdmc_finger_turn_hard\texpl_apt\t2\t5e5'

  submit_runlist pretrain_recover_001_p2e5_aptw23 32:00:00 \
    "$runlist_dir/immediate_001_p2e5_aptw23.tsv"
  submit_runlist pretrain_recover_002_aptw45 32:00:00 \
    "$runlist_dir/immediate_002_aptw45.tsv"
  submit_runlist pretrain_recover_003_aptw1 26:00:00 \
    "$runlist_dir/immediate_003_aptw1.tsv"
  submit_runlist pretrain_recover_004_aptf1 24:00:00 \
    "$runlist_dir/immediate_004_aptf1.tsv"
  submit_runlist pretrain_recover_005_aptf2 24:00:00 \
    "$runlist_dir/immediate_005_aptf2.tsv"
}

submit_later_after_inspect() {
  make_runlist_dir

  # Submit these only after the currently queued/running bundles have exited
  # or after you have verified they never touched these child RUN_IDs.
  write_runlist "$runlist_dir/later_001_aptf3.tsv" \
    $'pretrain_apt_finger_seed3\tdmc_finger_turn_hard\texpl_apt\t3\t5e5'

  write_runlist "$runlist_dir/later_002_aptc3.tsv" \
    $'pretrain_apt_cup_seed3\tdmc_cup_catch\texpl_apt\t3\t5e5'

  write_runlist "$runlist_dir/later_003_aptc2.tsv" \
    $'pretrain_apt_cup_seed2\tdmc_cup_catch\texpl_apt\t2\t5e5'

  write_runlist "$runlist_dir/later_004_aptc5.tsv" \
    $'pretrain_apt_cup_seed5\tdmc_cup_catch\texpl_apt\t5\t5e5'

  write_runlist "$runlist_dir/later_005_random_cup1.tsv" \
    $'pretrain_random_cup_seed1\tdmc_cup_catch\texpl_random\t1\t5e5'

  submit_runlist pretrain_recover_later_001_aptf3 24:00:00 \
    "$runlist_dir/later_001_aptf3.tsv"
  submit_runlist pretrain_recover_later_002_aptc3 24:00:00 \
    "$runlist_dir/later_002_aptc3.tsv"
  submit_runlist pretrain_recover_later_003_aptc2 16:00:00 \
    "$runlist_dir/later_003_aptc2.tsv"
  submit_runlist pretrain_recover_later_004_aptc5 16:00:00 \
    "$runlist_dir/later_004_aptc5.tsv"
  submit_runlist pretrain_recover_later_005_random_cup1 08:00:00 \
    "$runlist_dir/later_005_random_cup1.tsv"
}

case "$cmd" in
  cancel-bundle12)
    cancel_bundle12 ;;
  immediate)
    submit_immediate ;;
  later-after-inspect)
    submit_later_after_inspect ;;
  *)
    echo "usage: $0 {cancel-bundle12|immediate|later-after-inspect}"
    echo
    echo "Recommended:"
    echo "  $0 cancel-bundle12"
    echo "  DRYRUN=1 $0 immediate"
    echo "  $0 immediate"
    exit 1 ;;
esac
