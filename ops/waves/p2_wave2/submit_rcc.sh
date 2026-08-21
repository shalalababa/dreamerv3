#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# P2 Wave 2 -- fresh-cohort TM2 training, 32 cells.
# prereg/PREREG_p2_wave2_freshcohort_20260821.md (frozen 74bfeb7a)
#
# Grid VERBATIM from PREREG_w1_wave_20260809 with only the training seeds
# replaced: {cup,finger} x {e1,e4} x seeds 59-66 = 32 runs. The producer
# (scripts/tm2r3.sbatch, TM2R3_STAGE=train) is unchanged apart from review
# B5's widened seed guard, so every emitted command is the W1 command with a
# different seed.
#
# WHY A DRIP-FEED: the gpu QOS carries MaxSubmitJobsPU=12. Submitting 32 in
# one pass gets the surplus REJECTED, not deferred, while a naive loop reports
# success -- so this counts sbatch EXIT CODES and stops at DV3_MAX_SUBMIT.
# Driven by scripts/ops/rcc_dripfeed.sh, which tops the queue every interval.
#
# IDEMPOTENCY. Job names collide (the sbatch hardcodes --job-name=dv3_tm2r3),
# so a squeue name check cannot identify a cell. Two signals instead:
#   1. a realized counter -- ckpt_late.pt present means the cell trained;
#   2. a submit ledger -- cells this script successfully sbatch'd, so a cell
#      that is merely PENDING (no run dir yet) is never submitted twice.
# The ledger is submission bookkeeping ONLY. Completion is adjudicated by the
# read's own gates on realized checkpoints, never by this file or by the
# producer's TM2R3_TRAIN_DONE marker.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-/home/rickybao/projects/dreamerv3}"
cd "$REPO"
export PATH=/software/slurm-current-el8-x86_64/bin:$PATH

RUNROOT="${RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"
ROOT="$RUNROOT/tm2r3"
LEDGER="$ROOT/.wave2_submitted"
MAX="${DV3_MAX_SUBMIT:-32}"
mkdir -p "$ROOT"; touch "$LEDGER"

submitted=0; done_n=0; inflight=0; todo=0
for dom in cup finger; do
  for dose in e1 e4; do
    for seed in 59 60 61 62 63 64 65 66; do
      cell="${dom}_${dose}_seed${seed}"
      run="$ROOT/tm2r3_${cell}"
      if [ -f "$run/ckpt_late.pt" ]; then done_n=$((done_n+1)); continue; fi
      if grep -qx "$cell" "$LEDGER"; then inflight=$((inflight+1)); continue; fi
      todo=$((todo+1))
      [ "$submitted" -ge "$MAX" ] && continue
      out="$(TM2R3_STAGE=train TM2R3_DOM=$dom TM2R3_DOSE=$dose TM2R3_SEED=$seed \
             sbatch scripts/tm2r3.sbatch 2>&1)"
      if [ $? -eq 0 ] && printf '%s' "$out" | grep -q 'Submitted batch job'; then
        printf '%s\n' "$cell" >> "$LEDGER"
        submitted=$((submitted+1)); inflight=$((inflight+1)); todo=$((todo-1))
        echo "  submitted $cell -> $(printf '%s' "$out" | grep -o '[0-9]\+$')"
      else
        echo "  REFUSED $cell: $(printf '%s' "$out" | tail -1)" >&2
      fi
    done
  done
done

echo "wave2: trained=$done_n in-flight=$inflight not-yet-submitted=$todo (this pass: $submitted)"
[ "$todo" -eq 0 ] && echo "ALL CELLS ACCOUNTED FOR"
exit 0
