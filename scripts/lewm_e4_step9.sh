#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LeWM step 9 -- the E4 pass over the jp grafts. (PREREG_lewm_20260812 §9.)
#
# The registered command, verbatim:
#
#   PROBESET=$RUNROOT/e4_probesets/fingerpx_v1 GLOB='ax1wm_finger_jppx*' \
#   COLLATE=$RUNROOT/e4_fingerpx_v1_jp.csv sbatch $REPO/scripts/e4_measure.sbatch
#
# Two routing additions, neither of which touches the registered command:
#
#   --constraint=rtx6000  These are PIXEL world models.
#       cudnnConvolutionForward fails on Midway3's V100s in this JAX build
#       (`<unknown cudnn status: 5003>`), and the 12 fpx probe jobs of
#       2026-08-15 separate perfectly by architecture rather than by node:
#       midway3-0280 (v100) 9/9 FAILED, midway3-0282 (rtx6000) 3/3 clean.
#       So this leg stays on RCC as registered; it just may not land on a
#       Volta. The generated default (--constraint=v100) is right for the
#       proprio panels and wrong here, which is why this is explicit.
#
#   a single job          e4_measure.sbatch sweeps the whole GLOB itself and
#       is resumable (it skips any run with an existing summary.json), so the
#       entire 16-run panel is measured by ONE job on ONE GPU -- device-uniform
#       by construction, which is the property the panel needs.
#
# A file, not an inline ssh string: GLOB carries a `*` and ssh flattens
# arguments (§0.9).
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-$HOME/projects/dreamerv3}"
RUNROOT="${RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"

[ -d "$RUNROOT/e4_probesets/fingerpx_v1" ] || {
  echo "ERROR: probeset missing: $RUNROOT/e4_probesets/fingerpx_v1" >&2; exit 2; }
[ -f "$RUNROOT/e4_probesets/fingerpx_v1/FROZEN" ] || {
  echo "ERROR: probeset is not FROZEN" >&2; exit 2; }

n=$(ls -d "$RUNROOT"/ax1wm_finger_jppxq1ms[01]_seed[1-8] 2>/dev/null | wc -l)
[ "$n" -eq 16 ] || { echo "ERROR: expected 16 jp fits under GLOB, found $n" >&2; exit 2; }
echo "jp fits present: $n/16"

# Refuse to clobber a completed collate; the reader pins its sha.
if [ -f "$RUNROOT/e4_fingerpx_v1_jp.csv" ]; then
  echo "NOTE: $RUNROOT/e4_fingerpx_v1_jp.csv already exists -- per-run measures"
  echo "      are skipped individually, and the collate is rebuilt from them."
fi

sbatch --constraint=rtx6000 \
  --export=ALL,REPO="$REPO",RUNROOT="$RUNROOT",PROBESET="$RUNROOT/e4_probesets/fingerpx_v1",GLOB='ax1wm_finger_jppx*',COLLATE="$RUNROOT/e4_fingerpx_v1_jp.csv" \
  "$REPO/scripts/e4_measure.sbatch"
