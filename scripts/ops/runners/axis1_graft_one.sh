#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One axis1 graft: WM fit + adaptation in a single task.  (dv3ops P1, 2026-08-14)
#
# Repo-hosted replacement for /tmp/lewm_jp_axis1_skip_done.sh, which was written
# by heredoc at the top of every graft wave. Same behaviour, versioned.
#
#   axis1_graft_one.sh <run_id> <wm_run> <side> <seed> <donor> [replay_quad]
#
# Writes TWO run dirs:
#   $RUNROOT/$wm_run   the world-model fit   (UPDATES)
#   $RUNROOT/$run_id   the adaptation        (STEPS)
#
# The skip guard checks ADAPT_DONE *and* >=100 score lines, not the marker
# alone. Keep it that way: a resumed run that declares itself done with too few
# scores is precisely the silent truncation this project exists to catch, and a
# marker-only guard would make it permanent by refusing to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail

run_id="${1:?run_id}"
wm_run="${2:?wm_run}"
side="${3:?side}"
seed="${4:?seed}"
donor="${5:?donor}"
quad="${6:-pxq1m}"

source /root/.dreamer_vast_env 2>/dev/null || true
source "$REPO/scripts/env.sh"
cd "$REPO"

scores="$RUNROOT/$run_id/scores.jsonl"
if [ -f "$RUNROOT/$run_id/ADAPT_DONE" ] && [ -s "$scores" ] &&
   [ "$(wc -l < "$scores")" -ge 100 ]; then
  echo "SKIP done: $run_id"
  exit 0
fi

# Fail before burning GPU time rather than inside the driver.
[ -d "$RUNROOT/$donor/ckpt" ] || {
  echo "ERROR: no donor checkpoint dir: $RUNROOT/$donor/ckpt" >&2; exit 2; }
# Controlled replay buffers are parent-directory units: axis1.sbatch:77 requires
# manifest.json BESIDE the side dir, and a side-only sync leaves it missing.
[ -f "$RUNROOT/axis1_finger/$quad/manifest.json" ] || {
  echo "ERROR: no manifest.json beside $RUNROOT/axis1_finger/$quad/side${side}" >&2
  echo "       (replay was synced as a side dir without its parent manifest)" >&2
  exit 2; }

DREAMER_DISABLE_GPUFLAGS=1 \
RUN_ID="$run_id" \
WM_RUN="$wm_run" \
TASK=dmc_finger_turn_hard \
SEED="$seed" \
REPLAY="$RUNROOT/axis1_finger/$quad/side${side}" \
UPDATES=500000 \
STEPS=1.25e5 \
AXIS1_EXPL_MODE=task \
AXIS1_BASE_CONFIG=pixel_wm \
AXIS1_INIT_WM="$donor" \
AXIS1_ADAPT_CONFIG=frozen_readout \
bash scripts/axis1.sbatch
