#!/bin/bash
# ---------------------------------------------------------------------------
# R3 consumer-competence factorial: pilot training + corrected-instrument
# labeling driver (prereg/PREREG_r3_competence_20260725.md — freeze-commit
# the prereg + this driver + analysis/r3_read.py BEFORE the first pilot).
#
# Grid: {cup, finger[, reacher]} x {e1, e4} x seeds 31-38, each labeled at
# two maturities (early = 2.5e4-step snapshot, late = 1e5 final). Seeds
# 31-38 are disjoint from every prior D1 wave (21-23 shift, 1-3 gate
# pilots) by design. Dose e1 = no distractor; e4 = d0_dose3 (dim 32,
# scale 3.0) — the Gate-D1 dose levels. Base arm only (no shift arms).
#
# Stages (in order; `labels` refuses to start before the dosed smoke):
#   ./scripts/r3_local.sh pilots   [DOMS="cup finger"]   # train, 2-phase
#   ./scripts/r3_local.sh smoke                          # e1 AND e4 smokes
#   ./scripts/r3_local.sh labels   [DOMS=...]            # 2 mats x runs
# Then ONE read:
#   python -m analysis.r3_read --labels $R3_ROOT/r3_labels --output ...
#
# Reacher is a conditional-GO third domain: run `pilots`/`labels` with
# DOMS="reacher" ONLY as a complete cohort (the read enforces
# all-or-nothing). Env: activated dv3 env; R3_ROOT (default ~/r3_local).
# All labeling dials pinned by the prereg — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${R3_ROOT:-$HOME/r3_local}"
LABELS="$ROOT/r3_labels"
mkdir -p "$ROOT" "$LABELS"
STAGE="${1:?usage: r3_local.sh pilots|smoke|labels}"
DOMS="${DOMS:-cup finger}"
SEEDS="31 32 33 34 35 36 37 38"
DOSES="e1 e4"
EARLY_STEPS=2.5e4
LATE_STEPS=1e5

DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 0 --ref_stride 5 --oracle_all"

task_of() { case "$1" in cup) echo dmc_cup_catch;; finger) echo dmc_finger_turn_hard;; reacher) echo dmc_reacher_hard;; esac; }
cfg_of()  { case "$1" in e1) echo "dmc_proprio d0_probe";; e4) echo "dmc_proprio d0_probe d0_dose3";; esac; }

python -c "import jax; print('jax devices:', jax.devices())"

ckpt_step() {  # ckpt-dir -> realized step of the latest snapshot
  python - "$1" <<'PY' || echo "?"
import glob, pickle, sys, os
d = sys.argv[1]
subs = [p for p in glob.glob(os.path.join(d, '*')) if os.path.isdir(p)]
latest = max(subs, key=os.path.getmtime)
try:
  with open(os.path.join(latest, 'step.pkl'), 'rb') as f:
    print(pickle.load(f))
except Exception:
  print(os.path.basename(latest))
PY
}

train_run() {  # dom dose seed
  local RUN="r3_${1}_${2}_seed${3}" LOG="$ROOT/r3_${1}_${2}_seed${3}"
  if [ -f "$LOG/TRAINING_DONE" ]; then echo "skip $RUN (done)"; return; fi
  mkdir -p "$LOG"
  if [ ! -d "$LOG/ckpt_early" ]; then
    # save_every is CLOCK-based (seconds) and the loop does NOT save at
    # exit; a 60 s tick pins the early snapshot within ~1-2K steps of
    # EARLY_STEPS (realized step recorded, disclosed in the artifact).
    echo "[$(date)] $RUN phase A (to $EARLY_STEPS)"
    python dreamerv3/main.py --logdir "$LOG" \
        --configs $(cfg_of "$2") --task "$(task_of "$1")" --seed "$3" \
        --env.dmc.render False --run.steps "$EARLY_STEPS" \
        --run.save_every 60
    cp -r "$LOG/ckpt" "$LOG/ckpt_early"   # early snapshot, frozen
    ckpt_step "$LOG/ckpt_early" > "$LOG/ckpt_early/STEP"
    echo "  early snapshot realized step: $(cat "$LOG/ckpt_early/STEP")"
  fi
  # Phase B resumes FROM the early snapshot (maturity axis = the same
  # trajectory continued); 60 s ticks also pin the late endpoint.
  echo "[$(date)] $RUN phase B (resume to $LATE_STEPS)"
  python dreamerv3/main.py --logdir "$LOG" \
      --configs $(cfg_of "$2") --task "$(task_of "$1")" --seed "$3" \
      --env.dmc.render False --run.steps "$LATE_STEPS" \
      --run.save_every 60
  ckpt_step "$LOG/ckpt" > "$LOG/ckpt/STEP"
  echo "  late ckpt realized step: $(cat "$LOG/ckpt/STEP")"
  touch "$LOG/TRAINING_DONE"
}

label_run() {  # dom dose seed mat extra-args...
  local dom="$1" dose="$2" seed="$3" mat="$4"; shift 4
  local RUN="r3_${dom}_${dose}_seed${seed}" LOG="$ROOT/r3_${dom}_${dose}_seed${seed}"
  local OUT="$LABELS/${RUN}_${mat}.npz" CKPT=""
  [ "$mat" = early ] && CKPT="--checkpoint $LOG/ckpt_early"
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; return; }
  [ -f "$LOG/TRAINING_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] label $RUN $mat"
  python -m d0.oracle_labels --run_logdir "$LOG" $CKPT \
      --output "$OUT" $DIALS "$@"
}

if [ "$STAGE" = pilots ]; then
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    train_run "$dom" "$dose" "$seed"
  done; done; done
  echo "[$(date)] pilots done for: $DOMS"

elif [ "$STAGE" = smoke ]; then
  # Registered requirement (24-Jul audit): dosed labeling must smoke ON a
  # dosed env before labels — the OU/calibration snapshot path has never
  # touched a real dosed agent. Both doses, both maturities, 5 states.
  # SMOKE_DOM (default cup) selects the smoke domain: a NEW domain must
  # smoke on itself before its labels (reacher wave: SMOKE_DOM=reacher,
  # PREREG_r3_reacher_20260729.md; the reacher reader asserts the four
  # reacher smoke files exist).
  SMOKE_DOM="${SMOKE_DOM:-cup}"
  for dose in $DOSES; do for mat in early late; do
    LOG="$ROOT/r3_${SMOKE_DOM}_${dose}_seed31"
    [ -f "$LOG/TRAINING_DONE" ] || { echo "train r3_${SMOKE_DOM}_${dose}_seed31 first"; exit 1; }
    CKPT=""; [ "$mat" = early ] && CKPT="--checkpoint $LOG/ckpt_early"
    python -m d0.oracle_labels --run_logdir "$LOG" $CKPT \
        --output "$LABELS/r3_${SMOKE_DOM}_${dose}_seed31_${mat}_smoke.npz" \
        --states 5 --horizon 100 --label_every 25 --actions 8 \
        --rollouts 16 --seed 0 --ref_stride 5 --oracle_all
  done; done
  touch "$LABELS/SMOKE_OK"
  echo "[$(date)] smoke OK ($SMOKE_DOM e1+e4 x early+late; inspect stdout before labels)"

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  # Per-domain smoke guard (PREREG_r3_reacher_20260729): a stale SMOKE_OK
  # from the executed cup wave must not authorize labels for a NEW domain
  # — any domain beyond cup/finger needs its own four smoke files first.
  for dom in $DOMS; do case "$dom" in cup|finger) ;; *)
    for dose in $DOSES; do for mat in early late; do
      [ -f "$LABELS/r3_${dom}_${dose}_seed31_${mat}_smoke.npz" ] || {
        echo "run SMOKE_DOM=$dom smoke first"; exit 1; }
    done; done ;; esac; done
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    for mat in early late; do
      label_run "$dom" "$dose" "$seed" "$mat"
    done
  done; done; done
  echo "[$(date)] labels done for: $DOMS"

else
  echo "unknown stage $STAGE"; exit 1
fi
