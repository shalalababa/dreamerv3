#!/bin/bash
#SBATCH --job-name=dv3_tm2r3
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# TM2-R3 opportunity/competence cross-family replication
# (prereg/PREREG_tm2_competence_20260730.md — freeze-commit the prereg +
# probing/tdmpc2_oracle_labels.py + probing/tdmpc2_r3_train.py +
# analysis/tm2_r3_read.py + the tdmpc2_compat dose patch BEFORE any run).
#
# Grid: {cup, finger} x {e1, e4} x seeds 51-58 (disjoint from every prior
# cohort), each run labeled at two maturities (early = ckpt_early.pt
# ~2.5e4 realized steps, late = ckpt_late.pt at 1e5) = 32 fresh ONLINE
# runs, 64 oracle_all label passes. e1 = no distractor; e4 = d0_dose3
# (dim 32, scale 3.0) composed exactly as DreamerV3 composes it.
#
# Stages (TM2R3_STAGE; ordering enforced by marker files; `labels`
# refuses to start before BOTH smoke gates):
#   train    one run/job:   TM2R3_DOM, TM2R3_DOSE, TM2R3_SEED
#   apismoke once:          random-init API smoke (no checkpoint, no
#                           outcome) -> labels/tm2r3_apismoke_smoke.npz
#   smoke    once:          4 dosed real smokes, {cup,finger} x {e1,e4}
#                           on seed-51 LATE ckpts, 5 states each
#   labels   one pass/job:  TM2R3_DOM, TM2R3_DOSE, TM2R3_SEED, TM2R3_MAT
# Then ONE read:
#   python -m analysis.tm2_r3_read --labels $TM2R3_ROOT/labels --output ...
#
# Submit-all examples (login node, clean shell — module-leakage gotcha):
#   for d in cup finger; do for e in e1 e4; do for k in $(seq 51 58); do
#     TM2R3_STAGE=train TM2R3_DOM=$d TM2R3_DOSE=$e TM2R3_SEED=$k \
#       sbatch scripts/tm2r3.sbatch
#   done; done; done
#   TM2R3_STAGE=apismoke sbatch scripts/tm2r3.sbatch
#   TM2R3_STAGE=smoke sbatch scripts/tm2r3.sbatch
#   for d in cup finger; do for e in e1 e4; do for k in $(seq 51 58); do
#     for m in early late; do
#       TM2R3_STAGE=labels TM2R3_DOM=$d TM2R3_DOSE=$e TM2R3_SEED=$k \
#         TM2R3_MAT=$m sbatch scripts/tm2r3.sbatch
#   done; done; done; done
#
# All labeling dials pinned by the prereg — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"

TM2_CONDA_ENV="${TM2_CONDA_ENV:-/scratch/midway3/$USER/conda_envs/tdmpc2}"
TDMPC2_CHECKOUT="${TDMPC2_CHECKOUT:-/scratch/midway3/$USER/tdmpc2}"
export TDMPC2_ROOT="$TDMPC2_CHECKOUT/tdmpc2"
CONDA_ENV="$TM2_CONDA_ENV" activate_env
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Env-setup existence check (registered ordering step): pinned checkout.
TM2R3_PIN="e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
[ -d "$TDMPC2_ROOT" ] || { echo "ERROR: no TDMPC2_ROOT at $TDMPC2_ROOT (run scripts/tdmpc2_env_setup.sh)"; exit 2; }
HEAD="$(git -C "$TDMPC2_CHECKOUT" rev-parse HEAD)"
[ "$HEAD" = "$TM2R3_PIN" ] || { echo "ERROR: tdmpc2 checkout $HEAD != pin $TM2R3_PIN"; exit 2; }

TM2R3_ROOT="${TM2R3_ROOT:-$RUNROOT/tm2r3}"
LABELS="$TM2R3_ROOT/labels"
mkdir -p "$TM2R3_ROOT" "$LABELS"
STAGE="${TM2R3_STAGE:?train|apismoke|smoke|labels}"

DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --seed 0 --ref_stride 5 --oracle_all"
task_of() { case "$1" in cup) echo dmc_cup_catch;; finger) echo dmc_finger_turn_hard;; *) echo "ERROR: dom $1" >&2; exit 2;; esac; }
# Registered-grid validation: a typo'd cell would train/label a stray
# run whose file then hard-blocks the ONE read (unregistered-seed
# assert) until manually removed — refuse at submission instead.
check_cell() {
  case "${TM2R3_DOM:-}" in cup|finger) ;; *) echo "ERROR: TM2R3_DOM '${TM2R3_DOM:-}' not in {cup,finger}"; exit 2;; esac
  case "${TM2R3_DOSE:-}" in e1|e4) ;; *) echo "ERROR: TM2R3_DOSE '${TM2R3_DOSE:-}' not in {e1,e4}"; exit 2;; esac
  case "${TM2R3_SEED:-}" in 5[1-8]) ;; *) echo "ERROR: TM2R3_SEED '${TM2R3_SEED:-}' not in 51..58 (registered grid)"; exit 2;; esac
}

if [ "$STAGE" = train ]; then
  : "${TM2R3_DOM:?cup|finger}"; : "${TM2R3_DOSE:?e1|e4}"; : "${TM2R3_SEED:?51..58}"
  check_cell
  RUN="tm2r3_${TM2R3_DOM}_${TM2R3_DOSE}_seed${TM2R3_SEED}"
  LOG="$TM2R3_ROOT/$RUN"
  if [ -f "$LOG/TM2R3_TRAIN_DONE" ]; then echo "skip $RUN (done)"; exit 0; fi
  echo "[$(date)] train $RUN"
  python -m probing.tdmpc2_r3_train --logdir "$LOG" \
      --task "$(task_of "$TM2R3_DOM")" --dose "$TM2R3_DOSE" \
      --seed "$TM2R3_SEED" --steps 100000 --early_steps 25000
  echo "[$(date)] train $RUN rc=0 status=DONE"

elif [ "$STAGE" = apismoke ]; then
  # Registered smoke gate (i): random-init model, no checkpoint touched,
  # no outcome revealed (PREREG_tdmpc2_f2 precedent), on the DOSED env —
  # exercises the OU snapshot path + planner + torch-RNG CRN plumbing.
  OUT="$LABELS/tm2r3_apismoke_smoke.npz"
  [ -f "$OUT" ] && { echo "skip apismoke (exists)"; exit 0; }
  python -m probing.tdmpc2_oracle_labels --smoke_random_init \
      --task dmc_cup_catch --dose e4 \
      --states 2 --horizon 10 --label_every 25 --actions 8 --seed 0 \
      --ref_stride 5 --oracle_all --output "$OUT"
  echo "[$(date)] apismoke OK -> $OUT"

elif [ "$STAGE" = smoke ]; then
  # Registered smoke gate (ii): 4 dosed REAL smokes across {dom}x{dose}
  # on the seed-51 LATE checkpoints, 5 states each (the reader asserts
  # these four files + the apismoke exist).
  [ -f "$LABELS/tm2r3_apismoke_smoke.npz" ] || { echo "run apismoke first"; exit 1; }
  for dom in cup finger; do for dose in e1 e4; do
    RUN="tm2r3_${dom}_${dose}_seed51"
    LOG="$TM2R3_ROOT/$RUN"
    OUT="$LABELS/${RUN}_late_smoke.npz"
    [ -f "$LOG/TM2R3_TRAIN_DONE" ] || { echo "train $RUN first"; exit 1; }
    [ -f "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    python -m probing.tdmpc2_oracle_labels \
        --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
        --states 5 --horizon 100 --label_every 25 --actions 8 --seed 0 \
        --ref_stride 5 --oracle_all --output "$OUT"
  done; done
  touch "$LABELS/SMOKE_OK"
  echo "[$(date)] smoke OK (cup+finger x e1+e4 late; inspect stdout before labels)"

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  [ -f "$LABELS/tm2r3_apismoke_smoke.npz" ] || { echo "run apismoke first"; exit 1; }
  : "${TM2R3_DOM:?cup|finger}"; : "${TM2R3_DOSE:?e1|e4}"; : "${TM2R3_SEED:?51..58}"; : "${TM2R3_MAT:?early|late}"
  check_cell
  case "$TM2R3_MAT" in early|late) ;; *) echo "ERROR: TM2R3_MAT '$TM2R3_MAT' not in {early,late}"; exit 2;; esac
  RUN="tm2r3_${TM2R3_DOM}_${TM2R3_DOSE}_seed${TM2R3_SEED}"
  LOG="$TM2R3_ROOT/$RUN"
  OUT="$LABELS/${RUN}_${TM2R3_MAT}.npz"
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; exit 0; }
  [ -f "$LOG/TM2R3_TRAIN_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] label $RUN $TM2R3_MAT"
  python -m probing.tdmpc2_oracle_labels \
      --run_logdir "$LOG" --checkpoint "$LOG/ckpt_${TM2R3_MAT}.pt" \
      $DIALS --output "$OUT"
  echo "[$(date)] label $RUN $TM2R3_MAT rc=0 status=DONE"

else
  echo "unknown stage $STAGE"; exit 1
fi
