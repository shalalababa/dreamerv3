#!/bin/bash
#SBATCH --job-name=dv3_se
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# NFI Scale Exhibit (SE wave) — PREREG_nfi_scale_exhibit_20260812.md §3.
# One job = one Stage-1 run: DreamerV3 + p2e disagreement objective
# (reward-free) on a dmc proprio task with the planted channels
# (N = Distractor OU, dim 8; D-family = planted.py duplicate eps-ladder;
# C = constant) appended. Reuses the exploration-pretraining launch path
# incl. the checkpoint-retention watcher (M4 needs 25/50/100% ckpts).
#
# Required env (sbatch --export):
#   RUN_ID      e.g. se_cheetah_seed0
#   TASK        e.g. dmc_cheetah_run
#   SEED        integer 10..17 (prereg v2.1; smoke used 0)
#   SOURCE_KEY      pinned: position (smoke 12 Aug)
#   BASESD_PLANTED  pinned: 0.0976 (source-key mean per-dim sd)
#   BASESD_N        pinned: 1.215  (overall proprio mean per-dim sd)
# Optional: STEPS (default from prereg FILL; smoke: 2e4), SMOKE=1
#   (smoke additionally prints the proprio key inventory + per-key sd
#   over 2000 random steps BEFORE training — fills SOURCE_KEY/BASESD),
#   GATE_KEY/GATE_INDEX/GATE_THRESHOLD (Stage 2 only).
#
# GOTCHAS: clean shell (module purge; --export=ALL segfault rule);
# submit from repo root. Wrapper streams are keyed to SEED via main.py's
# SeedSequence blocks (0xD0 distractor, 0x5E planted).
#
# COLLECT: per results-sync policy v2 — bundle_manifest.sh generate on
# the run root, rsync, verify, cp manifest ->
#   manifests/uncfield_se_<date>.sha256
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env 2>/dev/null || true
PYTHON="${CONDA_ENV:-}/bin/python"; [ -x "$PYTHON" ] || PYTHON="$(command -v python)"
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${SEED:?set SEED}"
STEPS="${STEPS:-5e5}"           # pinned (prereg v2.1); smoke used 2e4
SAVE_EVERY="${SAVE_EVERY:-300}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/TRAINING_DONE"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[$(date)] SMOKE: proprio inventory + per-key sd for $TASK"
  "$PYTHON" - <<PY
import numpy as np
import elements
from embodied.envs import dmc
task = "$TASK".replace("dmc_", "")
env = dmc.DMC(task, image=False)
keys = [k for k, v in env.obs_space.items()
        if k not in ('reward', 'is_first', 'is_last', 'is_terminal')
        and np.issubdtype(v.dtype, np.floating) and len(v.shape) <= 1]
rng = np.random.default_rng(0)
vals = {k: [] for k in keys}
act = env.act_space['action']
obs = env.step({'action': np.zeros(act.shape, np.float32), 'reset': True})
for t in range(2000):
    a = np.float32(rng.uniform(-1, 1, act.shape))
    obs = env.step({'action': a, 'reset': bool(obs['is_last'])})
    for k in keys:
        vals[k].append(np.asarray(obs[k], np.float64).reshape(-1))
print("key inventory (dim, per-dim sd):")
allsd = []
for k in keys:
    v = np.stack(vals[k])
    sd = v.std(axis=0)
    allsd.extend(sd.tolist())
    print(f"  {k}: dim {v.shape[1]}, sd per dim {np.round(sd, 4).tolist()}")
print(f"MEAN per-dim proprio sd (candidate BASESD): {np.mean(allsd):.5f}")
PY
fi

: "${SOURCE_KEY:=position}"
: "${BASESD_PLANTED:=0.0976}"
: "${BASESD_N:=1.215}"

echo "[$(date)] SE run RUN_ID=$RUN_ID TASK=$TASK SEED=$SEED STEPS=$STEPS"
STOP_STEP=$("$PYTHON" -c "print(int(float('$STEPS')))")

"$PYTHON" -u dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio expl_p2e \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render False \
  --distractor.dim 8 --distractor.scale 1.0 --distractor.basesd "$BASESD_N" \
  --planted.source_key "$SOURCE_KEY" --planted.basesd "$BASESD_PLANTED" \
  --planted.gate_key "${GATE_KEY:-}" \
  --planted.gate_index "${GATE_INDEX:-0}" \
  --planted.gate_threshold "${GATE_THRESHOLD:-0.0}" \
  --run.steps "$STEPS" --run.save_every "$SAVE_EVERY" &
TRAIN_PID=$!

"$PYTHON" -u -m probing.checkpoint_watcher \
  --run_logdir "$LOGDIR" --stop_step "$STOP_STEP" \
  --stop_file "$LOGDIR/TRAINING_DONE" --poll_seconds 60 &
WATCH_PID=$!

TRAIN_RC=0; wait "$TRAIN_PID" || TRAIN_RC=$?
touch "$LOGDIR/TRAINING_DONE"
wait "$WATCH_PID" || true
echo "[$(date)] DONE rc=$TRAIN_RC"
exit "$TRAIN_RC"
