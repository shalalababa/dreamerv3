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
# Amendment-2 arm overrides (PREREG_nfi_scale_exhibit_amend2_20260814 §4).
# ALL DEFAULT TO THE STAGE-1 VALUES, so an unset environment reproduces the
# Stage-1 command exactly -- the defaults below are literally the config
# defaults (`distractor: {dim: 0, gate_key: '', gate_index: 0,
# gate_threshold: 0.0}` in configs.yaml), which is why adding the three
# distractor gate flags is inert for Stage-1 rather than a silent change:
#   DISTRACTOR_DIM             D-only arm sets 0 (the single registered delta)
#   DISTRACTOR_GATE_KEY        Stage-2 GATED sets the gate triple on the
#   DISTRACTOR_GATE_INDEX      distractor wrapper as well as the planted one
#   DISTRACTOR_GATE_THRESHOLD  ("the gate triple on BOTH wrappers")
#   DISTRACTOR_MOD_KEY         gradient arm (amendment 3): amplitude
#   DISTRACTOR_MOD_INDEX       modulation m = clip((x-lo)/(hi-lo),0,1);
#   DISTRACTOR_MOD_LO          defaults ''/0/0.0/1.0 = m==1 = the
#   DISTRACTOR_MOD_HI          config defaults, Stage-1-inert
#   DISTRACTOR_SCALE           flat arm (amendment 3): 0.35891393;
#                              default 1.0 = the Stage-1 hardcoded value
#   DISTRACTOR_THETA           PHI-SWEEP (PREREG_phisweep_20260830): the
#                              OU persistence knob, phi = 1 - theta.
#                              Arms 0.20 (phi 0.80) / 0.05 (phi 0.95);
#                              default 0.1 = phi 0.9 = the config default
#                              = every prior SE consumer, so passing it
#                              is inert for all earlier waves.
#   EXPL_CONFIG                B2 APT column: expl_apt; default
#                              expl_p2e = the Stage-1 hardcoded value
#   DISAG_BOOTSTRAP            noboot arm (amendment 4, seeds 40-43):
#                              set False (capitalized — elements flags
#                              reject lowercase); default True =
#                              the config/Stage-1 value
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
  --logdir "$LOGDIR" --configs dmc_proprio "${EXPL_CONFIG:-expl_p2e}" \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render False \
  --distractor.dim "${DISTRACTOR_DIM:-8}" --distractor.scale "${DISTRACTOR_SCALE:-1.0}" --distractor.basesd "$BASESD_N" \
  --distractor.theta "${DISTRACTOR_THETA:-0.1}" \
  --distractor.gate_key "${DISTRACTOR_GATE_KEY:-}" \
  --distractor.gate_index "${DISTRACTOR_GATE_INDEX:-0}" \
  --distractor.gate_threshold "${DISTRACTOR_GATE_THRESHOLD:-0.0}" \
  --distractor.mod_key "${DISTRACTOR_MOD_KEY:-}" \
  --distractor.mod_index "${DISTRACTOR_MOD_INDEX:-0}" \
  --distractor.mod_lo "${DISTRACTOR_MOD_LO:-0.0}" \
  --distractor.mod_hi "${DISTRACTOR_MOD_HI:-1.0}" \
  --agent.expl.disag_bootstrap "${DISAG_BOOTSTRAP:-True}" \
  --agent.expl.disag_head "${DISAG_HEAD:-det}" \
  --agent.expl.penalty_mix "${PENALTY_MIX:-False}" \
  --penalty.scale "${PENALTY_SCALE:-0.0}" \
  --penalty.gate_key "${PENALTY_GATE_KEY:-}" \
  --penalty.gate_index "${PENALTY_GATE_INDEX:-0}" \
  --penalty.gate_threshold "${PENALTY_GATE_THRESHOLD:-0.0}" \
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

# --- BIND-CHECK (incident 21 Aug: the NOBOOT wave's DISAG_BOOTSTRAP=False
# was silently unread by a STALE producer checkout on the instances — 4 runs
# / ~25 GPU-h produced the bootstrapped arm under noboot run_ids; the
# reader's config pin caught it only post-hoc). Assert that every
# wave-critical env override actually bound in the composed config, within
# minutes of launch; kill the run on any mismatch. NOTE this guard rides in
# the producer, so it cannot protect against a checkout that predates it —
# the wave preflight's repo-SHA + override-read content check is the
# stale-checkout defense.
for _i in $(seq 1 90); do [ -f "$LOGDIR/config.yaml" ] && break; sleep 10; done
if [ ! -f "$LOGDIR/config.yaml" ]; then
  echo "BIND-CHECK: config.yaml never appeared"; kill "$TRAIN_PID" "$WATCH_PID" 2>/dev/null; exit 90
fi
if ! "$PYTHON" - "$LOGDIR/config.yaml" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
env = os.environ
def want(path, val):
    node = cfg
    for p in path.split('.'):
        try:
            node = node[p]
        except (KeyError, TypeError):
            raise AssertionError(
                f"BIND-CHECK FAIL {path}: key missing from the "
                f"composed config — stale configs.yaml/producer "
                f"checkout?")
    assert node == val, (
        f"BIND-CHECK FAIL {path}: config={node!r} != wave={val!r}")
want('agent.expl.disag_bootstrap',
     env.get('DISAG_BOOTSTRAP', 'True') == 'True')
want('agent.expl.disag_head', env.get('DISAG_HEAD', 'det'))
want('agent.expl.penalty_mix',
     env.get('PENALTY_MIX', 'False') == 'True')
want('penalty.scale', float(env.get('PENALTY_SCALE', '0.0')))
want('penalty.gate_key', env.get('PENALTY_GATE_KEY', ''))
# #34 M8: the variables that DEFINE the regions must bind too
want('penalty.gate_index', int(env.get('PENALTY_GATE_INDEX', '0')))
want('penalty.gate_threshold',
     float(env.get('PENALTY_GATE_THRESHOLD', '0.0')))
want('distractor.mod_index',
     int(env.get('DISTRACTOR_MOD_INDEX', '0')))
want('distractor.mod_lo', float(env.get('DISTRACTOR_MOD_LO', '0.0')))
want('distractor.mod_hi', float(env.get('DISTRACTOR_MOD_HI', '1.0')))
want('distractor.dim', int(env.get('DISTRACTOR_DIM', '8')))
want('distractor.scale', float(env.get('DISTRACTOR_SCALE', '1.0')))
# PHI-SWEEP (PREREG_phisweep_20260830): the OU persistence knob is the
# WHOLE registered axis of that wave (phi = 1 - theta), so it must bind
# or the run is worthless. Default 0.1 = phi 0.9 = the config default =
# every prior consumer, so this line is bitwise-inert outside phisweep.
want('distractor.theta', float(env.get('DISTRACTOR_THETA', '0.1')))
want('distractor.mod_key', env.get('DISTRACTOR_MOD_KEY', ''))
want('distractor.gate_key', env.get('DISTRACTOR_GATE_KEY', ''))
want('planted.gate_key', env.get('GATE_KEY', ''))
# R3-D4: the remaining region-defining + identity variables
want('distractor.gate_index',
     int(env.get('DISTRACTOR_GATE_INDEX', '0')))
want('distractor.gate_threshold',
     float(env.get('DISTRACTOR_GATE_THRESHOLD', '0.0')))
want('planted.gate_index', int(env.get('GATE_INDEX', '0')))
want('planted.gate_threshold',
     float(env.get('GATE_THRESHOLD', '0.0')))
want('planted.source_key', env.get('SOURCE_KEY', 'position'))
want('planted.basesd', float(env.get('BASESD_PLANTED', '0.0976')))
want('distractor.basesd', float(env.get('BASESD_N', '1.215')))
want('seed', int(env['SEED']))
want('run.steps', float(env.get('STEPS', '5e5')))
if env.get('EXPL_CONFIG', 'expl_p2e') == 'expl_p2e':
    want('agent.expl.disag_scale', 1000.0)
# constants, but their PRESENCE proves the checkout carries the B4
# config keys (a stale configs.yaml would fail the key lookup)
want('agent.expl.disag_logvar_min', -8.0)
want('agent.expl.disag_logvar_max', 6.0)
want('agent.expl.mode',
     {'expl_p2e': 'p2e', 'expl_apt': 'apt'}[
         env.get('EXPL_CONFIG', 'expl_p2e')])
print('BIND-CHECK PASS: all wave-critical overrides bound')
PY
then
  echo "BIND-CHECK FAILED — killing the run before it burns GPU-hours"
  kill "$TRAIN_PID" "$WATCH_PID" 2>/dev/null
  exit 90
fi
# --- end BIND-CHECK

TRAIN_RC=0; wait "$TRAIN_PID" || TRAIN_RC=$?
touch "$LOGDIR/TRAINING_DONE"
wait "$WATCH_PID" || true
echo "[$(date)] DONE rc=$TRAIN_RC"
exit "$TRAIN_RC"
