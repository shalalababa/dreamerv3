#!/bin/bash
# ---------------------------------------------------------------------------
# Wave-B advd adaptation producer — PREREG_waveb_advd_20260822.
# A fresh actor-critic adapted on a FROZEN source WM+disag: expl mode
# advd (imagined reward = decoded distractor member-variance), loaded
# via run.from_checkpoint with the registered regex. Vars:
#   RUN_ID TASK SEED SRC_CKPT (source ckpt DIR)  [STEPS=1e5]
#   DISTRACTOR_* / GATE_* / PENALTY_* as in uncfield_se.sbatch
# BIND-CHECK mirrors the uncfield_se producer (NOBOOT lesson) plus
# the advd-specific pins; kills the run on any mismatch.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
RUNROOT="${RUNROOT:?set RUNROOT}"
PYTHON="${CONDA_ENV:-}/bin/python"; [ -x "$PYTHON" ] || PYTHON="$(command -v python)"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${SEED:?set SEED}"
# Source resolution: either SRC_CKPT (a ckpt DIR) directly, or
# SRCROOT + SRC_DIR — resolved via ckpt/latest WITH the
# stale-pointer guard (latest must equal the newest ckpt directory;
# the 22-Aug B1 s53 lesson)
if [ -z "${SRC_CKPT:-}" ]; then
  : "${SRCROOT:?set SRC_CKPT or SRCROOT+SRC_DIR}"
  : "${SRC_DIR:?set SRC_CKPT or SRCROOT+SRC_DIR}"
  latest="$(cat "$SRCROOT/$SRC_DIR/ckpt/latest" 2>/dev/null || true)"
  newest="$(find "$SRCROOT/$SRC_DIR/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1)"
  [ -n "$latest" ] && [ "$latest" = "$newest" ] || {
    echo "FATAL: source ckpt pointer unresolved/stale (latest='$latest' newest='$newest')"; exit 90; }
  SRC_CKPT="$SRCROOT/$SRC_DIR/ckpt/$latest"
fi
export SRC_CKPT
[ -d "$SRC_CKPT" ] || { echo "FATAL: SRC_CKPT $SRC_CKPT is not a dir"; exit 90; }
STEPS="${STEPS:-1e5}"
SAVE_EVERY="${SAVE_EVERY:-300}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/TRAINING_DONE"
# WB AMENDMENT 1 resume guards (20260823): a staged-back run dir
# must carry a resolvable ckpt pointer, its replay buffer, and no
# error file — elements Replay.load falls back SILENTLY to an empty
# buffer if replay/ is missing (ckpt/<id>/replay.pkl is a 4-byte
# None), and portal aborts opaquely on a leftover error file. The
# stale config.yaml must go, or the BIND-CHECK below reads leg-1's
# run.steps before main.py rewrites the file.
if [ -d "$LOGDIR/ckpt" ]; then
  [ ! -e "$LOGDIR/error" ] || { echo "FATAL: $LOGDIR/error exists — refusing resume"; exit 90; }
  r_latest="$(cat "$LOGDIR/ckpt/latest" 2>/dev/null || true)"
  [ -n "$r_latest" ] && [ -d "$LOGDIR/ckpt/$r_latest" ] || {
    echo "FATAL: resume without resolvable ckpt/latest (latest='$r_latest')"; exit 90; }
  n_npz="$(ls "$LOGDIR/replay/"*.npz 2>/dev/null | wc -l)"
  [ "$n_npz" -ge 150 ] || {
    echo "FATAL: resume with $n_npz replay chunks (<150) — replay/ not staged back"; exit 90; }
  rm -f "$LOGDIR/config.yaml"
fi

: "${SOURCE_KEY:=position}"
: "${BASESD_PLANTED:=0.0976}"
: "${BASESD_N:=1.215}"

echo "[$(date)] ADVD adapt RUN_ID=$RUN_ID SEED=$SEED SRC=$SRC_CKPT STEPS=$STEPS"
STOP_STEP=$("$PYTHON" -c "print(int(float('$STEPS')))")

"$PYTHON" -u dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio expl_advd \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render False \
  --distractor.dim "${DISTRACTOR_DIM:-8}" --distractor.scale "${DISTRACTOR_SCALE:-1.0}" --distractor.basesd "$BASESD_N" \
  --distractor.gate_key "${DISTRACTOR_GATE_KEY:-}" \
  --distractor.gate_index "${DISTRACTOR_GATE_INDEX:-0}" \
  --distractor.gate_threshold "${DISTRACTOR_GATE_THRESHOLD:-0.0}" \
  --distractor.mod_key "${DISTRACTOR_MOD_KEY:-}" \
  --distractor.mod_index "${DISTRACTOR_MOD_INDEX:-0}" \
  --distractor.mod_lo "${DISTRACTOR_MOD_LO:-0.0}" \
  --distractor.mod_hi "${DISTRACTOR_MOD_HI:-1.0}" \
  --agent.expl.penalty_mix "${PENALTY_MIX:-False}" \
  --penalty.scale "${PENALTY_SCALE:-0.0}" \
  --penalty.gate_key "${PENALTY_GATE_KEY:-}" \
  --penalty.gate_index "${PENALTY_GATE_INDEX:-0}" \
  --penalty.gate_threshold "${PENALTY_GATE_THRESHOLD:-0.0}" \
  --planted.source_key "$SOURCE_KEY" --planted.basesd "$BASESD_PLANTED" \
  --planted.gate_key "${GATE_KEY:-}" \
  --planted.gate_index "${GATE_INDEX:-0}" \
  --planted.gate_threshold "${GATE_THRESHOLD:-0.0}" \
  --run.from_checkpoint "$SRC_CKPT" \
  --run.steps "$STEPS" --run.save_every "$SAVE_EVERY" &
TRAIN_PID=$!

"$PYTHON" -u -m probing.checkpoint_watcher \
  --run_logdir "$LOGDIR" --stop_step "$STOP_STEP" \
  --stop_file "$LOGDIR/TRAINING_DONE" --poll_seconds 60 &
WATCH_PID=$!

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
                f"BIND-CHECK FAIL {path}: key missing — stale "
                f"configs.yaml/producer checkout?")
    assert node == val, (
        f"BIND-CHECK FAIL {path}: config={node!r} != wave={val!r}")
# advd identity
want('agent.expl.mode', 'advd')
want('agent.expl.advd_scale', 100.0)
want('run.from_checkpoint', env['SRC_CKPT'])
want('run.from_checkpoint_regex', '^(enc|dyn|dec|disag)/')
# the uncfield_se identity block (R3-D4 set)
want('agent.expl.penalty_mix', env.get('PENALTY_MIX', 'False') == 'True')
want('penalty.scale', float(env.get('PENALTY_SCALE', '0.0')))
want('penalty.gate_key', env.get('PENALTY_GATE_KEY', ''))
want('penalty.gate_index', int(env.get('PENALTY_GATE_INDEX', '0')))
want('penalty.gate_threshold', float(env.get('PENALTY_GATE_THRESHOLD', '0.0')))
want('distractor.dim', int(env.get('DISTRACTOR_DIM', '8')))
want('distractor.scale', float(env.get('DISTRACTOR_SCALE', '1.0')))
want('distractor.basesd', float(env.get('BASESD_N', '1.215')))
want('distractor.mod_key', env.get('DISTRACTOR_MOD_KEY', ''))
want('distractor.mod_index', int(env.get('DISTRACTOR_MOD_INDEX', '0')))
want('distractor.mod_lo', float(env.get('DISTRACTOR_MOD_LO', '0.0')))
want('distractor.mod_hi', float(env.get('DISTRACTOR_MOD_HI', '1.0')))
want('distractor.gate_key', env.get('DISTRACTOR_GATE_KEY', ''))
want('distractor.gate_index', int(env.get('DISTRACTOR_GATE_INDEX', '0')))
want('distractor.gate_threshold', float(env.get('DISTRACTOR_GATE_THRESHOLD', '0.0')))
want('planted.gate_key', env.get('GATE_KEY', ''))
want('planted.gate_index', int(env.get('GATE_INDEX', '0')))
want('planted.gate_threshold', float(env.get('GATE_THRESHOLD', '0.0')))
want('planted.source_key', env.get('SOURCE_KEY', 'position'))
want('planted.basesd', float(env.get('BASESD_PLANTED', '0.0976')))
want('seed', int(env['SEED']))
want('run.steps', float(env.get('STEPS', '1e5')))
print('BIND-CHECK PASS: all advd wave-critical overrides bound')
PY
then
  echo "BIND-CHECK FAILED — killing the run"
  kill "$TRAIN_PID" "$WATCH_PID" 2>/dev/null
  exit 90
fi

TRAIN_RC=0; wait "$TRAIN_PID" || TRAIN_RC=$?
touch "$LOGDIR/TRAINING_DONE"
wait "$WATCH_PID" || true
echo "[$(date)] DONE rc=$TRAIN_RC"
exit "$TRAIN_RC"
