#!/bin/bash
#SBATCH --job-name=dv3_pxpar
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Tier-0 EXPLORATORY: the pixel-parity SCRATCH cell
# (artifacts/algo_tier0_20260820/RECORD.md section 4; Algorithm_Ideation
# 20260819 section 4.1). Decides whether the pixel harness can score ANY
# pixel algorithm, and discharges the Limitation-4 capacity/budget
# confound either way.
#
# The cell: from-scratch pixel adaptation at HARNESS PARITY with the
# proprio cells, i.e. the study's two under-provisions removed --
#   run.train_ratio  256 -> 1024   (the study ran 1/4 the gradient budget)
#   .*\.depth          4 -> 32     (size1m's depth 4 = ~14K encoder,
#                                   ~250x under the standard CNN)
# RSSM stays size1m (deter 512 / hidden 64 / classes 4). The complaint on
# file is the ENCODER; moving one lever keeps the cell interpretable.
# VERIFIED locally against elements.Config: the regex override resolves to
# exactly agent.enc.simple.depth and agent.dec.simple.depth -- the decoder
# moves too, which is intended (spec names `.*\.depth`) and is disclosed
# here rather than discovered at read time.
#
# SCRATCH means scratch: no --run.from_checkpoint, no frozen_readout, no
# AXIS1_INIT_WM. Same shape as the proprio anchor (PREREG_scratch_anchor
# 20260730), which is a plain reward-on run at the adaptation budget.
# The saved per-run config.yaml (from_checkpoint: '', frozen_wm: false) is
# the audit trail.
#
# ROUTING -- rtx6000 pinned, load-bearing: cudnn 5003 is a VOLTA fact in
# this JAX build (ledger 08-16: pixel probes V100 9/9 fail, RTX6000 3/3
# clean). A pixel job that lands on a v100 node dies in the first conv.
# a100 also works; v100 does not. Do not remove the constraint.
#
# SCORING: run_id adapt_scratchpxpar_finger_seed<k>_ckpt0 parses under the
# frozen adaptation_auc RUN_RE as mode `scratchpxpar`, milestone 0. NO
# frozen reader consumes that mode (checked: horizon_read matches
# arm == 'scratch' exactly; review_waves_read constructs mode='scratch'
# literally), so collating it cannot perturb an executed read.
#
# WALLTIME: one seed is ~4x a study pixel cell (~6-12 h). With 36 h
# requested we can fit a second seed, but ONLY whole -- a truncated run is
# the failure mode this program has already been burned by (REALIZED_
# TRAINING_20260807), so a seed is not STARTED unless MIN_SEED_S remain.
# Resumable: a seed with ADAPT_DONE is skipped, so a re-submit continues.
#
# Env: SEEDS (default "1 2"), JOB_WALL_S (default 129600 = 36 h),
#      MIN_SEED_S (default 46800 = 13 h), STEPS (default 1.25e5).
# ---------------------------------------------------------------------------
set -uo pipefail
T0=$SECONDS
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
SEEDS="${SEEDS:-1 2}"
JOB_WALL_S="${JOB_WALL_S:-129600}"
MIN_SEED_S="${MIN_SEED_S:-46800}"
STEPS="${STEPS:-1.25e5}"
TASK="${TASK:-dmc_finger_turn_hard}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[warn] no nvidia-smi"
echo "[cell] pixel_wm SCRATCH ${TASK} steps=${STEPS} train_ratio=1024 depth=32 seeds='${SEEDS}'"
echo

# --- preflight: the overrides must actually bind ---------------------------
# An override that silently fails to apply is the expensive failure here: the
# job would run 12 h and produce the study's own under-provisioned cell under
# the parity cell's name. Assert on the resolved config BEFORE spending it.
python - <<'PY' || exit 2
import pathlib, sys
import ruamel.yaml as yaml, elements
cfg = yaml.YAML(typ='safe').load(
    pathlib.Path('dreamerv3/configs.yaml').read_text())
c = elements.Config(cfg['defaults']).update(cfg['pixel_wm'])
c = elements.Flags(c).parse(
    ['--.*\\.depth', '32', '--run.train_ratio', '1024',
     '--run.steps', '1.25e5'])
depths = {k: v for k, v in c.flat.items() if k.endswith('.depth')}
bad = []
if not depths or set(depths.values()) != {32}:
  bad.append(f'depth override did not bind: {depths}')
if float(c.run.train_ratio) != 1024.0:
  bad.append(f'train_ratio = {c.run.train_ratio}, want 1024')
if c.agent.model_obs != 'image':
  bad.append(f'model_obs = {c.agent.model_obs}, want image')
if c.agent.dyn.rssm.deter != 512 or c.agent.dyn.rssm.classes != 4:
  bad.append(f'rssm moved off size1m: deter={c.agent.dyn.rssm.deter} '
             f'classes={c.agent.dyn.rssm.classes}')
if c.run.from_checkpoint:
  bad.append(f'from_checkpoint is set: {c.run.from_checkpoint!r}')
if bad:
  print('PREFLIGHT FAILED:'); [print('  -', b) for b in bad]; sys.exit(1)
print(f'[preflight] OK  depths={depths}  train_ratio={c.run.train_ratio} '
      f'rssm.deter={c.agent.dyn.rssm.deter}  from_checkpoint=<empty>')
PY
echo

n_ok=0; n_skip=0; n_fail=0; n_defer=0; failed=""
for k in $SEEDS; do
  RUN_ID="adapt_scratchpxpar_finger_seed${k}_ckpt0"
  LOGDIR="$RUNROOT/$RUN_ID"
  if [ -f "$LOGDIR/ADAPT_DONE" ]; then
    echo "SKIP   $RUN_ID (ADAPT_DONE present)"; n_skip=$((n_skip+1)); continue
  fi
  left=$((JOB_WALL_S - (SECONDS - T0)))
  if [ "$left" -lt "$MIN_SEED_S" ]; then
    echo "DEFER  $RUN_ID -- ${left}s left < ${MIN_SEED_S}s needed for a whole seed"
    echo "       (re-submit to continue; finished seeds are skipped)"
    n_defer=$((n_defer+1)); continue
  fi
  echo "[$(date)] RUN $RUN_ID  (${left}s of walltime left)"
  mkdir -p "$LOGDIR"
  RC=0
  python -u dreamerv3/main.py \
    --logdir "$LOGDIR" --configs pixel_wm \
    --task "$TASK" --seed "$k" \
    --env.dmc.render True \
    --run.steps "$STEPS" \
    --run.train_ratio 1024 \
    '--.*\.depth' 32 || RC=$?
  if [ "$RC" -eq 0 ]; then
    # Post-hoc witness on what the run ACTUALLY used, from its own saved
    # config -- the preflight proves the override binds in principle, this
    # proves it bound in THIS run.
    python - "$LOGDIR/config.yaml" <<'PY'
import sys, ruamel.yaml as yaml
c = yaml.YAML(typ='safe').load(open(sys.argv[1]).read())
flat = {}
def walk(d, p=''):
  for k, v in (d or {}).items():
    walk(v, f'{p}{k}.') if isinstance(v, dict) else flat.setdefault(f'{p}{k}', v)
walk(c)
d = {k: v for k, v in flat.items() if k.endswith('.depth')}
print(f"[witness] depths={d} train_ratio={flat.get('run.train_ratio')} "
      f"from_checkpoint={flat.get('run.from_checkpoint')!r} "
      f"rssm.deter={flat.get('agent.dyn.rssm.deter')}")
PY
    touch "$LOGDIR/ADAPT_DONE"
    echo "OK     $RUN_ID"; n_ok=$((n_ok+1))
  else
    echo "FAIL   $RUN_ID (rc=$RC) -- ADAPT_DONE deliberately NOT written"
    n_fail=$((n_fail+1)); failed="$failed $RUN_ID"
  fi
done

echo
echo "[$(date)] pixel-parity: ok=$n_ok skipped=$n_skip deferred=$n_defer failed=$n_fail"
[ -n "$failed" ] && echo "FAILED:$failed"
echo "Score with: python -m analysis.adaptation_auc --runroot \$RUNROOT \\"
echo "  --output \$RUNROOT/_reads/pixel_parity_scratch   # its own OUTPUT DIR, not the canonical collation"
[ "$n_fail" -eq 0 ] || exit 1
exit 0
