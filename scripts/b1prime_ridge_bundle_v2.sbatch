#!/bin/bash
#SBATCH --job-name=b1p_ridge_v2
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=midway3-0278
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# B1' per-fit ridge panel, v2 -- ANCHOR RE-CHECK GATE + ALL 16 CELLS, ONE JOB.
# (PREREG_b1prime_20260815 registered per-fit form; platform decision note
# b1prime_platform_decision_20260816.md.)
#
# v1 (job 53411405) died at jax init on all 16 cells with
#   RuntimeError: Unable to initialize backend 'rocm':
#     module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
# and measured nothing. Two changes here; everything else is v1 verbatim.
#
# CHANGE 1 -- `--platform cuda`.
# probing/ridge_probe.py defaults --platform to 'gpu', which
# embodied/jax/internal.py:34 turns into jax.config jax_platforms='gpu'
# BEFORE jax.devices() is called in embodied/jax/agent.py:72. 'gpu' is an
# alias spanning cuda AND rocm, and this jaxlib build's rocm path calls
# _xla.GpuAllocatorConfig(), which it does not export. Backend probes 53415197
# (V100/midway3-0278) and 53415597 (rtx6000/midway3-0284) both bring CUDA up
# fine as-is, with JAX_PLATFORMS='' and with =cuda -- so neither the node nor
# the architecture is the blocker, and 'gpu' is the one value untested there.
# Per the owning chat's ruling this is EXECUTION of a registered conditional
# (the prereg's Vast-lane clause citing the sigma-ladder lesson), not an
# amendment: backend selection has zero arithmetic surface, and v1 crashed at
# init so nothing was observed.
#
# CHANGE 2 -- the anchor re-check gate, PHASE A below.
# The primary compares these 16 fresh AUROCs against 16 pinned anchor
# literals measured on 2026-08-10 by job 53243601 (sacct: all four array
# tasks on midway3-0278; its logs report `JAX devices (1): [cuda:0]`). Since
# same-node was the whole basis for treating the device term as zero, the
# owning chat gated consumption on re-measuring anchor cells under TODAY's
# env and comparing to the pins.
#
#   Ops note carried into that gate: the jax stack has NOT changed. Every
#   jax / jaxlib / jax-cuda12-{plugin,pjrt} / nvidia-cud* dist-info in the env
#   dates to 2026-05-09, three months BEFORE the anchor. The only
#   site-packages change since is 2026-08-15 10:32, which installed the
#   `vastai` CLI and its networking dependency tree (aiohttp, rich,
#   cryptography, ...) -- nothing numeric, no numpy/scipy/jax. So the
#   expected result is bitwise-identical, and this phase is a check, not a
#   suspicion.
#
# The gate re-measures ALL 16 anchors rather than the 3 the ruling requires:
# it is a superset, it costs ~5 min (the anchor did 43 cells in 13), it
# removes any question about which 3 were picked, and if it DOES diverge it
# has already produced the full re-measured panel the fallback amendment
# would need.
#
# NOTHING PINNED IS OVERWRITTEN: anchor re-measures go to
# $RUNROOT/b1p_anchor_recheck/<run>.json via --output. The pinned values live
# in each run dir's own ridge_probe_finger_v1.json and in the historical
# collate, and neither is touched. The comparison is done LOCALLY afterwards,
# because the historical collate is local-only under results-sync policy v2.
#
# ORDERING: Phase A runs first and aborts before Phase B if every anchor cell
# fails -- that is the signature of the platform fix not working, and there is
# no reason to touch the panel in that world.
#
# Resumable: a cell whose output already parses as json is skipped.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
PROBESET="$RUNROOT/e4_probesets/finger_v1"
[ -d "$PROBESET" ] || { echo "ERROR: no probeset $PROBESET" >&2; exit 2; }
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN" >&2; exit 2; }

RECHECK_DIR="$RUNROOT/b1p_anchor_recheck"
mkdir -p "$RECHECK_DIR"

host="$(hostname)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "[device] host=$host gpu=$gpu driver=$drv job=${SLURM_JOB_ID:-none}"
echo "[device] every cell below is measured on THIS gpu, with --platform cuda"
echo

# One sidecar writer for both phases. The probe json records no device and no
# command, which is what forced a day of sacct archaeology to identify the
# anchor's GPU; the command field is the disclosure witness the platform
# decision note asks for.
write_sidecar () {  # write_sidecar <out> <run_logdir>
  python - "$1" "$2" "$host" "$gpu" "$drv" "${SLURM_JOB_ID:-none}" "$PROBESET" <<'PY'
import hashlib, json, sys
out, run, host, gpu, drv, job, probeset = sys.argv[1:8]
json.dump({'output': out, 'run_logdir': run, 'probeset': probeset,
           'hostname': host, 'gpu_name': gpu, 'driver_version': drv,
           'slurm_job_id': job,
           'command': ('python -m probing.ridge_probe measure --run_logdir '
                       f'{run} --probeset {probeset} --output {out} '
                       '--platform cuda'),
           'platform_flag': 'cuda',
           'platform_flag_reason': ('registered conditional applied; '
                                    'default gpu alias unusable in this '
                                    'jaxlib build -- see '
                                    'b1prime_platform_decision_20260816.md'),
           'output_sha256': hashlib.sha256(open(out, 'rb').read()).hexdigest()},
          open(out + '.device.json', 'w'), indent=1)
PY
}

measure () {  # measure <run_logdir> <out>  -> 0 ok / 1 fail
  python -m probing.ridge_probe measure \
    --run_logdir "$1" --probeset "$PROBESET" --output "$2" --platform cuda
  local rc=$?
  [ "$rc" -eq 0 ] && [ -f "$2" ] || return 1
  write_sidecar "$2" "$1"
  return 0
}

# ---- PHASE A: anchor re-check gate ---------------------------------------
anchors=()
for armp in "" f; do
  for side in 0 1; do
    for seed in 1 2 3 4; do
      anchors+=("ax1wm_finger_${armp}q1s${side}_seed${seed}")
    done
  done
done

echo "=== PHASE A: re-measure ${#anchors[@]} anchor cells -> $RECHECK_DIR ==="
echo "    (pinned values are NOT touched; comparison happens locally)"
a_ok=0; a_fail=0; a_skip=0; a_failed=""
for r in "${anchors[@]}"; do
  d="$RUNROOT/$r"
  out="$RECHECK_DIR/$r.json"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; a_fail=$((a_fail+1)); a_failed="$a_failed $r"; continue
  fi
  if [ -f "$out" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then
    echo "SKIP   $r (already re-measured)"; a_skip=$((a_skip+1)); continue
  fi
  echo "[$(date)] RECHECK $r"
  if measure "$d" "$out"; then echo "OK     $r"; a_ok=$((a_ok+1))
  else echo "FAIL   $r"; a_fail=$((a_fail+1)); a_failed="$a_failed $r"; fi
done
echo
echo "[$(date)] PHASE A: ok=$a_ok skipped=$a_skip failed=$a_fail (of ${#anchors[@]})"
[ -n "$a_failed" ] && echo "PHASE A FAILED CELLS:$a_failed"

# Every anchor failing is the signature of the platform fix not working.
if [ "$((a_ok + a_skip))" -eq 0 ]; then
  echo "ABORT: no anchor cell could be measured -- not touching the panel." >&2
  exit 1
fi
echo

# ---- PHASE B: the 16 fresh B1' cells (registered outputs) -----------------
runs=()
for armp in "" f; do
  for side in 0 1; do
    for seed in 1 2 3 4; do
      runs+=("ax1wm_finger_${armp}sk6q1s${side}_seed${seed}")
    done
  done
done

echo "=== PHASE B: measure ${#runs[@]} B1' cells (registered outputs) ==="
n_ok=0; n_skip=0; n_fail=0; failed=""
for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  out="$d/ridge_probe_finger_v1.json"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  if [ -f "$out" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then
    echo "SKIP   $r (already measured)"; n_skip=$((n_skip+1)); continue
  fi
  echo "[$(date)] MEASURE $r"
  if measure "$d" "$out"; then echo "OK     $r"; n_ok=$((n_ok+1))
  else echo "FAIL   $r"; n_fail=$((n_fail+1)); failed="$failed $r"; fi
done

echo
echo "[$(date)] PHASE B: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#runs[@]})"
[ -n "$failed" ] && echo "PHASE B FAILED CELLS:$failed"
echo
echo "NEXT: pull $RECHECK_DIR and compare against the pinned anchor literals"
echo "      LOCALLY (historical collate is local-only). The panel must not be"
echo "      consumed until that comparison is adjudicated."
[ "$n_fail" -eq 0 ] && [ "$a_fail" -eq 0 ] || exit 1
exit 0
