#!/bin/bash
#SBATCH --job-name=se_advd_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# Wave-B evaluation probes — PREREG_waveb_advd_20260822. ALL 16
# (adapted, source) pairs in ONE GPU job (single-device rule: the
# panel must not mix devices). RUNS_ROOT = adapted run dirs;
# SRCS_ROOT = source run dirs (Stage-1 + A1). SMOKE=1 runs pair 0
# only, --smoke, to a separate output dir (#32 m2).
# SKIP RULE = ckpt-identity on the ADAPTED run (the 22-Aug lesson).
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUNS_ROOT="${RUNS_ROOT:?adapted runs parent dir}"
SRCS_ROOT="${SRCS_ROOT:?source runs parent dir}"
PAIRS=(
  "se_advd_s10:se_cheetah_seed10"      "se_advd_s11:se_cheetah_seed11"
  "se_advd_s12:se_cheetah_seed12"      "se_advd_s13:se_cheetah_seed13"
  "se_advd_s14:se_cheetah_seed14"      "se_advd_s15:se_cheetah_seed15"
  "se_advd_s16:se_cheetah_seed16"      "se_advd_s17:se_cheetah_seed17"
  "se_advd_het_s44:se_avoid_het_s44"   "se_advd_het_s45:se_avoid_het_s45"
  "se_advd_het_s46:se_avoid_het_s46"   "se_advd_het_s47:se_avoid_het_s47"
  "se_advd_flat_s48:se_avoid_flat_s48" "se_advd_flat_s49:se_avoid_flat_s49"
  "se_advd_flat_s50:se_avoid_flat_s50" "se_advd_flat_s51:se_avoid_flat_s51"
)
if [[ "${SMOKE:-0}" == "1" ]]; then
  IFS=: read -r adp src <<< "${PAIRS[0]}"
  echo "[$(date)] SMOKE pair $adp / $src"
  python -m uncfield.se_advd_probe --run_logdir "$RUNS_ROOT/$adp" \
    --source_logdir "$SRCS_ROOT/$src" --platform cuda --seed 0 \
    --smoke --output "$RUNS_ROOT/$adp/se_advd_smoke"
  exit 0
fi
n_ok=0; n_skip=0; n_fail=0
for pair in "${PAIRS[@]}"; do
  IFS=: read -r adp src <<< "$pair"
  d="$RUNS_ROOT/$adp"
  out="$d/se_advd"
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  newest="$(find "$d/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1)"
  if [ -z "$latest" ] || [ -z "$newest" ]; then
    echo "NO-CKPT $adp — refusing"; n_fail=$((n_fail+1)); continue
  fi
  if [ "$latest" != "$newest" ]; then
    echo "STALE-POINTER $adp (latest=$latest newest=$newest)"
    n_fail=$((n_fail+1)); continue
  fi
  # rev WB-n5/M2: the SOURCE pointer gets the same guard — a stale
  # source is invisible to the bitwise WM gate (jointly stale)
  s_latest="$(cat "$SRCS_ROOT/$src/ckpt/latest" 2>/dev/null || true)"
  s_newest="$(find "$SRCS_ROOT/$src/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1)"
  if [ -z "$s_latest" ] || [ "$s_latest" != "$s_newest" ]; then
    echo "SOURCE-STALE $src (latest='$s_latest' newest='$s_newest')"
    n_fail=$((n_fail+1)); continue
  fi
  probed="$(python -c "import json;print(json.load(open('$out/se_advd.json'))['ckpt'].rstrip('/').split('/')[-1])" 2>/dev/null || true)"
  if [ -n "$probed" ] && [ "$probed" = "$latest" ]; then
    echo "SKIP   $adp (probe matches ckpt/latest $latest)"
    n_skip=$((n_skip+1)); continue
  fi
  if [ -n "$probed" ]; then
    echo "STALE  $adp (probed $probed != latest $latest) — quarantining"
    mv "$out" "$d/_INVALID_STALE_CKPT_se_advd_$(date +%Y%m%d%H%M%S)"
  fi
  echo "PROBE  $adp vs $src"
  if python -m uncfield.se_advd_probe --run_logdir "$d" \
      --source_logdir "$SRCS_ROOT/$src" --platform cuda --seed 0; then
    n_ok=$((n_ok+1))
  else
    echo "FAIL   $adp"; n_fail=$((n_fail+1))
  fi
done
echo "[$(date)] se_advd probes: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#PAIRS[@]})"
[ "$n_fail" -eq 0 ]
