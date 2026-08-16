#!/bin/bash
#SBATCH --job-name=ridge_devnode_check
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Is pinning the GPU *model* enough, or must a panel pin the *node*?
# (2026-08-16, diagnostic — no registered decision consumes this.)
#
# ops/waves/lewm_probe_devcheck/FINDING.md measured the ACROSS-ARCHITECTURE
# term (V100/A100 vs a Vast 5060 Ti): up to 0.113 AUROC, with a zero
# same-GPU noise floor. It never tested SAME MODEL, DIFFERENT NODE — and that
# is exactly what `--constraint=v100` buys. B1' depends on the answer: its
# pinned anchor panel (job 53243601, all four array tasks on midway3-0278, a
# v100) is compared against a fresh ridge panel, and the device term sits
# inside its MDE80 window of 0.06-0.20.
#
#   identical bytes  -> --constraint=v100 is sufficient; B1' pins and proceeds
#   any difference   -> model pinning is NOT enough; B1' needs --nodelist, or
#                       the anchor must be re-measured beside the new panel
#
# Re-runs three anchor cells with the registered producer command verbatim
# (probeset finger_v1, ridge_probe's default ep_batch, no --platform), writing
# to a SIDE directory so no anchor json can be touched. Submit with
#   --constraint=v100 --exclude=midway3-0278
# so it cannot land on the node that produced the anchor.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
OUT="$RUNROOT/ridge_devnode_check"
mkdir -p "$OUT"

echo "[node] $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo

for r in ax1wm_finger_q1s0_seed1 ax1wm_finger_q1s0_seed2 ax1wm_finger_q1s1_seed1; do
  echo "=== $r ==="
  python -m probing.ridge_probe measure \
    --run_logdir "$RUNROOT/$r" \
    --probeset "$RUNROOT/e4_probesets/finger_v1" \
    --output "$OUT/$r.json"
  echo "  rc=$?"
done

echo
echo "=== compare against the pinned anchor jsons (produced on midway3-0278) ==="
for r in ax1wm_finger_q1s0_seed1 ax1wm_finger_q1s0_seed2 ax1wm_finger_q1s1_seed1; do
  a="$RUNROOT/$r/ridge_probe_finger_v1.json"
  b="$OUT/$r.json"
  printf '%-30s anchor=%s  here=%s  %s\n' "$r" \
    "$(sha256sum "$a" 2>/dev/null | cut -c1-16)" \
    "$(sha256sum "$b" 2>/dev/null | cut -c1-16)" \
    "$(cmp -s "$a" "$b" && echo IDENTICAL || echo DIFFERENT)"
done
