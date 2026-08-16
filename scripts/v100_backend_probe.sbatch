#!/bin/bash
#SBATCH --job-name=v100_backend_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=midway3-0278
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Can ANYTHING bring up a GPU backend on a Midway3 V100 in the current env?
# (2026-08-16 diagnostic.)
#
# B1's ridge panel must be measured on the anchor's hardware (midway3-0278, a
# V100), but all 16 cells died there with
#   RuntimeError: Unable to initialize backend 'rocm':
#     module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
#     (set JAX_PLATFORMS='' to automatically choose an available backend)
# The chat-history forensics confirm the anchor was produced by the plain
# default form -- no --platform, no --ep_batch, no DREAMER_DISABLE_GPUFLAGS --
# which is exactly what our script runs. So the command is NOT the difference;
# the environment is. The anchor ran here fine on 10 Aug.
#
# The error text names its own escape hatch, so test it rather than assume:
# if JAX_PLATFORMS='' lets the CUDA backend come up on Volta, B1' keeps the
# cheap resolution (pin 0278 = anchor hardware, no amendment). If nothing
# works, Volta is genuinely unusable and the anchor must be re-measured on
# rtx6000 beside the fresh panel.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

echo "[node] $(hostname)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null
echo "env JAX_PLATFORMS=${JAX_PLATFORMS-<unset>}"
python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
echo

probe () {
  echo "--- $1 ---"
  env $2 python - <<'PY' 2>&1 | tail -4
import os, traceback
try:
    import jax
    print('  devices:', jax.devices())
    import jax.numpy as jnp
    print('  matmul ok:', float(jnp.ones((8, 8)) @ jnp.ones((8, 8)))[0] if False else float((jnp.ones((8,8)) @ jnp.ones((8,8))).sum()))
except Exception:
    traceback.print_exc()
PY
  echo
}

probe "as-is (whatever the env sets)"            ""
probe "JAX_PLATFORMS=''"                          "JAX_PLATFORMS="
probe "JAX_PLATFORMS=cuda"                        "JAX_PLATFORMS=cuda"
probe "JAX_PLATFORMS=cpu (sanity: must work)"     "JAX_PLATFORMS=cpu"

echo "--- installed jax cuda plugins ---"
python -m pip list 2>/dev/null | grep -iE "jax|cuda|nvidia-cud(nn|a)" | head -20
