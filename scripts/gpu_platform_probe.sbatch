#!/bin/bash
#SBATCH --job-name=gpu_platform_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=midway3-0278
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Is the B1' ridge failure caused by the string 'gpu' rather than by the node?
# (2026-08-16 diagnostic, follow-up to v100_backend_probe 53415197.)
#
# That probe showed the CUDA backend comes up FINE on midway3-0278 (a V100)
# as-is, with JAX_PLATFORMS='' and with JAX_PLATFORMS=cuda. The rtx6000 twin
# (53415597, midway3-0284) is identical. So the node is not the problem, and
# neither is the driver.
#
# What none of those four cases tested is the value the ridge probe actually
# uses. `probing/ridge_probe.py --platform` DEFAULTS TO 'gpu', and
# embodied/jax/internal.py:34 turns that into
#     jax.config.update('jax_platforms', 'gpu')
# before `jax.devices()` is called in embodied/jax/agent.py:72. 'gpu' is an
# alias that expands to the GPU backends -- cuda AND rocm -- and jaxlib
# 0.4.33's rocm path calls `_xla.GpuAllocatorConfig()`, which this build does
# not export. When the platform list is requested EXPLICITLY a failing member
# is fatal; in the default path it is merely skipped. That would explain why a
# bare `import jax` works on the very node where the ridge job dies.
#
# This probe tests the four values side by side on one node, so the answer
# does not depend on comparing two jobs. Predicted: gpu FAILS, cuda works.
#
# If it holds, the fix is a one-word command change (--platform cuda) -- but
# that is a change to the REGISTERED per-fit form, so it is an amendment call
# for the owning chat, not an ops decision. Nothing is measured here.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

echo "[node] $(hostname)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null
python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
echo

# Exercise the SAME call path the ridge probe takes: set jax_platforms through
# jax.config (not the env var), then ask for devices.
for p in gpu cuda "" rocm; do
  echo "--- jax.config jax_platforms=${p:-<unset>} ---"
  P="$p" python - <<'PY' 2>&1 | tail -3
import os, traceback
p = os.environ.get('P', '')
try:
    import jax
    p and jax.config.update('jax_platforms', p)
    print('  devices:', jax.devices())
except Exception:
    traceback.print_exc()
PY
  echo
done

echo "--- when were the jax wheels last written? ---"
python - <<'PY'
import datetime, glob, os
import jax
sp = os.path.dirname(os.path.dirname(jax.__file__))
for pat in ('jax-*.dist-info', 'jaxlib-*.dist-info', 'jax_cuda12_plugin-*.dist-info',
            'jax_cuda12_pjrt-*.dist-info', 'nvidia_cudnn_cu12-*.dist-info'):
    for d in sorted(glob.glob(os.path.join(sp, pat))):
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(d))
        print(f"  {os.path.basename(d):46s} {ts:%Y-%m-%d %H:%M}")
PY
