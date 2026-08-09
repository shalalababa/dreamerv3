#!/bin/bash
# ---------------------------------------------------------------------------
# One-time setup of the TD-MPC2 conda env + pinned official checkout.
# Run on a login node from a CLEAN shell (module-leakage gotcha: do not
# submit jobs from a shell that has site modules loaded).
#
#   ./scripts/tdmpc2_env_setup.sh
#
# Overridables: TM2_CONDA_ENV (env path), TDMPC2_CHECKOUT (repo dir),
# TDMPC2_COMMIT (pin).
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TM2_CONDA_ENV="${TM2_CONDA_ENV:-/scratch/midway3/$USER/conda_envs/tdmpc2}"
TDMPC2_CHECKOUT="${TDMPC2_CHECKOUT:-/scratch/midway3/$USER/tdmpc2}"
TDMPC2_COMMIT="${TDMPC2_COMMIT:-e9f59321933cbc8e11a002b842adc7d4ffae8ff1}"

if [ ! -d "$TDMPC2_CHECKOUT/.git" ]; then
  git clone https://github.com/nicklashansen/tdmpc2 "$TDMPC2_CHECKOUT"
fi
git -C "$TDMPC2_CHECKOUT" fetch --depth 1 origin "$TDMPC2_COMMIT" || true
git -C "$TDMPC2_CHECKOUT" checkout "$TDMPC2_COMMIT"
echo "tdmpc2 pinned at $(git -C "$TDMPC2_CHECKOUT" rev-parse HEAD)"

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$TM2_CONDA_ENV" ]; then
  conda create -y -p "$TM2_CONDA_ENV" python=3.11
else
  echo "env $TM2_CONDA_ENV already exists; repairing deps if needed"
fi

conda activate "$TM2_CONDA_ENV"
python -m pip install -U pip setuptools wheel
if [ -n "${TM2_TORCH_INDEX_URL:-}" ]; then
  python -m pip install torch --index-url "$TM2_TORCH_INDEX_URL"
else
  python -m pip install -U torch
fi
python -m pip install tensordict torchrl gymnasium omegaconf hydra-core \
    dm_control mujoco numpy
# dv3-repo import chain for the env adapter (embodied -> elements/portal).
# Dv3TaskEnv imports embodied.envs.dmc, and embodied/__init__.py does
# `from .core import *` + `from . import jax`, so the whole Dreamer import
# chain is pulled in: embodied/jax/nets.py needs einops and
# embodied/core/driver.py needs cloudpickle. These must be installed BEFORE
# the smoke check below; otherwise the smoke dies on a bare
# "ModuleNotFoundError: No module named 'einops'" that reads like a TD-MPC2 or
# JAX bug rather than a missing adapter dep. gym is belt-and-braces for the
# upstream tdmpc2 checkout; the repo adapter itself uses gymnasium.
python -m pip install elements portal ninjax jaxtyping chex optax "jax[cpu]" \
    einops cloudpickle gym

# Fail loudly and specifically if the adapter chain is still incomplete, so a
# half-repaired env names its missing packages instead of dying inside the
# first Dv3TaskEnv construction.
echo "Preflight: Dreamer adapter import chain"
TM2_ENV="$TM2_CONDA_ENV" python - <<'EOF'
import importlib.util, os, sys
need = ('einops', 'cloudpickle', 'elements', 'portal', 'ninjax', 'chex',
        'optax', 'jax', 'gymnasium', 'dm_control', 'torch', 'tensordict')
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(
        'MISSING adapter deps in %s: %s\nrepair: %s/bin/python -m pip install %s'
        % (os.environ['TM2_ENV'], ' '.join(missing),
           os.environ['TM2_ENV'], ' '.join(missing)))
print('adapter deps OK')
EOF

echo "Smoke checks (GPU-independent):"
conda activate "$TM2_CONDA_ENV"
cd "$REPO"
python -m probing.tdmpc2_bridge selfcheck
TDMPC2_ROOT="$TDMPC2_CHECKOUT/tdmpc2" python - <<'EOF'
import os
from probing.tdmpc2_compat import add_tdmpc2_path, build_cfg
add_tdmpc2_path()
cfg = build_cfg(os.environ['TDMPC2_ROOT'], 'dmc_finger_turn_hard', 12, 2, 1000)
assert tuple(cfg.obs_shape['state']) == (12,), cfg.obs_shape
assert cfg.action_dim == 2, cfg.action_dim
assert cfg.latent_dim == 512, cfg.latent_dim
assert cfg.bin_size > 0, cfg.bin_size
print('cfg build OK:', cfg.task, 'latent', cfg.latent_dim)
from probing.tdmpc2_compat import Dv3TaskEnv
env = Dv3TaskEnv('dmc_finger_turn_hard', seed=0)
obs = env.reset(); assert tuple(obs.shape) == (12,)
o2, r, d, info = env.step(env.rand_act())
assert tuple(o2.shape) == (12,) and isinstance(r, float)
print('env adapter OK: obs', tuple(obs.shape), 'reward', r)
EOF
echo "TD-MPC2 env setup complete."
