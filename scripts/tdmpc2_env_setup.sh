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

if [ ! -d "$TM2_CONDA_ENV" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -y -p "$TM2_CONDA_ENV" python=3.11
  conda activate "$TM2_CONDA_ENV"
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install tensordict torchrl gymnasium omegaconf hydra-core \
      dm_control mujoco numpy
  # dv3-repo import chain for the env adapter (embodied -> elements, jax cpu)
  pip install elements "jax[cpu]"
else
  echo "env $TM2_CONDA_ENV already exists; skipping create"
fi

echo "Smoke checks (GPU-independent):"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TM2_CONDA_ENV"
cd "$REPO"
python -m probing.tdmpc2_bridge selfcheck
TDMPC2_ROOT="$TDMPC2_CHECKOUT/tdmpc2" python - <<'EOF'
import os
from probing.tdmpc2_compat import add_tdmpc2_path, build_cfg
add_tdmpc2_path()
cfg = build_cfg(os.environ['TDMPC2_ROOT'], 'dmc_finger_turn_hard', 12, 2, 1000)
assert cfg.obs_shape == {'state': (12,)} and cfg.action_dim == 2
assert cfg.latent_dim == 512 and cfg.bin_size > 0
print('cfg build OK:', cfg.task, 'latent', cfg.latent_dim)
from probing.tdmpc2_compat import Dv3TaskEnv
env = Dv3TaskEnv('dmc_finger_turn_hard', seed=0)
obs = env.reset(); assert tuple(obs.shape) == (12,)
o2, r, d, info = env.step(env.rand_act())
assert tuple(o2.shape) == (12,) and isinstance(r, float)
print('env adapter OK: obs', tuple(obs.shape), 'reward', r)
EOF
echo "TD-MPC2 env setup complete."
