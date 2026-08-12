#!/bin/bash
# One-time setup of the LeWM conda env + pinned official checkout
# (PREREG_lewm_20260812). Mirrors scripts/tdmpc2_env_setup.sh.
#
# Env vars: LEWM_CONDA_ENV (env prefix), LEWM_CHECKOUT (clone dir),
# LEWM_COMMIT (pin).
set -euo pipefail

LEWM_CONDA_ENV="${LEWM_CONDA_ENV:-/scratch/midway3/$USER/conda_envs/lewm}"
LEWM_CHECKOUT="${LEWM_CHECKOUT:-/scratch/midway3/$USER/le-wm}"
# main @ 2026-05-22 (audited 2026-08-12: MIT license; train.py/jepa.py;
# stable-worldmodel data interface)
LEWM_COMMIT="${LEWM_COMMIT:-8edfeb336732b5f3ce7b8b210d0ba370a09e2cac}"

if [ ! -d "$LEWM_CHECKOUT/.git" ]; then
  git clone https://github.com/lucas-maes/le-wm "$LEWM_CHECKOUT"
fi
git -C "$LEWM_CHECKOUT" fetch origin "$LEWM_COMMIT" || true
git -C "$LEWM_CHECKOUT" checkout "$LEWM_COMMIT"
echo "le-wm pinned at $(git -C "$LEWM_CHECKOUT" rev-parse HEAD)"

if [ ! -d "$LEWM_CONDA_ENV" ]; then
  conda create -y -p "$LEWM_CONDA_ENV" python=3.11
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEWM_CONDA_ENV"

# stable-worldmodel pins the data interface this wave's exporter writes
# (HDF5Writer schema audited from the 0.1.1 source 2026-08-12; validated
# locally by probing/lewm_export.py --selfcheck against the vendored
# source). The [format] extra carries h5py+hdf5plugin — WITHOUT it the
# HDF5 format is silently skipped by the format registry and LeWM's
# load_dataset cannot read our .h5 exports.
# FAIL-CLOSED (review C5): a soft-failed [train] extra would let the
# env smoke exit 0 with a broken training env.
python -m pip install 'stable-worldmodel[format]==0.1.1'
python -m pip install 'stable-worldmodel[train]==0.1.1'

# import smoke: the interfaces the wave depends on, INCLUDING format
# registration (guards the silent-skip failure mode)
python - <<'EOF'
from stable_worldmodel.data.formats.hdf5 import HDF5Writer, HDF5Dataset
from stable_worldmodel.data.format import FORMATS, detect_format
assert any(f.__name__ == 'HDF5' or getattr(f, 'name', '') == 'hdf5'
           for f in (FORMATS.values() if hasattr(FORMATS, 'values')
                     else FORMATS)), f'hdf5 format not registered: {FORMATS}'
import h5py
print('stable-worldmodel data interface OK (hdf5 registered)')
EOF

# checkout import smoke (review C5): probing/lewm_embed.py resolves
# Hydra model targets defined inside the checkout — it must be
# importable via LEWM_CHECKOUT on sys.path.
LEWM_CHECKOUT="$LEWM_CHECKOUT" python - <<'EOF'
import os, sys
sys.path.insert(0, os.environ['LEWM_CHECKOUT'])
import jepa    # noqa: F401 — the audited model module
import module  # noqa: F401
print('le-wm checkout modules import OK')
EOF
echo "lewm env ready: $LEWM_CONDA_ENV (export LEWM_CHECKOUT=$LEWM_CHECKOUT for embed/verify)"
