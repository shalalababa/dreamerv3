#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FB wave conda env (PREREG_fb_20260817; review-2 finding B-B).
#
# WHY A DEDICATED ENV (do not reuse tdmpc2): the pinned
# controllable_agent checkout (5a9950b0…) declares
#   physics: np.ndarray = dataclasses.field(default=np.ndarray([]), ...)
# in url_benchmark/dmc.py, which Python >= 3.11 rejects at import time
# (ValueError: mutable default for field) — reproduced on 3.11.15. The
# tdmpc2 env is python=3.11, so fb_fit/fb_zeroshot/fb_embed-fb can never
# import there. Upstream's own env.sh pins python=3.8; we use 3.10 (the
# newest that accepts the dataclass default) rather than patching the
# pinned checkout. Precedent: scripts/lewm_env_setup.sh.
#
# Import chain needing coverage (verified by both reviews): torch,
# dm_env, dm_control, tqdm, hydra-core, omegaconf. sklearn NOT needed.
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_DIR="${FB_CONDA_ENV:-/workspace/conda_envs/fb}"
CHECKOUT="${CONTROLLABLE_AGENT_ROOT:-/workspace/controllable_agent}"
PIN=5a9950b07f1edcb4bddd04f5819379541793d975

if [ ! -d "$CHECKOUT/.git" ]; then
  git clone https://github.com/facebookresearch/controllable_agent "$CHECKOUT"
fi
git -C "$CHECKOUT" checkout --quiet "$PIN"
[ "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$PIN" ] || {
  echo "ERROR: checkout is not the pinned sha" >&2; exit 2; }

conda create -y -p "$ENV_DIR" python=3.10
"$ENV_DIR/bin/pip" install --no-input torch tqdm hydra-core omegaconf \
  dm_env "dm_control>=1.0.8,<1.1"

# Smoke gate (registered): the import that 3.11 fails must pass here,
# plus one CUDA forward if a GPU is visible.
CONTROLLABLE_AGENT_ROOT="$CHECKOUT" "$ENV_DIR/bin/python" - <<'EOF'
import os, sys
sys.path.insert(0, os.environ['CONTROLLABLE_AGENT_ROOT'])
from url_benchmark.agent.fb_ddpg import FBDDPGAgent  # noqa: F401
from url_benchmark import dmc  # noqa: F401
import torch
print('import OK; torch', torch.__version__,
      'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(4, 8, device='cuda') @ torch.randn(8, 2, device='cuda')
    print('cuda forward OK', x.shape, torch.cuda.get_device_name(0))
EOF
echo "fb env ready at $ENV_DIR"
