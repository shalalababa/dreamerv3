"""Shared TD-MPC2 <-> DreamerV3 compatibility layer (cross-family arm).

Everything that must agree between the offline buffer bridge and the
online adapt environment lives here, so agreement holds by construction:

- ``OBS_ORDER``: per-task observation-key concatenation order (dm_control
  observation_spec order, which is also what the official TD-MPC2
  DMControlWrapper produces natively).
- ``flatten_obs`` / ``episode_obs_matrix``: the one concatenation rule.
- ``build_cfg``: constructs a TD-MPC2 config dataclass from the official
  repo's config.yaml plus explicit overrides, replicating the derivation
  steps of ``common.parser.parse_cfg`` without hydra.
- ``Dv3TaskEnv``: the DreamerV3 embodied DMC env (SAME wrapper family and
  action-repeat=1 semantics that produced every study buffer) behind the
  env API the TD-MPC2 trainer expects.
- Distractor-dose composition (TM2-R3 wave, 2026-07-30): ``dose`` in
  {e1, e4} mirrors EXACTLY how DreamerV3 applies the OU distractor
  (dreamerv3/main.py make_env: the ``Distractor`` wrapper is attached
  directly around the DMC env, seed derived via
  ``np.random.SeedSequence([run_seed, env_index, 0xD0])``, and the dose
  constants come from configs.yaml — e1 = no wrapper, e4 = ``d0_dose3``
  = dim 32, scale 3.0, with the shared defaults theta 0.1, basesd 0.0,
  calib 1000). The distractor observation key is appended LAST to the
  canonical concatenation order.

The official repo is used at a pinned commit (see PREREG); set
``TDMPC2_ROOT`` to its ``tdmpc2/`` package directory.
"""

import os
import sys

import numpy as np

# dm_control observation_spec order per study task (asserted at runtime).
OBS_ORDER = {
    'dmc_finger_turn_hard': (
        'position', 'velocity', 'touch', 'target_position', 'dist_to_target'),
    'dmc_cup_catch': ('position', 'velocity'),
}
META_KEYS = ('reward', 'is_first', 'is_last', 'is_terminal', 'action',
             'consec', 'stepid', 'regime', 'in_regime')

# Distractor doses (TM2-R3): value-for-value from dreamerv3/configs.yaml
# (`distractor:` defaults line + `d0_dose3`). e1 = wrapper not applied.
DISTRACTOR_KEY = 'distractor'
DOSE_TABLE = {
    'e1': dict(dim=0, scale=1.0, theta=0.1, basesd=0.0, calib=1000),
    'e4': dict(dim=32, scale=3.0, theta=0.1, basesd=0.0, calib=1000),
}


def dose_config(dose):
  if dose not in DOSE_TABLE:
    raise SystemExit(
        f'unknown distractor dose {dose!r}; registered: {sorted(DOSE_TABLE)}')
  return dict(DOSE_TABLE[dose])


def distractor_seed(run_seed, index=0):
  """Mirror of dreamerv3/main.py make_env: SeedSequence, not hash()
  (tuples containing strings hash differently per process, which would
  make the distractor stream irreproducible)."""
  return int(np.random.SeedSequence(
      [int(run_seed), int(index), 0xD0]).generate_state(1)[0])


def add_tdmpc2_path(root=None):
  root = root or os.environ.get('TDMPC2_ROOT')
  if not root or not os.path.isdir(root):
    raise SystemExit(
        'TDMPC2_ROOT must point at the tdmpc2/ package dir of the pinned '
        'official checkout (e.g. $HOME/tdmpc2/tdmpc2).')
  if root not in sys.path:
    sys.path.insert(0, root)
  return root


def obs_keys(task):
  if task not in OBS_ORDER:
    raise SystemExit(f'No OBS_ORDER registered for task {task}; '
                     f'add the dm_control spec order to tdmpc2_compat.')
  return OBS_ORDER[task]


def flatten_obs(obs, task, extra=()):
  """Single-step obs dict -> 1-D float32 vector in canonical order.
  ``extra`` appends further keys (e.g. the distractor) AFTER the
  canonical task keys; default () keeps the historical behavior
  bit-for-bit for every existing caller."""
  parts = []
  for k in obs_keys(task) + tuple(extra):
    v = np.asarray(obs[k], np.float32).reshape(-1)
    parts.append(v)
  return np.concatenate(parts, 0)


def episode_obs_matrix(ep, task):
  """Episode dict of [T, ...] arrays -> [T, D] float32 matrix."""
  keys = obs_keys(task)
  missing = [k for k in keys if k not in ep]
  if missing:
    raise SystemExit(f'episode lacks obs keys {missing}; has {sorted(ep)}')
  cols = []
  for k in keys:
    v = np.asarray(ep[k], np.float32)
    cols.append(v.reshape(v.shape[0], -1))
  return np.concatenate(cols, 1)


def build_cfg(tdmpc2_root, task, obs_dim, action_dim, episode_length,
              overrides=None):
  """Official config.yaml + explicit overrides -> TD-MPC2 cfg dataclass.

  Replicates parse_cfg's derivations (model-size expansion, bin_size,
  single-task fields) without hydra.
  """
  add_tdmpc2_path(tdmpc2_root)
  from omegaconf import OmegaConf, open_dict
  from common import MODEL_SIZE
  from common.parser import cfg_to_dataclass

  cfg = OmegaConf.load(os.path.join(tdmpc2_root, 'config.yaml'))
  with open_dict(cfg):
    cfg.pop('defaults', None)
    cfg.task = task
    cfg.obs = 'state'
    cfg.multitask = False
    cfg.tasks = [task]
    cfg.task_dim = 0
    cfg.task_title = task
    cfg.obs_shape = {'state': (int(obs_dim),)}
    cfg.action_dim = int(action_dim)
    cfg.episode_length = int(episode_length)
    cfg.seed_steps = max(1000, 5 * int(episode_length))
    cfg.model_size = 5
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = True
    cfg.compile = False
    cfg.wandb_project = 'none'
    cfg.wandb_entity = 'none'
    cfg.checkpoint = 'none'
    cfg.data_dir = 'none'
    cfg.work_dir = '.'
    for k, v in (overrides or {}).items():
      cfg[k] = v
    for k, v in MODEL_SIZE[cfg.model_size].items():
      cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
  return cfg_to_dataclass(cfg)


class Dv3TaskEnv:
  """DreamerV3 embodied DMC env behind the TD-MPC2 trainer's env API.

  Semantics identical to the study buffers by construction: same wrapper
  family (embodied.envs.dmc.DMC), action repeat 1, per-step rewards,
  1000-step episodes, and the canonical obs concatenation above.
  """

  def __init__(self, task, seed=0, dose='e1'):
    import torch  # deferred: torch lives in the tdmpc2 env
    import gymnasium as gym
    from embodied.envs import dmc
    self._torch = torch
    self.task = task
    self.dose = dose
    dcfg = dose_config(dose)
    name = task.removeprefix('dmc_')
    # DMC ctor takes no seed; episode randomness comes from dm_control's
    # per-reset seeding. Sampling seeds (rand_act) use self._rng below.
    self._env = dmc.DMC(name, repeat=1, proprio=True, image=False,
                        render=False)
    self._extra_keys = ()
    if dcfg['dim']:
      # EXACT mirror of dreamerv3/main.py make_env: Distractor wraps the
      # DMC env directly; seed from SeedSequence([seed, index=0, 0xD0]).
      from embodied.envs.distractor import Distractor
      self._env = Distractor(self._env, **dcfg,
                             seed=distractor_seed(seed, 0))
      self._extra_keys = (DISTRACTOR_KEY,)
    aspace = self._env.act_space['action']
    self._action_shape = aspace.shape
    self.action_space = gym.spaces.Box(
        low=np.asarray(aspace.low, np.float32),
        high=np.asarray(aspace.high, np.float32), dtype=np.float32)
    d = int(sum(np.prod(self._env.obs_space[k].shape) or 1
                for k in obs_keys(task) + self._extra_keys))
    self.observation_space = gym.spaces.Box(
        low=np.full((d,), -np.inf, np.float32),
        high=np.full((d,), np.inf, np.float32), dtype=np.float32)
    self.max_episode_steps = 1000
    self._rng = np.random.default_rng(seed)

  def _flat(self, obs):
    return self._torch.from_numpy(
        flatten_obs(obs, self.task, extra=self._extra_keys))

  def rand_act(self):
    a = self._rng.uniform(self.action_space.low, self.action_space.high)
    return self._torch.from_numpy(a.astype(np.float32))

  def reset(self):
    obs = self._env.step({
        'action': np.zeros(self._action_shape, np.float32), 'reset': True})
    return self._flat(obs)

  def step(self, action):
    a = np.asarray(action.detach().cpu().numpy(), np.float32)
    obs = self._env.step({'action': a, 'reset': False})
    done = bool(obs['is_last'])
    info = {'success': 0.0, 'terminated': bool(obs['is_terminal'])}
    return self._flat(obs), float(obs['reward']), done, info
