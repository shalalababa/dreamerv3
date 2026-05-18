"""Derive probe targets from collected probe trajectories.

Three families of targets, mirroring the project's "probe target" axis:

  * current simulator state   -- the full MuJoCo state (qpos | qvel);
  * future simulator state    -- the same array, shifted by the probing code;
  * derived dynamical quantities that are *not* a linear read-out of a single
    observation frame:
      - gait_phase  : instantaneous phase of the walking cycle, obtained as
                      the angle of the analytic (Hilbert) signal of the
                      antiphase hip oscillation, encoded as (cos, sin) so the
                      regression target is smooth across the 2*pi wrap;
      - return_to_go: discounted sum of future rewards (what DreamerV3's value
                      head estimates) -- a genuinely future-dependent target;
      - time_to_fall: steps until the torso first drops below a fall height
                      (heavily right-censored for a competent walker -- see
                      the `*_censored` mask).
  * downstream task reward    -- the dm_control Walker stand/walk/run reward
    recomputed analytically from the logged physics quantities, so the same
    common held-out set yields a reward-probe target for every task without
    re-collecting (research_procedure.md section G, probe 2).

`derive_targets` returns {name: (N, T, d) float32} plus a {name: (N, T) bool}
validity mask dict. The probing code handles horizon shifting and masking.
"""

import numpy as np
from scipy.signal import hilbert

FALL_HEIGHT = 0.6        # torso below this (m) counts as fallen (walker ~1.2)
RETURN_GAMMA = 0.99      # discount for return-to-go
TTF_CAP = 200            # cap (steps) for time-to-fall

# dm_control Walker reward constants (dm_control/suite/walker.py).
STAND_HEIGHT = 1.2
WALK_SPEED = 1.0
RUN_SPEED = 8.0


def _episode_phase(signal):
  """Instantaneous phase of a 1-D oscillation via the analytic signal."""
  signal = signal - signal.mean()
  if np.allclose(signal, 0.0):
    return np.zeros_like(signal)
  return np.angle(hilbert(signal))


def gait_phase(qpos, qpos_names):
  """(N, T) gait phase from the antiphase hip oscillation."""
  try:
    r = qpos_names.index('right_hip')
    l = qpos_names.index('left_hip')
    sig = qpos[..., r] - qpos[..., l]
  except (ValueError, AttributeError):
    sig = qpos[..., 3]                       # fallback: first non-root joint
  phase = np.stack([_episode_phase(sig[n]) for n in range(sig.shape[0])], 0)
  return phase.astype(np.float32)


def return_to_go(reward, gamma=RETURN_GAMMA):
  """(N, T) discounted sum of current and future rewards."""
  N, T = reward.shape
  out = np.zeros((N, T), np.float32)
  acc = np.zeros(N, np.float32)
  for t in reversed(range(T)):
    acc = reward[:, t] + gamma * acc
    out[:, t] = acc
  return out


def time_to_fall(torso_height, cap=TTF_CAP, fall_height=FALL_HEIGHT):
  """(N, T) steps until the next fallen frame, plus a censoring mask."""
  N, T = torso_height.shape
  fallen = torso_height < fall_height
  ttf = np.full((N, T), cap, np.float32)
  censored = np.ones((N, T), bool)
  for n in range(N):
    next_fall = cap
    for t in reversed(range(T)):
      next_fall = 0 if fallen[n, t] else next_fall + 1
      if next_fall <= cap:
        ttf[n, t] = next_fall
        censored[n, t] = (next_fall == cap and not fallen[n, t])
  return ttf, censored


def _tolerance(x, lo, hi, margin, value_at_margin=0.1, sigmoid='gaussian'):
  """NumPy reimplementation of dm_control's `rewards.tolerance`.

  Returns 1 inside [lo, hi] and a sigmoid-shaped falloff outside, reaching
  `value_at_margin` at `margin` units past the bound.
  """
  x = np.asarray(x, np.float64)
  in_bounds = np.logical_and(lo <= x, x <= hi)
  if margin == 0:
    return np.where(in_bounds, 1.0, 0.0).astype(np.float32)
  d = np.where(x < lo, lo - x, x - hi) / margin
  if sigmoid == 'gaussian':
    scale = np.sqrt(-2.0 * np.log(value_at_margin))
    val = np.exp(-0.5 * (d * scale) ** 2)
  elif sigmoid == 'linear':
    scale = 1.0 - value_at_margin
    val = np.maximum(1.0 - d * scale, 0.0)
  else:
    raise NotImplementedError(sigmoid)
  return np.where(in_bounds, 1.0, val).astype(np.float32)


def walker_task_rewards(traj):
  """Recompute the dm_control Walker stand/walk/run rewards from logged physics.

  Needs the named scalars `collect.py` logs (torso height/upright, horizontal
  velocity). Returns {} if any are missing. The reward is task-specific while
  the physics is not, so one common held-out set yields all three targets.
  """
  need = ('phys_torso_height', 'phys_torso_upright', 'phys_horizontal_velocity')
  if not all(k in traj for k in need):
    return {}
  height = np.asarray(traj['phys_torso_height'], np.float32)
  upright = np.asarray(traj['phys_torso_upright'], np.float32)
  velocity = np.asarray(traj['phys_horizontal_velocity'], np.float32)
  standing = _tolerance(height, STAND_HEIGHT, np.inf, STAND_HEIGHT / 2)
  upright01 = (1.0 + upright) / 2.0
  stand_reward = (3.0 * standing + upright01) / 4.0
  out = {'reward_stand': stand_reward.astype(np.float32)}
  for name, speed in (('reward_walk', WALK_SPEED), ('reward_run', RUN_SPEED)):
    move = _tolerance(velocity, speed, np.inf, speed / 2,
                      value_at_margin=0.5, sigmoid='linear')
    out[name] = (stand_reward * (5.0 * move + 1.0) / 6.0).astype(np.float32)
  return out


def derive_targets(traj, meta):
  """Return (targets, masks) dicts of (N, T, d) / (N, T) arrays."""
  phys = np.asarray(traj['phys_state'], np.float32)        # (N, T, S)
  N, T, S = phys.shape
  half = S // 2
  qpos, qvel = phys[..., :half], phys[..., half:]
  qpos_names = meta.get('qpos_names', None)

  targets, masks = {}, {}
  ones = np.ones((N, T), bool)

  # Current full simulator state and its components.
  targets['state'] = phys
  targets['qpos'] = qpos
  targets['qvel'] = qvel
  for k in ('state', 'qpos', 'qvel'):
    masks[k] = ones

  # Derived dynamical quantities.
  phase = gait_phase(qpos, qpos_names)
  targets['gait_phase'] = np.stack(
      [np.cos(phase), np.sin(phase)], -1).astype(np.float32)
  masks['gait_phase'] = ones

  targets['return_to_go'] = return_to_go(traj['reward'])[..., None]
  masks['return_to_go'] = ones

  if 'phys_torso_height' in traj:
    ttf, censored = time_to_fall(np.asarray(traj['phys_torso_height']))
    targets['time_to_fall'] = ttf[..., None]
    masks['time_to_fall'] = ~censored          # probe only uncensored steps

  # Downstream task reward (research_procedure.md section G, probe 2): one
  # reward-probe target per Walker task, recomputed from the logged physics.
  for name, value in walker_task_rewards(traj).items():
    targets[name] = value[..., None]
    masks[name] = ones

  # Simple single-frame physics references (sanity-check targets).
  for name in ('torso_height', 'torso_upright', 'horizontal_velocity'):
    key = f'phys_{name}'
    if key in traj:
      targets[name] = np.asarray(traj[key], np.float32)[..., None]
      masks[name] = ones

  return targets, masks
