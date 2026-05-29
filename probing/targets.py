"""Derive probe targets from collected Walker trajectories."""

import numpy as np
from scipy.signal import hilbert

FALL_HEIGHT = 0.6        # torso below this (m) counts as fallen (walker ~1.2)
RETURN_GAMMA = 0.99      # discount for return-to-go
TTF_CAP = 200            # cap (steps) for time-to-fall


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

  # Simple single-frame physics references (sanity-check targets).
  for name in ('torso_height', 'torso_upright', 'horizontal_velocity'):
    key = f'phys_{name}'
    if key in traj:
      targets[name] = np.asarray(traj[key], np.float32)[..., None]
      masks[name] = ones

  return targets, masks
