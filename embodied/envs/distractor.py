"""Ornstein-Uhlenbeck distractor observations for Gate D0 (E2 arm).

Implements the distractor wrapper pre-registered in
EVPI_theory_note_20260702.tex App. A.2: appended observation dimensions
driven by OU processes that are uncontrollable (no action coupling) and
reward-irrelevant by construction. The dose is (dim, scale): number of
distractor dimensions and their stationary sd as a multiple of the
per-dimension proprio sd. Dose zero (E1) means the wrapper is not applied.
"""

import functools

import elements
import embodied
import numpy as np


class Distractor(embodied.wrappers.Wrapper):

  def __init__(
      self, env, dim, scale=1.0, theta=0.1, basesd=0.0, calib=1000,
      key='distractor', seed=0):
    super().__init__(env)
    assert dim > 0, dim
    self._dim = int(dim)
    self._scale = float(scale)
    self._theta = float(theta)
    self._key = key
    self._rng = np.random.default_rng(seed)
    # Unit-stationary-sd AR(1): x' = (1 - theta) x + sqrt(1 - (1-theta)^2) eps,
    # so the emitted distractor sd is exactly scale * refsd at stationarity.
    self._ar = 1.0 - self._theta
    self._noisesd = np.sqrt(1.0 - self._ar ** 2)
    self._state = self._rng.standard_normal(self._dim)
    # Reference scale: mean per-dimension sd of the float proprio keys,
    # either fixed via basesd or estimated online (Welford) over the first
    # `calib` wrapper steps and frozen afterwards.
    self._basesd = float(basesd)
    self._calib = int(calib)
    self._calib_keys = sorted(
        k for k, v in env.obs_space.items()
        if k not in ('reward', 'is_first', 'is_last', 'is_terminal')
        and np.issubdtype(v.dtype, np.floating) and len(v.shape) <= 1)
    if not self._basesd:
      assert self._calib_keys, (
          'No float proprio keys to calibrate against; pass basesd > 0.',
          sorted(env.obs_space.keys()))
      pdims = sum(
          int(np.prod(env.obs_space[k].shape, dtype=int)) or 1
          for k in self._calib_keys)
      self._count = 0
      self._mean = np.zeros(pdims)
      self._m2 = np.zeros(pdims)
    self.frozen_refsd = self._basesd or None

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    assert self._key not in spaces, (self._key, sorted(spaces.keys()))
    spaces[self._key] = elements.Space(np.float32, (self._dim,))
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    if obs['is_first']:
      self._state = self._rng.standard_normal(self._dim)
    else:
      self._state = (
          self._ar * self._state +
          self._noisesd * self._rng.standard_normal(self._dim))
    if not self._basesd and self._count < self._calib:
      self._update_stats(obs)
    obs[self._key] = np.float32(self._state * self._scale * self._refsd())
    return obs

  def _update_stats(self, obs):
    values = np.concatenate(
        [np.asarray(obs[k], np.float64).reshape(-1) for k in self._calib_keys])
    self._count += 1
    delta = values - self._mean
    self._mean += delta / self._count
    self._m2 += delta * (values - self._mean)
    if self._count == self._calib:
      self.frozen_refsd = self._refsd()
      print(f'Distractor refsd frozen at {self.frozen_refsd:.5f} '
            f'({self._count} steps, {len(self._mean)} proprio dims).')

  def _refsd(self):
    if self._basesd:
      return self._basesd
    if self._count < 2:
      return 1.0
    persd = np.sqrt(self._m2 / (self._count - 1))
    return max(float(persd.mean()), 1e-6)

  # Restorable-state protocol for the oracle labeler (d0/oracle_labels
  # snapshot_env/restore_env). The OU value _state is mutable per-step
  # state: capturing only the RNG would replay identical noise increments
  # from branch-shifted starting values, breaking CRN in the distractor
  # dims. The Welford calibration triple is included so branch steps
  # cannot contaminate the reference-sd estimate.
  def oracle_get_state(self):
    import json
    state = dict(
        state=self._state.tolist(),
        rng=json.dumps(self._rng.bit_generator.state),
        frozen_refsd=self.frozen_refsd)
    if not self._basesd:
      state.update(count=self._count, mean=self._mean.tolist(),
                   m2=self._m2.tolist())
    return json.dumps(state)

  def oracle_set_state(self, blob):
    import json
    state = json.loads(blob)
    self._state = np.asarray(state['state'], np.float64)
    self._rng.bit_generator.state = json.loads(state['rng'])
    self.frozen_refsd = state['frozen_refsd']
    if not self._basesd:
      self._count = int(state['count'])
      self._mean = np.asarray(state['mean'], np.float64)
      self._m2 = np.asarray(state['m2'], np.float64)
