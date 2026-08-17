"""Ornstein-Uhlenbeck distractor observations for Gate D0 (E2 arm).

Implements the distractor wrapper pre-registered in
EVPI_theory_note_20260702.tex App. A.2: appended observation dimensions
driven by OU processes that are uncontrollable (no action coupling) and
reward-irrelevant by construction. The dose is (dim, scale): number of
distractor dimensions and their stationary sd as a multiple of the
per-dimension proprio sd (times m(x) when mod_key is set — the SE
gradient arm's amplitude modulation). Dose zero (E1) means the wrapper
is not applied.
"""

import functools

import elements
import embodied
import numpy as np


class Distractor(embodied.wrappers.Wrapper):

  def __init__(
      self, env, dim, scale=1.0, theta=0.1, basesd=0.0, calib=1000,
      key='distractor', gate_key='', gate_index=0, gate_threshold=0.0,
      mod_key='', mod_index=0, mod_lo=0.0, mod_hi=1.0, seed=0):
    super().__init__(env)
    assert dim > 0, dim
    self._dim = int(dim)
    self._scale = float(scale)
    self._theta = float(theta)
    self._key = key
    # Gating (SE Stage 2, PREREG_nfi_scale_exhibit_amend2_20260814):
    # if gate_key is set, the OU values are EMITTED only when
    # obs[gate_key][gate_index] > gate_threshold, else zeros. The OU
    # state recursion and rng consumption are gate-independent (no
    # stream desync — same rule as planted.py), so gated and ungated
    # wrappers with the same seed carry identical latent OU paths.
    # Default gate_key='' is bitwise-inert (all prior consumers).
    self._gate_key = gate_key
    self._gate_index = int(gate_index)
    self._gate_threshold = float(gate_threshold)
    if gate_key:
      assert gate_key in env.obs_space, gate_key
    # Amplitude modulation (SE gradient arm, amendment 3): if mod_key
    # is set, the emission is scaled by m = clip((x - mod_lo) /
    # (mod_hi - mod_lo), 0, 1) where x = obs[mod_key][mod_index] of
    # the SAME pre-append obs — a continuous spatial gradient of the
    # emitted amplitude (the gate is the binary special case). The OU
    # recursion and rng are mod-independent; the modulation coordinate
    # is itself observed, so the channel stays zero-marginal-
    # information. Default mod_key='' (m == 1) is bitwise-inert.
    self._mod_key = mod_key
    self._mod_index = int(mod_index)
    self._mod_lo = float(mod_lo)
    self._mod_hi = float(mod_hi)
    if mod_key:
      assert mod_key in env.obs_space, mod_key
      assert self._mod_hi > self._mod_lo, (mod_lo, mod_hi)
    self._rng = np.random.default_rng(seed)
    # Unit-stationary-sd AR(1): x' = (1 - theta) x + sqrt(1 - (1-theta)^2) eps,
    # so the emitted distractor sd is exactly scale * refsd at stationarity
    # (times m(x) when amplitude modulation is active).
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
    if self._gate_open(obs):
      obs[self._key] = np.float32(self._state * self._scale
                                  * self._refsd() * self._mod_factor(obs))
    else:
      obs[self._key] = np.zeros(self._dim, np.float32)
    return obs

  def _gate_open(self, obs):
    if not self._gate_key:
      return True
    val = np.asarray(obs[self._gate_key], np.float64).reshape(-1)
    return bool(val[self._gate_index] > self._gate_threshold)

  def _mod_factor(self, obs):
    if not self._mod_key:
      return 1.0
    x = float(np.asarray(obs[self._mod_key],
                         np.float64).reshape(-1)[self._mod_index])
    return min(max((x - self._mod_lo) / (self._mod_hi - self._mod_lo),
                   0.0), 1.0)

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


def selfcheck():
  """Gate extension checks (Amendment 2): default-inert, zeros-when-
  closed, no rng desync. Mirrors planted.py's selfcheck stub."""
  from embodied.envs.planted import _StubEnv

  # (1) default gate_key='' is bitwise-inert vs the pre-gate emission
  # rule (reference simulation of the same OU recursion)
  env = Distractor(_StubEnv(seed=3), dim=4, basesd=0.5, seed=11)
  rng = np.random.default_rng(11)
  state = rng.standard_normal(4)
  ar, noisesd = 1.0 - 0.1, np.sqrt(1.0 - (1.0 - 0.1) ** 2)
  for t in range(500):
    obs = env.step({'action': np.zeros(1, np.float32)})
    state = ar * state + noisesd * rng.standard_normal(4)
    assert np.array_equal(obs['distractor'],
                          np.float32(state * 1.0 * 0.5)), t

  # (2) gated wrapper: zeros exactly when closed, OU emission when open,
  # against the gate coordinate of the SAME obs
  base = _StubEnv(seed=5)
  env = Distractor(base, dim=4, basesd=0.5, seed=12,
                   gate_key='proprio', gate_index=1, gate_threshold=0.2)
  opens = closes = 0
  for t in range(2000):
    obs = env.step({'action': np.zeros(1, np.float32)})
    if obs['proprio'][1] > 0.2:
      opens += 1
      assert not np.array_equal(obs['distractor'], np.zeros(4)), t
    else:
      closes += 1
      assert np.array_equal(obs['distractor'],
                            np.zeros(4, np.float32)), t
  assert opens > 100 and closes > 100, (opens, closes)

  # (3) no rng desync: same seed, gated vs ungated — emissions agree
  # bitwise wherever the gate is open (identical latent OU paths)
  e1 = Distractor(_StubEnv(seed=7), dim=4, basesd=0.5, seed=13)
  e2 = Distractor(_StubEnv(seed=7), dim=4, basesd=0.5, seed=13,
                  gate_key='proprio', gate_index=0, gate_threshold=0.0)
  agree = 0
  for t in range(2000):
    o1 = e1.step({'action': np.zeros(1, np.float32)})
    o2 = e2.step({'action': np.zeros(1, np.float32)})
    if o2['proprio'][0] > 0.0:
      assert np.array_equal(o1['distractor'], o2['distractor']), t
      agree += 1
    else:
      assert np.array_equal(o2['distractor'], np.zeros(4, np.float32)), t
  assert agree > 100, agree

  # (4) Welford calibration x gate (review #23 m17): with basesd=0 the
  # reference sd is estimated from the INNER env's obs before the
  # emission is appended, so gating must not move frozen_refsd
  e3 = Distractor(_StubEnv(seed=9), dim=4, basesd=0.0, calib=300,
                  seed=14)
  e4 = Distractor(_StubEnv(seed=9), dim=4, basesd=0.0, calib=300,
                  seed=14, gate_key='proprio', gate_index=0,
                  gate_threshold=0.0)
  for t in range(400):
    e3.step({'action': np.zeros(1, np.float32)})
    e4.step({'action': np.zeros(1, np.float32)})
  assert e3.frozen_refsd == e4.frozen_refsd and e3.frozen_refsd, (
      e3.frozen_refsd, e4.frozen_refsd)

  # (5) amplitude modulation (amendment 3): emission tracks
  # m = clip((x-lo)/(hi-lo), 0, 1) of the SAME obs exactly; latent OU
  # path is mod-independent (same seed, mod vs no-mod agree after
  # dividing out m); m==1 default is inert (covered by check 1)
  e6 = Distractor(_StubEnv(seed=21), dim=4, basesd=0.5, seed=15,
                  mod_key='proprio', mod_index=1, mod_lo=-1.0, mod_hi=1.0)
  ref = np.random.default_rng(15)
  state6 = ref.standard_normal(4)
  interior = clip_lo = clip_hi = 0
  for t in range(3000):
    o6 = e6.step({'action': np.zeros(1, np.float32)})
    state6 = ar * state6 + noisesd * ref.standard_normal(4)
    x = float(np.asarray(o6['proprio'], np.float64)[1])
    m = min(max((x + 1.0) / 2.0, 0.0), 1.0)
    assert np.array_equal(o6['distractor'],
                          np.float32(state6 * 1.0 * 0.5 * m)), t
    if m == 0.0:
      clip_lo += 1
      assert np.array_equal(o6['distractor'], np.zeros(4, np.float32))
    elif m == 1.0:
      clip_hi += 1
    else:
      interior += 1
  assert interior > 1000 and clip_lo > 50 and clip_hi > 50, (
      interior, clip_lo, clip_hi)

  print(f'distractor gate selfcheck PASS (default-inert 500 steps; '
        f'gated zeros-when-closed {closes}/2000; no-desync agree '
        f'{agree}/2000; Welford gate-independent '
        f'refsd={e3.frozen_refsd:.4f}; modulation exact 3000 steps, '
        f'interior/lo/hi {interior}/{clip_lo}/{clip_hi})')


if __name__ == '__main__':
  selfcheck()
