"""Planted zero-information channels for the NFI scale exhibit (SE wave,
12 Aug 2026; PREREG_nfi_scale_exhibit_20260812.md §2).

Appends observation keys whose information content is zero BY
CONSTRUCTION:

  planted_dup0    exact duplicate of a pinned source proprio key (eps=0)
  planted_dup1    duplicate + iid N(0, (eps1*basesd)^2)   [default 0.05]
  planted_dup2    duplicate + iid N(0, (eps2*basesd)^2)   [default 0.5]
  planted_const   constant zeros (const_dim,)

The duplicate family carries zero MARGINAL information given the source
key; the constant carries none at all. The noisy-TV channel (N) is the
existing Distractor wrapper, composed separately with a PINNED basesd
(no online calibration window — prereg review minor 1).

`basesd` here is REQUIRED (pinned from the smoke run): the eps ladder is
defined in units of the frozen per-dim proprio sd, so the zero-marginal
statement holds from step 0 and the residual sd is exactly eps*basesd.

Gating (Stage 2): if `gate_key` is set, planted values are emitted only
when obs[gate_key][gate_index] > gate_threshold, else zeros. The gated
channels remain zero-marginal-information because the gating coordinate
is itself part of the observation vector (prereg §3).

Wrapper RNG is seeded from the caller (wire seed = training seed per
prereg review minor 2); it never touches the base env's RNG.

Selfcheck: python -m embodied.envs.planted
"""

import functools

import elements
import embodied
import numpy as np

EPS_LADDER = (0.0, 0.05, 0.5)


class Planted(embodied.wrappers.Wrapper):

  def __init__(self, env, source_key, basesd, eps=EPS_LADDER, const_dim=4,
               gate_key='', gate_index=0, gate_threshold=0.0, seed=0):
    super().__init__(env)
    assert basesd > 0, 'basesd must be PINNED (no online calibration)'
    assert source_key in env.obs_space, (source_key,
                                         sorted(env.obs_space.keys()))
    src = env.obs_space[source_key]
    assert np.issubdtype(src.dtype, np.floating) and len(src.shape) == 1, \
        (source_key, src)
    self._source_key = source_key
    self._src_dim = int(src.shape[0])
    self._basesd = float(basesd)
    self._eps = tuple(float(e) for e in eps)
    self._const_dim = int(const_dim)
    self._gate_key = gate_key
    self._gate_index = int(gate_index)
    self._gate_threshold = float(gate_threshold)
    if gate_key:
      assert gate_key in env.obs_space, gate_key
    self._rng = np.random.default_rng(seed)

  @property
  def keys(self):
    return ([f'planted_dup{i}' for i in range(len(self._eps))]
            + ['planted_const'])

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    for i in range(len(self._eps)):
      key = f'planted_dup{i}'
      assert key not in spaces, key
      spaces[key] = elements.Space(np.float32, (self._src_dim,))
    assert 'planted_const' not in spaces
    spaces['planted_const'] = elements.Space(np.float32, (self._const_dim,))
    return spaces

  def _gate_open(self, obs):
    if not self._gate_key:
      return True
    val = np.asarray(obs[self._gate_key], np.float64).reshape(-1)
    return bool(val[self._gate_index] > self._gate_threshold)

  def step(self, action):
    obs = self.env.step(action)
    src = np.asarray(obs[self._source_key], np.float64).reshape(-1)
    gate = self._gate_open(obs)
    for i, e in enumerate(self._eps):
      # RNG always consumed (gate must not desynchronize the stream).
      noise = self._rng.standard_normal(self._src_dim)
      val = src + e * self._basesd * noise if gate else np.zeros(self._src_dim)
      obs[f'planted_dup{i}'] = np.float32(val)
    obs['planted_const'] = np.zeros(self._const_dim, np.float32)
    return obs


# ------------------------------------------------------------------ selfcheck

class _StubEnv:
  """Minimal base env exposing one proprio key with known values."""

  def __init__(self, dim=3, seed=0):
    self._rng = np.random.default_rng(seed)
    self._dim = dim
    self.obs_space = {
        'proprio': elements.Space(np.float32, (dim,)),
        'reward': elements.Space(np.float32, ()),
        'is_first': elements.Space(bool, ()),
        'is_last': elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
    }
    self.act_space = {'action': elements.Space(np.float32, (1,))}
    self._rng_state_probe = None

  def step(self, action):
    return {
        'proprio': np.float32(self._rng.standard_normal(self._dim)),
        'reward': np.float32(0.0), 'is_first': False, 'is_last': False,
        'is_terminal': False,
    }


def selfcheck():
  basesd = 0.7
  # (1) construction identities, ungated
  env = Planted(_StubEnv(seed=1), 'proprio', basesd=basesd, seed=42)
  resid = {i: [] for i in range(3)}
  for t in range(4000):
    obs = env.step({'action': np.zeros(1, np.float32)})
    src = obs['proprio'].astype(np.float64)
    for i, e in enumerate(EPS_LADDER):
      resid[i].append(obs[f'planted_dup{i}'].astype(np.float64) - src)
    assert (obs['planted_const'] == 0).all()
  for i, e in enumerate(EPS_LADDER):
    r = np.concatenate(resid[i])
    if e == 0.0:
      assert np.abs(r).max() < 1e-6, 'dup0 must be EXACT duplicate'
    else:
      sd = r.std()
      assert abs(sd - e * basesd) / (e * basesd) < 0.05, (i, sd, e * basesd)
      assert abs(r.mean()) < 4 * e * basesd / np.sqrt(len(r))
  # (2) obs_space correctness
  sp = env.obs_space
  assert sp['planted_dup0'].shape == (3,) and sp['planted_const'].shape == (4,)
  # (3) RNG isolation: base env stream unchanged by wrapper presence
  bare = _StubEnv(seed=7)
  a = [bare.step(None)['proprio'] for _ in range(3)]
  wenv = Planted(_StubEnv(seed=7), 'proprio', basesd=1.0, seed=0)
  b = [wenv.step({'action': np.zeros(1)})['proprio'] for _ in range(3)]
  assert all((x == y).all() for x, y in zip(a, b)), 'base RNG contaminated'
  # (4) determinism: same wrapper seed -> identical planted streams
  e1 = Planted(_StubEnv(seed=3), 'proprio', basesd=1.0, seed=9)
  e2 = Planted(_StubEnv(seed=3), 'proprio', basesd=1.0, seed=9)
  for _ in range(50):
    o1, o2 = e1.step(None), e2.step(None)
    assert (o1['planted_dup2'] == o2['planted_dup2']).all()
  # (5) gating: emits iff gate coordinate above threshold; RNG stream
  # consumption is gate-independent (no desync)
  g = Planted(_StubEnv(seed=5), 'proprio', basesd=1.0, seed=11,
              gate_key='proprio', gate_index=0, gate_threshold=0.0)
  n_open = n_closed = 0
  for _ in range(2000):
    obs = g.step(None)
    if obs['proprio'][0] > 0.0:
      n_open += 1
      assert (obs['planted_dup0'] == obs['proprio']).all()
    else:
      n_closed += 1
      assert (obs['planted_dup0'] == 0).all()
  assert n_open > 500 and n_closed > 500
  # gate-independence of the noise stream: two wrappers, same seed, one
  # gated one not, must agree on dup1 wherever the gate is open
  u = Planted(_StubEnv(seed=5), 'proprio', basesd=1.0, seed=11)
  gg = Planted(_StubEnv(seed=5), 'proprio', basesd=1.0, seed=11,
               gate_key='proprio', gate_index=0, gate_threshold=0.0)
  for _ in range(200):
    ou, og = u.step(None), gg.step(None)
    if og['proprio'][0] > 0.0:
      assert (ou['planted_dup1'] == og['planted_dup1']).all()
  print('planted selfcheck PASS (dup0 exact; eps-ladder sds within 5%; '
        'const zero; base-RNG isolated; deterministic; gate clean, '
        'stream gate-independent)')


if __name__ == '__main__':
  selfcheck()
