"""HiddenStakes — reward-relevant hidden-hazard wrapper (new-paper track).

ThreeAxes item 12: the OU distractor axis is reward-irrelevant by
construction, so information about it can never be worth anything —
the wrong load axis for a value-of-information paper. HiddenStakes
replaces it with a channel whose knowledge has GENUINE decision value:

  - a hidden 2-state Markov hazard h_t (P(stay) = p_stay; stationary
    start), h = 1 meaning the environment is in a hazardous regime;
  - reward relevance: r_out = r * (1 - scale * h) - cost * h — during
    hazard, achieved reward is scaled down and a flat cost accrues, so
    the optimal policy WAITS OUT hazards it can detect;
  - observability: an appended `stakes` key of `dim` dimensions, each
    emitting h * signal + N(0, obs_noise), plus a `log/hazard` truth
    key (log/-prefixed: excluded from the agent's obs by make_agent,
    available to analyses).

**Which knob is the information axis (build-review finding 9):** at
scale = 1.0 on a dense-reward task the REWARD itself reveals h almost
perfectly (measured classifier ~0.998 from reward alone), so obs_noise
does NOT hold reward relevance fixed while varying information — the
obs_noise sweep is confounded. The leak-immune axis is **p_stay**:
corr(stakes_t, h_{t+1}) ~ 0.64 at p_stay=.95 and ~0 at p_stay=.5, so
p_stay = 0.5 is a clean no-VoI control REGARDLESS of what the agent
can infer about h_t (there is nothing to predict). Experiments should
dial p_stay, treating obs_noise as a secondary knob.

The hazard does not touch dynamics or observations of the base env —
only the reward and the appended keys — so a scale=0, cost=0 dose is
reward-inert and isolates the pure-channel effect. The reward
transform is NOT applied on is_first steps (review finding 11: no
hazard cost on a step the agent did not act in).

Restorable-state protocol (d0.oracle_labels snapshot/restore_env):
oracle_get_state/oracle_set_state carry (h, rng), mirroring
distractor.py, so labeler branches replay identical hazard paths (CRN).

Config: `hiddenstakes: {dim: 0, ...}` in configs.yaml; dim 0 disables
(default path unchanged). Applied in dreamerv3.main.make_env with a
SeedSequence-derived stream, same rationale as the distractor.
"""

import functools
import json

import elements
import embodied
import numpy as np


class HiddenStakes(embodied.wrappers.Wrapper):

  def __init__(self, env, dim, p_stay=0.95, scale=1.0, cost=0.0,
               signal=1.0, obs_noise=0.5, key='stakes', seed=0):
    super().__init__(env)
    assert dim > 0, dim
    assert 0.5 <= p_stay < 1.0, p_stay
    self._dim = int(dim)
    self._p_stay = float(p_stay)
    self._scale = float(scale)
    self._cost = float(cost)
    self._signal = float(signal)
    self._obs_noise = float(obs_noise)
    self._key = key
    self._rng = np.random.default_rng(seed)
    # stationary distribution of the symmetric 2-state chain is uniform
    self._h = int(self._rng.random() < 0.5)

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    assert self._key not in spaces, (self._key, sorted(spaces.keys()))
    spaces[self._key] = elements.Space(np.float32, (self._dim,))
    spaces['log/hazard'] = elements.Space(np.float32, ())
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    if obs['is_first']:
      self._h = int(self._rng.random() < 0.5)
    else:
      if self._rng.random() > self._p_stay:
        self._h = 1 - self._h
    if not obs['is_first']:
      obs['reward'] = np.float32(
          float(obs['reward']) * (1.0 - self._scale * self._h)
          - self._cost * self._h)
    noise = self._rng.standard_normal(self._dim)
    obs[self._key] = np.float32(
        self._h * self._signal + self._obs_noise * noise)
    obs['log/hazard'] = np.float32(self._h)
    return obs

  # Restorable-state protocol for the oracle labeler.
  def oracle_get_state(self):
    return json.dumps(dict(
        h=self._h, rng=json.dumps(self._rng.bit_generator.state)))

  def oracle_set_state(self, blob):
    state = json.loads(blob)
    self._h = int(state['h'])
    self._rng.bit_generator.state = json.loads(state['rng'])


def selfcheck():
  from embodied.envs.planted import _StubEnv

  class _Rew(_StubEnv):
    def step(self, action):
      obs = super().step(action)
      obs['reward'] = np.float32(2.0)
      return obs

  act = {'action': np.zeros(1, np.float32)}

  # (1) reward transform exact: safe -> r; hazard -> r*(1-scale)-cost
  env = HiddenStakes(_Rew(seed=0), dim=3, p_stay=0.9, scale=0.5,
                     cost=0.25, obs_noise=0.0, seed=7)
  saw = {0: 0, 1: 0}
  for t in range(3000):
    obs = env.step(act)
    h = env._h
    saw[h] += 1
    want = 2.0 * (1.0 - 0.5 * h) - 0.25 * h
    assert abs(float(obs['reward']) - want) < 1e-6, (t, h, obs['reward'])
    # noiseless channel reveals h exactly
    assert np.allclose(obs['stakes'], float(h)), (h, obs['stakes'])
  assert saw[0] > 500 and saw[1] > 500, saw

  # (1b) truth key mirrors the hidden state; is_first steps are
  # transform-exempt (review finding 11)
  class _First(_Rew):
    def __init__(self, seed=0):
      super().__init__(seed=seed)
      self._t = 0

    def step(self, action):
      obs = super().step(action)
      obs['is_first'] = (self._t % 7 == 0)
      self._t += 1
      return obs

  envf = HiddenStakes(_First(seed=5), dim=2, p_stay=0.8, scale=0.5,
                      cost=0.25, seed=17)
  for t in range(500):
    obs = envf.step(act)
    assert float(obs['log/hazard']) == float(envf._h)
    if obs['is_first']:
      assert float(obs['reward']) == 2.0, (t, obs['reward'])

  # (2) hazard dwell: mean run length ~ 1/(1-p_stay)
  runs, cur, prev = [], 1, env._h
  env2 = HiddenStakes(_Rew(seed=1), dim=1, p_stay=0.95, seed=8)
  prev = env2._h
  for _ in range(20000):
    env2.step(act)
    if env2._h == prev:
      cur += 1
    else:
      runs.append(cur)
      cur, prev = 1, env2._h
  mean_run = float(np.mean(runs))
  assert 14.0 < mean_run < 28.0, mean_run   # true 20

  # (3) scale=0, cost=0 dose is reward-inert
  env3 = HiddenStakes(_Rew(seed=2), dim=2, scale=0.0, cost=0.0, seed=9)
  for _ in range(200):
    obs = env3.step(act)
    assert float(obs['reward']) == 2.0

  # (4) snapshot/restore CRN: branches replay identical hazard + noise
  from d0.oracle_labels import restore_env, snapshot_env
  env4 = HiddenStakes(_Rew(seed=3), dim=2, obs_noise=0.5, seed=10)
  for _ in range(50):
    env4.step(act)
  snap = snapshot_env(env4)
  assert any(isinstance(n, HiddenStakes) for n, _ in snap['custom'])
  path1 = [np.array(env4.step(act)['stakes']) for _ in range(20)]
  restore_env(env4, snap)
  path2 = [np.array(env4.step(act)['stakes']) for _ in range(20)]
  for a, b in zip(path1, path2):
    assert np.array_equal(a, b), 'restore did not replay the stream'

  # (5) obs noise dials informativeness: channel/hazard correlation
  # falls with obs_noise
  def corr(noise):
    e = HiddenStakes(_Rew(seed=4), dim=1, obs_noise=noise, seed=11)
    hs, ss = [], []
    for _ in range(4000):
      obs = e.step(act)
      hs.append(e._h)
      ss.append(float(obs['stakes'][0]))
    return float(np.corrcoef(hs, ss)[0, 1])
  assert corr(0.05) > 0.95 and corr(5.0) < 0.3, (corr(0.05), corr(5.0))

  print('hiddenstakes selfcheck PASS (reward transform exact + '
        'noiseless channel reveals h; dwell ~ 1/(1-p_stay); zero-dose '
        'reward-inert; snapshot/restore replays hazard+noise streams '
        'bitwise; obs_noise dials channel informativeness)')


if __name__ == '__main__':
  selfcheck()
