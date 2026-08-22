"""Region-penalty wrapper (Track A2 SCARECROW baseline;
PREREG_trackA_scarecrow_20260822.md).

REPLACES the env reward with a pure region penalty:
    reward = -scale * 1[obs[gate_key][gate_index] > gate_threshold]
evaluated on the PRE-STEP observation's successor (i.e. the obs this
step returns), using the same gate-triple convention as the
distractor/planted wrappers (embodied/envs/distractor.py:112,
planted.py:81). Replacement (not addition) is deliberate and pinned:
the comparator arms are otherwise reward-free explorers, so the task
reward must not leak into the penalty arms — the penalty is the ONLY
extrinsic signal.

Default-inert: scale == 0.0 disables the wrapper entirely (main.py
skips construction, mirroring the hiddenstakes dim-0 convention), so
every existing arm's command line reproduces byte-identical behavior.

Deterministic; no rng.

Run `python -m embodied.envs.regionpenalty` for the selfcheck.
"""

from __future__ import annotations

import numpy as np


class RegionPenalty:

  def __init__(self, env, scale=0.0, gate_key='', gate_index=0,
               gate_threshold=0.0):
    assert scale >= 0.0, scale
    assert gate_key, 'regionpenalty needs a gate_key when enabled'
    assert gate_key in env.obs_space, (gate_key, sorted(env.obs_space))
    self.env = env
    self._scale = float(scale)
    self._gate_key = gate_key
    self._gate_index = int(gate_index)
    self._gate_threshold = float(gate_threshold)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self.env, name)

  @property
  def obs_space(self):
    return self.env.obs_space

  @property
  def act_space(self):
    return self.env.act_space

  def _in_region(self, obs):
    val = np.asarray(obs[self._gate_key], np.float64).reshape(-1)
    return bool(val[self._gate_index] > self._gate_threshold)

  def step(self, action):
    obs = self.env.step(action)
    obs = dict(obs)
    pen = -self._scale if self._in_region(obs) else 0.0
    obs['reward'] = np.asarray(pen, dtype=np.asarray(obs['reward']).dtype)
    return obs

  def close(self):
    return self.env.close()


def selfcheck():
  class FakeEnv:
    obs_space = {'position': None, 'reward': None}
    act_space = {}

    def __init__(self):
      self._t = 0

    def step(self, action):
      self._t += 1
      # position[0] alternates below/above 0.5
      return {'position': np.array([1.0 if self._t % 2 else 0.0, 9.9],
                                   np.float32),
              'reward': np.float32(123.0)}

  env = RegionPenalty(FakeEnv(), scale=2.5, gate_key='position',
                      gate_index=0, gate_threshold=0.5)
  o1 = env.step({})   # t=1, position[0]=1.0 -> in region
  o2 = env.step({})   # t=2, position[0]=0.0 -> out
  assert float(o1['reward']) == -2.5, o1['reward']
  assert float(o2['reward']) == 0.0, o2['reward']
  assert o1['reward'].dtype == np.float32           # dtype preserved
  assert float(o1['position'][0]) == 1.0            # obs untouched
  # task reward fully REPLACED, never added
  assert float(o1['reward']) != 123.0 - 2.5
  # gate on the second index
  env2 = RegionPenalty(FakeEnv(), scale=1.0, gate_key='position',
                       gate_index=1, gate_threshold=5.0)
  o = env2.step({})
  assert float(o['reward']) == -1.0                 # 9.9 > 5.0 always
  # enabled wrapper without gate_key refuses
  try:
    RegionPenalty(FakeEnv(), scale=1.0)
    raise SystemExit('gate_key assert FAILED to fire')
  except AssertionError:
    pass
  print('regionpenalty selfcheck PASS (replace-not-add, dtype, '
        'gate-triple on both indices, no-gate refusal)')


if __name__ == '__main__':
  selfcheck()
