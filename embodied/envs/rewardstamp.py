"""RewardStamp — in-vivo certified-injection wrapper (Stage B).

Registered under PREREG_p2_stageb_injection_20260821.md (ThreeAxes item
8): a known, referee-computable reward bonus planted into the LIVE env
so the labeling chain can be calibrated against a positive control that
flows through the real probe/selection path (unlike Stage A's
arithmetic injection into archived arrays).

Semantics — ARM-ON-RESTORE, one step:
  Every labeler branch (G rollout, op_real probe, the restore-resume of
  the base trajectory) begins with d0.oracle_labels.restore_env, which
  calls oracle_set_state on every wrapper that exposes it. This wrapper
  uses that call as its trigger: after a restore, exactly the NEXT
  non-reset step's reward is stamped with

      reward += delta * clip(action[dim], -1, 1)

  and the stamp disarms. Consequences, by construction:
  - each candidate branch's G and the probe's r_real carry the planted
    per-candidate value delta * clip(a_m[dim]) exactly once (the
    referee is exact: planted(s, m) = delta * clip(cands[s, m, dim]));
  - follower steps are never stamped (disarmed after one step);
  - the value bootstrap Vhat comes from the frozen critic, which knows
    nothing of the stamp — the designed dissociation is
    probe-harvests / consumer-does-not;
  - the base trajectory's post-label resume step is stamped too, but
    reward does not enter physics and the policy is frozen, so state
    visitation is bit-identical to an unstamped pass and base-trajectory
    rewards are not stored by the labeler (disclosed in the prereg).

  The stamp changes no dynamics and no observation key; delta=0 wrapping
  is bitwise-inert on rewards.
"""

import json

import embodied
import numpy as np


class RewardStamp(embodied.wrappers.Wrapper):

  def __init__(self, env, delta, act_key='action', dim=0):
    super().__init__(env)
    self._delta = float(delta)
    self._act_key = act_key
    self._dim = int(dim)
    self._armed = False
    self.stamp_count = 0

  def step(self, action):
    armed, self._armed = self._armed, False
    obs = self.env.step(action)
    if armed and not bool(np.asarray(action.get('reset', False)).any()):
      a = float(np.asarray(action[self._act_key]).reshape(-1)[self._dim])
      obs['reward'] = np.float32(
          float(obs['reward']) + self._delta * float(np.clip(a, -1.0, 1.0)))
      self.stamp_count += 1
    return obs

  # Restorable-state protocol (d0.oracle_labels snapshot/restore_env).
  # The wrapper carries no stochastic state; the restore call itself is
  # the information: it marks the start of a branch, so it ARMS the
  # one-step stamp.
  def oracle_get_state(self):
    return json.dumps(dict(delta=self._delta, dim=self._dim))

  def oracle_set_state(self, blob):
    state = json.loads(blob)
    assert float(state['delta']) == self._delta, (state, self._delta)
    self._armed = True


def selfcheck():
  from embodied.envs.planted import _StubEnv

  class _Rec(_StubEnv):
    """StubEnv that reports a constant reward so stamping is visible."""

    def step(self, action):
      obs = super().step(action)
      obs['reward'] = np.float32(1.0)
      return obs

  env = RewardStamp(_Rec(seed=0), delta=2.0)
  act = {'action': np.asarray([0.25], np.float32),
         'reset': np.array(False)}
  # (1) unarmed by default: no stamp
  obs = env.step(act)
  assert float(obs['reward']) == 1.0, obs['reward']
  # (2) arm-on-restore stamps EXACTLY the next non-reset step, then
  # disarms; clip applies
  for a, want in ((0.25, 1.0 + 2.0 * 0.25), (3.0, 1.0 + 2.0 * 1.0),
                  (-0.5, 1.0 - 2.0 * 0.5)):
    env.oracle_set_state(env.oracle_get_state())
    obs = env.step({'action': np.asarray([a], np.float32),
                    'reset': np.array(False)})
    assert abs(float(obs['reward']) - want) < 1e-6, (a, obs['reward'])
    obs = env.step(act)
    assert float(obs['reward']) == 1.0, ('no disarm', obs['reward'])
  assert env.stamp_count == 3, env.stamp_count
  # (3) armed + reset: consumed without stamping
  env.oracle_set_state(env.oracle_get_state())
  env.step({'action': np.asarray([0.9], np.float32),
            'reset': np.array(True)})
  obs = env.step(act)
  assert float(obs['reward']) == 1.0 and env.stamp_count == 3
  # (4) delta identity guard on restore blobs
  try:
    env.oracle_set_state(json.dumps(dict(delta=99.0, dim=0)))
    raise AssertionError('delta mismatch not refused')
  except AssertionError as e:
    if 'not refused' in str(e):
      raise
  # (5) snapshot/restore integration: snapshot_env captures the wrapper,
  # restore_env arms it
  from d0.oracle_labels import restore_env, snapshot_env
  env2 = RewardStamp(_Rec(seed=1), delta=2.0)
  snap = snapshot_env(env2)
  assert any(isinstance(n, RewardStamp) for n, _ in snap['custom'])
  assert not env2._armed
  restore_env(env2, snap)
  assert env2._armed, 'restore did not arm the stamp'
  obs = env2.step(act)
  assert abs(float(obs['reward']) - (1.0 + 2.0 * 0.25)) < 1e-6
  # (6) delta=0 wrapping is reward-inert even when armed
  env3 = RewardStamp(_Rec(seed=2), delta=0.0)
  env3.oracle_set_state(env3.oracle_get_state())
  obs = env3.step(act)
  assert float(obs['reward']) == 1.0
  print('rewardstamp selfcheck PASS (unarmed default; arm-on-restore '
        'one-step stamp + clip + disarm; reset consumes without stamp; '
        'delta identity guard; snapshot_env/restore_env integration '
        'arms; delta=0 inert)')


if __name__ == '__main__':
  selfcheck()
