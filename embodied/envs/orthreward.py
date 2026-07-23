"""Orthogonal-objective reward override (Paper-1 orthogonal test).

Replaces the environment's scalar reward with a DIFFERENT dm_control
task's reward computed on the SAME physics. Observation space, dynamics,
embodiment, action space, and episode structure are untouched — only the
reward channel changes, so frozen world-model checkpoints trained on the
original task load exactly.

Supported objectives:
  finger_spin  the dm_control finger:spin sparse reward
               (hinge_velocity <= -15) evaluated on the finger Physics
               shared by the turn tasks. Canonical: calls the same
               Physics method and threshold constant as dm_control's
               Spin.get_reward — no re-implemented reward math.
"""

import numpy as np

import embodied


class OrthReward(embodied.wrappers.Wrapper):

  def __init__(self, env, task):
    super().__init__(env)
    assert task == 'finger_spin', f'unsupported orthreward task {task!r}'
    from dm_control.suite import finger as dm_finger
    self._threshold = float(dm_finger._SPIN_VELOCITY)
    # vars()-based walk: embodied Wrapper.__getattr__ delegates down the
    # chain, so plain getattr would mis-attribute the inner env's _dmenv.
    node, dmenv, seen = env, None, set()
    while node is not None and id(node) not in seen:
      seen.add(id(node))
      dmenv = vars(node).get('_dmenv') or dmenv
      node = vars(node).get('env') or vars(node).get('_env')
    assert dmenv is not None, 'orthreward: no dm_control env in the chain'
    assert hasattr(dmenv.physics, 'hinge_velocity'), (
        'orthreward finger_spin needs the finger domain Physics')
    self._physics = dmenv.physics

  def step(self, action):
    obs = self.env.step(action)
    if not bool(obs['is_first']):
      obs = dict(obs)
      obs['reward'] = np.float32(
          float(self._physics.hinge_velocity()) <= -self._threshold)
    return obs
