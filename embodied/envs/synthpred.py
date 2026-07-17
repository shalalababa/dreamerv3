"""Synthetic predictable-heads domain (`synth` suite, task `synth_reach`).

Purpose (Plan_PredictableHeads_20260717.md; triage #6): the engineered
third domain for the causal law "transfer is governed by usable
task-aligned information delivered through trunk-directed gradients".
Every factor the DMC domains entangle is an explicit dial here:

  * variance spectrum: D Ornstein-Uhlenbeck distractor dims (count and
    scale set the dominant-variance bulk = the g_k competitors of
    Theory_SpectralTransfer_20260717);
  * regime rarity: sparse reward inside a fixed target region of radius
    `radius` around a fixed goal -- occupancy of that region is set by
    the scripted collectors (probing/synth_buffers.py), not by the env;
  * dynamics coupling: `coupling` couples the distractor dims into the
    point-mass acceleration (0 = fully decoupled; >0 makes distractor
    modeling task-relevant -- the value-path wake-up dial).

State: 2-D point mass (position, velocity) in the [-1, 1]^2 box, force
actions, damped Euler integration, walls clip position and zero the
normal velocity. Obs are flat float vectors (dmc_proprio pipeline):
`position` (2), `velocity` (2), `distractor` (D), `to_target` (2 =
goal - position; a GOAL key, excluded from coverage by
probing/regimes.py). Reward = 1.0 iff ||to_target|| < radius.

The regime spec in probing/regimes.py ('synth') uses threshold 0.1
'below' on ||to_target|| -- it matches the DEFAULT radius; change the
radius dial only together with that spec.

Episodes are fixed length (default 1000 steps -> 1001 frames), matching
the fixed-length-episode assumptions of the frozen AUC pipeline.
"""

import elements
import embodied
import numpy as np

GOAL = np.array([0.6, 0.6], np.float64)


class SynthPred(embodied.Env):

  def __init__(
      self, task, length=1000, distractors=8, distractor_scale=1.0,
      coupling=0.0, radius=0.1, theta=0.1, dt=0.05, damping=0.9,
      accel=2.0, seed=0):
    assert task == 'reach', task
    self._length = int(length)
    self._d = int(distractors)
    self._scale = float(distractor_scale)
    self._coupling = float(coupling)
    self._radius = float(radius)
    self._dt = float(dt)
    self._damping = float(damping)
    self._accel = float(accel)
    self._ar = 1.0 - float(theta)
    self._noisesd = np.sqrt(1.0 - self._ar ** 2)
    self._rng = np.random.default_rng(seed)
    self._pos = np.zeros(2)
    self._vel = np.zeros(2)
    self._distr = np.zeros(self._d)
    self._count = 0
    self._done = True

  @property
  def obs_space(self):
    return {
        'position': elements.Space(np.float32, (2,)),
        'velocity': elements.Space(np.float32, (2,)),
        'distractor': elements.Space(np.float32, (self._d,)),
        'to_target': elements.Space(np.float32, (2,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (2,), -1.0, 1.0),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._reset()
      return self._obs(is_first=True)
    force = np.clip(np.asarray(action['action'], np.float64), -1.0, 1.0)
    acc = self._accel * force
    if self._coupling and self._d >= 2:
      acc = acc + self._coupling * self._distr[:2]
    self._vel = self._damping * self._vel + self._dt * acc
    self._pos = self._pos + self._dt * self._vel
    hit = (self._pos < -1.0) | (self._pos > 1.0)
    self._pos = np.clip(self._pos, -1.0, 1.0)
    self._vel = np.where(hit, 0.0, self._vel)
    self._distr = self._ar * self._distr + \
        self._noisesd * self._scale * self._rng.standard_normal(self._d)
    self._count += 1
    self._done = self._count >= self._length
    return self._obs(is_last=self._done)

  def _reset(self):
    while True:
      self._pos = self._rng.uniform(-0.8, 0.8, 2)
      if np.linalg.norm(GOAL - self._pos) > self._radius + 0.05:
        break
    self._vel = np.zeros(2)
    self._distr = self._scale * self._rng.standard_normal(self._d)
    self._count = 0
    self._done = False

  def _obs(self, is_first=False, is_last=False):
    to_target = GOAL - self._pos
    reward = 0.0 if is_first else \
        float(np.linalg.norm(to_target) < self._radius)
    return dict(
        position=self._pos.astype(np.float32),
        velocity=self._vel.astype(np.float32),
        distractor=self._distr.astype(np.float32),
        to_target=to_target.astype(np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=False,
    )
