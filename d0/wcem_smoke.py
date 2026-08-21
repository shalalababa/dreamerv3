"""Local smoke for the CEM consumer path (PREREG_p2_cem_consumer_20260821).

Builds a TINY agent with random weights (no checkpoint, no env, no
MuJoCo) and drives the real jitted policy path with d0.cem_iters > 0:
shape/finiteness/range checks on 'd0/cem_act', counter-CRN determinism
(same n_actions mark => bit-identical CEM action), and default-path
inertness (cem_iters 0 emits no cem keys). This validates the jax
plumbing only; head values are meaningless under random weights. The
checkpoint-level smoke (real R3 run, one labeled state) runs on
cluster per the prereg ops chain.

Usage: python -m d0.wcem_smoke
"""

import os
import pathlib
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import elements
import numpy as np
import ruamel.yaml as yaml


def make_tiny_agent(cem_iters):
  from dreamerv3.agent import Agent
  with open(REPO / 'dreamerv3' / 'configs.yaml') as f:
    full = yaml.YAML(typ='safe').load(f)
  config = elements.Config(full['defaults']['agent'])
  config = config.update({
      r'.*\.rssm': dict(deter=64, hidden=16, classes=4),
      r'.*\.depth': 2,
      r'.*\.units': 16,
      r'.*\.layers': 1,
      'valens.k': 2,
      'expl.disag_task': True,
      'expl.disag_ens': 2,
      'expl.disag_units': 16,
      'expl.disag_layers': 1,
      'd0.signals': True,
      'd0.cem_iters': cem_iters,
      'd0.cem_samples': 8,
      'd0.cem_horizon': 3,
      'd0.cem_elites': 3,
  })
  jax_cfg = dict(full['defaults']['jax'])
  jax_cfg['platform'] = 'cpu'
  config = elements.Config(
      **config, logdir='/tmp/wcem_smoke', seed=0, jax=jax_cfg,
      batch_size=2, batch_length=8, replay_context=0, report_length=8,
      replica=0, replicas=1)
  obs_space = {
      'proprio': elements.Space(np.float32, (6,)),
      'reward': elements.Space(np.float32, ()),
      'is_first': elements.Space(bool, ()),
      'is_last': elements.Space(bool, ()),
      'is_terminal': elements.Space(bool, ()),
  }
  act_space = {'action': elements.Space(np.float32, (2,), -1.0, 1.0)}
  return Agent(obs_space, act_space, config)


def _obs(batch=1):
  rng = np.random.default_rng(0)
  return {
      'proprio': rng.standard_normal((batch, 6)).astype(np.float32),
      'reward': np.zeros((batch,), np.float32),
      'is_first': np.ones((batch,), bool),
      'is_last': np.zeros((batch,), bool),
      'is_terminal': np.zeros((batch,), bool),
  }


def main():
  import jax
  agent = make_tiny_agent(cem_iters=3)
  jax.config.update('jax_transfer_guard', 'allow')
  carry = agent.init_policy(batch_size=1)
  obs = _obs()
  with agent.n_actions.lock:
    mark = int(agent.n_actions.value)
  # mode gate (review finding 2): plain eval calls pay nothing
  _, _, out_eval = agent.policy(carry, obs, mode='eval')
  assert not any(k.startswith('d0/cem') for k in out_eval), sorted(out_eval)
  with agent.n_actions.lock:
    agent.n_actions.value = mark
  carry1, acts, out = agent.policy(carry, obs, mode='cemplan')
  assert 'd0/cem_act' in out and 'd0/cem_score' in out, sorted(out)
  cem1 = np.asarray(out['d0/cem_act'])
  assert cem1.shape == (1, 2), cem1.shape
  assert np.isfinite(cem1).all() and np.isfinite(
      np.asarray(out['d0/cem_score'])).all()
  assert np.all(np.abs(cem1) <= 1.0 + 1e-6), cem1
  assert 'd0/qfull' in out, 'cem must compose with d0.signals'
  # counter-CRN: restore the mark, replay -> bit-identical CEM action
  with agent.n_actions.lock:
    agent.n_actions.value = mark
  _, _, out2 = agent.policy(carry, obs, mode='cemplan')
  cem2 = np.asarray(out2['d0/cem_act'])
  assert np.array_equal(cem1, cem2), (cem1, cem2)
  # ...and a DIFFERENT mark gives a different plan (stochastic search)
  _, _, out3 = agent.policy(carry, obs, mode='cemplan')
  cem3 = np.asarray(out3['d0/cem_act'])
  assert not np.array_equal(cem1, cem3), 'CEM ignored the RNG counter'
  # CRN corollary (review): the SAMPLED action at a given mark is
  # byte-identical whether or not the cem block ran (it runs after
  # sample(policy))
  with agent.n_actions.lock:
    agent.n_actions.value = mark
  _, acts_eval, _ = agent.policy(carry, obs, mode='eval')
  with agent.n_actions.lock:
    agent.n_actions.value = mark
  _, acts_cem, _ = agent.policy(carry, obs, mode='cemplan')
  assert np.array_equal(np.asarray(acts_eval['action']),
                        np.asarray(acts_cem['action']))
  # default path: cem_iters 0 emits no cem keys
  agent0 = make_tiny_agent(cem_iters=0)
  jax.config.update('jax_transfer_guard', 'allow')
  _, _, out0 = agent0.policy(agent0.init_policy(1), _obs(), mode='eval')
  assert not any(k.startswith('d0/cem') for k in out0), sorted(out0)
  print('WCEM SMOKE PASS (mode gate: eval emits nothing, cemplan emits '
        'cem_act/score; shape/finite/range; composes with d0.signals; '
        'counter-CRN bit-identical replay + fresh-mark divergence; '
        'sampled action mark-identical across modes; cem_iters=0 '
        'emits nothing)')


if __name__ == '__main__':
  main()
