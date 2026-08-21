"""Offline Gate D0 signal sweep over a trained run's checkpoint.

Rolls out held-out evaluation episodes with the frozen policy and records
the per-state raw signals emitted by the agent's d0 path (per-head,
per-candidate-action one-step Q matrices at two rollout counts, plus
dynamics-ensemble disagreement), then attaches the derived signals from
d0.signals. One output npz per (task x dose x seed) cell; the analysis in
d0.analysis consumes these tables (thresholds always come from the
matching dose-zero cell, note App. A.5).

Usage:
  python -m d0.sweep --run_logdir <dir> --output <cell>.npz \
      --episodes 50 [--actions 8] [--rollouts 16] [--platform cpu]

The run must have been trained with the d0_probe config (valens.k > 0 and
expl.disag_task True); the distractor dose is read from the run's own
config, so E1/E2 need no extra flags here. For E2 cells the wrapper's
refsd calibration transient is discarded via a warmup before recording;
note the sweep's converged refsd is calibrated under the *trained* policy
and may differ from the training-time value (calibrated under the early
policy) -- it is recorded in the metadata for auditability.
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent, make_env
from d0 import signals


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run_logdir', required=True,
                 help='Run dir containing config.yaml and ckpt.')
  p.add_argument('--output', required=True, help='Output .npz path.')
  p.add_argument('--checkpoint', default='',
                 help='Checkpoint path (default: <run_logdir>/ckpt).')
  p.add_argument('--episodes', type=int, default=50,
                 help='Held-out eval episodes (App. A.5: >= 50).')
  p.add_argument('--max_steps', type=int, default=1000)
  p.add_argument('--actions', type=int, default=8,
                 help='Candidate actions M: policy mode + M-1 samples '
                      '[prov.; confirm at freeze].')
  p.add_argument('--rollouts', type=int, default=16,
                 help='One-step rollouts R per (head, action); signals are '
                      'also recorded at R/2 [prov.; confirm at freeze].')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def _flat(d, prefix=''):
  out = {}
  for k, v in d.items():
    key = f'{prefix}{k}'
    if isinstance(v, dict):
      out.update(_flat(v, key + '.'))
    else:
      out[key] = v
  return out


def _backfill_defaults(saved):
  """Add config keys the CURRENT code requires but an older run's saved
  config predates, using the repo defaults (PREREG_d1_relabel_amend2:
  schema-drift compatibility; e.g. `agent.model_obs` default '.*'
  reproduces the pre-key include-everything behavior). Only ABSENT keys
  are added — saved values always win — and every backfilled key is
  printed so label logs record exactly what was filled."""
  path = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      'dreamerv3', 'configs.yaml')
  with open(path) as f:
    defaults = yaml.YAML(typ='safe').load(f)['defaults']
  flat_saved = _flat(saved)
  missing = {
      k: v for k, v in _flat(defaults).items()
      if k not in flat_saved
      and not any(s.startswith(k + '.') for s in flat_saved)
      and not any(k.startswith(s + '.') for s in flat_saved)}
  for key, value in missing.items():
    node = saved
    parts = key.split('.')
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = value
  if missing:
    print('[load_config] saved config predates current schema; '
          'backfilled from defaults:', sorted(missing))
  return saved


def load_config(args, out_dir):
  """Returns (sweep config, train_seed).

  The saved config's seed identifies the training run (the analysis pairs
  E2 cells with the matching dose-zero run by task and seed index); the
  sweep overrides it with its own seed for env/policy RNG, so both are
  kept and recorded separately in the output metadata.
  """
  with open(os.path.join(args.run_logdir, 'config.yaml')) as f:
    saved = yaml.YAML(typ='safe').load(f)
  saved = _backfill_defaults(saved)
  config = elements.Config(saved)
  train_seed = int(config.seed)
  updates = {
      'logdir': out_dir,
      'agent.d0.signals': True,
      'agent.d0.actions': args.actions,
      'agent.d0.rollouts': args.rollouts,
      'seed': args.seed,
  }
  if getattr(args, 'cem_consumer', False):
    # WCEM consumer pins (PREREG_p2_cem_consumer_20260821); the labeler
    # meta records them and the frozen reader refuses any other values.
    updates.update({
        'agent.d0.cem_iters': 4,
        'agent.d0.cem_samples': 64,
        'agent.d0.cem_horizon': 6,
        'agent.d0.cem_elites': 8,
        'agent.d0.cem_std': 0.5,
    })
  if args.platform:
    updates['jax'] = {'platform': args.platform}
  return config.update(updates), train_seed


def load_frozen_agent(agent, ckpt):
  cp = elements.Checkpoint()
  cp.agent = agent
  if os.path.exists(os.path.join(ckpt, 'done')):
    cp.load(ckpt, keys=['agent'])
  else:
    cp = elements.Checkpoint(ckpt)
    cp.agent = agent
    cp.load(keys=['agent'])
  return agent


def distractor_refsd(env):
  """Frozen calibration scale of the distractor wrapper, if any."""
  try:
    return float(env.frozen_refsd)
  except (AttributeError, ValueError, TypeError):
    return None


def rollout_episode(agent, env, max_steps):
  carry = agent.init_policy(batch_size=1)
  zero_act = {k: np.zeros(v.shape, v.dtype)
              for k, v in env.act_space.items() if k != 'reset'}
  obs = env.step({**zero_act, 'reset': np.array(True)})
  rec = {k: [] for k in ('qfull', 'qhalf', 'udyn', 'reward')}
  for _ in range(max_steps):
    agent_obs = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
    carry, acts, out = agent.policy(carry, agent_obs, mode='eval')
    rec['qfull'].append(np.asarray(out['d0/qfull'][0], np.float32))
    rec['qhalf'].append(np.asarray(out['d0/qhalf'][0], np.float32))
    rec['udyn'].append(np.float32(out['d0/udyn'][0]))
    rec['reward'].append(np.float32(obs['reward']))
    if bool(obs['is_last']):
      break
    act = {k: np.asarray(acts[k][0]) for k in zero_act}
    obs = env.step({**act, 'reset': np.array(False)})
  return {k: np.stack(v, 0) for k, v in rec.items()}


def main():
  args = parse_args()
  out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
  os.makedirs(out_dir, exist_ok=True)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')

  config, train_seed = load_config(args, out_dir)
  assert config.agent.valens.k > 0, 'run was trained without valens heads'
  assert config.agent.expl.disag_task or config.agent.expl.mode == 'p2e', (
      'run was trained without a dynamics-disagreement ensemble')
  dose = dict(config.distractor)
  print(f'Task: {config.task} | dose: {dose} | train seed: {train_seed} | '
        f'checkpoint: {ckpt}')

  agent = make_agent(config)
  env = make_env(config, 0)
  load_frozen_agent(agent, ckpt)
  print('Loaded frozen DreamerV3 checkpoint.')

  # The sweep's fresh wrapper instance recalibrates refsd online over its
  # first `calib` steps; discard that transient before recording, or the
  # early E2 states carry a drifting dose (~calib/N of the sample --
  # material next to the 5% Q-I occupancy floor).
  warmup = int(config.distractor.calib) if config.distractor.dim else 0
  done = 0
  while done < warmup:
    traj = rollout_episode(agent, env, min(args.max_steps, warmup - done))
    done += len(traj['reward'])
  if warmup:
    print(f'Discarded {done} warmup steps; distractor refsd frozen at '
          f'{distractor_refsd(env)}.')

  eps = []
  for ep in range(args.episodes):
    traj = rollout_episode(agent, env, args.max_steps)
    eps.append(traj)
    print(f'  episode {ep:>3}: len={len(traj["reward"]):>4}  '
          f'return={float(traj["reward"].sum()):7.1f}  '
          f'udyn mean={float(traj["udyn"].mean()):.4f}')
  refsd = distractor_refsd(env)
  env.close()

  episode_ids = np.concatenate(
      [np.full(len(t['reward']), i, np.int32) for i, t in enumerate(eps)])
  qfull = np.concatenate([t['qfull'] for t in eps], 0)
  qhalf = np.concatenate([t['qhalf'] for t in eps], 0)
  udyn = np.concatenate([t['udyn'] for t in eps], 0)
  derived = signals.compute(
      qfull.astype(np.float64), qhalf.astype(np.float64), udyn)

  meta = dict(
      run_logdir=args.run_logdir, checkpoint=ckpt, task=config.task,
      distractor=dose, train_seed=train_seed, sweep_seed=args.seed,
      warmup_steps=warmup, distractor_refsd=refsd, episodes=len(eps),
      states=int(len(udyn)), heads=int(qfull.shape[1]),
      actions=args.actions, rollouts=args.rollouts)
  np.savez_compressed(
      args.output, qfull=qfull, qhalf=qhalf, episode=episode_ids,
      reward=np.concatenate([t['reward'] for t in eps], 0),
      meta=json.dumps(meta), **derived)
  print(f'Wrote {len(udyn)} states ({len(eps)} episodes) -> {args.output}')


if __name__ == '__main__':
  main()
