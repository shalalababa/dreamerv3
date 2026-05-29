"""Collect DMC Walker trajectories with observations, actions, and physics."""

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

import embodied
from dreamerv3.main import make_agent, make_env

PHYS_QUANTITIES = ('torso_height', 'torso_upright', 'horizontal_velocity')


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run_logdir', required=True,
                 help='DreamerV3 run dir containing config.yaml and ckpt.')
  p.add_argument('--output', required=True, help='Output .npz path.')
  p.add_argument('--checkpoint', default='',
                 help='Checkpoint path (default: <run_logdir>/ckpt).')
  p.add_argument('--episodes', type=int, default=40)
  p.add_argument('--max_steps', type=int, default=1000)
  p.add_argument('--mode', default='eval', choices=['eval', 'train'],
                 help='Policy mode; eval is the standard frozen-policy probe.')
  p.add_argument('--task', default='',
                 help='Override the env task, e.g. dmc_walker_run, for the '
                      'cross-task probing extension (same walker body).')
  p.add_argument('--random', action='store_true',
                 help='Use a random policy instead of the trained one.')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def load_run_config(run_logdir, platform, output_dir, random_agent, task=''):
  cfg_path = os.path.join(run_logdir, 'config.yaml')
  with open(cfg_path) as f:
    saved = yaml.YAML(typ='safe').load(f)
  config = elements.Config(saved)
  updates = {'logdir': output_dir, 'random_agent': random_agent}
  if platform:
    updates['jax'] = {'platform': platform}
  if task:
    updates['task'] = task
  config = config.update(updates)
  return config


def load_frozen_agent(agent, ckpt):
  """Load a DreamerV3 checkpoint into `agent`."""
  cp = elements.Checkpoint()
  cp.agent = agent
  if os.path.exists(os.path.join(ckpt, 'done')):
    cp.load(ckpt, keys=['agent'])
  else:
    cp = elements.Checkpoint(ckpt)
    cp.agent = agent
    cp.load(keys=['agent'])
  return agent


def get_physics(env):
  """Return the underlying MuJoCo physics object."""
  try:
    return env._dmenv.physics
  except Exception as e:
    raise RuntimeError(f'Could not access MuJoCo physics: {e}')


def read_physics(physics):
  state = np.asarray(physics.get_state(), np.float32)
  named = {}
  for name in PHYS_QUANTITIES:
    fn = getattr(physics, name, None)
    if callable(fn):
      named[name] = np.asarray(fn(), np.float32).reshape(())
  return state, named


def rollout_episode(agent, env, physics, obs_keys, act_keys, mode, max_steps):
  carry = agent.init_policy(batch_size=1)
  zero_act = {k: np.zeros(v.shape, v.dtype)
              for k, v in env.act_space.items() if k != 'reset'}
  obs = env.step({**zero_act, 'reset': np.array(True)})
  rec = {f'obs_{k}': [] for k in obs_keys}
  rec.update({k: [] for k in ('action', 'reward', 'is_first', 'is_last',
                              'phys_state')})
  named_keys = None
  for _ in range(max_steps):
    state, named = read_physics(physics)
    if named_keys is None:
      named_keys = sorted(named.keys())
      rec.update({f'phys_{k}': [] for k in named_keys})
    agent_obs = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
    carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)
    act = {k: np.asarray(acts[k][0]) for k in act_keys}
    for k in obs_keys:
      rec[f'obs_{k}'].append(np.asarray(obs[k], np.float32))
    rec['action'].append(np.concatenate(
        [np.asarray(act[k], np.float32).reshape(-1) for k in act_keys]))
    rec['reward'].append(np.float32(obs['reward']))
    rec['is_first'].append(bool(obs['is_first']))
    rec['is_last'].append(bool(obs['is_last']))
    rec['phys_state'].append(state)
    for k in named_keys:
      rec[f'phys_{k}'].append(named[k])
    if bool(obs['is_last']):
      break
    obs = env.step({**act, 'reset': np.array(False)})
  return {k: np.stack(v, 0) for k, v in rec.items()}


def main():
  args = parse_args()
  out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
  os.makedirs(out_dir, exist_ok=True)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')

  config = load_run_config(args.run_logdir, args.platform, out_dir,
                           args.random, args.task)
  print(f'Task: {config.task}  |  checkpoint: {ckpt}  |  mode: {args.mode}')

  agent = make_agent(config)
  env = make_env(config, 0)
  physics = get_physics(env)

  if not args.random:
    load_frozen_agent(agent, ckpt)
    print('Loaded frozen DreamerV3 checkpoint.')

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  obs_keys = sorted(k for k, v in agent.obs_space.items()
                    if k not in exclude and len(v.shape) <= 1)
  act_keys = sorted(agent.act_space.keys())
  print(f'Observation keys: {obs_keys}  |  action keys: {act_keys}')

  episodes = []
  for ep in range(args.episodes):
    traj = rollout_episode(agent, env, physics, obs_keys, act_keys,
                           args.mode, args.max_steps)
    episodes.append(traj)
    ret = float(traj['reward'].sum())
    print(f'  episode {ep:>3}: len={len(traj["reward"]):>4}  return={ret:7.1f}')
  env.close()

  lengths = sorted({len(e['reward']) for e in episodes})
  T = lengths[0]
  if len(lengths) > 1:
    print(f'Episodes have unequal lengths {lengths}; truncating to {T}.')
  data = {k: np.stack([e[k][:T] for e in episodes], 0)
          for k in episodes[0].keys()}

  meta = dict(
      run_logdir=args.run_logdir, checkpoint=ckpt, task=config.task,
      mode=args.mode, random=args.random, seed=args.seed,
      episodes=len(episodes), length=int(T), obs_keys=obs_keys,
      act_keys=act_keys, act_dim=int(data['action'].shape[-1]),
      phys_state_dim=int(data['phys_state'].shape[-1]),
      phys_named=[k[5:] for k in data if k.startswith('phys_')
                  and k != 'phys_state'])
  try:
    meta['qpos_names'] = list(physics.named.data.qpos.axes.row.names)
    meta['qvel_names'] = list(physics.named.data.qvel.axes.row.names)
  except Exception:
    pass

  np.savez_compressed(args.output, **data)
  with open(args.output + '.meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
  ret = data['reward'].sum(1)
  print(f'Saved {len(episodes)} x {T} steps -> {args.output}')
  print(f'Return over episodes: {ret.mean():.1f} +/- {ret.std():.1f}')


if __name__ == '__main__':
  main()
