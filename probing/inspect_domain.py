"""Inspect DMC domains to fix the MuJoCo accessors for regime quantities.

Throwaway reconnaissance tool for Phase 2 / Gate 0.  For each task it loads the
raw ``dm_control`` suite env (no rendering, no EGL) and prints:

  * the observation dict keys/shapes and one sample observation,
  * qpos / qvel row names,
  * site / geom / body names (the pool from which regime distances are built),
  * candidate regime quantities and their value ranges over a random rollout,
  * task reward range (for the reward-derived regime).

Use it to confirm which observation components carry the mechanism-derived regime
quantity (ball-cup distance, fingertip-target distance, spinner-angle error) and
whether that quantity is recoverable from the proprio obs alone -- which decides
whether Phase 2 needs extra physics logging at all.

    python -m probing.inspect_domain \
        --tasks dmc_cup_catch dmc_finger_turn_hard dmc_reacher_hard --steps 300
"""

import argparse

import numpy as np
from dm_control import suite


def parse_task(dmc_task):
  """dmc_<domain>_<task> -> (domain, task) with the ball_in_cup alias."""
  assert dmc_task.startswith('dmc_'), dmc_task
  domain, task = dmc_task[len('dmc_'):].split('_', 1)
  if domain == 'cup':
    domain = 'ball_in_cup'
  return domain, task


def named_positions(physics, kind):
  """{name: xyz} for sites / geoms / bodies, best-effort."""
  try:
    axis = getattr(physics.named.data, kind)
    names = list(axis.axes.row.names)
    return {n: np.asarray(axis[n]).copy() for n in names}
  except Exception as e:
    return {'<error>': str(e)}


def regime_candidates(domain, obs, physics):
  """Domain-specific candidate regime scalars, from obs and/or physics."""
  out = {}
  if domain == 'ball_in_cup':
    pos = np.asarray(obs['position'], np.float32)  # cup(x,z), ball(x,z) (to check)
    out['pos[0:2]-pos[2:4] dist'] = float(np.linalg.norm(pos[:2] - pos[2:4]))
    try:
      sp = physics.named.data.site_xpos
      if 'target' in sp.axes.row.names and 'ball' in sp.axes.row.names:
        out['site ball-target dist'] = float(
            np.linalg.norm(sp['ball'] - sp['target']))
    except Exception:
      pass
  elif domain == 'finger':
    if 'dist_to_target' in obs:
      out['obs dist_to_target'] = float(np.asarray(obs['dist_to_target']))
    if 'target_position' in obs and 'position' in obs:
      out['|target_position|'] = float(np.linalg.norm(obs['target_position']))
  elif domain == 'reacher':
    if 'to_target' in obs:
      out['|to_target|'] = float(np.linalg.norm(obs['to_target']))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--tasks', nargs='+',
                 default=['dmc_cup_catch', 'dmc_finger_turn_hard',
                          'dmc_reacher_hard'])
  p.add_argument('--steps', type=int, default=300)
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args()
  rng = np.random.default_rng(args.seed)

  for dmc_task in args.tasks:
    domain, task = parse_task(dmc_task)
    print('\n' + '=' * 78)
    print(f'{dmc_task}   ->   suite.load({domain!r}, {task!r})')
    print('=' * 78)
    env = suite.load(domain, task)
    physics = env.physics
    ts = env.reset()
    obs = ts.observation

    print('-- observation dict --')
    for k, v in obs.items():
      v = np.asarray(v)
      print(f'   {k:20s} shape={v.shape}  sample={np.round(v.reshape(-1)[:6], 3)}')

    print('-- qpos / qvel names --')
    try:
      print('   qpos:', list(physics.named.data.qpos.axes.row.names))
      print('   qvel:', list(physics.named.data.qvel.axes.row.names))
    except Exception as e:
      print('   <error>', e)

    print('-- site names --', list(named_positions(physics, 'site_xpos').keys()))
    print('-- geom names --', list(named_positions(physics, 'geom_xpos').keys()))
    print('-- body names --', list(named_positions(physics, 'xpos').keys()))

    # Random rollout to measure candidate regime and reward ranges.
    spec = env.action_spec()
    stats = {}
    rewards = []
    for _ in range(args.steps):
      a = rng.uniform(spec.minimum, spec.maximum).astype(np.float32)
      ts = env.step(a)
      if ts.reward is not None:
        rewards.append(float(ts.reward))
      for k, v in regime_candidates(domain, ts.observation, physics).items():
        stats.setdefault(k, []).append(v)
      if ts.last():
        env.reset()

    print(f'-- candidate regime quantities over {args.steps} random steps --')
    for k, vals in stats.items():
      a = np.asarray(vals)
      print(f'   {k:28s} min={a.min():.4f}  max={a.max():.4f}  '
            f'mean={a.mean():.4f}  std={a.std():.4f}')
    if rewards:
      r = np.asarray(rewards)
      nz = (r > 0).mean()
      print(f'   {"reward":28s} min={r.min():.4f}  max={r.max():.4f}  '
            f'mean={r.mean():.4f}  frac>0={nz:.3f}')


if __name__ == '__main__':
  main()
