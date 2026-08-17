"""Zero-shot task evaluation of an offline-trained FB fit.

FB wave (FB_Wave_Plan_20260816; prereg PREREG_fb_20260817) — the P-FB1
behavioral leg. Loads fb_ckpt.pt (probing/fb_fit.py), infers the task
z from (obs, reward) pairs drawn from the fit's OWN training export
(the buffer stores the env task reward even under reward-free
collection), then rolls the z-conditioned policy on dmc finger_turn_hard
via the pinned checkout's own dmc.make wrapper (obs_type='states',
action_repeat=1 — their default; returns are episode reward sums and
comparable to dv3-side returns at the return level; disclosed).

Registered constants: N_INFER=5120 reward-labeled transitions (the
agent's own num_inference_steps default), drawn without replacement by
a seeded rng from non-reset rows; N_EPISODES=10; env seed = 1000+seed;
act(eval_mode=True).

Usage:
  CONTROLLABLE_AGENT_ROOT=... MUJOCO_GL=disabled \
  python -m probing.fb_zeroshot eval --ckpt <logdir>/fb_ckpt.pt \
      --data $RUNROOT/fb_data/finger_q1_side<side>.npz \
      --task finger_turn_hard --seed <fit seed> --output <logdir>/zeroshot.json
Selfcheck (torch env, CPU, no dm_control):
  CONTROLLABLE_AGENT_ROOT=... python -m probing.fb_zeroshot --selfcheck
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np

from probing.fb_fit import PIN, load_agent, _import_ca  # noqa: F401
from probing.fb_embed import sha256_file

N_INFER = 5120
N_EPISODES = 10
ENV_SEED_BASE = 1000


def infer_task_z(agent, data_path, seed):
  import torch
  with np.load(data_path) as z:
    obs, reward = z['obs'], z['reward']
  E, T, D = obs.shape
  flat_obs = obs[:, 1:].reshape(-1, D)          # drop reset rows
  flat_rew = reward[:, 1:].reshape(-1)
  assert len(flat_obs) >= N_INFER, ('too few transitions', len(flat_obs))
  rng = np.random.default_rng(seed)
  idx = rng.choice(len(flat_obs), size=N_INFER, replace=False)
  with contextlib.redirect_stdout(io.StringIO()):  # their reward prints
    meta = agent.infer_meta_from_obs_and_rewards(
        torch.as_tensor(flat_obs[idx], dtype=torch.float32,
                        device=agent.cfg.device),
        torch.as_tensor(flat_rew[idx], dtype=torch.float32,
                        device=agent.cfg.device).unsqueeze(1))
  zvec = np.asarray(meta['z'], np.float32).reshape(-1)
  assert zvec.shape == (agent.cfg.z_dim,) and np.isfinite(zvec).all()
  frac_pos = float((flat_rew[idx] > 0).mean())
  z_norm = float(np.linalg.norm(zvec))
  assert frac_pos > 0, ('no positive reward in the inference sample - '
                        'z would be degenerate; REFUSE', frac_pos)
  assert z_norm > 0, 'zero-norm z - degenerate inference'
  stats = dict(n_infer=N_INFER, frac_pos_reward=frac_pos, z_norm=z_norm)
  return meta, stats


def rollout(agent, meta, env, n_episodes):
  returns = []
  for _ in range(n_episodes):
    ts = env.reset()
    total, steps = 0.0, 0
    while not ts.last():
      action = agent.act(ts.observation, meta, step=0, eval_mode=True)
      ts = env.step(np.asarray(action))
      total += float(ts.reward or 0.0)
      steps += 1
    returns.append(dict(ret=total, steps=steps))
  return returns


def run(args):
  agent, ck = load_agent(args.ckpt, device=args.device)
  meta, zstats = infer_task_z(agent, args.data, args.seed)
  sys_root = os.environ['CONTROLLABLE_AGENT_ROOT']
  from url_benchmark import dmc
  env = dmc.make(args.task, obs_type='states', frame_stack=1,
                 action_repeat=1, seed=ENV_SEED_BASE + args.seed)
  obs0 = env.reset().observation
  assert obs0.shape[-1] == ck['agent_kwargs']['obs_shape'][0], \
      ('env obs dim != trained obs dim — key-order/spec mismatch',
       obs0.shape, ck['agent_kwargs']['obs_shape'])
  from probing.fb_export import OBS_KEYS
  env_keys = list(env.task.get_observation(env.physics).keys())
  assert env_keys == list(OBS_KEYS), \
      ('env observation ORDER != pinned OBS_KEYS - zero-shot would be '
       'silently corrupted; REFUSE', env_keys, list(OBS_KEYS))
  eps = rollout(agent, meta, env, N_EPISODES)
  out = dict(tool='fb_zeroshot_v1', ckpt=os.path.abspath(args.ckpt),
             ckpt_sha256=sha256_file(args.ckpt),
             data=os.path.abspath(args.data), task=args.task,
             seed=args.seed, env_seed=ENV_SEED_BASE + args.seed,
             action_repeat=1, n_episodes=N_EPISODES,
             checkout_pin=PIN, checkout_root=sys_root,
             obs_key_order_ok=True, z_inference=zstats, episodes=eps,
             mean_return=float(np.mean([e['ret'] for e in eps])),
             updates=ck['updates'])
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=1)
  print(f"mean_return={out['mean_return']:.1f} -> {args.output}")


def run_random_floor(args):
  """Harness positive control (review B7): the SAME rollout loop with
  uniform random actions. Validates action scaling, episode length,
  reward accumulation and task identity end-to-end, and measures the
  no-learning floor INSIDE this protocol (retires the action_repeat
  cross-protocol caveat)."""
  _import_ca()  # review-2 B-A: this path never loaded the checkout
  from url_benchmark import dmc
  env = dmc.make(args.task, obs_type='states', frame_stack=1,
                 action_repeat=1, seed=ENV_SEED_BASE + args.seed)
  env.reset()  # symmetry with run()'s obs-dim probe reset (review-2 m7)
  spec = env.action_spec()
  rng = np.random.default_rng(args.seed)

  class _Rand:
    def act(self, obs, meta, step, eval_mode):
      return rng.uniform(spec.minimum, spec.maximum,
                         size=spec.shape).astype(np.float32)
  eps = rollout(_Rand(), {'z': None}, env, N_EPISODES)
  out = dict(tool='fb_random_floor_v1', task=args.task, seed=args.seed,
             env_seed=ENV_SEED_BASE + args.seed, action_repeat=1,
             n_episodes=N_EPISODES, episodes=eps,
             checkout_pin=PIN,
             checkout_root=os.environ.get('CONTROLLABLE_AGENT_ROOT'),
             mean_return=float(np.mean([e['ret'] for e in eps])))
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=1)
  print(f"random-floor mean_return={out['mean_return']:.1f} -> "
        f"{args.output}")


class _StubTS:
  def __init__(self, obs, reward, last):
    self.observation, self.reward = obs, reward
    self._last = last

  def last(self):
    return self._last


class _StubEnv:
  """8-step episode, reward 1 when action[0] > 0 — harness bookkeeping."""

  def __init__(self, dim):
    self.dim, self.t = dim, 0

  def reset(self):
    self.t = 0
    return _StubTS(np.zeros(self.dim, np.float32), None, False)

  def step(self, action):
    self.t += 1
    r = 1.0 if float(np.asarray(action).reshape(-1)[0]) > 0 else 0.0
    return _StubTS(np.zeros(self.dim, np.float32), r, self.t >= 8)


def selfcheck():
  import tempfile
  import torch
  from probing import fb_fit
  base = tempfile.mkdtemp(prefix='fb_zs_sc_')
  rng = np.random.default_rng(0)
  E, T, D = 4, 33, 12
  obs = rng.standard_normal((E, T, D)).astype(np.float32)
  action = rng.uniform(-1, 1, (E, T, 2)).astype(np.float32)
  reward = (obs[..., 0] > 1.0).astype(np.float32)  # obs-correlated reward
  data = os.path.join(base, 'data.npz')
  np.savez(data, obs=obs, action=action, reward=reward)
  import hashlib
  json.dump(dict(tool='fb_export_v1', obs_dim=D, action_dim=2,
                 n_episodes=E, ep_len=T, obs_keys=[],
                 output_sha256=hashlib.sha256(open(data, 'rb').read())
                 .hexdigest()),
            open(data + '.manifest.json', 'w'))
  logdir = os.path.join(base, 'run')
  fb_fit.fit(argparse.Namespace(data=data, logdir=logdir, seed=1,
                                updates=4, device='cpu'))
  agent, _ = load_agent(os.path.join(logdir, 'fb_ckpt.pt'), device='cpu')
  keep_n = N_INFER
  globals()['N_INFER'] = 64  # tiny data
  meta, zstats = infer_task_z(agent, data, seed=1)
  assert zstats['z_norm'] > 0
  # same seed -> identical z (deterministic inference)
  meta2, _ = infer_task_z(agent, data, seed=1)
  assert np.array_equal(np.asarray(meta['z']), np.asarray(meta2['z']))
  eps = rollout(agent, meta, _StubEnv(D), 3)
  assert len(eps) == 3 and all(e['steps'] == 8 for e in eps)
  globals()['N_INFER'] = keep_n
  print('fb_zeroshot selfcheck PASS (z inference deterministic + '
        'rollout bookkeeping on stub env)')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  ev = sub.add_parser('eval')
  ev.add_argument('--ckpt', required=True)
  ev.add_argument('--data', required=True)
  ev.add_argument('--task', default='finger_turn_hard')
  ev.add_argument('--seed', type=int, required=True)
  ev.add_argument('--output', required=True)
  ev.add_argument('--device', default='cuda')
  rf = sub.add_parser('random_floor')
  rf.add_argument('--task', default='finger_turn_hard')
  rf.add_argument('--seed', type=int, default=0)
  rf.add_argument('--output', required=True)
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  elif args.cmd == 'random_floor':
    run_random_floor(args)
  else:
    assert args.cmd == 'eval'
    run(args)


if __name__ == '__main__':
  main()
