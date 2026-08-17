"""Offline FB (forward-backward) fit on an exported proprio side buffer.

FB wave (FB_Wave_Plan_20260816; prereg PREREG_fb_20260817). Trains
controllable_agent's FBDDPGAgent (pinned checkout
5a9950b07f1edcb4bddd04f5819379541793d975) offline on a
probing/fb_export.py npz. Bypasses their hydra/train scripts: the agent
is constructed directly from kwargs (verified: FBDDPGAgent(**kwargs)
builds FBDDPGAgentConfig(**kwargs)) and updated via
agent.update(buffer, step) on an in-memory ReplayBuffer assembled from
the export.

Buffer-assembly contract (verified against the pinned checkout,
in_memory_replay_buffer.py):
- episode arrays are (T+1)-row with a dummy reset row; our export's
  dv3-convention row 0 IS that reset row, so arrays map 1:1 and
  episode_length = T - 1;
- sample() unconditionally indexes _storage['physics'] -> a zero stub
  array is REQUIRED even though custom_reward is never used;
- update() ignores stored meta z (samples its own); future_ratio
  defaults to 0.0 so future_obs is never read (buffer future=0.99 only
  costs a geometric draw per batch).

Env: torch env with the pinned checkout on CONTROLLABLE_AGENT_ROOT
(reuse the tdmpc2 conda env + hydra-core; never bare python).

Usage:
  CONTROLLABLE_AGENT_ROOT=/workspace/controllable_agent \
  python -m probing.fb_fit fit --data $RUNROOT/fb_data/finger_q1_side0.npz \
      --logdir $RUNROOT/fbwm_finger_q1s0_seed1 --seed 1 \
      --updates 500000 [--device cuda]
Selfcheck (torch env, CPU ok):
  CONTROLLABLE_AGENT_ROOT=... python -m probing.fb_fit --selfcheck
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PIN = '5a9950b07f1edcb4bddd04f5819379541793d975'
# Frozen agent kwargs: the COMPLETE FBDDPGAgentConfig field set with the
# omegaconf interpolations resolved explicitly. SOLE declared deviation
# from upstream defaults: use_tb=True — a pure logging flag (it gates
# only the metric-dict blocks in update_fb/update_actor) required so
# agent.update() returns non-empty metrics; without it the training
# loop has no NaN guard and no per-update signal (review finding B2).
AGENT_KWARGS = dict(
    _target_='url_benchmark.agent.fb_ddpg.FBDDPGAgent', name='fb_ddpg',
    obs_type='states', device='cuda', lr=1e-4, lr_coef=1,
    fb_target_tau=0.01, update_every_steps=2, use_tb=True,
    use_wandb=False, use_hiplog=False, num_expl_steps=0,
    num_inference_steps=5120, hidden_dim=1024, backward_hidden_dim=526,
    feature_dim=512, z_dim=50, stddev_schedule='0.2', stddev_clip=0.3,
    update_z_every_step=300, update_z_proba=1.0, nstep=1,
    batch_size=1024, init_fb=True, update_encoder=True, goal_space=None,
    ortho_coef=1.0, log_std_bounds=(-5, 2), temp=1, boltzmann=False,
    debug=False, future_ratio=0.0, mix_ratio=0.5, rand_weight=False,
    preprocess=True, norm_z=True, q_loss=False, q_loss_coef=0.01,
    additional_metric=False, add_trunk=False)
BUFFER_DISCOUNT = 0.99  # upstream pretrain.py default (review M8)
BUFFER_FUTURE = 0.99
PROGRESS_EVERY = 2000  # gradient updates


def _import_ca():
  root = os.environ.get('CONTROLLABLE_AGENT_ROOT')
  assert root and os.path.isdir(root), \
      'CONTROLLABLE_AGENT_ROOT must point at the pinned checkout'
  head = os.path.join(root, '.git')
  if os.path.isdir(head):
    import subprocess
    sha = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    assert sha == PIN, ('checkout is not the pinned sha', sha, PIN)
  sys.path.insert(0, root)
  from url_benchmark.agent.fb_ddpg import FBDDPGAgent
  from url_benchmark.in_memory_replay_buffer import ReplayBuffer
  return FBDDPGAgent, ReplayBuffer


def make_buffer(ReplayBuffer, obs, action, reward):
  """(E,T,12)/(E,T,A)/(E,T) export arrays -> a sample()-ready buffer."""
  E, T, _ = obs.shape
  buf = ReplayBuffer(max_episodes=E, discount=BUFFER_DISCOUNT,
                     future=BUFFER_FUTURE)
  buf._storage = {
      'observation': obs.astype(np.float32),
      'action': action.astype(np.float32),
      'reward': reward.astype(np.float32)[..., None],
      'discount': np.ones((E, T, 1), np.float32),
      'physics': np.zeros((E, T, 1), np.float32),  # stub; sample() indexes it
  }
  buf._episodes_length = np.full(E, T - 1, np.int32)
  buf._idx = 0
  buf._full = True
  buf._collected_episodes = E
  buf._is_fixed_episode_length = True
  buf._episodes_selection_probability = None
  return buf


def fit(args):
  import torch
  FBDDPGAgent, ReplayBuffer = _import_ca()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  with np.load(args.data) as z:
    obs, action, reward = z['obs'], z['action'], z['reward']
  man = json.load(open(args.data + '.manifest.json'))
  assert man['obs_dim'] == obs.shape[2] and man['tool'] == 'fb_export_v1'
  os.makedirs(args.logdir, exist_ok=True)
  kwargs = dict(AGENT_KWARGS, device=args.device,
                obs_shape=(obs.shape[2],), action_shape=(action.shape[2],))
  agent = FBDDPGAgent(**kwargs)
  import dataclasses
  resolved = dataclasses.asdict(agent.cfg)
  for k, v in kwargs.items():
    got = resolved.get(k)
    got = tuple(got) if isinstance(got, (list, tuple)) else got
    want = tuple(v) if isinstance(v, (list, tuple)) else v
    assert got == want, ('constructed cfg drifted from AGENT_KWARGS',
                         k, got, want)
  buf = make_buffer(ReplayBuffer, obs, action, reward)
  b = buf.sample(8)
  assert b.obs.shape == (8, obs.shape[2]) and np.isfinite(
      np.asarray(b.obs)).all(), 'buffer smoke failed'

  cfg_snap = dict(agent_kwargs={k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in kwargs.items()},
                  buffer=dict(discount=BUFFER_DISCOUNT, future=BUFFER_FUTURE),
                  data=os.path.abspath(args.data),
                  data_sha256=man['output_sha256'],
                  data_manifest=man, seed=args.seed, updates=args.updates,
                  checkout_pin=PIN)
  with open(os.path.join(args.logdir, 'fb_config.json'), 'w') as f:
    json.dump(cfg_snap, f, indent=1)

  def write_progress(u):
    with open(os.path.join(args.logdir, 'FB_FIT_PROGRESS'), 'w') as f:
      f.write(f'update={u}\ntotal_updates={args.updates}\n'
              f'updated_at={time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')

  updates, step, t0 = 0, 0, time.time()
  last_metrics, tail = {}, []
  write_progress(0)
  while updates < args.updates:
    did_update = (step % AGENT_KWARGS['update_every_steps'] == 0)
    metrics = agent.update(buf, step)
    step += 1
    if did_update:
      updates += 1
      assert metrics, ('update() returned no metrics on a gradient step '
                       '- use_tb flag regression?', step)
      assert all(np.isfinite(v) for v in metrics.values()), \
          ('non-finite training metric', updates, metrics)
      last_metrics = {k: float(v) for k, v in metrics.items()}
      if updates > args.updates - 100:
        tail.append(last_metrics.get('fb_loss', float('nan')))
      if updates % PROGRESS_EVERY == 0 or updates == args.updates:
        write_progress(updates)
        if updates % (PROGRESS_EVERY * 10) == 0:
          print(f'[{time.time()-t0:.0f}s] update {updates}/{args.updates} '
                f'fb_loss={last_metrics.get("fb_loss")}', flush=True)

  import torch as _t
  _t.save(dict(forward_net=agent.forward_net.state_dict(),
               backward_net=agent.backward_net.state_dict(),
               actor=agent.actor.state_dict(),
               agent_kwargs=cfg_snap['agent_kwargs'],
               updates=updates, seed=args.seed, checkout_pin=PIN),
          os.path.join(args.logdir, 'fb_ckpt.pt'))
  write_progress(updates)
  with open(os.path.join(args.logdir, 'fb_metrics.json'), 'w') as f:
    json.dump(dict(final=last_metrics,
                   fb_loss_tail_median=float(np.nanmedian(tail))
                   if tail else None, n_tail=len(tail)), f, indent=1)
  with open(os.path.join(args.logdir, 'FB_FIT_DONE'), 'w') as f:
    f.write(f'updates={updates}\nwall_s={time.time()-t0:.0f}\n')
  print(f'DONE {updates} updates in {time.time()-t0:.0f}s '
        f'-> {args.logdir}/fb_ckpt.pt')


def load_agent(ckpt_path, device='cuda'):
  """Rebuild the agent from fb_ckpt.pt (shared by zeroshot/embed)."""
  import torch
  FBDDPGAgent, _ = _import_ca()
  ck = torch.load(ckpt_path, map_location=device)
  assert ck.get('checkout_pin') == PIN, ('ckpt from wrong checkout',
                                         ck.get('checkout_pin'))
  kwargs = dict(ck['agent_kwargs'])
  kwargs['obs_shape'] = tuple(kwargs['obs_shape'])
  kwargs['action_shape'] = tuple(kwargs['action_shape'])
  kwargs['log_std_bounds'] = tuple(kwargs.get('log_std_bounds', (-5, 2)))
  kwargs['device'] = device
  agent = FBDDPGAgent(**kwargs)
  agent.forward_net.load_state_dict(ck['forward_net'])
  agent.backward_net.load_state_dict(ck['backward_net'])
  agent.actor.load_state_dict(ck['actor'])
  for net in (agent.forward_net, agent.backward_net, agent.actor):
    net.eval()
  return agent, ck


def selfcheck():
  """Torch env required (CPU fine): tiny fit end-to-end + refusals."""
  import tempfile
  base = tempfile.mkdtemp(prefix='fb_fit_sc_')
  rng = np.random.default_rng(0)
  E, T = 4, 33
  obs = rng.standard_normal((E, T, 12)).astype(np.float32)
  action = rng.uniform(-1, 1, (E, T, 2)).astype(np.float32)
  reward = (rng.random((E, T)) > .9).astype(np.float32)
  data = os.path.join(base, 'data.npz')
  np.savez(data, obs=obs, action=action, reward=reward)
  import hashlib
  man = dict(tool='fb_export_v1', obs_dim=12, action_dim=2,
             n_episodes=E, ep_len=T, obs_keys=[],
             output_sha256=hashlib.sha256(open(data, 'rb').read())
             .hexdigest())
  json.dump(man, open(data + '.manifest.json', 'w'))
  logdir = os.path.join(base, 'run')
  ns = argparse.Namespace(data=data, logdir=logdir, seed=1, updates=6,
                          device='cpu')
  fit(ns)
  assert os.path.exists(os.path.join(logdir, 'FB_FIT_DONE'))
  agent, ck = load_agent(os.path.join(logdir, 'fb_ckpt.pt'), device='cpu')
  assert ck['updates'] == 6
  import torch
  with torch.no_grad():
    z = agent.backward_net(torch.zeros(3, 12))
  assert tuple(z.shape) == (3, 50), z.shape
  # determinism: same seed twice -> identical ckpt bytes? (state_dicts)
  logdir2 = os.path.join(base, 'run2')
  fit(argparse.Namespace(data=data, logdir=logdir2, seed=1, updates=6,
                         device='cpu'))
  a2, _ = load_agent(os.path.join(logdir2, 'fb_ckpt.pt'), device='cpu')
  with torch.no_grad():
    z2 = a2.backward_net(torch.zeros(3, 12))
  assert torch.equal(z, z2), 'same-seed CPU refit not deterministic'
  # refusal: wrong obs_dim manifest
  man2 = dict(man, obs_dim=11)
  data2 = os.path.join(base, 'data2.npz')
  np.savez(data2, obs=obs, action=action, reward=reward)
  json.dump(man2, open(data2 + '.manifest.json', 'w'))
  try:
    fit(argparse.Namespace(data=data2, logdir=os.path.join(base, 'r3'),
                           seed=1, updates=2, device='cpu'))
    raise SystemExit('FAIL: wrong-dim manifest accepted')
  except AssertionError:
    pass
  print('fb_fit selfcheck PASS (tiny fit + ckpt roundtrip + B shape + '
        'same-seed determinism + wrong-dim refusal)')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  ft = sub.add_parser('fit')
  ft.add_argument('--data', required=True)
  ft.add_argument('--logdir', required=True)
  ft.add_argument('--seed', type=int, required=True)
  ft.add_argument('--updates', type=int, required=True)
  ft.add_argument('--device', default='cuda')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  else:
    assert args.cmd == 'fit'
    fit(args)


if __name__ == '__main__':
  main()
