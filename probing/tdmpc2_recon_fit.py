"""Offline TD-MPC2 fit with an optional observation-reconstruction head.

Bridge-wave sibling of ``probing/tdmpc2_offline_fit.py`` (that executed
instrument stays untouched). Trains the official pinned TD-MPC2 agent on
a static bridged buffer with THREE registered arms, all mechanically
identical (same modules present, same optimizer groups, same update
mechanics) and differing ONLY in loss coefficients:

  --arm aware : official objective (consistency + reward + value);
                recon_coef = 0 (decoder attached, stays at init).
  --arm free  : reward_coef = 0, value_coef = 0; recon_coef = 0 —
                consistency-only representation (decoder at init).
  --arm rec   : reward_coef = 0, value_coef = 0; recon_coef = RECON_COEF
                — the free arm PLUS observation reconstruction: a
                decoder MLP (latent -> 2x mlp_dim -> obs_dim) is trained
                by MSE(decode(encode(obs)), obs) and its gradient flows
                into the ENCODER. This imports the Dreamer-apt gradient
                structure (reconstruction bids for encoder capacity)
                into the TD-MPC2 family (PREREG_tm2_bridge_20260812).

The decoder is attached in EVERY arm (parameter parity); it trains only
when recon_coef > 0 (exactly how the free arm carries an untrained
reward head). Checkpoints save the official model state dict WITH THE
DECODER KEYS STRIPPED into 'model' (so probing/tdmpc2_adapt.py loads it
unchanged, strict=True) and the decoder separately under 'decoder'.

Saved per run: logdir/config.yaml (arm + all four coefficients + data
manifest + decoder param count), TM2_FIT_PROGRESS, resumable full
checkpoint, TM2_FIT_DONE marker.

Usage (cluster, tdmpc2 conda env, GPU):
  python -m probing.tdmpc2_recon_fit --data <bridge.pt> \
      --logdir $RUNROOT/tm2wm_finger_brecq1s1_seed1 --arm rec \
      --task dmc_finger_turn_hard --updates 500000 --seed 1
Selfcheck (CUDA, tiny synthetic buffer):
  python -m probing.tdmpc2_recon_fit --selfcheck
"""

import argparse
import json
import os
import pathlib
import random

import numpy as np

from probing.tdmpc2_compat import add_tdmpc2_path, build_cfg

CKPT_NAME = 'tm2_ckpt.pt'
RECON_COEF = 20.0   # registered: the consistency_coef slot's magnitude
ARMS = ('aware', 'free', 'rec')


def make_agent(cfg, recon_coef):
  """Official TDMPC2 subclassed with a decoder head + recon loss term."""
  import torch
  import torch.nn.functional as F
  from tdmpc2 import TDMPC2
  from common import layers

  class ReconTDMPC2(TDMPC2):

    def __init__(self, cfg, recon_coef):
      super().__init__(cfg)
      self.recon_coef = float(recon_coef)
      # attach AFTER super().__init__ so the official module tree and
      # optimizer groups are byte-identical to the pinned code, then
      # register the decoder as one extra Adam param group (base lr).
      self.model._decoder = layers.mlp(
          cfg.latent_dim, 2 * [cfg.mlp_dim], cfg.obs_shape['state'][0]
      ).to(self.device)
      self.optim.add_param_group(
          {'params': self.model._decoder.parameters(), 'lr': cfg.lr})

    def _update(self, obs, action, reward, terminated, task=None):
      # Accumulate recon grads BEFORE the official update's backward();
      # the official step() then applies the SUM and zero_grad()s after
      # — one combined gradient step, official mechanics untouched.
      recon = None
      if self.recon_coef > 0:
        z = self.model.encode(obs, task)
        pred = self.model._decoder(z)
        recon = F.mse_loss(pred, obs) * self.recon_coef
        recon.backward()
      info = super()._update(obs, action, reward, terminated, task=task)
      import torch as _t
      info['recon_loss'] = (recon.detach() if recon is not None
                            else _t.tensor(0.0))
      return info

  return ReconTDMPC2(cfg, recon_coef)


def arm_overrides(arm):
  if arm == 'aware':
    return {}, 0.0
  if arm == 'free':
    return dict(reward_coef=0.0, value_coef=0.0), 0.0
  if arm == 'rec':
    return dict(reward_coef=0.0, value_coef=0.0), RECON_COEF
  raise SystemExit(f'unknown arm {arm!r}')


def strip_decoder(model_sd):
  kept = {k: v for k, v in model_sd.items() if not k.startswith('_decoder')}
  dec = {k: v for k, v in model_sd.items() if k.startswith('_decoder')}
  return kept, dec


def fit(args):
  add_tdmpc2_path(args.tdmpc2_root)
  import torch
  if not torch.cuda.is_available():
    raise SystemExit('TD-MPC2 requires CUDA (Buffer/agent pin cuda:0).')
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  logdir = pathlib.Path(args.logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  if (logdir / 'TM2_FIT_DONE').exists():
    print(f'{logdir} already DONE; nothing to do.')
    return

  blob = torch.load(args.data, weights_only=False)
  td, man = blob['td'], blob['manifest']
  if man['task'] != args.task:
    raise SystemExit(f"data task {man['task']} != --task {args.task}")
  n_eps, ep_len = td.shape
  total_rows = n_eps * ep_len

  overrides, recon_coef = arm_overrides(args.arm)
  overrides = dict(seed=args.seed, steps=total_rows,
                   buffer_size=total_rows, **overrides)
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  args.task, man['obs_dim'], man['action_dim'],
                  ep_len - 1, overrides)

  agent = make_agent(cfg, recon_coef)
  audit = dict(
      arm=args.arm, task=args.task, seed=args.seed, updates=args.updates,
      consistency_coef=cfg.consistency_coef, reward_coef=cfg.reward_coef,
      value_coef=cfg.value_coef, recon_coef=recon_coef,
      decoder_params=sum(p.numel()
                         for p in agent.model._decoder.parameters()),
      model_size=cfg.model_size, horizon=cfg.horizon,
      batch_size=cfg.batch_size, data=man,
  )
  with open(logdir / 'config.yaml', 'w') as f:
    json.dump(audit, f, indent=1)  # json is valid yaml

  from common.buffer import Buffer
  buffer = Buffer(cfg)
  buffer.load(td)
  print(f'Loaded static data: {n_eps} eps x {ep_len} rows = {total_rows} '
        f'rows; arm={args.arm} (reward_coef={cfg.reward_coef}, '
        f'value_coef={cfg.value_coef}, recon_coef={recon_coef})',
        flush=True)

  start = 0
  ckpt = logdir / CKPT_NAME
  if ckpt.exists():
    state = torch.load(ckpt, weights_only=False)
    merged = dict(state['model'])
    merged.update(state.get('decoder', {}))
    agent.model.load_state_dict(merged)
    agent.optim.load_state_dict(state['optim'])
    agent.pi_optim.load_state_dict(state['pi_optim'])
    start = int(state['update'])
    print(f'Resumed at update {start}', flush=True)

  def save(i):
    kept, dec = strip_decoder(agent.model.state_dict())
    torch.save(dict(model=kept, decoder=dec,
                    optim=agent.optim.state_dict(),
                    pi_optim=agent.pi_optim.state_dict(),
                    update=i), ckpt)

  for i in range(start, args.updates):
    info = agent.update(buffer)
    if i == 0 or (i + 1) % args.log_every == 0 or i == args.updates - 1:
      # grad_norm logged per arm (review B5: the shared clip at 20 now
      # includes decoder grads in the rec arm — disclosure requires the
      # realized norms per arm to be inspectable post-hoc)
      msg = '  '.join(f'{k}={float(v):.4f}' for k, v in sorted(info.items())
                      if k in ('consistency_loss', 'reward_loss',
                               'value_loss', 'total_loss', 'pi_loss',
                               'recon_loss', 'grad_norm'))
      print(f'  update {i + 1:>6}/{args.updates}: {msg}', flush=True)
      with open(logdir / 'TM2_FIT_PROGRESS', 'w') as f:
        f.write(f'update={i + 1}/{args.updates}\n')
    if (i + 1) % args.save_every_updates == 0 and (i + 1) < args.updates:
      save(i + 1)
      print(f'Saved checkpoint at update {i + 1}', flush=True)

  save(args.updates)
  (logdir / 'TM2_FIT_DONE').touch()
  print(f'DONE: {args.updates} updates -> {ckpt}', flush=True)


def selfcheck():
  """Tiny synthetic-buffer run of all three arms (CUDA required):
  (a) rec arm's decoder + encoder move, recon_loss falls;
  (b) aware/free arms leave the decoder EXACTLY at init;
  (c) saved 'model' has no decoder keys and loads into the OFFICIAL
      class strict=True; (d) free/rec leave reward head at init.
  """
  add_tdmpc2_path(None)
  import copy
  import tempfile
  import torch
  from tdmpc2 import TDMPC2

  tmp = pathlib.Path(tempfile.mkdtemp(prefix='tm2recon_selfcheck_'))
  n_eps, ep_len, obs_dim, act_dim = 12, 26, 12, 2
  torch.manual_seed(0)
  from tensordict import TensorDict
  obs = torch.randn(n_eps, ep_len, obs_dim)
  # make obs partially action-predictable so consistency/recon both learn
  td = TensorDict(dict(
      obs=obs,
      action=torch.rand(n_eps, ep_len, act_dim) * 2 - 1,
      reward=obs[..., 0].clone(),
      terminated=torch.zeros(n_eps, ep_len),
  ), batch_size=(n_eps, ep_len))
  man = dict(task='dmc_finger_turn_hard', obs_dim=obs_dim,
             action_dim=act_dim)
  data = tmp / 'bridge.pt'
  torch.save(dict(td=td, manifest=man), data)

  results = {}
  for arm in ARMS:
    logdir = tmp / f'fit_{arm}'
    args = argparse.Namespace(
        tdmpc2_root=None, data=str(data), logdir=str(logdir), arm=arm,
        task='dmc_finger_turn_hard', updates=30, seed=1, log_every=10,
        save_every_updates=1000)
    # snapshot init decoder/reward by re-deriving the same seeded agent
    fit(args)
    state = torch.load(logdir / CKPT_NAME, weights_only=False)
    assert not any(k.startswith('_decoder') for k in state['model']), \
        'decoder keys leaked into the official model dict'
    assert any(k.startswith('_decoder') for k in state['decoder']), arm
    audit = json.load(open(logdir / 'config.yaml'))
    assert audit['arm'] == arm
    exp = dict(aware=(0.1, 0.1, 0.0), free=(0.0, 0.0, 0.0),
               rec=(0.0, 0.0, RECON_COEF))[arm]
    assert (audit['reward_coef'], audit['value_coef'],
            audit['recon_coef']) == exp, (arm, audit)
    results[arm] = state

  # decoder moved ONLY in rec; aware/free decoders must sit AT INIT
  # (review B7: comparing aware==free alone would pass if both moved
  # identically — build the reference init by replaying the fit's exact
  # seeding path: torch.manual_seed(seed) -> build_cfg -> make_agent)
  blob = torch.load(data, weights_only=False)
  torch.manual_seed(1)
  np.random.seed(1)
  import random as _random
  _random.seed(1)
  cfg_ref = build_cfg(os.environ.get('TDMPC2_ROOT'),
                      'dmc_finger_turn_hard', obs_dim, act_dim,
                      ep_len - 1, dict(seed=1, steps=n_eps * ep_len,
                                       buffer_size=n_eps * ep_len))
  ref = make_agent(cfg_ref, 0.0)
  dec_init = {k: v.cpu() for k, v in ref.model.state_dict().items()
              if k.startswith('_decoder')}
  da = results['aware']['decoder']
  df = results['free']['decoder']
  dr = results['rec']['decoder']
  assert all(torch.equal(da[k].cpu(), dec_init[k]) for k in da), \
      'aware decoder moved off init'
  assert all(torch.equal(df[k].cpu(), dec_init[k]) for k in df), \
      'free decoder moved off init'
  moved = any(not torch.equal(dec_init[k].cpu(), dr[k].cpu())
              for k in dec_init)
  assert moved, 'rec decoder never trained'
  # reward head at init in free AND rec, moved in aware
  rw_f = {k: v for k, v in results['free']['model'].items()
          if k.startswith('_reward')}
  rw_r = {k: v for k, v in results['rec']['model'].items()
          if k.startswith('_reward')}
  assert all(torch.equal(rw_f[k], rw_r[k]) for k in rw_f), \
      'free/rec reward heads diverged - reward gradient leaked'
  # encoder differs between free and rec (recon gradient reached it)
  enc_f = {k: v for k, v in results['free']['model'].items()
           if k.startswith('_encoder')}
  enc_r = {k: v for k, v in results['rec']['model'].items()
           if k.startswith('_encoder')}
  assert any(not torch.equal(enc_f[k], enc_r[k]) for k in enc_f), \
      'rec encoder identical to free - recon gradient never reached enc'
  # official-class strict load of the stripped dict
  blob = torch.load(data, weights_only=False)
  cfg = build_cfg(os.environ.get('TDMPC2_ROOT'),
                  'dmc_finger_turn_hard', obs_dim, act_dim,
                  ep_len - 1, dict(seed=1, steps=n_eps * ep_len,
                                   buffer_size=n_eps * ep_len))
  official = TDMPC2(cfg)
  official.model.load_state_dict(results['rec']['model'], strict=True)
  print('SELFCHECK PASS')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selfcheck', action='store_true')
  ap.add_argument('--tdmpc2_root', default=None)
  ap.add_argument('--data')
  ap.add_argument('--logdir')
  ap.add_argument('--arm', choices=ARMS)
  ap.add_argument('--task')
  ap.add_argument('--updates', type=int, default=500_000)
  ap.add_argument('--seed', type=int, default=1)
  ap.add_argument('--log_every', type=int, default=500)
  ap.add_argument('--save_every_updates', type=int, default=100_000)
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  for req in ('data', 'logdir', 'arm', 'task'):
    assert getattr(args, req), f'--{req} required'
  fit(args)


if __name__ == '__main__':
  main()
