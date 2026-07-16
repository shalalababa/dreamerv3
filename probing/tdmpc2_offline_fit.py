"""Offline TD-MPC2 world-model fit on a bridged DreamerV3 buffer.

Cross-family analog of ``probing/offline_fit.py``: trains the official
TD-MPC2 agent (pinned checkout via TDMPC2_ROOT) for a fixed number of
gradient updates on a static dataset produced by
``probing.tdmpc2_bridge``, with the reward-supervision arm switched by
loss coefficients only (identical parameter counts and update mechanics):

  --arm aware : official objective (consistency + reward + value; the
                policy update never shapes the encoder - zs detached).
  --arm free  : reward_coef=0, value_coef=0 - representation is shaped
                by the latent self-consistency loss alone (reward and Q
                heads receive zero gradient and stay at init).

Saved per run (the audit record): logdir/config.yaml (records arm +
reward_coef/value_coef/consistency_coef + data manifest), stdout loss
lines every --log_every, TM2_FIT_PROGRESS, resumable full checkpoint
(model + both optimizers + update counter), TM2_FIT_DONE marker.

Usage (cluster, tdmpc2 conda env, GPU):
  python -m probing.tdmpc2_offline_fit --data <bridge.pt> \
      --logdir $RUNROOT/tm2wm_finger_awareq1s1_seed1 --arm aware \
      --task dmc_finger_turn_hard --updates 500000 --seed 1
"""

import argparse
import json
import os
import pathlib
import random

import numpy as np

from probing.tdmpc2_compat import add_tdmpc2_path, build_cfg

CKPT_NAME = 'tm2_ckpt.pt'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tdmpc2_root', default=None)
  ap.add_argument('--data', required=True)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--arm', required=True, choices=('aware', 'free'))
  ap.add_argument('--task', required=True)
  ap.add_argument('--updates', type=int, default=500_000)
  ap.add_argument('--seed', type=int, default=1)
  ap.add_argument('--log_every', type=int, default=500)
  ap.add_argument('--save_every_updates', type=int, default=100_000)
  args = ap.parse_args()

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

  overrides = dict(seed=args.seed, steps=total_rows, buffer_size=total_rows)
  if args.arm == 'free':
    overrides.update(reward_coef=0.0, value_coef=0.0)
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  args.task, man['obs_dim'], man['action_dim'],
                  ep_len - 1, overrides)

  # audit record: arm + effective coefficients + data identity
  audit = dict(
      arm=args.arm, task=args.task, seed=args.seed, updates=args.updates,
      consistency_coef=cfg.consistency_coef, reward_coef=cfg.reward_coef,
      value_coef=cfg.value_coef, model_size=cfg.model_size,
      horizon=cfg.horizon, batch_size=cfg.batch_size,
      data=man,
  )
  with open(logdir / 'config.yaml', 'w') as f:
    json.dump(audit, f, indent=1)  # json is valid yaml

  from tdmpc2 import TDMPC2
  from common.buffer import Buffer
  agent = TDMPC2(cfg)
  buffer = Buffer(cfg)
  buffer.load(td)
  print(f'Loaded static data: {n_eps} eps x {ep_len} rows = {total_rows} '
        f'rows; arm={args.arm} (reward_coef={cfg.reward_coef}, '
        f'value_coef={cfg.value_coef})', flush=True)

  start = 0
  ckpt = logdir / CKPT_NAME
  if ckpt.exists():
    state = torch.load(ckpt, weights_only=False)
    agent.model.load_state_dict(state['model'])
    agent.optim.load_state_dict(state['optim'])
    agent.pi_optim.load_state_dict(state['pi_optim'])
    start = int(state['update'])
    print(f'Resumed at update {start}', flush=True)

  def save(i):
    torch.save(dict(model=agent.model.state_dict(),
                    optim=agent.optim.state_dict(),
                    pi_optim=agent.pi_optim.state_dict(),
                    update=i), ckpt)

  for i in range(start, args.updates):
    info = agent.update(buffer)
    if i == 0 or (i + 1) % args.log_every == 0 or i == args.updates - 1:
      msg = '  '.join(f'{k}={float(v):.4f}' for k, v in sorted(info.items())
                      if k in ('consistency_loss', 'reward_loss',
                               'value_loss', 'total_loss', 'pi_loss'))
      print(f'  update {i + 1:>6}/{args.updates}: {msg}', flush=True)
      with open(logdir / 'TM2_FIT_PROGRESS', 'w') as f:
        f.write(f'update={i + 1}/{args.updates}\n')
    if (i + 1) % args.save_every_updates == 0 and (i + 1) < args.updates:
      save(i + 1)
      print(f'Saved checkpoint at update {i + 1}', flush=True)

  save(args.updates)
  (logdir / 'TM2_FIT_DONE').touch()
  print(f'DONE: {args.updates} updates -> {ckpt}', flush=True)


if __name__ == '__main__':
  main()
