"""Frozen-representation online adaptation for a pretrained TD-MPC2 WM.

Cross-family analog of the DreamerV3 frozen-readout adapt stage: loads
the encoder + latent dynamics from a ``tdmpc2_offline_fit`` checkpoint,
FREEZES them, and trains fresh reward / Q / policy heads online on the
task for --steps env steps (action repeat 1, matching the study
protocol). Planning (MPC) runs through the frozen dynamics with the
learned reward/Q — the transfer test is whether the frozen
representation supports task learning.

Writes scores.jsonl rows ``{"step": N, "episode/score": R}`` (one per
eval episode, eval bursts of --eval_episodes every --eval_freq steps) so
the frozen ``analysis/adaptation_auc.py`` pipeline computes AUC100k
unchanged, plus metrics.jsonl, config.yaml (audit), and ADAPT_DONE.

Usage (cluster, tdmpc2 conda env, GPU):
  python -m probing.tdmpc2_adapt --pretrained $RUNROOT/tm2wm_..._seed1 \
      --logdir $RUNROOT/adapt_tm2awareq1s1_finger_seed1_ckpt500000 \
      --task dmc_finger_turn_hard --seed 1
"""

import argparse
import json
import os
import pathlib
import random

import numpy as np

from probing.tdmpc2_compat import Dv3TaskEnv, add_tdmpc2_path, build_cfg

FROZEN_PREFIXES = ('_encoder', '_dynamics')


class JsonlLogger:
  """Minimal stand-in for the official Logger (no wandb/csv deps)."""

  def __init__(self, logdir):
    self._metrics = open(pathlib.Path(logdir) / 'metrics.jsonl', 'a')

  def log(self, d, category='train'):
    row = {f'{category}/{k}': (float(v) if hasattr(v, '__float__') else v)
           for k, v in d.items()}
    self._metrics.write(json.dumps(row) + '\n')
    self._metrics.flush()
    if category == 'eval':
      print(f'[eval] {row}', flush=True)

  def finish(self, agent):
    self._metrics.close()


def load_frozen_wm(agent, fit_dir):
  """Copy encoder+dynamics from the fit checkpoint; freeze them."""
  import torch
  ckpt = pathlib.Path(fit_dir) / 'tm2_ckpt.pt'
  if not (pathlib.Path(fit_dir) / 'TM2_FIT_DONE').exists():
    raise SystemExit(f'{fit_dir}: no TM2_FIT_DONE marker; fit incomplete.')
  state = torch.load(ckpt, weights_only=False)['model']
  keep = {k: v for k, v in state.items()
          if k.split('.')[0] in FROZEN_PREFIXES}
  if not keep:
    raise SystemExit(f'{ckpt}: found no {FROZEN_PREFIXES} params')
  # Newer tensordict versions can fail inside TensorDictParams when loading a
  # partial state dict with strict=False. Merge into the model's full current
  # state first so every submodule receives its expected state object.
  merged = agent.model.state_dict()
  bad = [k for k, v in keep.items()
         if k not in merged or tuple(merged[k].shape) != tuple(v.shape)]
  if bad:
    raise SystemExit(f'{ckpt}: incompatible frozen keys: {bad[:10]}')
  merged.update(keep)
  missing, unexpected = agent.model.load_state_dict(merged, strict=True)
  if missing:
    raise SystemExit(f'missing keys after merged load: {missing[:10]}')
  if unexpected:
    raise SystemExit(f'unexpected keys in checkpoint slice: {unexpected}')
  frozen = 0
  for prefix in FROZEN_PREFIXES:
    module = getattr(agent.model, prefix)
    module.requires_grad_(False)
    frozen += sum(p.numel() for p in module.parameters())
  print(f'Loaded + froze {len(keep)} tensors ({frozen:,} params) from {ckpt}',
        flush=True)
  return sorted(keep)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tdmpc2_root', default=None)
  ap.add_argument('--pretrained', required=True)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--task', required=True)
  ap.add_argument('--seed', type=int, default=1)
  ap.add_argument('--steps', type=int, default=125_000)
  ap.add_argument('--eval_freq', type=int, default=16_000)
  ap.add_argument('--eval_episodes', type=int, default=16)
  ap.add_argument('--seed_steps', type=int, default=5_000)
  def _bool(s):
    v = s.strip().lower()
    if v in ('true', '1', 'yes'): return True
    if v in ('false', '0', 'no'): return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {s!r}')
  ap.add_argument('--mpc', type=_bool, default=True,
                  help='2026-08-08 (review #24): pass --mpc False to adapt '
                       'with the learned actor-critic readout instead of the '
                       'MPPI planner (family-boundary de-confound).')
  args = ap.parse_args()

  add_tdmpc2_path(args.tdmpc2_root)
  import torch
  if not torch.cuda.is_available():
    raise SystemExit('TD-MPC2 requires CUDA.')
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  logdir = pathlib.Path(args.logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  if (logdir / 'ADAPT_DONE').exists():
    print(f'{logdir} already DONE; nothing to do.')
    return

  env = Dv3TaskEnv(args.task, seed=args.seed)
  obs_dim = int(env.observation_space.shape[0])
  act_dim = int(env.action_space.shape[0])
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  args.task, obs_dim, act_dim, env.max_episode_steps,
                  overrides=dict(
                      seed=args.seed, steps=args.steps,
                      buffer_size=args.steps + env.max_episode_steps + 1,
                      eval_freq=args.eval_freq,
                      eval_episodes=args.eval_episodes,
                      seed_steps=args.seed_steps, mpc=args.mpc))

  from tdmpc2 import TDMPC2
  from common.buffer import Buffer
  from trainer.online_trainer import OnlineTrainer

  agent = TDMPC2(cfg)
  frozen_keys = load_frozen_wm(agent, args.pretrained)

  with open(logdir / 'config.yaml', 'w') as f:
    json.dump(dict(
        task=args.task, seed=args.seed, steps=args.steps,
        eval_freq=args.eval_freq, eval_episodes=args.eval_episodes,
        seed_steps=args.seed_steps, pretrained=os.path.abspath(args.pretrained),
        frozen_prefixes=list(FROZEN_PREFIXES), n_frozen_tensors=len(frozen_keys),
        mpc=bool(cfg.mpc), horizon=cfg.horizon,
    ), f, indent=1)

  scores_path = logdir / 'scores.jsonl'

  class AdaptTrainer(OnlineTrainer):

    def to_td(self, obs, action=None, reward=None, terminated=None):
      """TensorDict construction compatible with newer tensordict releases."""
      import torch
      from tensordict.tensordict import TensorDict

      if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device='cpu')
      else:
        obs = obs.unsqueeze(0).cpu()
      if action is None:
        action = torch.full_like(self.env.rand_act(), float('nan'))
      if reward is None:
        reward = torch.tensor(float('nan'), dtype=torch.float32)
      elif not torch.is_tensor(reward):
        reward = torch.tensor(float(reward), dtype=torch.float32)
      if terminated is None:
        terminated = torch.tensor(float('nan'), dtype=torch.float32)
      elif not torch.is_tensor(terminated):
        terminated = torch.tensor(float(terminated), dtype=torch.float32)
      return TensorDict(dict(
          obs=obs,
          action=action.unsqueeze(0).cpu(),
          reward=reward.unsqueeze(0).cpu(),
          terminated=terminated.unsqueeze(0).cpu(),
      ), batch_size=(1,))

    def eval(self):
      rewards = []
      with open(scores_path, 'a') as sf:
        for _ in range(self.cfg.eval_episodes):
          obs, done, ep_reward, t = self.env.reset(), False, 0.0, 0
          while not done:
            action = self.agent.act(obs, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = self.env.step(action)
            ep_reward += reward
            t += 1
          rewards.append(ep_reward)
          sf.write(json.dumps(
              {'step': int(self._step), 'episode/score': float(ep_reward)})
              + '\n')
      return dict(episode_reward=float(np.mean(rewards)),
                  episode_success=0.0,
                  episode_length=float(env.max_episode_steps))

  trainer = AdaptTrainer(cfg=cfg, env=env, agent=agent,
                         buffer=Buffer(cfg), logger=JsonlLogger(logdir))
  trainer.train()
  agent.save(logdir / 'tm2_adapt_final.pt')
  (logdir / 'ADAPT_DONE').touch()
  print(f'ADAPT DONE -> {logdir}', flush=True)


if __name__ == '__main__':
  main()
