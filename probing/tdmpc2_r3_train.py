"""Fresh online TD-MPC2 training for the TM2-R3 competence wave.

Registered in prereg/PREREG_tm2_competence_20260730.md. Trains the
official TD-MPC2 agent (pinned checkout via TDMPC2_ROOT, model_size 5,
state obs, mpc=True — NOTHING frozen, official loss coefficients) fully
online on a dosed ``Dv3TaskEnv`` (dose e1 = clean, e4 = OU distractor
dim 32 scale 3.0, composed exactly as DreamerV3 composes it — see
probing/tdmpc2_compat.py), for --steps env steps with action repeat 1
and 1000-step episodes.

Two-maturity checkpointing (mirrors scripts/r3_local.sh's ckpt_early
convention): at the FIRST episode boundary with env step >=
--early_steps the model is snapshotted to ckpt_early.pt (realized step
recorded in ckpt_early.STEP — episode boundaries are every 1000 steps,
so the realized step is within one episode of the nominal 2.5e4); the
final model is ckpt_late.pt (+ ckpt_late.STEP). Checkpoints are
``agent.save`` files ({"model": state_dict}) — exactly what
probing/tdmpc2_oracle_labels.py loads. config.json records the audit
trail (task, dose, seed, dims, planner dials, realized steps).

This is a fresh-online driver, NOT the frozen-representation adapt
stage: no dependency on any tm2wm_* offline-fit checkpoint exists
anywhere in this wave.

Usage (cluster, tdmpc2 conda env, GPU; see scripts/tm2r3.sbatch):
  python -m probing.tdmpc2_r3_train --logdir $RUNROOT/tm2r3/tm2r3_cup_e4_seed51 \
      --task dmc_cup_catch --dose e4 --seed 51
"""

import argparse
import json
import os
import pathlib
import random

import numpy as np

from probing.tdmpc2_compat import (
    Dv3TaskEnv, add_tdmpc2_path, build_cfg, dose_config)

TRAINER_VERSION = 'tm2r3_train_20260730'
DONE_MARKER = 'TM2R3_TRAIN_DONE'
TASKS = ('dmc_cup_catch', 'dmc_finger_turn_hard')


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


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--tdmpc2_root', default=None)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--task', required=True, choices=TASKS)
  ap.add_argument('--dose', required=True, choices=('e1', 'e4'))
  ap.add_argument('--seed', type=int, required=True)
  ap.add_argument('--steps', type=int, default=100_000)
  ap.add_argument('--early_steps', type=int, default=25_000)
  ap.add_argument('--eval_freq', type=int, default=25_000)
  ap.add_argument('--eval_episodes', type=int, default=3)
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
  if (logdir / DONE_MARKER).exists():
    print(f'{logdir} already DONE; nothing to do.')
    return

  env = Dv3TaskEnv(args.task, seed=args.seed, dose=args.dose)
  obs_dim = int(env.observation_space.shape[0])
  act_dim = int(env.action_space.shape[0])
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  args.task, obs_dim, act_dim, env.max_episode_steps,
                  overrides=dict(
                      seed=args.seed, steps=args.steps,
                      buffer_size=args.steps + env.max_episode_steps + 1,
                      eval_freq=args.eval_freq,
                      eval_episodes=args.eval_episodes, mpc=True))

  from tdmpc2 import TDMPC2
  from common.buffer import Buffer
  from trainer.online_trainer import OnlineTrainer

  agent = TDMPC2(cfg)

  audit = dict(
      trainer_version=TRAINER_VERSION, task=args.task, dose=args.dose,
      dose_config=dose_config(args.dose), seed=args.seed,
      steps=args.steps, early_steps=args.early_steps,
      obs_dim=obs_dim, action_dim=act_dim,
      episode_length=env.max_episode_steps, seed_steps=int(cfg.seed_steps),
      model_size=int(cfg.model_size), num_q=int(cfg.num_q),
      latent_dim=int(cfg.latent_dim), discount=float(agent.discount),
      planner=dict(mpc=bool(cfg.mpc), horizon=int(cfg.horizon),
                   iterations=int(cfg.iterations),
                   num_samples=int(cfg.num_samples),
                   num_elites=int(cfg.num_elites),
                   num_pi_trajs=int(cfg.num_pi_trajs),
                   temperature=float(cfg.temperature),
                   min_std=float(cfg.min_std), max_std=float(cfg.max_std)),
      early_step_realized=None, late_step_realized=None)

  def write_audit():
    with open(logdir / 'config.json', 'w') as f:
      json.dump(audit, f, indent=1)

  write_audit()
  scores_path = logdir / 'scores.jsonl'

  class R3Trainer(OnlineTrainer):

    def to_td(self, obs, action=None, reward=None, terminated=None):
      """TensorDict construction compatible with newer tensordict
      releases (same repair as probing/tdmpc2_adapt.py)."""
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

    def common_metrics(self):
      # Called at EVERY episode boundary (train log + eval blocks):
      # the early-maturity snapshot hook. Episodes are 1000 steps, so
      # the realized early step lands within one episode of nominal.
      self._maybe_early_snapshot()
      return super().common_metrics()

    def _maybe_early_snapshot(self):
      if audit['early_step_realized'] is None and \
          self._step >= args.early_steps:
        self.agent.save(logdir / 'ckpt_early.pt')
        audit['early_step_realized'] = int(self._step)
        with open(logdir / 'ckpt_early.STEP', 'w') as f:
          f.write(f'{int(self._step)}\n')
        write_audit()
        print(f'[{self._step}] early snapshot -> ckpt_early.pt', flush=True)

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

  trainer = R3Trainer(cfg=cfg, env=env, agent=agent,
                      buffer=Buffer(cfg), logger=JsonlLogger(logdir))
  trainer.train()
  if audit['early_step_realized'] is None:
    raise SystemExit('training ended without an early snapshot; '
                     'refusing to mark DONE (early_steps > steps?)')
  agent.save(logdir / 'ckpt_late.pt')
  audit['late_step_realized'] = int(trainer._step)
  with open(logdir / 'ckpt_late.STEP', 'w') as f:
    f.write(f'{int(trainer._step)}\n')
  write_audit()
  (logdir / DONE_MARKER).touch()
  print(f'TM2R3 TRAIN DONE -> {logdir} '
        f"(early @{audit['early_step_realized']}, "
        f"late @{audit['late_step_realized']})", flush=True)


if __name__ == '__main__':
  main()
