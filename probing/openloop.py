"""Open-loop prediction: what the RSSM world model actually predicts.

This is the "how the system works" figure. For a held-out episode we feed the
first `--context` steps to the RSSM to establish a posterior, then roll the
*prior* forward with the recorded actions for `--horizon` steps -- the model
no longer sees observations -- and decode both back to observation units.
Plotting predicted-vs-true torso height / velocities / orientations
(posterior solid, prior dashed, truth black) directly visualises the
predictive component that the RSSM-prior probes and held-out likelihood
quantify, and shows the prior drifting from truth as the horizon grows.

It also reports the open-loop prediction RMSE as a function of horizon.

Run from the repository root:

    python -m probing.openloop \
        --traj       <RUN>/probing_walker_walk_seed0/traj.npz \
        --run_logdir <RUN>/dmc_proprio_walker_walk_seed0 \
        --output     results/probing_walker_walk_seed0 \
        --context 25 --horizon 75 --episode 0
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

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ninjax as nj
import numpy as np

import embodied.jax.nets as nets
from dreamerv3.main import make_agent
from probing.collect import load_run_config, load_frozen_agent

f32 = jnp.float32


def openloop_forward(model, obs, action_dict, reset, obs_keys, context,
                     horizon):
  """Posterior over [0, context); open-loop prior over [context, context+H).

  Returns decoded predictions in observation units for both windows.
  """
  B = reset.shape[0]
  enc_carry = model.enc.initial(B)
  dyn_carry = model.dyn.initial(B)
  enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
  prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
             for k, v in action_dict.items()}
  sl = lambda x, a, b: jax.tree.map(lambda v: v[:, a:b], x)

  dyn_carry, _, post = model.dyn.observe(
      dyn_carry, sl(tokens, 0, context), sl(prevact, 0, context),
      reset[:, :context], training=False)
  _, _, post_rec = model.dec(
      model.dec.initial(B), post, reset[:, :context], training=False)

  # Imagine forward using the actions actually taken from step context-1 on.
  imag_act = {k: v[:, context - 1:context - 1 + horizon]
              for k, v in action_dict.items()}
  _, prior_feat, _ = model.dyn.imagine(
      dyn_carry, imag_act, horizon, training=False)
  _, _, prior_rec = model.dec(
      model.dec.initial(B), prior_feat, jnp.zeros((B, horizon), bool),
      training=False)

  out = {}
  for k in obs_keys:
    out[f'{k}/post'] = nets.symexp(f32(post_rec[k].pred()))
    out[f'{k}/prior'] = nets.symexp(f32(prior_rec[k].pred()))
  return out


def select_panels(obs_keys, dims, n_velocity=3, n_orient=2):
  panels = []
  if 'height' in obs_keys:
    panels.append(('height', None, 'torso height'))
  if 'velocity' in obs_keys:
    for d in range(min(n_velocity, dims['velocity'])):
      panels.append(('velocity', d, f'velocity[{d}]'))
  if 'orientations' in obs_keys:
    for d in range(min(n_orient, dims['orientations'])):
      panels.append(('orientations', d, f'orientation[{d}]'))
  return panels


def plot_episode(traj, pred, obs_keys, dims, ep, context, horizon, outpath):
  panels = select_panels(obs_keys, dims)
  ncol = 3
  nrow = int(np.ceil(len(panels) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow),
                           squeeze=False, constrained_layout=True)
  axes = axes.flatten()
  tp = np.arange(context)
  ti = np.arange(context, context + horizon)
  for ax, (key, d, label) in zip(axes, panels):
    true = np.asarray(traj[f'obs_{key}'][ep])
    post = pred[f'{key}/post'][ep]
    prior = pred[f'{key}/prior'][ep]
    if d is not None:
      true, post, prior = true[:, d], post[:, d], prior[:, d]
    ax.plot(np.arange(context + horizon), true[:context + horizon],
            color='#222222', lw=1.6, label='true')
    ax.plot(tp, post, color='#1f77b4', lw=1.6, label='posterior')
    ax.plot(ti, prior, color='#d1495b', lw=1.6, ls='--', label='open-loop prior')
    ax.axvline(context, color='#999999', lw=0.8, ls=':')
    ax.set_title(label, fontsize=10)
  for ax in axes[len(panels):]:
    ax.axis('off')
  axes[0].legend(fontsize=8, loc='best')
  fig.suptitle(f'RSSM open-loop prediction (episode {ep}, '
               f'context {context} -> imagine {horizon})', fontweight='bold')
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


def plot_rmse(rmse, outpath):
  fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
  ax.plot(np.arange(1, len(rmse) + 1), rmse, 'o-', color='#d1495b', ms=3)
  ax.set_xlabel('open-loop horizon (steps after context)')
  ax.set_ylabel('prediction RMSE (obs units)')
  ax.set_title('Open-loop prior error grows with horizon')
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--traj', required=True)
  p.add_argument('--run_logdir', required=True)
  p.add_argument('--output', required=True, help='Output directory.')
  p.add_argument('--checkpoint', default='')
  p.add_argument('--context', type=int, default=25)
  p.add_argument('--horizon', type=int, default=75)
  p.add_argument('--episode', type=int, default=0)
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def main():
  args = parse_args()
  os.makedirs(args.output, exist_ok=True)
  traj = {k: np.asarray(v) for k, v in np.load(args.traj).items()}
  with open(args.traj + '.meta.json') as f:
    meta = json.load(f)
  N, T = traj['reward'].shape
  context = max(1, min(args.context, T - 1))
  horizon = min(args.horizon, T - context)
  print(f'Episodes {N} x {T}; context {context}, horizon {horizon}')

  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  config = load_run_config(args.run_logdir, args.platform, args.output, False)
  agent = make_agent(config)
  load_frozen_agent(agent, ckpt)
  model = agent.model
  jax.config.update('jax_transfer_guard', 'allow')

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  obs_keys = sorted(k for k, v in agent.obs_space.items()
                    if k not in exclude and len(v.shape) <= 1)
  act_keys = sorted(agent.act_space.keys())
  act_dims = [int(np.prod(agent.act_space[k].shape)) for k in act_keys]
  offs = np.cumsum([0] + act_dims)
  dims = {k: (traj[f'obs_{k}'].shape[-1] if traj[f'obs_{k}'].ndim == 3 else 1)
          for k in obs_keys}

  params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), agent.params)
  pure = nj.pure(lambda o, a, r: openloop_forward(
      model, o, a, r, obs_keys, context, horizon))
  jit = jax.jit(lambda p, o, a, r, s: pure(p, o, a, r, seed=s))

  obs = {k: jnp.asarray(traj[f'obs_{k}'], np.float32) for k in obs_keys}
  action = traj['action']
  action_dict = {ak: jnp.asarray(action[..., offs[i]:offs[i + 1]], np.float32)
                 for i, ak in enumerate(act_keys)}
  reset = jnp.asarray(traj['is_first'], bool)
  _, pred = jit(params, obs, action_dict, reset,
                jnp.array([args.seed, 1], np.uint32))
  pred = {k: np.asarray(v, np.float32) for k, v in pred.items()}

  ep = min(args.episode, N - 1)
  plot_episode(traj, pred, obs_keys, dims, ep, context, horizon,
               os.path.join(args.output, f'openloop_ep{ep}.png'))

  # Open-loop RMSE vs horizon, pooled over episodes and all obs dims.
  rmse = []
  for h in range(horizon):
    sq, cnt = 0.0, 0
    for k in obs_keys:
      true = np.asarray(traj[f'obs_{k}'])[:, context + h]
      pr = pred[f'{k}/prior'][:, h]
      sq += float(np.sum((true - pr) ** 2))
      cnt += true.size
    rmse.append((sq / cnt) ** 0.5)
  plot_rmse(rmse, os.path.join(args.output, 'openloop_rmse_vs_horizon.png'))
  with open(os.path.join(args.output, 'openloop.json'), 'w') as f:
    json.dump({'context': context, 'horizon': horizon,
               'rmse_vs_horizon': rmse}, f, indent=2)
  print(f'Open-loop RMSE: step1={rmse[0]:.3f} -> step{horizon}={rmse[-1]:.3f}')
  print(f'Wrote figures -> {args.output}')


if __name__ == '__main__':
  main()
