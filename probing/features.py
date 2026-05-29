"""Extract RSSM and optional VAE features for probe trajectories."""

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
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn
from dreamerv3.main import make_agent
from probing.collect import load_run_config, load_frozen_agent
from probing import vae as vae_mod

f32 = jnp.float32


def imagine_k(dyn, post_deter, post_stoch, action_dict, k):
  """Open-loop k-step RSSM prior, anchored at every timestep."""
  B, T = post_deter.shape[:2]
  carry = {
      'deter': post_deter.reshape((B * T,) + post_deter.shape[2:]),
      'stoch': post_stoch.reshape((B * T,) + post_stoch.shape[2:]),
  }
  windows = {}
  for ak, av in action_dict.items():
    A = av.shape[-1]
    pad = jnp.concatenate([av, jnp.zeros((B, k, A), av.dtype)], 1)
    win = jnp.stack([pad[:, i:i + T] for i in range(k)], 2)   # (B, T, k, A)
    windows[ak] = win.reshape((B * T, k, A))
  _, feat, _ = dyn.imagine(carry, windows, k, training=False)
  deter_k = feat['deter'][:, -1].reshape((B, T, -1))
  probs_k = jax.nn.softmax(f32(feat['logit'][:, -1]), -1)
  return f32(deter_k), f32(probs_k).reshape((B, T, -1))


def world_model_features(model, obs, action_dict, reset, obs_keys, horizons):
  B, T = reset.shape
  enc_carry = model.enc.initial(B)
  dyn_carry = model.dyn.initial(B)
  enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)

  prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
             for k, v in action_dict.items()}
  dyn_carry, _, post = model.dyn.observe(
      dyn_carry, tokens, prevact, reset, training=False)
  post_deter = f32(post['deter'])
  post_probs = jax.nn.softmax(f32(post['logit']), -1)
  prior_logit = model.dyn._prior(post['deter'])
  prior_probs = jax.nn.softmax(f32(prior_logit), -1)

  out = {
      'wm_enc': f32(tokens),
      'wm_post_deter': post_deter,
      'wm_post_stoch': post_probs.reshape((B, T, -1)),
      'wm_prior_deter': post_deter,            # prior shares the deter state
      'wm_prior_stoch': prior_probs.reshape((B, T, -1)),
  }

  _, _, post_rec = model.dec(
      model.dec.initial(B), post, reset, training=False)
  prior_feat = {'deter': post['deter'], 'stoch': prior_probs}
  _, _, prior_rec = model.dec(
      model.dec.initial(B), prior_feat, reset, training=False)
  out['wm_post_nll'] = f32(sum(
      post_rec[k].loss(f32(obs[k])) for k in obs_keys))
  out['wm_prior_nll'] = f32(sum(
      prior_rec[k].loss(f32(obs[k])) for k in obs_keys))

  for k in horizons:
    dk, sk = imagine_k(model.dyn, post['deter'], post['stoch'], action_dict, k)
    out[f'wm_imag{k}_deter'] = dk
    out[f'wm_imag{k}_stoch'] = sk
  return out


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--traj', required=True, help='Probe trajectory .npz.')
  p.add_argument('--run_logdir', required=True,
                 help='DreamerV3 run dir (config.yaml + ckpt).')
  p.add_argument('--output', required=True, help='Output features .npz.')
  p.add_argument('--vae_ckpt', default='', help='StaticVAE checkpoint.')
  p.add_argument('--checkpoint', default='',
                 help='DreamerV3 checkpoint (default: <run_logdir>/ckpt).')
  p.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 20])
  p.add_argument('--ep_batch', type=int, default=8,
                 help='Episodes processed per forward pass.')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def load_traj(path):
  data = {k: np.asarray(v) for k, v in np.load(path).items()}
  with open(path + '.meta.json') as f:
    meta = json.load(f)
  return data, meta


def extract_world_model(args, traj, meta):
  out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  config = load_run_config(args.run_logdir, args.platform, out_dir, False)
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
  print(f'World model: obs={obs_keys}  act={act_keys}  '
        f'horizons={args.horizons}')

  horizons = tuple(args.horizons)

  def fn(obs, action_dict, reset):
    return world_model_features(model, obs, action_dict, reset, obs_keys,
                                horizons)

  pure = nj.pure(fn)
  jit = jax.jit(lambda p, o, a, r, s: pure(p, o, a, r, seed=s))

  params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), agent.params)

  N, T = traj['reward'].shape
  feats = {}
  for lo in range(0, N, args.ep_batch):
    hi = min(lo + args.ep_batch, N)
    obs = {k: jnp.asarray(traj[f'obs_{k}'][lo:hi], np.float32)
           for k in obs_keys}
    action = traj['action'][lo:hi]
    action_dict = {ak: jnp.asarray(action[..., offs[i]:offs[i + 1]],
                                   np.float32)
                   for i, ak in enumerate(act_keys)}
    reset = jnp.asarray(traj['is_first'][lo:hi], bool)
    seed = jnp.array([args.seed, lo + 1], np.uint32)
    _, out = jit(params, obs, action_dict, reset, seed)
    for k, v in out.items():
      feats.setdefault(k, []).append(np.asarray(v, np.float32))
    print(f'  world-model features: episodes {lo}-{hi - 1}')
  return {k: np.concatenate(v, 0) for k, v in feats.items()}


def extract_vae(args, traj):
  state, meta = vae_mod.load(args.vae_ckpt)
  nn.COMPUTE_DTYPE = {'float32': jnp.float32,
                      'bfloat16': jnp.bfloat16}[meta.get('dtype', 'float32')]
  vec_space = {k: elements.Space(np.float32, (meta['obs_dims'][k],))
               for k in meta['obs_keys']}
  vae = vae_mod.build(
      vec_space, meta['enc_kw'], latent=meta['latent'], beta=meta['beta'],
      free_nats=meta['free_nats'], dec_layers=meta['dec_layers'],
      dec_units=meta['dec_units'], act=meta['act'], norm=meta['norm'],
      name='vae')
  pure = nj.pure(vae.featurize)
  jit = jax.jit(lambda p, o, s: pure(p, o, seed=s))

  N, T = traj['reward'].shape
  obs = {k: traj[f'obs_{k}'].reshape(N * T, -1).astype(np.float32)
         for k in meta['obs_keys']}
  feats, batch = {}, 4096
  for lo in range(0, N * T, batch):
    hi = min(lo + batch, N * T)
    chunk = {k: jnp.asarray(v[lo:hi]) for k, v in obs.items()}
    _, out = jit(state, chunk, jnp.array([0, lo + 1], np.uint32))
    for k, v in out.items():
      feats.setdefault(f'vae_{k}', []).append(np.asarray(v, np.float32))
  feats = {k: np.concatenate(v, 0).reshape((N, T) + v[0].shape[1:])
           for k, v in feats.items()}
  return feats, meta


def main():
  args = parse_args()
  out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
  os.makedirs(out_dir, exist_ok=True)
  traj, traj_meta = load_traj(args.traj)
  N, T = traj['reward'].shape
  print(f'Probe trajectories: {N} episodes x {T} steps')

  features = extract_world_model(args, traj, traj_meta)
  out_meta = dict(traj=args.traj, run_logdir=args.run_logdir,
                  horizons=args.horizons, episodes=int(N), length=int(T),
                  sites=['enc', 'posterior', 'prior'] +
                  [f'imag{h}' for h in args.horizons])

  if args.vae_ckpt:
    vae_features, vae_meta = extract_vae(args, traj)
    features.update(vae_features)
    out_meta['vae_ckpt'] = args.vae_ckpt
    out_meta['vae_latent'] = vae_meta['latent']
    out_meta['sites'] += ['vae_enc', 'vae_latent']
  else:
    print('No --vae_ckpt given; extracting world-model features only.')

  np.savez_compressed(args.output, **features)
  with open(args.output + '.meta.json', 'w') as f:
    json.dump(out_meta, f, indent=2)
  print('Feature arrays:')
  for k, v in sorted(features.items()):
    print(f'  {k:22s} {v.shape}')
  print(f'Saved -> {args.output}')


if __name__ == '__main__':
  main()
