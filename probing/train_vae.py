"""Train a static frame-level VAE on DreamerV3 replay frames."""

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import elements
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ninjax as nj
import ruamel.yaml as yaml

import embodied
import embodied.jax.nets as nn
from probing import vae as vae_mod
from probing import replay_dataset as rd

DEFAULT_ENC_KW = dict(
    depth=4, mults=(2, 3, 4, 4), layers=3, units=64, act='silu', norm='rms',
    winit='trunc_normal_in', symlog=True, outer=False, kernel=5,
    strided=False)

BASIC = ('is_first', 'is_last', 'is_terminal', 'reward')
NONOBS = ('action', 'reset', 'consec', 'stepid', 'seed')


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--replay_dirs', nargs='+', required=True,
                 help='DreamerV3 run logdirs (or replay dirs) for VAE frames.')
  p.add_argument('--logdir', required=True, help='Output dir for the VAE.')
  p.add_argument('--run_config', default='',
                 help='DreamerV3 config.yaml to read encoder hparams from.')
  p.add_argument('--obs_keys', default='',
                 help='Comma-separated vector obs keys (default: auto-detect).')
  p.add_argument('--latent', type=int, default=128)
  p.add_argument('--beta', type=float, default=1.0)
  p.add_argument('--free_nats', type=float, default=0.0)
  p.add_argument('--steps', type=int, default=60000)
  p.add_argument('--batch_size', type=int, default=256)
  p.add_argument('--lr', type=float, default=3e-4)
  p.add_argument('--grad_clip', type=float, default=100.0)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--max_frames', type=int, default=2_000_000)
  p.add_argument('--val_frac', type=float, default=0.05)
  p.add_argument('--log_every', type=int, default=500)
  p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
  return p.parse_args()


def resolve_replay_dir(path):
  if rd.list_chunks(path):
    return path
  return rd.find_replay_dir(path)


def get_enc_kw(run_config):
  if not run_config:
    print('No --run_config given; using DEFAULT_ENC_KW (dmc_proprio/size1m).')
    return dict(DEFAULT_ENC_KW)
  with open(run_config) as f:
    cfg = yaml.YAML(typ='safe').load(f)
  enc = dict(cfg['agent']['enc']['simple'])
  enc['mults'] = tuple(enc['mults'])
  print(f'Loaded encoder hparams from {run_config}: {enc}')
  return enc


def detect_obs_keys(replay_dir, override):
  if override:
    return [k.strip() for k in override.split(',') if k.strip()]
  keys = []
  for key, shape in rd.peek_keys(replay_dir).items():
    if key in BASIC or key in NONOBS:
      continue
    if key.startswith(('log/', 'enc/', 'dyn/', 'dec/')):
      continue
    if len(shape) > 1:
      continue
    keys.append(key)
  keys = sorted(keys)
  if not keys:
    raise RuntimeError(f'Could not auto-detect vector obs keys in {replay_dir}')
  print(f'Auto-detected vector obs keys: {keys}')
  return keys


def main():
  args = parse_args()
  os.makedirs(args.logdir, exist_ok=True)
  nn.COMPUTE_DTYPE = {'float32': jnp.float32,
                      'bfloat16': jnp.bfloat16}[args.dtype]

  replay_dirs = [resolve_replay_dir(p) for p in args.replay_dirs]
  obs_keys = detect_obs_keys(replay_dirs[0], args.obs_keys)
  enc_kw = get_enc_kw(args.run_config)

  frames = rd.load_frames(replay_dirs, obs_keys, args.max_frames, args.seed)
  n = next(iter(frames.values())).shape[0]
  n_val = max(args.batch_size, int(args.val_frac * n))
  perm = np.random.default_rng(args.seed).permutation(n)
  val_idx, train_idx = perm[:n_val], perm[n_val:]
  train = {k: v[train_idx] for k, v in frames.items()}
  val = {k: jnp.asarray(v[val_idx[:args.batch_size]]) for k, v in frames.items()}
  print(f'Train frames: {len(train_idx)}  |  Val frames: {n_val}')

  vec_space = {k: elements.Space(np.float32, (v.shape[1],))
               for k, v in frames.items()}
  vae = vae_mod.build(
      vec_space, enc_kw, latent=args.latent, beta=args.beta,
      free_nats=args.free_nats, name='vae')
  optchain = optax.chain(
      optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
  opt = embodied.jax.Optimizer([vae], optchain, name='opt')

  def train_fn(obs):
    optmets, auxmets = opt(vae.loss, obs, has_aux=True)
    return {**{f'opt_{k.split("/")[-1]}': v for k, v in optmets.items()},
            **auxmets}

  def eval_fn(obs):
    return vae.loss(obs, training=False)[1]

  pure_train = nj.pure(train_fn)
  pure_eval = nj.pure(eval_fn)

  seed0 = jnp.array([args.seed, 0], np.uint32)
  example = {k: jnp.asarray(v[:args.batch_size]) for k, v in train.items()}
  state, _ = pure_train({}, example, seed=seed0, create=True)
  n_params = sum(int(np.prod(v.shape)) for k, v in state.items()
                 if k.startswith('vae/'))
  print(f'StaticVAE parameters: {n_params:,}')

  jit_train = jax.jit(lambda s, o, k: pure_train(s, o, seed=k))
  jit_eval = jax.jit(lambda s, o, k: pure_eval(
      s, o, seed=k, create=False, modify=False, ignore=True))

  meta = dict(
      obs_keys=obs_keys, obs_dims={k: v.shape[1] for k, v in frames.items()},
      enc_kw=enc_kw, latent=args.latent, beta=args.beta,
      free_nats=args.free_nats, dec_layers=vae.dec_layers,
      dec_units=vae.dec_units, act=vae.act, norm=vae.norm,
      dtype=args.dtype, seed=args.seed)

  metrics_path = os.path.join(args.logdir, 'metrics.jsonl')
  open(metrics_path, 'w').close()
  step, start = 0, time.time()
  rng = np.random.default_rng(args.seed)
  print(f'Training StaticVAE for {args.steps} steps...')
  while step < args.steps:
    for batch in rd.iterate_batches(train, args.batch_size, rng.integers(1 << 30)):
      batch = {k: jnp.asarray(v) for k, v in batch.items()}
      key = jnp.array([args.seed, step + 1], np.uint32)
      state, mets = jit_train(state, batch, key)
      step += 1
      if step % args.log_every == 0 or step == 1:
        _, valmets = jit_eval(state, val, key)
        row = dict(step=step,
                   fps=round(step / (time.time() - start), 1),
                   **{f'train/{k}': float(v) for k, v in mets.items()},
                   **{f'val/{k}': float(v) for k, v in valmets.items()})
        with open(metrics_path, 'a') as f:
          f.write(json.dumps(row) + '\n')
        print(f'step {step:>7}  loss {row["train/loss"]:.4f}  '
              f'recon {row["train/recon"]:.4f}  kl {row["train/kl"]:.4f}  '
              f'val_elbo {row["val/elbo"]:.4f}  '
              f'active {row["train/active_units"]:.0f}/{args.latent}')
      if step >= args.steps:
        break

  model_state = {k: v for k, v in state.items() if k.startswith('vae/')}
  ckpt_path = os.path.join(args.logdir, 'vae.ckpt')
  vae_mod.save(ckpt_path, model_state, meta)
  with open(os.path.join(args.logdir, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
  print(f'Saved VAE checkpoint -> {ckpt_path}')


if __name__ == '__main__':
  main()
