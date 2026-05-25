"""Action-conditioned dynamics in the frozen VAE's latent space.

The RSSM's open-loop prior is a learned latent dynamics model; to compare the
static VAE representation against it on equal footing for prediction, we give
the VAE its own dynamics. We fit a small forward model

    g(z_t, a_t) ~= z_{t+1} - z_t

on the VAE latents of the *training* episodes (the same split the probes train
on, so the held-out episodes are never seen), roll it open-loop with the
recorded actions, and write `vae_imag{k}` features. The probing script
(`probing.probe`) then treats these as a VAE probe site and scores them against
the future simulator state, directly alongside the RSSM `imag{k}` sites.

Inputs are the artifacts the pipeline already produces: `vae_mean` from
`features.npz` and `action`/`is_first` from `traj.npz`. No DreamerV3 model or
checkpoint is needed.

Run from the repository root:

    python -m probing.vae_dynamics \
        --features <RUN>/probing_walker_walk_seed0/features.npz \
        --traj     <RUN>/probing_walker_walk_seed0/traj.npz \
        --output   <RUN>/probing_walker_walk_seed0/features_vaedyn.npz \
        --horizons 1 5 20 --test_frac 0.3
"""

import argparse
import functools
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np
import optax

f32 = jnp.float32


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--features', required=True, help='features.npz (has vae_mean).')
  p.add_argument('--traj', required=True, help='traj.npz (has action, is_first).')
  p.add_argument('--output', required=True, help='Augmented features .npz.')
  p.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 20])
  p.add_argument('--test_frac', type=float, default=0.3,
                 help='Held-out episode fraction; must match probing.probe so '
                      'the dynamics model never sees the probe test episodes.')
  p.add_argument('--hidden', type=int, default=256)
  p.add_argument('--layers', type=int, default=2)
  p.add_argument('--rollout_len', type=int, default=0,
                 help='Max training rollout horizon (0 = max of --horizons). A '
                      'multi-step loss keeps open-loop imagination stable.')
  p.add_argument('--ramp_frac', type=float, default=0.5,
                 help='Fraction of steps over which the rollout length ramps '
                      'from 1 to rollout_len (curriculum).')
  p.add_argument('--steps', type=int, default=6000)
  p.add_argument('--batch', type=int, default=512)
  p.add_argument('--lr', type=float, default=1e-3)
  p.add_argument('--grad_clip', type=float, default=1.0)
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def init_mlp(d_in, hidden, layers, d_out, rng):
  sizes = [d_in] + [hidden] * layers + [d_out]
  params = []
  for i in range(len(sizes) - 1):
    w = rng.normal(size=(sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])
    params.append([jnp.asarray(w, f32), jnp.zeros(sizes[i + 1], f32)])
  return params


def forward(params, x):
  for w, b in params[:-1]:
    x = jax.nn.relu(x @ w + b)
  w, b = params[-1]
  return x @ w + b


def residual(params, z, a):
  """Predicted next standardized latent: z + g([z, a])."""
  return z + forward(params, jnp.concatenate([z, a], -1))


def main():
  args = parse_args()
  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

  features = {k: np.asarray(v) for k, v in np.load(args.features).items()}
  traj = {k: np.asarray(v) for k, v in np.load(args.traj).items()}
  if 'vae_mean' not in features:
    raise SystemExit('features.npz has no vae_mean; run features.py with --vae_ckpt.')

  Z = features['vae_mean'].astype(np.float32)          # (N, T, L)
  A = traj['action'].astype(np.float32)                # (N, T, Adim)
  N, T, L = Z.shape
  Adim = A.shape[-1]
  n_test = max(1, round(args.test_frac * N))
  n_train = N - n_test
  print(f'Episodes {N} (train {n_train}, test {n_test}); latent {L}, act {Adim}')

  # Standardize latents by train-episode statistics; work in standardized space.
  flat_tr = Z[:n_train].reshape(-1, L)
  zmean = flat_tr.mean(0, keepdims=True)
  zstd = flat_tr.std(0, keepdims=True)
  zstd = np.where(zstd < 1e-6, 1.0, zstd)
  Zn = (Z - zmean) / zstd                              # (N, T, L) standardized

  # Train with a multi-step open-loop rollout loss so the model is stable when
  # imagined forward: a model fit only on one-step transitions compounds error
  # and diverges over a long horizon. We roll H = max(horizons) steps from
  # sampled windows of the train episodes and average the per-step error.
  H = args.rollout_len or max(args.horizons)
  assert T > H + 1, (T, H)
  Zn_tr = jnp.asarray(Zn[:n_train])                    # (n_train, T, L)
  A_tr = jnp.asarray(A[:n_train])                      # (n_train, T, Adim)

  rng = np.random.default_rng(args.seed)
  params = init_mlp(L + Adim, args.hidden, args.layers, L, rng)
  opt = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                    optax.adam(args.lr))
  ostate = opt.init(params)

  def rollout_loss(params, ep, st, h):
    idx = st[:, None] + jnp.arange(h + 1)[None, :]      # (B, h+1) time indices
    zwin = Zn_tr[ep[:, None], idx]                      # (B, h+1, L)
    awin = A_tr[ep[:, None], idx[:, :h]]               # (B, h, Adim)
    z = zwin[:, 0]
    total = 0.0
    for j in range(h):
      z = residual(params, z, awin[:, j])
      total = total + jnp.mean((z - zwin[:, j + 1]) ** 2)
    return total / h

  @functools.partial(jax.jit, static_argnums=(4,))
  def step(params, ostate, ep, st, h):
    loss, grad = jax.value_and_grad(rollout_loss)(params, ep, st, h)
    updates, ostate = opt.update(grad, ostate)
    return optax.apply_updates(params, updates), ostate, loss

  # Curriculum: ramp the rollout horizon from 1 to H, then hold at H. This
  # reaches the good short-horizon model first and then refines for stability.
  ramp = max(1, int(args.ramp_frac * args.steps))
  print(f'Training rollout model (curriculum 1->{H}) on {n_train} episodes...')
  for s in range(args.steps):
    h = int(min(H, 1 + (H - 1) * s // ramp))
    ep = jnp.asarray(rng.integers(0, n_train, args.batch))
    st = jnp.asarray(rng.integers(0, T - H, args.batch))
    params, ostate, loss = step(params, ostate, ep, st, h)
    if s == 0 or (s + 1) % 1000 == 0:
      print(f'  step {s + 1:>5}  h={h:<2}  rollout MSE {float(loss):.4f}')

  # Open-loop rollout: from each anchor t, apply g for k steps with a_t..a_{t+k-1}.
  Zn_j = jnp.asarray(Zn)
  A_j = jnp.asarray(A)

  @functools.partial(jax.jit, static_argnums=(3,))
  def rollout(params, Zn_all, A_all, k):
    n, t, l = Zn_all.shape
    z = Zn_all.reshape(n * t, l)
    pad = jnp.concatenate([A_all, jnp.zeros((n, k, A_all.shape[-1]), f32)], 1)
    for j in range(k):
      aj = jax.lax.dynamic_slice_in_dim(pad, j, t, axis=1).reshape(n * t, -1)
      z = residual(params, z, aj)
    return z.reshape(n, t, l)

  out = dict(features)
  for k in args.horizons:
    zk = np.asarray(rollout(params, Zn_j, A_j, int(k)))   # standardized z_{t+k}
    out[f'vae_imag{k}'] = (zk * zstd + zmean).astype(np.float32)
    print(f'  wrote vae_imag{k}: {out[f"vae_imag{k}"].shape}')

  np.savez_compressed(args.output, **out)
  # Carry the features meta forward so probing.probe can read it.
  meta = {}
  meta_path = args.features + '.meta.json'
  if os.path.exists(meta_path):
    with open(meta_path) as fh:
      meta = json.load(fh)
  meta['vae_dynamics'] = dict(horizons=args.horizons, test_frac=args.test_frac,
                              hidden=args.hidden, layers=args.layers,
                              steps=args.steps)
  meta.setdefault('horizons', list(args.horizons))
  meta['sites'] = sorted(set(meta.get('sites', []) +
                             [f'vae_imag{k}' for k in args.horizons]))
  with open(args.output + '.meta.json', 'w') as fh:
    json.dump(meta, fh, indent=2)
  print(f'Saved augmented features -> {args.output}')


if __name__ == '__main__':
  main()
