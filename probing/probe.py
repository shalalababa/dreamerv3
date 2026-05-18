"""Probing battery: what does each frozen representation encode?

Given the probe trajectories and the extracted features, this script sweeps
the three axes of the experimental design:

  * probe target   : current state / future state at horizon k / a derived
                      dynamical quantity (gait phase, return-to-go, ...);
  * probe site     : RSSM encoder / RSSM posterior / RSSM prior (one-step and
                      open-loop k-step) / VAE encoder / VAE latent mean;
  * receptive field: the VAE latent probed with 1, 4, 16 or all past frames
                      (post-hoc temporal aggregation of a static model).

For every (target, site, horizon, receptive-field) configuration it fits a
linear ridge probe (closed form) and a small MLP probe, and reports the
held-out coefficient of determination R^2. It also reports held-out
reconstruction / predictive log-likelihoods so probe quality and generative
quality can be correlated.

R^2 is computed on held-out episodes against the *train-set* mean baseline:

    R^2 = 1 - sum (y - y_hat)^2 / sum (y - mean_train(y))^2

so a representation that carries no information about the target scores ~0.

Run from the repository root:

    python -m probing.probe \
        --traj     /scratch/.../probe_walker_walk_seed0.npz \
        --features /scratch/.../features_walker_walk_seed0.npz \
        --output   /scratch/.../probes_walker_walk_seed0
"""

import argparse
import csv
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from probing import targets as targets_mod

LAMBDAS = (1e-3, 1e-1, 1e0, 1e1, 1e3)        # ridge regularization grid


# -- feature assembly -----------------------------------------------------

def site_features(features, fmeta):
  """Assemble {site: (N, T, d)} probe-site feature arrays."""
  sites = {}
  if 'wm_enc' in features:
    sites['enc'] = features['wm_enc']
    sites['posterior'] = np.concatenate(
        [features['wm_post_deter'], features['wm_post_stoch']], -1)
    sites['prior'] = np.concatenate(
        [features['wm_prior_deter'], features['wm_prior_stoch']], -1)
    # Posterior components: deter (recurrent state) and stoch (the categorical
    # latent) probed alone -- stoch is the fair same-dimension analog of the
    # VAE's Gaussian latent.
    sites['post_deter'] = features['wm_post_deter']
    sites['post_stoch'] = features['wm_post_stoch']
    for h in fmeta.get('horizons', []):
      if f'wm_imag{h}_deter' in features:
        sites[f'imag{h}'] = np.concatenate(
            [features[f'wm_imag{h}_deter'], features[f'wm_imag{h}_stoch']], -1)
  if 'vae_mean' in features:
    sites['vae_enc'] = features['vae_enc']
    sites['vae_latent'] = features['vae_mean']
  return sites


def receptive_window(feat, k):
  """Causal temporal aggregation of a per-frame feature (N, T, d).

  k in {1,4,16}: concatenate the last k frames (start padded by repetition).
  k == 'all'   : concatenate the current frame with the causal mean of all
                 past-and-current frames.
  """
  N, T, d = feat.shape
  if k == 'all':
    csum = np.cumsum(feat, axis=1)
    causal_mean = csum / np.arange(1, T + 1)[None, :, None]
    return np.concatenate([feat, causal_mean], -1)
  out = [np.concatenate([feat[:, :1]] * j + [feat[:, :T - j]], 1)
         for j in range(k)]                       # shifts 0..k-1
  return np.concatenate(out[::-1], -1)


# -- probes ---------------------------------------------------------------

def r2_score(y, yhat, baseline):
  ss_res = float(np.sum((y - yhat) ** 2))
  ss_tot = float(np.sum((y - baseline) ** 2))
  return 1.0 - ss_res / max(ss_tot, 1e-12)


def standardize(train, *others):
  mean = train.mean(0, keepdims=True)
  std = train.std(0, keepdims=True)
  std = np.where(std < 1e-6, 1.0, std)
  return tuple((a - mean) / std for a in (train, *others))


def fit_ridge(Xtr, Ytr, lam):
  """Closed-form ridge with an unregularized bias column."""
  X = np.concatenate([Xtr, np.ones((len(Xtr), 1), Xtr.dtype)], 1)
  D = X.shape[1]
  reg = lam * np.eye(D, dtype=X.dtype)
  reg[-1, -1] = 0.0                                # do not penalize bias
  W = np.linalg.solve(X.T @ X + reg, X.T @ Ytr)
  return W


def predict_ridge(W, X):
  return np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1) @ W


def ridge_probe(Xtr, Ytr, Xte, Yte, val_frac=0.2):
  """Ridge probe with lambda chosen on a validation split of the train set."""
  n_val = max(1, int(val_frac * len(Xtr)))
  Xf, Yf, Xv, Yv = Xtr[:-n_val], Ytr[:-n_val], Xtr[-n_val:], Ytr[-n_val:]
  best_lam, best_r2 = LAMBDAS[0], -np.inf
  for lam in LAMBDAS:
    W = fit_ridge(Xf, Yf, lam)
    r2 = r2_score(Yv, predict_ridge(W, Xv), Yf.mean(0, keepdims=True))
    if r2 > best_r2:
      best_lam, best_r2 = lam, r2
  W = fit_ridge(Xtr, Ytr, best_lam)
  r2 = r2_score(Yte, predict_ridge(W, Xte), Ytr.mean(0, keepdims=True))
  return r2, best_lam


def mlp_probe(Xtr, Ytr, Xte, Yte, steps=2000, width=256, lr=1e-3, seed=0):
  """Small 2-hidden-layer MLP probe trained with Adam (JAX)."""
  import jax
  import jax.numpy as jnp
  import optax

  ym = Ytr.mean(0, keepdims=True)
  ys = np.where(Ytr.std(0, keepdims=True) < 1e-6, 1.0,
                Ytr.std(0, keepdims=True))
  Ytr_n = (Ytr - ym) / ys
  rng = np.random.default_rng(seed)
  din, dout = Xtr.shape[1], Ytr.shape[1]

  def init(d_in, d_out, scale):
    return [jnp.asarray(rng.normal(size=(d_in, d_out)) * scale, jnp.float32),
            jnp.zeros(d_out, jnp.float32)]

  params = [init(din, width, np.sqrt(2 / din)),
            init(width, width, np.sqrt(2 / width)),
            init(width, dout, np.sqrt(1 / width))]
  opt = optax.adam(lr)
  ostate = opt.init(params)

  def forward(params, x):
    for w, b in params[:-1]:
      x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b

  def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

  @jax.jit
  def step(params, ostate, x, y):
    loss, grad = jax.value_and_grad(loss_fn)(params, x, y)
    updates, ostate = opt.update(grad, ostate)
    return optax.apply_updates(params, updates), ostate, loss

  Xtr_j, Ytr_j = jnp.asarray(Xtr), jnp.asarray(Ytr_n)
  batch = min(512, len(Xtr))
  for s in range(steps):
    idx = rng.integers(0, len(Xtr), batch)
    params, ostate, _ = step(params, ostate, Xtr_j[idx], Ytr_j[idx])
  pred = np.asarray(forward(params, jnp.asarray(Xte))) * ys + ym
  return r2_score(Yte, pred, Ytr.mean(0, keepdims=True))


# -- sweep ----------------------------------------------------------------

def flatten(X, Y, mask):
  m = mask.reshape(-1)
  return X.reshape(-1, X.shape[-1])[m], Y.reshape(-1, Y.shape[-1])[m]


def shift_target(target, mask, horizon):
  """Align representation_t with target_{t+horizon}; drop invalid tails."""
  if horizon == 0:
    return target, mask
  N, T = mask.shape
  tgt = np.concatenate([target[:, horizon:], target[:, :horizon]], 1)
  valid = mask.copy()
  valid[:, T - horizon:] = False
  valid &= np.concatenate([mask[:, horizon:],
                           np.zeros_like(mask[:, :horizon])], 1)
  return tgt, valid


def run_job(site_feat, target, mask, horizon, n_test, recfield, do_mlp,
            mlp_steps):
  feat = receptive_window(site_feat, recfield) if recfield not in (None, 1) \
      else site_feat
  tgt, valid = shift_target(target, mask, horizon)
  Xtr, Ytr = flatten(feat[:-n_test], tgt[:-n_test], valid[:-n_test])
  Xte, Yte = flatten(feat[-n_test:], tgt[-n_test:], valid[-n_test:])
  if len(Xte) < 8 or len(Xtr) < 32:
    return None
  Xtr_s, Xte_s = standardize(Xtr, Xte)
  r2_ridge, lam = ridge_probe(Xtr_s, Ytr, Xte_s, Yte)
  row = dict(r2_ridge=round(r2_ridge, 5), ridge_lambda=lam,
             feat_dim=feat.shape[-1], n_train=len(Xtr), n_test=len(Xte))
  if do_mlp:
    row['r2_mlp'] = round(
        mlp_probe(Xtr_s, Ytr, Xte_s, Yte, steps=mlp_steps), 5)
  return row


def build_jobs(sites, targets, masks, horizons, recfields):
  """Enumerate (model, site, target, horizon, receptive_field) probe jobs."""
  jobs = []
  rssm_sites = [s for s in ('enc', 'posterior', 'post_deter', 'post_stoch',
                            'prior') if s in sites]

  def add(model, site, target, horizon, rf):
    jobs.append(dict(model=model, site=site, target=target,
                     horizon=horizon, receptive_field=rf))

  # Axis 1+2: current/future simulator state vs probe site.
  if 'state' in targets:
    for h in horizons:
      for s in rssm_sites:
        add('rssm', s, 'state', h, 1)
      if 'vae_enc' in sites:
        add('vae', 'vae_enc', 'state', h, 1)
      if 'vae_latent' in sites:
        for rf in recfields:
          add('vae', 'vae_latent', 'state', h, rf)
      # open-loop RSSM prior predicting the matching future state.
      if h > 0 and f'imag{h}' in sites:
        add('rssm', f'imag{h}', 'state', h, 1)

  # Derived dynamical targets and downstream task-reward probes, horizon 0.
  for tname in ('gait_phase', 'return_to_go', 'time_to_fall',
                'reward_stand', 'reward_walk', 'reward_run'):
    if tname not in targets:
      continue
    for s in rssm_sites:
      add('rssm', s, tname, 0, 1)
    if 'vae_enc' in sites:
      add('vae', 'vae_enc', tname, 0, 1)
    if 'vae_latent' in sites:
      for rf in recfields:
        add('vae', 'vae_latent', tname, 0, rf)
  return jobs


# -- likelihoods ----------------------------------------------------------

def held_out_likelihood(features, n_test):
  """Mean held-out NLL (lower is better) for each generative quantity."""
  out = {}
  names = dict(wm_post_nll='rssm_posterior_recon',
               wm_prior_nll='rssm_prior_predictive',
               vae_recon_nll='vae_recon')
  for key, label in names.items():
    if key in features:
      arr = features[key][-n_test:]
      out[label] = dict(mean=float(arr.mean()), std=float(arr.std()))
  return out


# -- plots ----------------------------------------------------------------

def plot_horizon(rows, outpath):
  fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
  series = {}
  for r in rows:
    if r['target'] != 'state' or r['r2_ridge'] is None:
      continue
    label = r['site'] if r['receptive_field'] in (1, None) \
        else f"{r['site']}[K={r['receptive_field']}]"
    series.setdefault(label, []).append((r['horizon'], r['r2_ridge']))
  for label, pts in sorted(series.items()):
    pts = sorted(pts)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=label)
  ax.set_xlabel('prediction horizon k (steps)')
  ax.set_ylabel('held-out R^2  (target: simulator state)')
  ax.set_title('Probe R^2 vs horizon: per-frame vs predictive content')
  ax.axhline(0, color='#999999', lw=0.8)
  ax.legend(fontsize=7, ncol=2)
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


def plot_latent_pca(sites, targets, outpath, max_points=4000, seed=0):
  """PCA scatter of the RSSM posterior latent, coloured by a task variable
  (research_procedure.md section G, probe 5 -- qualitative separability)."""
  if 'posterior' not in sites:
    return
  feat = sites['posterior'].reshape(-1, sites['posterior'].shape[-1])
  color_key = next((k for k in ('reward_walk', 'torso_height', 'return_to_go')
                    if k in targets), None)
  if color_key is None:
    return
  color = targets[color_key].reshape(-1, targets[color_key].shape[-1])[:, 0]
  rng = np.random.default_rng(seed)
  if len(feat) > max_points:
    idx = rng.choice(len(feat), max_points, replace=False)
    feat, color = feat[idx], color[idx]
  X = feat - feat.mean(0, keepdims=True)
  _, _, Vt = np.linalg.svd(X, full_matrices=False)        # top-2 PCs via SVD
  proj = X @ Vt[:2].T
  fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
  sc = ax.scatter(proj[:, 0], proj[:, 1], c=color, s=4, cmap='viridis')
  fig.colorbar(sc, ax=ax, label=color_key)
  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  ax.set_title(f'RSSM posterior latent (PCA), coloured by {color_key}')
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


def plot_receptive_field(rows, outpath):
  fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
  rf_order = {1: 0, 4: 1, 16: 2, 'all': 3}
  for tname in ('state', 'gait_phase', 'return_to_go'):
    pts = [(rf_order.get(r['receptive_field'], 0), r['r2_ridge'])
           for r in rows if r['site'] == 'vae_latent'
           and r['target'] == tname and r['horizon'] == 0]
    if pts:
      pts = sorted(pts)
      ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=tname)
    post = [r['r2_ridge'] for r in rows if r['site'] == 'posterior'
            and r['target'] == tname and r['horizon'] == 0]
    if post:
      ax.axhline(post[0], ls='--', lw=1,
                 color=ax.lines[-1].get_color() if ax.lines else 'k')
  ax.set_xticks(list(rf_order.values()))
  ax.set_xticklabels(list(rf_order.keys()))
  ax.set_xlabel('VAE receptive field (past frames)')
  ax.set_ylabel('held-out R^2')
  ax.set_title('Static VAE closing the gap via post-hoc aggregation\n'
               '(dashed = RSSM posterior)')
  ax.legend(fontsize=8)
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


# -- driver ---------------------------------------------------------------

def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--traj', required=True)
  p.add_argument('--features', required=True)
  p.add_argument('--output', required=True, help='Output directory.')
  p.add_argument('--horizons', type=int, nargs='+', default=[0, 1, 5, 20])
  p.add_argument('--receptive_fields', nargs='+', default=['1', '4', '16',
                                                           'all'])
  p.add_argument('--test_frac', type=float, default=0.3)
  p.add_argument('--mlp', action='store_true', default=True)
  p.add_argument('--no_mlp', dest='mlp', action='store_false')
  p.add_argument('--mlp_steps', type=int, default=2000)
  return p.parse_args()


def main():
  args = parse_args()
  os.makedirs(args.output, exist_ok=True)
  features = {k: np.asarray(v) for k, v in np.load(args.features).items()}
  with open(args.features + '.meta.json') as f:
    fmeta = json.load(f)
  traj = {k: np.asarray(v) for k, v in np.load(args.traj).items()}
  with open(args.traj + '.meta.json') as f:
    tmeta = json.load(f)

  targets, masks = targets_mod.derive_targets(traj, tmeta)
  sites = site_features(features, fmeta)
  N = traj['reward'].shape[0]
  n_test = max(1, round(args.test_frac * N))
  recfields = [int(x) if x != 'all' else 'all' for x in args.receptive_fields]
  print(f'Episodes: {N} (test {n_test})  |  sites: {sorted(sites)}')
  print(f'Targets: {sorted(targets)}')

  jobs = build_jobs(sites, targets, masks, args.horizons, recfields)
  rows = []
  for i, job in enumerate(jobs):
    res = run_job(
        sites[job['site']], targets[job['target']], masks[job['target']],
        job['horizon'], n_test, job['receptive_field'], args.mlp,
        args.mlp_steps)
    row = {**job, **(res or {'r2_ridge': None})}
    rows.append(row)
    tag = f"{job['model']:>4}/{job['site']:<10} {job['target']:<13}" \
          f" h={job['horizon']:<2} K={job['receptive_field']}"
    print(f"  [{i + 1:>3}/{len(jobs)}] {tag}  "
          f"R2_ridge={row.get('r2_ridge')}  R2_mlp={row.get('r2_mlp')}")

  # Write the long-form results table.
  cols = ['model', 'site', 'target', 'horizon', 'receptive_field',
          'r2_ridge', 'r2_mlp', 'ridge_lambda', 'feat_dim', 'n_train',
          'n_test']
  csv_path = os.path.join(args.output, 'probe_results.csv')
  with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
      w.writerow({c: r.get(c, '') for c in cols})

  likelihood = held_out_likelihood(features, n_test)
  with open(os.path.join(args.output, 'likelihood.json'), 'w') as f:
    json.dump(likelihood, f, indent=2)

  plot_horizon(rows, os.path.join(args.output, 'r2_vs_horizon.png'))
  if 'vae_latent' in sites:                  # VAE-comparison plot, course only
    plot_receptive_field(
        rows, os.path.join(args.output, 'r2_vs_receptive_field.png'))
  plot_latent_pca(sites, targets, os.path.join(args.output, 'latent_pca.png'))
  write_summary(rows, likelihood, args.output)
  print(f'Wrote results -> {args.output}')


def write_summary(rows, likelihood, outdir):
  def get(site, target, horizon=0, rf=1):
    for r in rows:
      if (r['site'] == site and r['target'] == target
          and r['horizon'] == horizon and r['receptive_field'] == rf
          and r.get('r2_ridge') is not None):
        return r['r2_ridge']
    return None

  lines = ['Probing summary (held-out ridge R^2)', '=' * 38, '']
  post0 = get('posterior', 'state', 0)
  vae0 = get('vae_latent', 'state', 0, 1)
  lines.append('Per-frame content (target: state, horizon 0):')
  lines.append(f'  RSSM posterior          R^2 = {post0}')
  lines.append(f'  static VAE latent (K=1) R^2 = {vae0}')
  if post0 is not None and vae0 is not None:
    lines.append(f'  static-vs-sequential gap    = {post0 - vae0:+.4f}')
  lines.append('')
  lines.append('Predictive content (RSSM posterior, target: state):')
  for h in (0, 1, 5, 20):
    lines.append(f'  horizon {h:>2}: R^2 = {get("posterior", "state", h)}')
  lines.append('')
  lines.append('VAE receptive field (target: state, horizon 0):')
  for rf in (1, 4, 16, 'all'):
    lines.append(f'  K={str(rf):>3}: R^2 = {get("vae_latent", "state", 0, rf)}')
  lines.append('')
  lines.append('Reward probe (RSSM posterior, target: task reward, h=0):')
  for tname in ('reward_stand', 'reward_walk', 'reward_run'):
    lines.append(f'  {tname:<13} R^2 = {get("posterior", tname, 0)}')
  lines.append('')
  lines.append('Held-out likelihood (mean NLL, lower is better):')
  for k, v in likelihood.items():
    lines.append(f'  {k:<26} {v["mean"]:.4f}')
  text = '\n'.join(lines) + '\n'
  with open(os.path.join(outdir, 'probe_summary.txt'), 'w') as f:
    f.write(text)
  print('\n' + text)


if __name__ == '__main__':
  main()
