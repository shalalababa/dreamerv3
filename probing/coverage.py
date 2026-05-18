"""Latent / state-space coverage of the pretraining replay buffers.

research_procedure.md section G, probe 3: each reward-free condition collects
its own data, and *how much of the state space that data covers* is one of the
representational properties the study correlates against adaptation speed.

This script reads the per-condition pretraining replay buffers (the data each
condition actually collected) and, for every buffer, estimates how spread out
its observations are. DreamerV3's replay stores only observations, but for the
Walker proprio task the observation is an almost-complete function of the
simulator state, so the proprio observation *is* the low-dimensional state
projection the procedure asks for.

Two coverage estimators, both made comparable across conditions by fitting a
single shared standardiser and PCA on the pooled frames:

  * particle_entropy -- the APT estimator itself (mean log(c + mean k-NN
    distance)) over standardised observations; higher = more spread.
  * hist_entropy_2d  -- Shannon entropy of a 2-D histogram over the top two
    shared principal components; higher = more uniform occupancy.

Run from the repository root:

    python -m probing.coverage \
        --replay random_seed1=/scratch/.../pretrain_random_seed1/replay \
                 p2e_seed1=/scratch/.../pretrain_p2e_seed1/replay \
                 apt_seed1=/scratch/.../pretrain_apt_seed1/replay \
        --output /scratch/.../coverage
"""

import argparse
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

from probing import replay_dataset

# Replay keys that are not part of the proprio observation vector.
NON_OBS = {'reward', 'is_first', 'is_last', 'is_terminal', 'action', 'reset',
           'stepid', 'consec'}


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--replay', nargs='+', required=True,
                 help='One or more label=replay_dir entries.')
  p.add_argument('--output', required=True, help='Output directory.')
  p.add_argument('--max_frames', type=int, default=3000,
                 help='Frames per buffer used for the k-NN estimate.')
  p.add_argument('--knn', type=int, default=12)
  p.add_argument('--logc', type=float, default=1.0)
  p.add_argument('--bins', type=int, default=24)
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def obs_keys(replay_dir):
  """Proprio observation keys in a replay buffer (drop flags / context)."""
  peek = replay_dataset.peek_keys(replay_dir)
  keys = [k for k, shp in peek.items()
          if len(shp) <= 1 and k not in NON_OBS and '/' not in k]
  return sorted(keys)


def load_obs(replay_dir, keys, max_frames, seed):
  """Flatten a replay buffer into a single (N, D) observation matrix."""
  frames = replay_dataset.load_frames(
      replay_dir, keys, max_frames=max_frames, rng=seed, verbose=True)
  return np.concatenate([frames[k] for k in keys], -1).astype(np.float32)


def particle_entropy(x, knn, logc):
  """APT k-NN entropy estimate: mean log(c + mean distance to k neighbours)."""
  d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
  dist = np.sqrt(np.maximum(d2, 0.0))
  knn = min(knn, max(len(x) - 1, 1))
  part = np.partition(dist, knn, axis=1)[:, 1:knn + 1]   # drop self (col 0)
  return float(np.log(logc + part.mean(1)).mean())


def hist_entropy_2d(proj, edges_x, edges_y):
  """Shannon entropy (nats) of a 2-D occupancy histogram."""
  hist, _, _ = np.histogram2d(proj[:, 0], proj[:, 1], bins=[edges_x, edges_y])
  p = hist.reshape(-1)
  p = p[p > 0] / p.sum()
  return float(-(p * np.log(p)).sum())


def main():
  args = parse_args()
  os.makedirs(args.output, exist_ok=True)
  entries = []
  for item in args.replay:
    assert '=' in item, f'Expected label=dir, got {item!r}'
    label, directory = item.split('=', 1)
    entries.append((label, directory))

  # Use the first buffer's keys; assume all conditions share the obs space.
  keys = obs_keys(entries[0][1])
  print(f'Observation keys: {keys}')

  obs = {}
  for label, directory in entries:
    print(f'[{label}] loading {directory}')
    obs[label] = load_obs(directory, keys, args.max_frames, args.seed)

  # Shared standardiser + shared PCA fit on the pooled frames, so the
  # per-condition numbers live on one common scale.
  pooled = np.concatenate(list(obs.values()), 0)
  mean = pooled.mean(0, keepdims=True)
  std = pooled.std(0, keepdims=True)
  std = np.where(std < 1e-6, 1.0, std)
  pooled_std = (pooled - mean) / std
  _, _, Vt = np.linalg.svd(
      pooled_std - pooled_std.mean(0, keepdims=True), full_matrices=False)
  pcs = Vt[:2].T
  proj_pooled = pooled_std @ pcs
  edges_x = np.linspace(*np.percentile(proj_pooled[:, 0], [1, 99]),
                        args.bins + 1)
  edges_y = np.linspace(*np.percentile(proj_pooled[:, 1], [1, 99]),
                        args.bins + 1)

  results = {}
  for label, _ in entries:
    x = (obs[label] - mean) / std
    proj = x @ pcs
    results[label] = dict(
        n_frames=int(len(x)),
        particle_entropy=particle_entropy(x, args.knn, args.logc),
        hist_entropy_2d=hist_entropy_2d(proj, edges_x, edges_y))
    r = results[label]
    print(f'  {label:<18} particle_entropy={r["particle_entropy"]:.4f}  '
          f'hist_entropy_2d={r["hist_entropy_2d"]:.4f}')

  meta = dict(obs_keys=keys, obs_dim=int(pooled.shape[1]),
              knn=args.knn, logc=args.logc, bins=args.bins,
              max_frames=args.max_frames, results=results)
  with open(os.path.join(args.output, 'coverage.json'), 'w') as f:
    json.dump(meta, f, indent=2)

  # Comparison scatter: pretraining occupancy in the shared PCA plane.
  n = len(entries)
  fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False,
                           sharex=True, sharey=True, constrained_layout=True)
  for ax, (label, _) in zip(axes[0], entries):
    proj = ((obs[label] - mean) / std) @ pcs
    ax.scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.3, color='#3366cc')
    ax.set_title(f'{label}\nH_knn={results[label]["particle_entropy"]:.3f}',
                 fontsize=8)
    ax.set_xlabel('PC1')
  axes[0][0].set_ylabel('PC2')
  fig.suptitle('Pretraining-replay state coverage (shared PCA)', fontsize=10)
  fig.savefig(os.path.join(args.output, 'coverage.png'), dpi=150)
  plt.close(fig)
  print(f'Wrote -> {args.output}/coverage.json')


if __name__ == '__main__':
  main()
