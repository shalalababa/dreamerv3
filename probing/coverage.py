"""Estimate replay-buffer coverage from vector observations."""

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

from probing import regimes
from probing import replay_dataset

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
  p.add_argument('--time_buckets', type=int, default=1,
                 help='If > 1, also report coverage per time window of the '
                      'time-ordered replay (coverage-over-time curve).')
  p.add_argument('--task', default='',
                 help='dmc task (e.g. dmc_cup_catch); restricts the coverage '
                      'vector to body-state keys and enables occupancy under '
                      'the mechanism-derived regime (see probing/regimes.py).')
  p.add_argument('--occ_threshold', type=float, default=None,
                 help='Override the regime threshold (default: calibrated in '
                      'probing/regimes.py).')
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


def load_obs_from_chunks(chunk_paths, keys, max_frames, seed):
  """Flatten an explicit, time-ordered list of replay chunks into (N, D)."""
  buffers = {k: [] for k in keys}
  for path in chunk_paths:
    with np.load(path) as data:
      for k in keys:
        arr = np.asarray(data[k], np.float32)
        buffers[k].append(arr.reshape(arr.shape[0], -1))
  x = np.concatenate(
      [np.concatenate(buffers[k], 0) for k in keys], -1).astype(np.float32)
  if max_frames and len(x) > max_frames:
    idx = np.random.default_rng(seed).choice(len(x), max_frames, replace=False)
    idx.sort()
    x = x[idx]
  return x


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

  all_keys = obs_keys(entries[0][1])
  keys = regimes.coverage_keys(args.task, all_keys) if args.task else all_keys
  regime_needs = ()
  regime_name = None
  if args.task and regimes.has_regime(args.task):
    spec = regimes.spec(args.task)
    regime_name = spec['name']
    regime_needs = tuple(k for k in spec['needs'] if k in all_keys)
    missing = [k for k in spec['needs'] if k not in all_keys]
    if missing:
      raise SystemExit(f'Regime for {args.task} needs {missing}, not in buffer '
                       f'(has {all_keys}).')
  load_keys = sorted(set(keys) | set(regime_needs))
  print(f'Coverage keys (body-state): {keys}')
  if regime_name:
    print(f'Regime "{regime_name}" occupancy from: {list(regime_needs)}')

  obs = {}       # label -> coverage matrix (N, D)
  occupancy = {}
  for label, directory in entries:
    print(f'[{label}] loading {directory}')
    fr = replay_dataset.load_frames(
        directory, load_keys, max_frames=args.max_frames, rng=args.seed)
    obs[label] = np.concatenate(
        [fr[k].reshape(len(fr[k]), -1) for k in keys], -1).astype(np.float32)
    if regime_needs:
      occupancy[label] = regimes.occupancy(args.task, fr, args.occ_threshold)

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
    if label in occupancy:
      results[label]['occupancy'] = occupancy[label]
    r = results[label]
    occ_str = f'  occupancy={r["occupancy"]:.4f}' if 'occupancy' in r else ''
    print(f'  {label:<18} particle_entropy={r["particle_entropy"]:.4f}  '
          f'hist_entropy_2d={r["hist_entropy_2d"]:.4f}{occ_str}')

  over_time = {}
  if args.time_buckets > 1:
    print(f'Coverage over {args.time_buckets} time buckets (0 = earliest):')
    for label, directory in entries:
      chunks = replay_dataset.list_chunks(directory)
      idx_groups = np.array_split(np.arange(len(chunks)), args.time_buckets)
      curve = []
      for bi, idxs in enumerate(idx_groups):
        if len(idxs) == 0:
          curve.append(None)
          continue
        group = [chunks[i] for i in idxs]
        xb = (load_obs_from_chunks(group, keys, args.max_frames, args.seed)
              - mean) / std
        curve.append(dict(
            bucket=bi, n_frames=int(len(xb)),
            particle_entropy=particle_entropy(xb, args.knn, args.logc),
            hist_entropy_2d=hist_entropy_2d(xb @ pcs, edges_x, edges_y)))
      over_time[label] = curve
      pe = [f'{c["particle_entropy"]:.3f}' if c else 'na' for c in curve]
      print(f'  {label:<18} particle_entropy/bucket = [{", ".join(pe)}]')

  meta = dict(obs_keys=keys, obs_dim=int(pooled.shape[1]),
              knn=args.knn, logc=args.logc, bins=args.bins,
              max_frames=args.max_frames, time_buckets=args.time_buckets,
              task=args.task, regime=regime_name,
              occ_threshold=args.occ_threshold,
              results=results, coverage_over_time=over_time)
  with open(os.path.join(args.output, 'coverage.json'), 'w') as f:
    json.dump(meta, f, indent=2)

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

  if over_time:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for label, curve in over_time.items():
      pts = [(c['bucket'], c['particle_entropy']) for c in curve if c]
      if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=label)
    ax.set_xlabel(f'pretraining time bucket '
                  f'(0 = earliest .. {args.time_buckets - 1} = latest)')
    ax.set_ylabel('particle entropy (k-NN coverage)')
    ax.set_title('Replay coverage over pretraining time')
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(args.output, 'coverage_over_time.png'), dpi=150)
    plt.close(fig)
  print(f'Wrote -> {args.output}/coverage.json')


if __name__ == '__main__':
  main()
