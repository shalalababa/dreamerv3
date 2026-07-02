"""Gate 0: quantitative dissociability feasibility check (plan Sec. 3.3).

The causal step (Phase 6) needs confound-matched buffers that are
*coverage-matched / regime-different* and *coverage-different / regime-matched*.
Resampling cannot manufacture a (Cov, Occ) quadrant the exploration never
visited, so before any full run we check that pilot buffers populate the needed
region of coverage x occupancy space.

Given several labelled pilot buffers for one task (one per policy family: P2E,
APT, random, and a goal-reacher for the high-occupancy corner), this:

  1. places each buffer as a point in (coverage, occupancy) space on a shared
     standardizer/PCA, and plots it;
  2. splits each buffer into contiguous windows and reports the window-level
     occupancy distribution and window counts (criterion a: >= min_windows in
     both a high- and a low-occupancy bin);
  3. estimates the occupancy measurement noise by bootstrapping over windows
     (criterion c: regime separation must exceed occ_sep_mult x this noise);
  4. searches buffer pairs for the two matched-quadrant patterns
     (coverage-matched/regime-different and vice versa) under the calibrated
     thresholds (criterion b: coverage mismatch <= cov_match_frac of the pooled
     coverage range);
  5. emits gate0.json + gate0.png and a GO / NO-GO verdict with the reason.

Coverage is measured on body-state keys only and occupancy on the
mechanism-derived regime R^phys -- both via probing/regimes.py.

Example::

    python -m probing.gate0 --task dmc_cup_catch --window 50 \
        --replay p2e=$RUN/pilot_p2e_cup/replay \
                 apt=$RUN/pilot_apt_cup/replay \
                 random=$RUN/pilot_random_cup/replay \
                 goal=$RUN/pilot_goal_cup/replay \
        --output $RUN/gate0_cup
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

from probing import regimes
from probing import replay_dataset
from probing.coverage import particle_entropy, obs_keys


def parse_args():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--replay', nargs='+', required=True,
                 help='label=replay_dir entries (>=2; include a goal-reacher).')
  p.add_argument('--task', required=True, help='dmc task, e.g. dmc_cup_catch.')
  p.add_argument('--output', required=True)
  p.add_argument('--window', type=int, default=50,
                 help='Contiguous window length for window-level occupancy.')
  p.add_argument('--max_frames', type=int, default=5000,
                 help='Frames per buffer for the coverage k-NN estimate.')
  p.add_argument('--knn', type=int, default=12)
  p.add_argument('--logc', type=float, default=1.0)
  p.add_argument('--occ_threshold', type=float, default=None)
  # Criteria (provisional defaults from the runbook; calibrated to observed scale).
  p.add_argument('--min_windows', type=int, default=200)
  p.add_argument('--cov_match_frac', type=float, default=0.05)
  p.add_argument('--occ_sep_mult', type=float, default=3.0)
  p.add_argument('--boot', type=int, default=200)
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def load_ordered(replay_dir, keys):
  """Load frames in chunk (time) order, keeping temporal contiguity.

  Returns {key: (N, d)} concatenated over sorted chunks (no subsampling), so it
  can be split into contiguous windows.
  """
  chunks = replay_dataset.list_chunks(replay_dir)
  if not chunks:
    raise FileNotFoundError(f'No chunks in {replay_dir}')
  buffers = {k: [] for k in keys}
  for path in chunks:
    with np.load(path) as data:
      for k in keys:
        arr = np.asarray(data[k], np.float32)
        buffers[k].append(arr.reshape(arr.shape[0], -1))
  return {k: np.concatenate(v, 0) for k, v in buffers.items()}


def window_occupancy(task, frames, window, threshold):
  """Per-window occupancy = fraction of in-regime frames in each contiguous window."""
  mask = regimes.in_regime(task, frames, threshold).astype(np.float32)
  n = (len(mask) // window) * window
  if n == 0:
    return np.array([mask.mean()]) if len(mask) else np.array([])
  return mask[:n].reshape(-1, window).mean(1)


def main():
  args = parse_args()
  os.makedirs(args.output, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  entries = []
  for item in args.replay:
    assert '=' in item, f'Expected label=dir, got {item!r}'
    label, directory = item.split('=', 1)
    entries.append((label, directory))
  assert len(entries) >= 2, 'Need >= 2 pilot buffers.'

  if not regimes.has_regime(args.task):
    raise SystemExit(f'No regime spec for {args.task}; Gate 0 needs occupancy.')
  spec = regimes.spec(args.task)

  all_keys = obs_keys(entries[0][1])
  cov_keys = regimes.coverage_keys(args.task, all_keys)
  regime_needs = [k for k in spec['needs'] if k in all_keys]
  load_keys = sorted(set(cov_keys) | set(regime_needs))
  print(f'task={args.task}  regime={spec["name"]}  '
        f'threshold={spec["threshold"] if args.occ_threshold is None else args.occ_threshold}')
  print(f'coverage keys (body-state): {cov_keys}')

  # Load ordered frames per buffer; build coverage matrix + window occupancy.
  ordered, cov_mat, wocc = {}, {}, {}
  for label, directory in entries:
    fr = load_ordered(directory, load_keys)
    ordered[label] = fr
    cov_mat[label] = np.concatenate(
        [fr[k] for k in cov_keys], -1).astype(np.float32)
    wocc[label] = window_occupancy(args.task, fr, args.window, args.occ_threshold)
    print(f'  [{label}] frames={len(cov_mat[label])}  windows={len(wocc[label])}')

  # Shared standardizer for comparable coverage across buffers.
  pooled = np.concatenate(list(cov_mat.values()), 0)
  mean = pooled.mean(0, keepdims=True)
  std = pooled.std(0, keepdims=True)
  std = np.where(std < 1e-6, 1.0, std)

  buffers = {}
  for label, _ in entries:
    x = (cov_mat[label] - mean) / std
    if len(x) > args.max_frames:
      idx = rng.choice(len(x), args.max_frames, replace=False)
      x = x[idx]
    cov = particle_entropy(x, args.knn, args.logc)
    occ = float(regimes.occupancy(args.task, ordered[label], args.occ_threshold))
    # Occupancy noise: bootstrap over this buffer's windows.
    w = wocc[label]
    if len(w) > 1:
      boots = [w[rng.integers(0, len(w), len(w))].mean() for _ in range(args.boot)]
      occ_noise = float(np.std(boots))
    else:
      occ_noise = float('nan')
    # Window counts in high / low occupancy bins (criterion a).
    n_high = int((w >= 0.5).sum())
    n_low = int((w <= 1e-6).sum())
    buffers[label] = dict(
        coverage=float(cov), occupancy=occ, occ_noise=occ_noise,
        n_windows=int(len(w)), n_high_occ_windows=n_high,
        n_low_occ_windows=n_low, n_frames=int(len(cov_mat[label])))
    print(f'  [{label}] coverage={cov:.4f}  occupancy={occ:.4f} '
          f'(+/-{occ_noise:.4f})  hi_win={n_high} lo_win={n_low}')

  # --- Criteria evaluation ---
  covs = np.array([b['coverage'] for b in buffers.values()])
  cov_range = float(covs.max() - covs.min()) or 1.0
  cov_tol = args.cov_match_frac * cov_range
  typ_noise = float(np.nanmedian([b['occ_noise'] for b in buffers.values()]))
  occ_sep_min = args.occ_sep_mult * (typ_noise if np.isfinite(typ_noise) else 0.0)

  labels = list(buffers)
  cov_matched_regime_diff = []   # same coverage, different occupancy
  cov_diff_regime_matched = []   # different coverage, same occupancy
  for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
      a, b = buffers[labels[i]], buffers[labels[j]]
      dcov = abs(a['coverage'] - b['coverage'])
      docc = abs(a['occupancy'] - b['occupancy'])
      pair = (labels[i], labels[j], round(dcov, 4), round(docc, 4))
      if dcov <= cov_tol and docc >= occ_sep_min and docc > 0:
        cov_matched_regime_diff.append(pair)
      if dcov > cov_tol and docc < max(occ_sep_min, 1e-9):
        cov_diff_regime_matched.append(pair)

  # Enough windows to build a non-tiny matched buffer in each regime bin.
  total_high = sum(b['n_high_occ_windows'] for b in buffers.values())
  total_low = sum(b['n_low_occ_windows'] for b in buffers.values())
  windows_ok = total_high >= args.min_windows and total_low >= args.min_windows

  quad1_ok = len(cov_matched_regime_diff) > 0
  quad2_ok = len(cov_diff_regime_matched) > 0
  go = windows_ok and quad1_ok and quad2_ok

  reasons = []
  if not windows_ok:
    reasons.append(f'insufficient windows: high={total_high}, low={total_low} '
                   f'(need >= {args.min_windows} each)')
  if not quad1_ok:
    reasons.append(f'no coverage-matched/regime-different pair '
                   f'(cov_tol={cov_tol:.4f}, occ_sep_min={occ_sep_min:.4f})')
  if not quad2_ok:
    reasons.append('no coverage-different/regime-matched pair')

  verdict = dict(
      task=args.task, regime=spec['name'],
      threshold=spec['threshold'] if args.occ_threshold is None
      else args.occ_threshold,
      window=args.window, coverage_range=cov_range, cov_tol=cov_tol,
      occ_noise_typical=typ_noise, occ_sep_min=occ_sep_min,
      criteria=dict(min_windows=args.min_windows,
                    cov_match_frac=args.cov_match_frac,
                    occ_sep_mult=args.occ_sep_mult),
      total_high_occ_windows=int(total_high),
      total_low_occ_windows=int(total_low),
      windows_ok=bool(windows_ok),
      coverage_matched_regime_different=cov_matched_regime_diff,
      coverage_different_regime_matched=cov_diff_regime_matched,
      buffers=buffers,
      decision='GO' if go else 'NO-GO',
      reasons=reasons)
  with open(os.path.join(args.output, 'gate0.json'), 'w') as f:
    json.dump(verdict, f, indent=2)

  # --- Feasibility plot: (coverage, occupancy) with occupancy error bars ---
  fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
  for label, b in buffers.items():
    ax.errorbar(b['coverage'], b['occupancy'],
                yerr=(b['occ_noise'] if np.isfinite(b['occ_noise']) else None),
                fmt='o', ms=9, capsize=3)
    ax.annotate(label, (b['coverage'], b['occupancy']),
                textcoords='offset points', xytext=(6, 4), fontsize=9)
  ax.set_xlabel('coverage  (k-NN particle entropy, body-state)')
  ax.set_ylabel(f'occupancy  Pr[{spec["name"]} in regime]')
  ax.set_title(f'Gate 0 feasibility: {args.task}\ndecision = {verdict["decision"]}')
  ax.grid(True, alpha=0.3)
  fig.savefig(os.path.join(args.output, 'gate0.png'), dpi=150)
  plt.close(fig)

  print('\n' + '=' * 60)
  print(f'GATE 0 DECISION: {verdict["decision"]}')
  for r in reasons:
    print(f'  - {r}')
  if go:
    print(f'  coverage-matched/regime-different pairs: {cov_matched_regime_diff}')
    print(f'  coverage-different/regime-matched pairs: {cov_diff_regime_matched}')
  print(f'Wrote {args.output}/gate0.json and gate0.png')


if __name__ == '__main__':
  main()
