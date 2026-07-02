"""Gate 0, segment level: matched-buffer *constructability* (plan Sec. 3.6 item 3).

probing/gate0.py screens whole pilot buffers for the two matched quadrants.
That test is sufficient but not necessary: Phase 6 (Axis 1) builds its
intervention buffers *segment-level*, by re-composing replay windows drawn
across the pilot policies.  The pre-registered feasibility criterion (plan
Sec. 3.6 item 3; runbook Phase 3: "compute on the same segment granularity
Phase 6 will use") is therefore whether matched buffers are constructable from
segments -- not whether four raw pilot buffers happen to align pairwise.

This script maps the constructable region.  It composes many fixed-size
candidate buffers (windows sampled without replacement from strata = pilot
buffer x occupancy bin), places every candidate in (coverage, occupancy) space,
and searches candidate *pairs* for the two quadrants:

  Q1  coverage-matched / occupancy-different:
        |dCov| <= cov_match_frac x pilot coverage range,
        dOcc >= occ_sep_mult x bootstrap occupancy noise;
  Q2  coverage-different / occupancy-matched: the reverse;

with bounded window overlap between the two sides of a pair (they must be
materially different datasets).  GO-compose iff both quadrants are
constructable.  Composition only re-weights frames the pilots actually
visited; it cannot manufacture unvisited states -- exactly the feasibility
question Gate 0 owns.  Run before any transfer outcome is observed (same
pre-registration boundary as gate0.py); coverage numbers are internal to this
script (fixed estimator sample size across candidates and pilots).

Example::

    python -m probing.gate0_compose --task dmc_cup_catch --window 50 \
        --replay p2e=$RUN/pilot_p2e_cup_seed1/replay \
                 apt=$RUN/pilot_apt_cup_seed1/replay \
                 random=$RUN/pilot_random_cup_seed1/replay \
                 goal=$RUN/pilot_goal_cup_seed1/replay \
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
from probing.coverage import obs_keys
from probing.gate0 import load_ordered, window_occupancy


def parse_args():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--replay', nargs='+', required=True,
                 help='label=replay_dir entries (the Gate 0 pilot buffers).')
  p.add_argument('--task', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--window', type=int, default=50)
  p.add_argument('--occ_threshold', type=float, default=None)
  # Composed-candidate settings.
  p.add_argument('--n_windows', type=int, default=200,
                 help='Windows per composed candidate buffer (fixed size, so '
                      'coverage/occupancy noise are comparable).')
  p.add_argument('--n_candidates', type=int, default=400)
  p.add_argument('--dirichlet', type=float, default=0.3,
                 help='Dirichlet concentration over strata (small = sparse '
                      'mixtures, explores the corners of the region).')
  p.add_argument('--max_overlap', type=float, default=0.2,
                 help='Max shared-window fraction between the two sides of a '
                      'matched pair.')
  # Coverage estimator (same APT k-NN entropy as probing/coverage.py).
  p.add_argument('--max_frames', type=int, default=2000,
                 help='Frames per coverage estimate; identical for all '
                      'candidates and pilots so entropies are comparable.')
  p.add_argument('--knn', type=int, default=12)
  p.add_argument('--logc', type=float, default=1.0)
  # Criteria (same semantics as gate0.py).
  p.add_argument('--cov_match_frac', type=float, default=0.05)
  p.add_argument('--occ_sep_mult', type=float, default=3.0)
  p.add_argument('--boot', type=int, default=100)
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def knn_entropy(x, knn, logc):
  """APT k-NN entropy (as coverage.particle_entropy), via the Gram trick so it
  is cheap enough to run on hundreds of composed candidates."""
  x = np.asarray(x, np.float64)
  g = x @ x.T
  sq = np.diag(g)
  d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0)
  dist = np.sqrt(d2)
  k = min(knn, len(x) - 1)
  # k+1 smallest per row, sorted, dropping the self-distance (the zero min).
  part = np.sort(np.partition(dist, k, axis=1)[:, :k + 1], axis=1)[:, 1:]
  return float(np.log(logc + part.mean(1)).mean())


def occ_bin(occ):
  if occ <= 1e-6:
    return 'zero'
  return 'high' if occ >= 0.5 else 'mid'


def allocate(weights, caps, total):
  """Integer allocation of `total` draws across strata: proportional to
  weights, capped by stratum sizes; None if capacity is insufficient."""
  w = np.asarray(weights, np.float64)
  if w.sum() <= 0:
    return None
  w = w / w.sum()
  counts = np.minimum(np.floor(w * total).astype(int), caps)
  deficit = total - counts.sum()
  order = np.argsort(-w)
  while deficit > 0:
    progressed = False
    for i in order:
      if deficit == 0:
        break
      if counts[i] < caps[i]:
        counts[i] += 1
        deficit -= 1
        progressed = True
    if not progressed:
      return None
  return counts


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
  W = args.window
  print(f'task={args.task}  regime={spec["name"]}  window={W}  '
        f'coverage keys: {cov_keys}')

  # Load ordered frames, per-window occupancy, and the coverage matrices.
  cov_mat, wocc = {}, {}
  for label, directory in entries:
    fr = load_ordered(directory, load_keys)
    cov_mat[label] = np.concatenate(
        [fr[k] for k in cov_keys], -1).astype(np.float32)
    wocc[label] = window_occupancy(args.task, fr, W, args.occ_threshold)
    print(f'  [{label}] frames={len(cov_mat[label])}  windows={len(wocc[label])}')

  # Shared standardizer (as gate0.py) so coverage is comparable everywhere.
  pooled = np.concatenate(list(cov_mat.values()), 0)
  mean = pooled.mean(0, keepdims=True)
  std = pooled.std(0, keepdims=True)
  std = np.where(std < 1e-6, 1.0, std)
  cov_std = {label: (m - mean) / std for label, m in cov_mat.items()}

  def coverage_of(frames):
    idx = rng.choice(len(frames), min(args.max_frames, len(frames)),
                     replace=False)
    return knn_entropy(frames[idx], args.knn, args.logc)

  # Raw pilot buffers, re-measured at this script's estimator settings; their
  # coverage range defines cov_tol (same definition as gate0.py).
  raw = {}
  for label, _ in entries:
    raw[label] = dict(coverage=coverage_of(cov_std[label]),
                      occupancy=float(wocc[label].mean()))
    print(f'  [{label}] coverage={raw[label]["coverage"]:.4f}  '
          f'occupancy={raw[label]["occupancy"]:.4f}')
  covs = np.array([r['coverage'] for r in raw.values()])
  cov_range = float(covs.max() - covs.min()) or 1.0
  cov_tol = args.cov_match_frac * cov_range

  # Strata: (pilot buffer, occupancy bin) -> window indices.
  strata = {}
  for label, _ in entries:
    for widx, occ in enumerate(wocc[label]):
      strata.setdefault((label, occ_bin(occ)), []).append(widx)
  strata = {k: np.array(v) for k, v in strata.items()}
  snames = sorted(strata)
  caps = np.array([len(strata[s]) for s in snames])
  print('strata: ' + '  '.join(f'{l}/{b}={c}' for (l, b), c
                               in zip(snames, caps)))
  if caps.sum() < args.n_windows:
    raise SystemExit(f'Only {caps.sum()} windows total; need n_windows='
                     f'{args.n_windows}. Collect longer pilots.')

  # Candidate weight vectors: deterministic corners (pure occupancy bin, pure
  # pilot buffer) first, then sparse Dirichlet mixtures.
  weight_sets = []
  for group_key, group_fn in (
      ('bin', lambda s, g: s[1] == g), ('buffer', lambda s, g: s[0] == g)):
    for g in sorted({s[1] if group_key == 'bin' else s[0] for s in snames}):
      w = np.array([float(len(strata[s])) if group_fn(s, g) else 0.0
                    for s in snames])
      if w.sum() >= args.n_windows:
        weight_sets.append(w)
  while len(weight_sets) < args.n_candidates:
    weight_sets.append(rng.dirichlet(np.full(len(snames), args.dirichlet)))

  # Compose and evaluate the candidates.
  candidates = []
  for ci, w in enumerate(weight_sets):
    counts = allocate(w, caps, args.n_windows)
    if counts is None:
      continue
    members, occs = [], []
    for s, c in zip(snames, counts):
      if c == 0:
        continue
      label = s[0]
      for widx in rng.choice(strata[s], c, replace=False):
        members.append((label, int(widx)))
        occs.append(wocc[label][widx])
    occs = np.array(occs)
    boots = [occs[rng.integers(0, len(occs), len(occs))].mean()
             for _ in range(args.boot)]
    frames = np.concatenate(
        [cov_std[l][widx * W:(widx + 1) * W] for l, widx in members], 0)
    mixture = {f'{s[0]}/{s[1]}': int(c) for s, c in zip(snames, counts) if c}
    candidates.append(dict(
        coverage=coverage_of(frames), occupancy=float(occs.mean()),
        occ_noise=float(np.std(boots)), mixture=mixture,
        members=frozenset(members)))
    if (ci + 1) % 50 == 0:
      print(f'  composed {ci + 1}/{len(weight_sets)} candidates')

  typ_noise = float(np.median([c['occ_noise'] for c in candidates]))
  occ_sep_min = args.occ_sep_mult * typ_noise

  # Pair search over composed candidates (Q1 and Q2 quadrants).
  best_q1, best_q2, n_q1, n_q2 = None, None, 0, 0
  for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
      a, b = candidates[i], candidates[j]
      overlap = len(a['members'] & b['members']) / args.n_windows
      if overlap > args.max_overlap:
        continue
      dcov = abs(a['coverage'] - b['coverage'])
      docc = abs(a['occupancy'] - b['occupancy'])
      if dcov <= cov_tol and docc >= occ_sep_min:
        n_q1 += 1
        if best_q1 is None or docc > best_q1[2]:
          best_q1 = (i, j, docc, dcov, overlap)
      if dcov > cov_tol and docc < max(occ_sep_min, 1e-9):
        n_q2 += 1
        if best_q2 is None or dcov > best_q2[2]:
          best_q2 = (i, j, dcov, docc, overlap)

  def pair_record(best, primary, secondary):
    if best is None:
      return None
    i, j, val, other, overlap = best
    sides = []
    for c in (candidates[i], candidates[j]):
      sides.append(dict(coverage=round(c['coverage'], 4),
                        occupancy=round(c['occupancy'], 4),
                        occ_noise=round(c['occ_noise'], 4),
                        mixture=c['mixture']))
    return {primary: round(val, 4), secondary: round(other, 4),
            'overlap_frac': round(overlap, 4), 'sides': sides}

  q1 = pair_record(best_q1, 'docc', 'dcov')
  q2 = pair_record(best_q2, 'dcov', 'docc')
  go = q1 is not None and q2 is not None
  reasons = []
  if q1 is None:
    reasons.append(f'no constructable coverage-matched/occupancy-different '
                   f'pair (cov_tol={cov_tol:.4f}, occ_sep_min={occ_sep_min:.4f})')
  if q2 is None:
    reasons.append('no constructable coverage-different/occupancy-matched pair')

  verdict = dict(
      task=args.task, regime=spec['name'],
      threshold=spec['threshold'] if args.occ_threshold is None
      else args.occ_threshold,
      window=W, n_windows=args.n_windows,
      n_candidates=len(candidates),
      criteria=dict(cov_match_frac=args.cov_match_frac,
                    occ_sep_mult=args.occ_sep_mult,
                    max_overlap=args.max_overlap,
                    dirichlet=args.dirichlet, max_frames=args.max_frames,
                    knn=args.knn, logc=args.logc),
      strata={f'{l}/{b}': int(c) for (l, b), c in zip(snames, caps)},
      pilots={l: dict(coverage=round(r['coverage'], 4),
                      occupancy=round(r['occupancy'], 4))
              for l, r in raw.items()},
      coverage_range=cov_range, cov_tol=cov_tol,
      occ_noise_typical=typ_noise, occ_sep_min=occ_sep_min,
      n_q1_pairs=n_q1, n_q2_pairs=n_q2,
      best_coverage_matched_occupancy_different=q1,
      best_coverage_different_occupancy_matched=q2,
      decision='GO-compose' if go else 'NO-GO-compose',
      reasons=reasons)
  with open(os.path.join(args.output, 'gate0_compose.json'), 'w') as f:
    json.dump(verdict, f, indent=2)

  # Plot: constructable cloud + pilot stars + the best matched pairs.
  fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
  ax.scatter([c['coverage'] for c in candidates],
             [c['occupancy'] for c in candidates],
             s=14, alpha=0.35, color='0.55', label='composed candidates')
  for label, r in raw.items():
    ax.scatter(r['coverage'], r['occupancy'], marker='*', s=220, zorder=5)
    ax.annotate(label, (r['coverage'], r['occupancy']),
                textcoords='offset points', xytext=(7, 5), fontsize=9)
  for best, color, name in ((best_q1, 'tab:red', 'Q1 cov-matched'),
                            (best_q2, 'tab:blue', 'Q2 occ-matched')):
    if best is None:
      continue
    a, b = candidates[best[0]], candidates[best[1]]
    ax.plot([a['coverage'], b['coverage']], [a['occupancy'], b['occupancy']],
            'o-', color=color, ms=8, lw=2, zorder=6, label=name)
  ax.set_xlabel('coverage  (k-NN particle entropy, body-state)')
  ax.set_ylabel(f'occupancy  Pr[{spec["name"]} in regime]')
  ax.set_title(f'Gate 0 segment-level constructability: {args.task}\n'
               f'decision = {verdict["decision"]}')
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=8, loc='best')
  fig.savefig(os.path.join(args.output, 'gate0_compose.png'), dpi=150)
  plt.close(fig)

  print('\n' + '=' * 60)
  print(f'GATE 0 (segment-level) DECISION: {verdict["decision"]}')
  for r in reasons:
    print(f'  - {r}')
  if q1:
    print(f'  Q1 best: docc={q1["docc"]} at dcov={q1["dcov"]} '
          f'(n_q1_pairs={n_q1})')
  if q2:
    print(f'  Q2 best: dcov={q2["dcov"]} at docc={q2["docc"]} '
          f'(n_q2_pairs={n_q2})')
  print(f'Wrote {args.output}/gate0_compose.json and gate0_compose.png')


if __name__ == '__main__':
  main()
