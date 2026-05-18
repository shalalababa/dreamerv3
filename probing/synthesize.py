"""Synthesis: early-adaptation AUC vs. representation metrics (procedure H).

Builds the single table the study is designed to produce -- one row per
(condition, seed) carrying the early-adaptation AUC for each downstream task
alongside every representation metric from section G -- and then correlates
adaptation speed against each metric across the pretrained conditions. That
correlation, not the side-by-side table, is what turns RQ2 from "restated"
into "answered" (procedure H).

It reads three sources, tolerating any that are missing (e.g. adaptation runs
still in flight) and filling absent cells with NaN:

  * adaptation / from-scratch runs -> scores.jsonl  -> early-adaptation AUC;
  * probes_<cond>_seed<s>/probe_results.csv         -> probe R^2 metrics;
  * coverage/coverage.json                          -> replay coverage.

Run from the repository root:

    python -m probing.synthesize \
        --run_root  /scratch/midway3/$USER/dreamerv3_runs \
        --output    /scratch/midway3/$USER/dreamerv3_runs/g_synthesis
"""

import argparse
import csv
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

# C1 has no pretrained world model, so it carries no representation metrics;
# it still contributes adaptation AUC as the no-pretraining lower bound.
CONDITIONS = ('c1', 'random', 'p2e', 'apt')
PRETRAINED = ('random', 'p2e', 'apt')
TASKS = ('stand', 'walk', 'run')
REPR_METRICS = ('state_h0', 'state_h5', 'state_h20', 'gait_phase',
                'time_to_fall', 'reward_stand', 'reward_walk', 'reward_run',
                'particle_entropy', 'hist_entropy_2d')


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  default_root = os.path.expandvars('/scratch/midway3/$USER/dreamerv3_runs')
  p.add_argument('--run_root', default=default_root)
  p.add_argument('--probe_root', default='',
                 help='Where probes_<cond>_seed<s>/ live (default: run_root).')
  p.add_argument('--coverage', default='',
                 help='coverage.json path (default: <run_root>/coverage/...).')
  p.add_argument('--output', default='',
                 help='Output dir (default: <run_root>/g_synthesis).')
  p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
  p.add_argument('--early_window', type=int, default=125000,
                 help='Early-AUC window: first ~half of the 250K budget A.')
  return p.parse_args()


def run_dir(run_root, cond, task, seed):
  """Locate the run directory for one (condition, task, seed)."""
  if cond == 'c1':
    if task == 'walk':                       # reused existing from-scratch run
      return os.path.join(run_root, f'dmc_proprio_walker_walk_seed{seed}')
    return os.path.join(run_root, f'c1_{task}_seed{seed}')
  return os.path.join(run_root, f'adapt_{cond}_{task}_seed{seed}')


def early_auc(scores_path, window):
  """Mean episode score over the early-adaptation window (normalised AUC)."""
  if not os.path.exists(scores_path):
    return float('nan')
  steps, scores = [], []
  with open(scores_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      d = json.loads(line)
      if 'step' in d and 'episode/score' in d:
        steps.append(d['step'])
        scores.append(d['episode/score'])
  if len(steps) < 2:
    return float('nan')
  steps = np.asarray(steps, np.float64)
  scores = np.asarray(scores, np.float64)
  order = np.argsort(steps)
  steps, scores = steps[order], scores[order]
  keep = steps <= window
  if keep.sum() < 2:
    return float('nan')
  s, sc = steps[keep], scores[keep]
  return float(np.trapz(sc, s) / max(s[-1] - s[0], 1.0))


def read_probe_csv(path):
  """Extract the section-G probe R^2 metrics from one probe_results.csv."""
  if not os.path.exists(path):
    return {}
  want = {
      'state_h0': ('state', '0'), 'state_h5': ('state', '5'),
      'state_h20': ('state', '20'), 'gait_phase': ('gait_phase', '0'),
      'time_to_fall': ('time_to_fall', '0'),
      'reward_stand': ('reward_stand', '0'), 'reward_walk': ('reward_walk', '0'),
      'reward_run': ('reward_run', '0')}
  out = {}
  with open(path) as f:
    for row in csv.DictReader(f):
      if row.get('site') != 'posterior' or str(row.get('receptive_field')) \
          not in ('1', 'None', ''):
        continue
      for name, (tgt, hor) in want.items():
        if row.get('target') == tgt and str(row.get('horizon')) == hor:
          try:
            out[name] = float(row['r2_ridge'])
          except (ValueError, KeyError, TypeError):
            pass
  return out


def pearson(x, y):
  """Pearson r over finite pairs; returns (r, n)."""
  x, y = np.asarray(x, float), np.asarray(y, float)
  m = np.isfinite(x) & np.isfinite(y)
  if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
    return float('nan'), int(m.sum())
  return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
  args = parse_args()
  probe_root = args.probe_root or args.run_root
  coverage_path = args.coverage or os.path.join(
      args.run_root, 'coverage', 'coverage.json')
  output = args.output or os.path.join(args.run_root, 'g_synthesis')
  os.makedirs(output, exist_ok=True)

  coverage = {}
  if os.path.exists(coverage_path):
    coverage = json.load(open(coverage_path)).get('results', {})
  else:
    print(f'No coverage file at {coverage_path} (coverage cells -> NaN)')

  # One record per (condition, seed): per-task AUC + representation metrics.
  records = []
  for cond in CONDITIONS:
    for seed in args.seeds:
      rec = dict(condition=cond, seed=seed)
      for task in TASKS:
        rec[f'auc_{task}'] = early_auc(
            os.path.join(run_dir(args.run_root, cond, task, seed),
                         'scores.jsonl'), args.early_window)
      probe = {} if cond == 'c1' else read_probe_csv(os.path.join(
          probe_root, f'probes_{cond}_seed{seed}', 'probe_results.csv'))
      cov = coverage.get(f'{cond}_seed{seed}', {})
      for m in REPR_METRICS:
        if m in ('particle_entropy', 'hist_entropy_2d'):
          rec[m] = float(cov.get(m, float('nan')))
        else:
          rec[m] = probe.get(m, float('nan'))
      records.append(rec)

  cols = (['condition', 'seed'] + [f'auc_{t}' for t in TASKS]
          + list(REPR_METRICS))
  table_path = os.path.join(output, 'g_table.csv')
  with open(table_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for rec in records:
      w.writerow({c: rec.get(c, '') for c in cols})

  # Correlation: per task, early-AUC vs each representation metric, across the
  # pretrained conditions x seeds (C1 excluded -- it has no representation).
  pre = [r for r in records if r['condition'] in PRETRAINED]
  lines = ['Section-H synthesis: early-adaptation AUC vs representation',
           '=' * 60, '',
           f'Early-AUC window: first {args.early_window} steps.',
           f'Correlations over pretrained conditions {PRETRAINED} x '
           f'seeds {args.seeds}.', '']
  for task in TASKS:
    auc = [r[f'auc_{task}'] for r in pre]
    lines.append(f'[task: {task}]  AUC available for '
                 f'{int(np.isfinite(auc).sum())}/{len(auc)} runs')
    for m in REPR_METRICS:
      r, n = pearson([rr[m] for rr in pre], auc)
      rtxt = 'n/a' if not np.isfinite(r) else f'{r:+.3f}'
      lines.append(f'  corr(AUC, {m:<16}) = {rtxt}   (n={n})')
    lines.append('')
  report = '\n'.join(lines) + '\n'
  with open(os.path.join(output, 'correlations.txt'), 'w') as f:
    f.write(report)
  print('\n' + report)
  print(f'Wrote -> {table_path}')
  print(f'Wrote -> {os.path.join(output, "correlations.txt")}')


if __name__ == '__main__':
  main()
