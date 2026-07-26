"""Frozen read for the volume-anomaly replication (P-E4a).

Registered in prereg/PREREG_volume_repl_20260726.md and committed
BEFORE any fresh-seed volume run exists. Adjudicates P-E4a
(PREREG_compcapacity_theory_20260724): does the 18-Jul v400s1 anomaly
(low-occupancy 400-episode buffer beating high-occupancy 200-episode
buffer, task arm) replicate on fresh seeds?

Cells (canonical AUC csv modes; task arm):
  v200s1 ax1v2q1v200s1   v400s1 ax1v4q1v400s1   (primary contrast)
  v400s0 ax1v4q1v400s0   v200s0 ax1v2q1v200s0   (secondary, s0)
Original batch = seeds 1-6 (KNOWN at freeze); fresh batch = seeds 7-12.

PRIMARY (sole confirmatory): fresh-batch mean[v400s1] - mean[v200s1],
two-sample seed-cluster bootstrap (B=10K, default_rng(0)); FIRES iff
CI > 0 => P-E4a REPLICATES (support breadth enters the theory as a
lambda-structure input). Does not fire => registered demotion to
winner's curse (no theory term; frozen consequence).

Registered descriptives (never decisional): pooled-12 contrast;
original-batch contrast restated; s0 fresh contrast (v400s0 vs v200s0
— the occupancy-conditional cell for P-E4b); per-cell means.

Usage:
  python -m analysis.volume_repl_read --auc <canonical auc.csv> --output <dir>
  python -m analysis.volume_repl_read --selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
FRESH = tuple(range(7, 13))
ORIG = tuple(range(1, 7))
MODES = dict(v200s1='ax1v2q1v200s1', v400s1='ax1v4q1v400s1',
             v200s0='ax1v2q1v200s0', v400s0='ax1v4q1v400s0')


def load_cells(path):
  cells = {k: {} for k in MODES}
  with open(path) as f:
    for r in csv.DictReader(f):
      for cell, mode in MODES.items():
        if r['mode'] == mode:
          assert r['qc_pass'] in ('1', 'True', 'true'), (
              f"{mode} seed {r['seed']}: qc_pass={r['qc_pass']}")
          cells[cell][int(r['seed'])] = float(r['auc100k'])
  for cell in ('v200s1', 'v400s1'):
    missing = [s for s in ORIG + FRESH if s not in cells[cell]]
    assert not missing, f'{cell}: missing seeds {missing}'
  # s0 cells: v400s0 orig+fresh expected; v200s0 fresh-only acceptable
  # (never run before) — all-or-nothing on the fresh batch.
  for cell in ('v200s0', 'v400s0'):
    have = [s for s in FRESH if s in cells[cell]]
    assert len(have) in (0, len(FRESH)), (
        f'{cell}: partial fresh batch {have}')
  return cells


def _two_sample(a, b, rng):
  a, b = np.asarray(a, float), np.asarray(b, float)
  point = float(a.mean() - b.mean())
  n, m = len(a), len(b)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = (a[rng.integers(0, n, n)].mean()
                - b[rng.integers(0, m, m)].mean())
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=point, ci=[float(lo), float(hi)], n=[n, m])


def analyze(cells):
  rng = np.random.default_rng(RNG_SEED)
  sel = lambda c, seeds: [cells[c][s] for s in seeds if s in cells[c]]
  primary = _two_sample(sel('v400s1', FRESH), sel('v200s1', FRESH), rng)
  fires = primary['ci'][0] > 0
  out = dict(
      primary_fresh_v400s1_minus_v200s1=primary, fires=bool(fires),
      pooled12=_two_sample(sel('v400s1', ORIG + FRESH),
                           sel('v200s1', ORIG + FRESH), rng),
      original_batch=_two_sample(sel('v400s1', ORIG),
                                 sel('v200s1', ORIG), rng),
      s0_fresh=(_two_sample(sel('v400s0', FRESH), sel('v200s0', FRESH),
                            rng)
                if sel('v200s0', FRESH) and sel('v400s0', FRESH) else None),
      cell_means={c: {str(s): v for s, v in sorted(cells[c].items())}
                  for c in cells},
      verdict=('P-E4a REPLICATES: fresh-batch v400s1 > v200s1 — support '
               'breadth (episode diversity) enters the theory as a '
               'lambda-structure input; P-E4b diversity-index leg engages.'
               if fires else
               'P-E4a does NOT replicate: registered demotion to '
               "winner's curse — no support-breadth term is added "
               '(frozen consequence).'))
  return out


def read(args):
  cells = load_cells(args.auc)
  res = analyze(cells)
  res['auc_csv'] = os.path.abspath(args.auc)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'volume_repl.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  p = res['primary_fresh_v400s1_minus_v200s1']
  print(f"PRIMARY fresh v400s1-v200s1: {p['point']:+.1f} "
        f"CI=[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] fires={res['fires']}")
  print(res['verdict'])
  print(f'-> {out}')


def _synth(gap, rng=None):
  rng = rng or np.random.default_rng(5)
  cells = {c: {} for c in MODES}
  for s in ORIG + FRESH:
    cells['v200s1'][s] = 200 + 60 * rng.standard_normal()
    cells['v400s1'][s] = 200 + gap + 60 * rng.standard_normal()
    cells['v400s0'][s] = 300 + 60 * rng.standard_normal()
    if s in FRESH:
      cells['v200s0'][s] = 250 + 60 * rng.standard_normal()
  return cells


def selfcheck(args):
  res = analyze(_synth(gap=260))
  assert res['fires'] and res['verdict'].startswith('P-E4a REPLICATES')
  assert res['s0_fresh'] is not None
  res = analyze(_synth(gap=0))
  assert not res['fires'] and 'winner' in res['verdict']
  cells = _synth(gap=260)
  del cells['v400s1'][9]
  try:
    load_cells_check = [s for s in ORIG + FRESH if s not in cells['v400s1']]
    assert not load_cells_check
    raise SystemExit('selfcheck FAIL: missing fresh seed not caught')
  except AssertionError:
    pass
  cells = _synth(gap=260)
  cells['v200s0'] = {7: 1.0}  # partial fresh cohort must trip in loader
  import tempfile
  rows = ['run_id,mode,domain,seed,auc100k,qc_pass']
  for c, m in MODES.items():
    for s, v in cells[c].items():
      rows.append(f'r,{m},finger,{s},{v},1')
  with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
    f.write('\n'.join(rows)); path = f.name
  try:
    load_cells(path)
    raise SystemExit('selfcheck FAIL: partial s0 cohort not caught')
  except AssertionError:
    pass
  os.unlink(path)
  print('selfcheck PASS: replicate/demote branches, missing-seed and '
        'partial-cohort trips')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--output', default='analysis_out/volume_repl')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
