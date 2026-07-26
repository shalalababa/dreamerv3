"""Frozen read for the Paper-3 confirmatory 25m point (A->B transition).

Registered in prereg/PREREG_compcapacity_confirm_20260726.md and
committed BEFORE any size25m run exists. Adjudicates the registered
consequence of the Scaling B read (P-E1 holds, 12m = regime A =>
"the 25m point targets the A->B transition (task-lo rise)").

Cells (canonical AUC csv modes, seeds 1-8 EXPLICIT — the seed-99
timing-smoke rows present in the canonical csv are excluded by rule):
  task@25m ax1s25q1s{0,1}   apt@25m ax1fs25q1s{0,1}
  task@12m ax1s12q1s{0,1}   apt@12m ax1fs12q1s{0,1}

PRIMARY (sole confirmatory): task-lo rise
  R = mean[task@25m,s0] - mean[task@12m,s0]  (AUC100k)
  two-sample seed-cluster bootstrap (independent resampling of the 8
  seeds within each stratum), B=10K, default_rng(0); FIRES iff CI > 0.
  (Both strata share the zero-init ctx buffer protocol, so this level
  comparison is protocol-clean in a way 1m<->12m never was.)

GUARD (P-E1 forbidden pattern at 25m, decisional for the ordering
law): apt rise A = mean[apt@25m, sides pooled] - mean[apt@12m, sides
pooled], same bootstrap. FORBIDDEN iff A's CI > 0 while the PRIMARY
CI is not > 0 (apt un-nulls while task-lo stays floored) => the
ordering law is REFUTED as registered.

Registered secondaries (never decisional): interaction@25m
(within-seed B_task - B_apt at 25m); B_task shrinkage 25m vs 12m with
its decomposition (task-lo rise vs task-hi change); apt side simples
at 25m; all cell means; sensitivity t-intervals.

Usage:
  python -m analysis.compcapacity_read --auc <canonical auc.csv> \
      --output <dir>
  python -m analysis.compcapacity_read --selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats as sps

B_BOOT = 10_000
RNG_SEED = 0
SEEDS = tuple(range(1, 9))
MODES = dict(task25s0='ax1s25q1s0', task25s1='ax1s25q1s1',
             apt25s0='ax1fs25q1s0', apt25s1='ax1fs25q1s1',
             task12s0='ax1s12q1s0', task12s1='ax1s12q1s1',
             apt12s0='ax1fs12q1s0', apt12s1='ax1fs12q1s1')


def load_cells(path):
  cells = {k: {} for k in MODES}
  with open(path) as f:
    for r in csv.DictReader(f):
      for cell, mode in MODES.items():
        if r['mode'] == mode and int(r['seed']) in SEEDS:
          assert r['qc_pass'] in ('1', 'True', 'true'), (
              f"{mode} seed {r['seed']}: qc_pass={r['qc_pass']}")
          cells[cell][int(r['seed'])] = float(r['auc100k'])
  for cell, vals in cells.items():
    missing = [s for s in SEEDS if s not in vals]
    assert not missing, f'{cell} ({MODES[cell]}): missing seeds {missing}'
  return {k: np.array([v[s] for s in SEEDS]) for k, v in cells.items()}


def _two_sample(a, b, rng):
  """Bootstrap CI of mean(a) - mean(b), independent seed clusters."""
  point = float(a.mean() - b.mean())
  n, m = len(a), len(b)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = (a[rng.integers(0, n, n)].mean()
                - b[rng.integers(0, m, m)].mean())
  lo, hi = np.percentile(stats, [2.5, 97.5])
  t = sps.ttest_ind(a, b, equal_var=False)
  se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / m)
  df = t.df if hasattr(t, 'df') else n + m - 2
  tcrit = sps.t.ppf(0.975, df)
  return dict(point=point, ci=[float(lo), float(hi)], n=[n, m],
              sensitivity=dict(t_p=float(t.pvalue),
                               t_ci=[point - tcrit * se, point + tcrit * se]))


def _paired(deltas, rng):
  d = np.asarray(deltas, float)
  n = len(d)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = d[rng.integers(0, n, n)].mean()
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=float(d.mean()), ci=[float(lo), float(hi)], n=n,
              pos=int((d > 0).sum()))


def analyze(cells):
  rng = np.random.default_rng(RNG_SEED)
  primary = _two_sample(cells['task25s0'], cells['task12s0'], rng)
  apt25 = np.concatenate([cells['apt25s0'], cells['apt25s1']])
  apt12 = np.concatenate([cells['apt12s0'], cells['apt12s1']])
  guard = _two_sample(apt25, apt12, rng)
  fires = primary['ci'][0] > 0
  forbidden = (guard['ci'][0] > 0) and not fires

  inter25 = _paired((cells['task25s1'] - cells['task25s0'])
                    - (cells['apt25s1'] - cells['apt25s0']), rng)
  btask25 = _paired(cells['task25s1'] - cells['task25s0'], rng)
  btask12 = _paired(cells['task12s1'] - cells['task12s0'], rng)
  btask_change = _two_sample(cells['task25s1'] - cells['task25s0'],
                             cells['task12s1'] - cells['task12s0'], rng)
  taskhi_change = _two_sample(cells['task25s1'], cells['task12s1'], rng)
  apt_simple25 = _paired(cells['apt25s1'] - cells['apt25s0'], rng)

  if forbidden:
    verdict = ('FORBIDDEN PATTERN at 25m: apt rises off its floor while '
               'task-lo does not — the ordering law (P-E1) is REFUTED as '
               'registered; Paper 3 pivots to measuring where the model '
               'fails; Paper 1 legibility framing must weaken.')
  elif fires:
    verdict = ('PRIMARY FIRES: task-lo rises at 25m — the A->B transition '
               'is DETECTED within the registered ordering (capacity '
               'repairs the dose deficit first); the flagship headline '
               'is "which deficit capacity repairs, in which order."')
  else:
    verdict = ('PRIMARY does not fire: regime A persists through 25m — '
               'the transition bracket widens (G1 crossing above 25m); '
               'reported as a bounded null with the E4 membership '
               'counterpart; no ordering violation.')
  return dict(primary_task_lo_rise=primary, fires=bool(fires),
              guard_apt_rise=guard, forbidden_pattern=bool(forbidden),
              interaction_at_25m=inter25, b_task_25m=btask25,
              b_task_12m=btask12, b_task_change=btask_change,
              task_hi_change=taskhi_change, apt_simple_25m=apt_simple25,
              cell_means={k: float(v.mean()) for k, v in cells.items()},
              seeds=list(SEEDS), verdict=verdict)


def read(args):
  cells = load_cells(args.auc)
  res = analyze(cells)
  res['auc_csv'] = os.path.abspath(args.auc)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'compcapacity_read.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  p = res['primary_task_lo_rise']
  g = res['guard_apt_rise']
  print(f"PRIMARY task-lo rise (25m-12m): {p['point']:+.1f} "
        f"CI=[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] fires={res['fires']}")
  print(f"GUARD apt rise: {g['point']:+.1f} "
        f"CI=[{g['ci'][0]:+.1f},{g['ci'][1]:+.1f}] "
        f"forbidden={res['forbidden_pattern']}")
  i = res['interaction_at_25m']
  print(f"interaction@25m: {i['point']:+.1f} "
        f"CI=[{i['ci'][0]:+.1f},{i['ci'][1]:+.1f}] pos={i['pos']}/8")
  print(res['verdict'])
  print(f'-> {out}')


def _synth(task25s0, apt25=None, rng=None):
  rng = rng or np.random.default_rng(3)
  base = dict(task12s0=97.0, task12s1=236.0, apt12s0=87.0, apt12s1=100.0,
              task25s1=236.0, apt25s0=87.0, apt25s1=100.0)
  base['task25s0'] = task25s0
  if apt25 is not None:
    base['apt25s0'] = base['apt25s1'] = apt25
  return {k: v + 12 * rng.standard_normal(8) for k, v in base.items()}


def selfcheck(args):
  res = analyze(_synth(task25s0=180.0))
  assert res['fires'] and not res['forbidden_pattern'], res['verdict']
  assert res['verdict'].startswith('PRIMARY FIRES')
  res = analyze(_synth(task25s0=97.0))
  assert not res['fires'] and not res['forbidden_pattern']
  assert res['verdict'].startswith('PRIMARY does not fire')
  res = analyze(_synth(task25s0=97.0, apt25=170.0))
  assert res['forbidden_pattern'], res['verdict']
  assert res['verdict'].startswith('FORBIDDEN')
  # csv loader: seed-99 rows ignored, missing seed trips, qc trips.
  import io, tempfile
  rows = ['run_id,mode,domain,seed,auc100k,qc_pass']
  for cell, mode in MODES.items():
    for s in SEEDS:
      rows.append(f'r,{mode},finger,{s},100.0,1')
  rows.append(f'r,{MODES["task25s0"]},finger,99,999.0,1')
  with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
    f.write('\n'.join(rows)); path = f.name
  cells = load_cells(path)
  assert cells['task25s0'].mean() == 100.0, 'seed-99 row not excluded'
  with open(path, 'w') as f:
    f.write('\n'.join(r for r in rows if ',3,' not in r or 's25q1s0' not in r))
  try:
    load_cells(path)
    raise SystemExit('selfcheck FAIL: missing seed not caught')
  except AssertionError:
    pass
  os.unlink(path)
  print('selfcheck PASS: fire/null/forbidden branches, seed-99 exclusion, '
        'missing-seed trip')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--output', default='analysis_out/compcapacity')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
