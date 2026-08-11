"""Frozen reader — evaluator-selection follow-ups (ensemble + laws).

Registration: PREREG_evaluator_selection_20260811.md (rules verbatim).
ONE execution:

  python -m analysis.evaluator_selection_read \
      --pool artifacts/goodhart_partial_reg_20260811/pool_ids.json \
      --metrics EV_ax1wm_finger_q1s0_seed1=<...>/pool_metrics.json ... \
      [--action_stats action_stats.json] --output <dir>
Selfcheck: python -m analysis.evaluator_selection_read --selfcheck
"""

import argparse
import json
import math
import os

import numpy as np

from analysis.goodhart_partial_read import (_midranks, perm_p,
                                            regret_stats, spearman)

B_PERM = 10_000
RNG_SEED = 0
POOL_FLOOR = 88
RHO_BAR = 0.4
SELECTION_FACTOR = 0.5
REAL_TOL = 1e-6
LAMBDA_PRIMARY = 1.0
LAMBDAS_DESC = (0.5, 2.0)
EVALUATORS = tuple(f'ax1wm_finger_q1s{s}_seed{k}'
                   for s in (0, 1) for k in range(1, 9))
# Review B1: the executed-Goodhart evaluator, BY NAME (never by index —
# EVALUATORS[9] is q1s1_seed2, not E2).
E2_NAME = 'ax1wm_finger_q1s1_seed1'
assert E2_NAME in EVALUATORS
CURVE_NS = (5, 10, 20, 40, 88)
CURVE_B = 2000


def _check_rows(rows, pool):
  ids = [r['policy_id'] for r in rows]
  assert len(set(ids)) == len(ids), 'duplicate policy rows'
  unknown = [i for i in ids if i not in pool]
  assert not unknown, ('unregistered ids', unknown[:3])
  assert len(ids) >= POOL_FLOOR, ('pool floor', len(ids))
  for r in rows:
    assert abs(float(r['real']) - pool[r['policy_id']]['final10']) \
        <= REAL_TOL, (r['policy_id'], 'real-return provenance')
  return sorted(ids)


def analyse(metrics_by_eval, pool, e2_rows=None, action_stats=None):
  assert set(metrics_by_eval) == set(EVALUATORS), (
      'evaluator set mismatch',
      sorted(set(EVALUATORS) ^ set(metrics_by_eval)))
  ids = None
  for name, m in metrics_by_eval.items():
    got = _check_rows(m['rows'], pool)
    if ids is None:
      ids = got
    assert got == ids, (name, 'policy sets differ across evaluators')
  reals = np.asarray([pool[i]['final10'] for i in ids])
  # score matrix (E, P), z-scored within evaluator
  Z = []
  singles = {}
  for name in EVALUATORS:
    by_id = {r['policy_id']: float(r['m_return'])
             for r in metrics_by_eval[name]['rows']}
    s = np.asarray([by_id[i] for i in ids])
    singles[name] = dict(
        rho=spearman(s, reals),
        top1_regret_frac=regret_stats(s, reals)['top1_regret_frac'])
    sd = s.std()
    Z.append((s - s.mean()) / sd if sd > 1e-12 else np.zeros_like(s))
  Z = np.stack(Z)                                    # (16, P)
  res = {'n_policies': len(ids),
         'singles_panel': singles,
         'singles_rho_max': float(max(v['rho'] for v in
                                      singles.values())),
         'singles_rho_mean': float(np.mean([v['rho'] for v in
                                            singles.values()]))}
  def grade_score(score, tag):
    rho = spearman(score, reals)
    p = perm_p(score, reals)
    reg = regret_stats(score, reals)
    rescues = (p < 0.05 and rho >= RHO_BAR
               and reg['top1_regret_frac'] is not None
               and reg['random_pick_mean_regret_frac'] is not None
               and reg['top1_regret_frac']
               <= SELECTION_FACTOR * reg['random_pick_mean_regret_frac'])
    improves = (not rescues and p < 0.05
                and rho > res['singles_rho_max'])
    verdict = ('ENSEMBLE-RESCUES' if rescues else
               'ENSEMBLE-IMPROVES' if improves else 'ENSEMBLE-FLAT')
    return dict(rho=rho, perm_p=p, verdict=verdict,
                singles_rho_max=res['singles_rho_max'],
                singles_rho_mean=res['singles_rho_mean'],
                top1_regret_frac=reg['top1_regret_frac'],
                random_pick_mean_regret_frac=reg[
                    'random_pick_mean_regret_frac'],
                top3_regret_frac=reg['top3_regret_frac'],
                top5_regret_frac=reg['top5_regret_frac'])
  res['e_r1_ensemble_mean'] = grade_score(Z.mean(0), 'mean')
  res['e_r2_penalized'] = grade_score(
      Z.mean(0) - LAMBDA_PRIMARY * Z.std(0), 'pen')
  res['penalized_desc'] = {
      f'lam{lam:g}': grade_score(Z.mean(0) - lam * Z.std(0), 'd')
      for lam in LAMBDAS_DESC}
  # leg 3a: selection-pressure curve (value-aware descriptive) — always
  # computed under the TRUE E2 (review B1); --e2_metrics is only an
  # optional override for cross-checking the executed collate file.
  e2_rows = e2_rows if e2_rows is not None else \
      metrics_by_eval[E2_NAME]['rows']
  if True:
    by_id = {r['policy_id']: float(r['m_return']) for r in e2_rows}
    s2 = np.asarray([by_id[i] for i in ids])
    rng = np.random.default_rng(RNG_SEED)
    curve = {}
    best_all = None
    for n in CURVE_NS:
      fr, base = [], []
      for _ in range(CURVE_B):
        pick = rng.choice(len(ids), n, replace=False)
        rr, ss = reals[pick], s2[pick]
        best = rr.max()
        if best <= 0:
          continue
        fr.append((best - rr[ss.argmax()]) / best)
        base.append((best - rr.mean()) / best)
      curve[str(n)] = dict(top1_regret_frac=float(np.mean(fr)),
                           random_mean_frac=float(np.mean(base)))
    res['leg3a_pressure_curve'] = dict(
        curve=curve, b=CURVE_B,
        note='VALUE-AWARE descriptive on executed E2 scores; no verdict')
  # leg 3b: shift law under E2 (review B1); missing/partial stats
  # DEGRADE to SKIPPED-disclosed, never abort legs 1-2 (review M9)
  if action_stats is None:
    res['leg3b_shift_law'] = 'SKIPPED-disclosed (no action stats)'
  else:
    missing = [i for i in ids if i not in action_stats]
    if 'evaluator' not in action_stats or missing:
      res['leg3b_shift_law'] = (
          f'SKIPPED-disclosed (stats missing: evaluator='
          f'{"evaluator" not in action_stats}, '
          f'{len(missing)} policies)')
    else:
      ev = np.asarray(action_stats['evaluator'], float)
      by_id = {r['policy_id']: float(r['m_return']) for r in e2_rows}
      s2b = np.asarray([by_id[i] for i in ids])
      rm = _midranks(s2b)
      rr = _midranks(reals)
      dist = [float(np.linalg.norm(
          np.asarray(action_stats[i], float) - ev)) for i in ids]
      disc = [abs(rm[k] - rr[k]) for k in range(len(ids))]
      rho = spearman(np.asarray(dist), np.asarray(disc))
      p = perm_p(np.asarray(dist), np.asarray(disc))
      res['leg3b_shift_law'] = dict(
          rho=rho, perm_p=p,
          rule_note='rho > 0 and two-sided perm p < .05 (no CI for '
                    'this leg, per the amended prereg wording)',
          verdict=('SHIFT-LAW-SUPPORTED' if (rho > 0 and p < 0.05)
                   else 'NULL'))
  return res


# --------------------------------------------------------------------------

def _pool(n=88, rng=None):
  rng = rng or np.random.default_rng(1)
  return {f'adapt_pol{i:03d}': {'final10': float(v)}
          for i, v in enumerate(rng.uniform(0, 900, n))}


def _metrics(pool, signal, noise, rng):
  ids = sorted(pool)
  reals = np.asarray([pool[i]['final10'] for i in ids])
  z = (reals - reals.mean()) / reals.std()
  out = {}
  for name in EVALUATORS:
    s = signal * z + rng.normal(0, noise, len(ids))
    out[name] = {'rows': [dict(policy_id=i, m_return=float(s[k]),
                               real=float(reals[k]))
                          for k, i in enumerate(ids)]}
  return out

def selfcheck():
  pool = _pool()
  rng = np.random.default_rng(3)
  # each single is weak (signal 0.25/noise 1) but the 16-average is strong
  m = _metrics(pool, 0.25, 1.0, rng)
  r = analyse(m, pool, e2_rows=m[E2_NAME]['rows'])
  assert r['e_r1_ensemble_mean']['verdict'] in (
      'ENSEMBLE-RESCUES', 'ENSEMBLE-IMPROVES'), r['e_r1_ensemble_mean']
  assert r['e_r1_ensemble_mean']['rho'] > r['singles_rho_max']
  assert 'leg3a_pressure_curve' in r
  c = r['leg3a_pressure_curve']['curve']
  assert set(c) == {str(n) for n in CURVE_NS}
  # pure noise -> FLAT
  m = _metrics(pool, 0.0, 1.0, np.random.default_rng(5))
  r = analyse(m, pool)
  assert r['e_r1_ensemble_mean']['verdict'] == 'ENSEMBLE-FLAT', \
      r['e_r1_ensemble_mean']
  # shift law: planted distance-error correlation
  m = _metrics(pool, 0.3, 0.8, np.random.default_rng(7))
  ids = sorted(pool)
  reals = np.asarray([pool[i]['final10'] for i in ids])
  # corrupt scores of far policies in the E2 slot
  stats = {'evaluator': [0.0, 1.0]}
  rows = m[E2_NAME]['rows']
  rng2 = np.random.default_rng(9)
  for k, row in enumerate(rows):
    d = float(rng2.uniform(0, 3))
    stats[row['policy_id']] = [d, 1.0]
    row['m_return'] = float(0.3 * reals[k] * max(0.1, 1 - d / 3)
                            + rng2.normal(0, 20) * (1 + d))
  r = analyse(m, pool, e2_rows=rows, action_stats=stats)
  assert r['leg3b_shift_law']['verdict'] == 'SHIFT-LAW-SUPPORTED', \
      r['leg3b_shift_law']
  # gates: evaluator-set mismatch + provenance
  try:
    analyse({k: m[k] for k in list(m)[:15]}, pool)
    raise SystemExit('selfcheck FAIL: evaluator-set gate not tripped')
  except AssertionError:
    pass
  bad = _metrics(pool, 0.2, 1.0, np.random.default_rng(11))
  bad[EVALUATORS[0]]['rows'][0]['real'] += 1.0
  try:
    analyse(bad, pool)
    raise SystemExit('selfcheck FAIL: provenance gate not tripped')
  except AssertionError:
    pass
  print('evaluator_selection_read selfcheck PASS (weak-singles/strong-'
        'ensemble recovered; noise -> FLAT; planted shift-law fires; '
        'evaluator-set + provenance gates trip; pressure-curve emitted)')


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--pool')
  ap.add_argument('--metrics', nargs='+',
                  help='EV_<fit>=path/to/pool_metrics.json x16')
  ap.add_argument('--e2_metrics',
                  help="E2's executed pool_metrics.json (leg 3a)")
  ap.add_argument('--action_stats')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.pool and args.metrics and args.output
  pool = json.load(open(args.pool))
  metrics = {}
  for spec in args.metrics:
    name, path = spec.split('=', 1)
    metrics[name.removeprefix('EV_')] = json.load(open(path))
  e2_rows = (json.load(open(args.e2_metrics))['rows']
             if args.e2_metrics else None)
  stats = json.load(open(args.action_stats)) if args.action_stats else None
  res = analyse(metrics, pool, e2_rows=e2_rows, action_stats=stats)
  res['prereg'] = 'PREREG_evaluator_selection_20260811.md'
  import hashlib
  res['inputs'] = dict(
      pool=args.pool,
      pool_sha256=hashlib.sha256(open(args.pool, 'rb').read()).hexdigest(),
      metrics={spec.split('=', 1)[0]: spec.split('=', 1)[1]
               for spec in args.metrics},
      e2_metrics=args.e2_metrics, action_stats=args.action_stats)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'e_r1': res['e_r1_ensemble_mean'],
                    'e_r2': res['e_r2_penalized'],
                    'singles_rho_max': res['singles_rho_max'],
                    'leg3b': res['leg3b_shift_law']}, indent=1))


if __name__ == '__main__':
  main()
