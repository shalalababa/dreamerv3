"""Frozen reader for the Goodhart fixed-code partial re-test (2026-08-11).

Registration: PREREG_goodhart_partial_20260811.md (rules verbatim). The
archived sprint/p2elong conclusions are PERMANENTLY WITHDRAWN regardless
of this read (original pools destroyed); this adjudicates the FIXED
instrument on the surviving 106-policy pool only.

Execution (ONE):
  python -m analysis.goodhart_partial_read \
      --pool artifacts/goodhart_partial_reg_20260811/pool_ids.json \
      --metrics E1=<bundle>/E1/pool_metrics.json E2=<bundle>/E2/pool_metrics.json \
      --out artifacts/goodhart_partial_read_<date>
  (an evaluator may be absent; E1 dropped is disclosed per the prereg)
Selfcheck: python -m analysis.goodhart_partial_read --selfcheck
"""

import argparse
import json
import os

import numpy as np

B_PERM = 10_000
RNG_SEED = 0
# Amendment 1 (2026-08-11, pre-execution): floor 95 -> 88. 18 pinned
# members (rif 1-8, p2eof/p2eou 1-5) have no loadable checkpoint; the 88
# s x w_r survivors preserve the full final10 spread (0-904.5, sd 181).
# Verdicts are scoped "within-family policy pool" per the amendment.
POOL_FLOOR = 88
RHO_BAR = 0.4
SELECTION_FACTOR = 0.5
REAL_TOL = 1e-6


def _midranks(x):
  x = np.asarray(x, float)
  order = np.argsort(x, kind='mergesort')
  ranks = np.empty(len(x))
  i = 0
  xs = x[order]
  while i < len(x):
    j = i
    while j + 1 < len(x) and xs[j + 1] == xs[i]:
      j += 1
    ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
    i = j + 1
  return ranks


def spearman(a, b):
  ra, rb = _midranks(a), _midranks(b)
  sa, sb = ra.std(), rb.std()
  if sa < 1e-12 or sb < 1e-12:
    return float('nan')
  return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def perm_p(scores, reals, rng_seed=RNG_SEED, b=B_PERM):
  obs = abs(spearman(scores, reals))
  rng = np.random.default_rng(rng_seed)
  reals = np.asarray(reals, float)
  cnt = 0
  for _ in range(b):
    cnt += abs(spearman(scores, rng.permutation(reals))) >= obs - 1e-15
  return float((cnt + 1) / (b + 1))


def regret_stats(scores, reals):
  scores = np.asarray(scores, float)
  reals = np.asarray(reals, float)
  best = float(reals.max())
  order = np.argsort(-scores, kind='mergesort')
  out = {}
  for k in (1, 3, 5):
    sel = float(reals[order[:k]].max())
    out[f'top{k}_regret'] = best - sel
    out[f'top{k}_regret_frac'] = (best - sel) / best if best > 0 else None
  out['random_pick_mean_regret'] = float(best - reals.mean())
  out['random_pick_mean_regret_frac'] = (
      (best - float(reals.mean())) / best if best > 0 else None)
  # inversion rate (descriptive): discordant pair fraction
  n = len(scores)
  disc = tot = 0
  for i in range(n):
    for j in range(i + 1, n):
      ds, dr = scores[i] - scores[j], reals[i] - reals[j]
      if ds == 0 or dr == 0:
        continue
      tot += 1
      disc += (ds * dr) < 0
  out['inversion_rate'] = disc / tot if tot else None
  return out


def grade(rho, p, reg):
  if p < 0.05 and rho >= RHO_BAR:
    v = ('RANKING-RECOVERED: the fixed evaluator ranks (suggestive that '
         'the withdrawn lottery finding was bug-inflated)')
  elif p < 0.05 and rho > 0:
    v = ('WEAK-RANKING: fixed-code ranking at the withdrawn-era '
         'magnitude - the bug was not the main story')
  elif p < 0.05 and rho < 0:
    v = 'ANTI-RANKING: registered anomaly, disclosed, no wording'
  else:
    v = ('LOTTERY-REPLICATES: the evaluator cannot rank even with fixed '
         'code (cleanly attributable to WM evaluation; forward wording '
         'only)')
  rm = reg['random_pick_mean_regret_frac']
  t1 = reg['top1_regret_frac']
  sel = (t1 is not None and rm is not None
         and t1 <= SELECTION_FACTOR * rm)
  return v, ('SELECTION-USEFUL' if sel else 'SELECTION-NOT-USEFUL')


ORDER = {'LOTTERY-REPLICATES': 0, 'ANTI-RANKING': 0, 'WEAK-RANKING': 1,
         'RANKING-RECOVERED': 2}


def analyse_evaluator(rows, pool):
  ids = [r['policy_id'] for r in rows]
  assert len(set(ids)) == len(ids), 'duplicate policy rows'
  unknown = [i for i in ids if i not in pool]
  assert not unknown, ('unregistered policy ids', unknown[:5])
  missing = sorted(set(pool) - set(ids))
  assert len(ids) >= POOL_FLOOR, ('pool floor violated', len(ids))
  for r in rows:
    pinned = pool[r['policy_id']]['final10']
    assert abs(float(r['real']) - pinned) <= REAL_TOL, (
        r['policy_id'], r['real'], pinned, 'real-return provenance')
  scores = [float(r['m_return']) for r in rows]
  reals = [float(r['real']) for r in rows]
  rho = spearman(scores, reals)
  p = perm_p(scores, reals)
  reg = regret_stats(scores, reals)
  rank_v, sel_v = grade(rho, p, reg)
  return {'n': len(ids), 'dropped_missing_ckpt': missing,
          'spearman': rho, 'perm_p': p, **reg,
          'verdict_ranking': rank_v, 'verdict_selection': sel_v}


def analyse(metrics_by_eval, pool):
  res = {'evaluators': {}}
  grades = []
  for name, m in metrics_by_eval.items():
    r = analyse_evaluator(m['rows'], pool)
    r['collate_crosscheck'] = {
        'spearman_collate': m.get('spearman'),
        'delta': (abs(m['spearman'] - r['spearman'])
                  if m.get('spearman') is not None else None)}
    res['evaluators'][name] = r
    grades.append(r['verdict_ranking'].split(':')[0])
  if not grades:
    raise AssertionError('no evaluator metrics supplied')
  weakest = min(grades, key=lambda g: ORDER[g])
  res['forward_wording'] = (
      f'weaker-of-evaluators rule: {weakest}'
      + ('' if len(grades) == 2 else ' (single evaluator - E1 dropped, '
                                     'disclosed)'))
  res['withdrawn_note'] = (
      'the archived sprint/p2elong Goodhart conclusions are PERMANENTLY '
      'WITHDRAWN independent of this read (original pools destroyed)')
  return res


# --------------------------------------------------------------------------

def _pool(n=106, rng=None):
  rng = rng or np.random.default_rng(1)
  return {f'adapt_pol{i:03d}': {'mode': 'm', 'final10': float(v)}
          for i, v in enumerate(rng.uniform(0, 900, n))}


def _rows(pool, rho_target, rng=None):
  rng = rng or np.random.default_rng(2)
  ids = sorted(pool)
  reals = np.array([pool[i]['final10'] for i in ids])
  z = (reals - reals.mean()) / reals.std()
  noise = rng.normal(0, 1, len(ids))
  if rho_target == 0:
    s = noise
  else:
    lam = abs(rho_target)
    s = np.sign(rho_target) * lam * z + (1 - lam) * noise
  return [dict(policy_id=i, m_return=float(s[k]), real=float(reals[k]))
          for k, i in enumerate(ids)]


def selfcheck():
  pool = _pool()
  # RANKING-RECOVERED
  r = analyse({'E1': {'rows': _rows(pool, 0.9), 'spearman': None}}, pool)
  assert r['evaluators']['E1']['verdict_ranking'].startswith(
      'RANKING-RECOVERED'), r['evaluators']['E1']['verdict_ranking']
  assert r['evaluators']['E1']['verdict_selection'] == 'SELECTION-USEFUL'
  # WEAK-RANKING (mixture lam 0.12 -> realized rho ~0.28, n=106 significant)
  r = analyse({'E1': {'rows': _rows(pool, 0.12), 'spearman': None}}, pool)
  assert r['evaluators']['E1']['verdict_ranking'].startswith('WEAK'), \
      r['evaluators']['E1']
  # LOTTERY
  r = analyse({'E1': {'rows': _rows(pool, 0.0), 'spearman': None}}, pool)
  assert r['evaluators']['E1']['verdict_ranking'].startswith('LOTTERY')
  # ANTI
  r = analyse({'E1': {'rows': _rows(pool, -0.6), 'spearman': None}}, pool)
  assert r['evaluators']['E1']['verdict_ranking'].startswith('ANTI')
  # weaker-of rule
  r = analyse({'E1': {'rows': _rows(pool, 0.9), 'spearman': None},
               'E2': {'rows': _rows(pool, 0.0), 'spearman': None}}, pool)
  assert 'LOTTERY' in r['forward_wording'], r['forward_wording']
  # regret arithmetic on a hand case: scores pick the worst policy
  hp = {f'p{i}': {'mode': 'm', 'final10': v}
        for i, v in enumerate([100.0, 50.0, 0.0])}
  rows = [dict(policy_id='p0', m_return=0.0, real=100.0),
          dict(policy_id='p1', m_return=1.0, real=50.0),
          dict(policy_id='p2', m_return=2.0, real=0.0)]
  reg = regret_stats([r_['m_return'] for r_ in rows],
                     [r_['real'] for r_ in rows])
  assert reg['top1_regret'] == 100.0 and reg['top3_regret'] == 0.0
  assert abs(reg['random_pick_mean_regret'] - 50.0) < 1e-12
  # gate trips: pool floor (amendment-1 pin: 87 refuses, 88 passes),
  # unknown id, real mismatch
  small = dict(list(pool.items())[:87])
  try:
    analyse({'E1': {'rows': _rows(small, 0.5), 'spearman': None}}, small)
    raise SystemExit('selfcheck FAIL: pool floor not enforced')
  except AssertionError:
    pass
  ok88 = dict(list(pool.items())[:88])
  r = analyse({'E1': {'rows': _rows(ok88, 0.5), 'spearman': None}}, ok88)
  assert r['evaluators']['E1']['n'] == 88
  rows = _rows(pool, 0.5)
  rows[0]['policy_id'] = 'adapt_intruder'
  try:
    analyse({'E1': {'rows': rows, 'spearman': None}}, pool)
    raise SystemExit('selfcheck FAIL: unregistered id accepted')
  except AssertionError:
    pass
  rows = _rows(pool, 0.5)
  rows[3]['real'] += 1.0
  try:
    analyse({'E1': {'rows': rows, 'spearman': None}}, pool)
    raise SystemExit('selfcheck FAIL: real-provenance mismatch accepted')
  except AssertionError:
    pass
  # dropped-but-above-floor path: 100 of 106 scored is accepted+recorded
  rows = _rows(pool, 0.5)[:100]
  r = analyse({'E1': {'rows': rows, 'spearman': None}}, pool)
  assert len(r['evaluators']['E1']['dropped_missing_ckpt']) == 6
  print('goodhart_partial_read selfcheck PASS (all four P-GH1 branches, '
        'weaker-of rule, regret arithmetic, floor/intruder/provenance '
        'gates, drop-with-record path)')


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--pool')
  ap.add_argument('--metrics', nargs='+',
                  help='NAME=path/to/pool_metrics.json per evaluator')
  ap.add_argument('--out')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.pool and args.metrics and args.out
  pool = json.load(open(args.pool))
  metrics = {}
  for spec in args.metrics:
    name, path = spec.split('=', 1)
    metrics[name] = json.load(open(path))
  res = analyse(metrics, pool)
  res['prereg'] = 'PREREG_goodhart_partial_20260811.md'
  res['pool_sha_note'] = ('pool pinned in '
                          'artifacts/goodhart_partial_reg_20260811/'
                          'pool_ids.json (committed)')
  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({k: dict(verdict_ranking=v['verdict_ranking'],
                            verdict_selection=v['verdict_selection'],
                            spearman=v['spearman'], perm_p=v['perm_p'],
                            top1_regret_frac=v['top1_regret_frac'])
                    for k, v in res['evaluators'].items()}
                   | {'forward_wording': res['forward_wording']},
                   indent=1))


if __name__ == '__main__':
  main()
