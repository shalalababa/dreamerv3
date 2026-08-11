"""Frozen reader for the cup parity equivalence re-test (P-C3-EQ).

Registration: PREREG_cup_parity_tost_20260810.md (successor to the
executed P-C3 read; value-aware, disclosed there). Same bundle, gates,
pairing, clustering as analysis.cup_refit_read. ONE execution:

  python -m analysis.cup_parity_tost \
      --bundle local_results/regen_ridge_e4_20260810_220849 \
      --out artifacts/cup_parity_tost_20260810
Selfcheck: python -m analysis.cup_parity_tost selfcheck
"""

import argparse
import json
import os

import numpy as np

from analysis.cup_refit_read import (B_BOOT, RNG_SEED, _ndtri, bca,
                                     load_rows)

BAND = 0.10
ALPHA = 0.05


def paired_diffs(rows):
  cells = {}
  for r in rows:
    cells.setdefault((r['side'], r['seed']), {})[r['arm']] = r['auroc']
  diffs, clusters = [], []
  for (side, seed), d in sorted(cells.items()):
    diffs.append(d['task'] - d['apt'])
    clusters.append(seed)
  return np.asarray(diffs, float), clusters


def bca_ci(vals, clusters, lo_pt, hi_pt, rng_seed=RNG_SEED, b=B_BOOT):
  """Cluster BCa at arbitrary quantiles (reuses the house machinery via a
  local re-implementation of the quantile map around cup_refit_read.bca's
  internals)."""
  import math
  vals = np.asarray(vals, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  idx = {c: np.where(cl == c)[0] for c in uniq}
  rng = np.random.default_rng(rng_seed)
  boots = np.empty(b)
  for i in range(b):
    pick = rng.choice(len(uniq), len(uniq), replace=True)
    sel = np.concatenate([idx[uniq[p]] for p in pick])
    boots[i] = np.nanmean(vals[sel])
  theta = float(np.nanmean(vals))
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
  z0 = _ndtri(prop)
  jack = np.asarray([float(np.nanmean(vals[cl != c])) for c in uniq])
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  a_acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha_pt):
    z = _ndtri(alpha_pt)
    adj = z0 + (z0 + z) / (1.0 - a_acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2.0)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  return theta, (q(lo_pt), q(hi_pt))


def shifted_signflip_p(diffs, clusters, mu0, side, rng_seed=RNG_SEED,
                       b=B_BOOT):
  """One-sided cluster sign-flip test of H0: mean == mu0 against
  side='below' (alternative mean < mu0) or 'above' (mean > mu0)."""
  d = np.asarray(diffs, float) - mu0
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  masks = np.stack([cl == c for c in uniq])
  rng = np.random.default_rng(rng_seed)
  obs = float(np.nanmean(d))
  flips = rng.choice([-1.0, 1.0], size=(b, len(uniq)))
  null = np.nanmean((flips @ masks) * d[None], axis=1)
  if side == 'below':
    return float((np.sum(null <= obs + 1e-15) + 1) / (b + 1))
  return float((np.sum(null >= obs - 1e-15) + 1) / (b + 1))


def analyse(cup_rows, finger_rows):
  cd, ccl = paired_diffs(cup_rows)
  fd, fcl = paired_diffs(finger_rows)
  cup_mean, cup_ci90 = bca_ci(cd, ccl, ALPHA, 1.0 - ALPHA)
  p_upper = shifted_signflip_p(cd, ccl, +BAND, 'below')
  p_lower = shifted_signflip_p(cd, ccl, -BAND, 'above')
  fin_mean, fin_ci95 = bca_ci(fd, fcl, 0.025, 0.975)
  cup_leg = (-BAND < cup_ci90[0] and cup_ci90[1] < BAND
             and p_upper < ALPHA and p_lower < ALPHA)
  finger_leg = fin_ci95[0] > BAND
  established = bool(cup_leg and finger_leg)
  return {
      'band': BAND,
      'cup': {'mean': cup_mean, 'ci90': list(cup_ci90),
              'tost_p_vs_upper': p_upper, 'tost_p_vs_lower': p_lower,
              'n_pairs': len(cd),
              'perm_note': 'cluster sign flips, 2^8 patterns floor p '
                           'at ~1/257'},
      'finger': {'mean': fin_mean, 'ci95': list(fin_ci95),
                 'bar': f'CI entirely > +{BAND} (stricter than the '
                        'original > 0)'},
      'cup_leg_pass': bool(cup_leg),
      'finger_leg_pass': bool(finger_leg),
      'verdict': ('P-C3-EQ ESTABLISHED: cup parity within the '
                  'registered +-0.10 AUROC band (TOST) with the finger '
                  'contrast exceeding the band' if established else
                  'P-C3-EQ NOT-ESTABLISHED: no parity wording; the '
                  'executed AMBIGUOUS P-C3 record stands'),
      'established': established,
  }


def _fake(delta, jitter, n_seeds=8, rng=None):
  rng = rng or np.random.default_rng(5)
  rows = []
  for arm, off in (('task', delta), ('apt', 0.0)):
    for side in (0, 1):
      for seed in range(1, n_seeds + 1):
        rows.append(dict(arm=arm, side=side, seed=seed,
                         auroc=0.8 + off + rng.normal(0, jitter)))
  return rows


def selfcheck():
  fin_hi = _fake(0.5, 0.01)
  fin_low = _fake(0.05, 0.01)
  r = analyse(_fake(0.0, 0.01), fin_hi)
  assert r['established'], r
  r = analyse(_fake(0.2, 0.01), fin_hi)           # cup diff outside band
  assert not r['established'] and not r['cup_leg_pass'], r
  r = analyse(_fake(0.0, 0.30), fin_hi)           # CI too wide for TOST
  assert not r['cup_leg_pass'], r
  r = analyse(_fake(0.0, 0.01), fin_low)          # finger leg fails
  assert not r['established'] and r['cup_leg_pass'], r
  # boundary: point inside band but CI crossing the band edge
  r = analyse(_fake(0.08, 0.08), fin_hi)
  if not (-BAND < r['cup']['ci90'][0] and r['cup']['ci90'][1] < BAND):
    assert not r['cup_leg_pass'], r
  print('cup_parity_tost selfcheck PASS')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  sub.add_parser('selfcheck')
  ap.add_argument('--bundle')
  ap.add_argument('--out')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck()
    return
  assert args.bundle and args.out
  res = analyse(load_rows(args.bundle, 'cup'),
                load_rows(args.bundle, 'finger'))
  res['bundle'] = args.bundle
  res['prereg'] = 'PREREG_cup_parity_tost_20260810.md'
  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps(res, indent=1))


if __name__ == '__main__':
  main()
