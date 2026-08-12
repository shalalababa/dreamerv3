"""Frozen reader — TM2 legibility diagnostics (P-F1 premise check).

Registration: PREREG_tm2_diag_20260811.md (REVISED post-review: the
dv3 anchor is the EXECUTED finger-refit read's pooled AUROCs — pinned
constants below, no dv3 inputs consumed; the TM2 scalar is the
fold-centered pooled-OOF AUROC from probing/tm2_legibility_probe). ONE
execution:

  python -m analysis.tm2_diag_read \
      --tm2_glob '<dir>/tm2wm_finger_*q1s*_seed*.json' --output <dir>
Selfcheck: python -m analysis.tm2_diag_read --selfcheck
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np

MIN_PER_ARM = 6
N_FOLDS_EXPECT = 4
# EXECUTED dv3 anchor (artifacts/finger_refit_read_20260810/read.json:
# task.auroc.mean / apt.auroc.mean, pooled n=16 each, finger_v1
# probeset, witness_match 32/32). Value-aware: pinned, disclosed.
DV3_TASK_AUROC = 0.93477
DV3_APT_AUROC = 0.40107
DV3_GAP = DV3_TASK_AUROC - DV3_APT_AUROC          # 0.53370
DV3_SOURCE = 'artifacts/finger_refit_read_20260810/read.json'
TM2_RE = re.compile(r'tm2wm_finger_(?P<arm>aware|free)q1s(?P<side>[01])'
                    r'_seed(?P<seed>\d+)')


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def bca_two_sample(a, b, rng_seed=0, nboot=10_000):
  """House two-sample BCa 95% for mean(a) - mean(b) (dose_task form)."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  rng = np.random.default_rng(rng_seed)
  boots = np.array([
      a[rng.integers(0, len(a), len(a))].mean()
      - b[rng.integers(0, len(b), len(b))].mean() for _ in range(nboot)])
  theta = float(a.mean() - b.mean())
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (nboot + 1)), 1.0 - 1.0 / (nboot + 1))
  z0 = math.sqrt(2) * _erfinv(2 * prop - 1)
  jack = [np.delete(a, i).mean() - b.mean() for i in range(len(a))]
  jack += [a.mean() - np.delete(b, j).mean() for j in range(len(b))]
  jack = np.asarray(jack)
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha):
    z = math.sqrt(2) * _erfinv(2 * alpha - 1)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  return theta, (q(0.025), q(0.975))


def grade(gap_tm2, ci):
  if not math.isfinite(gap_tm2):
    return 'DEGENERATE: non-finite TM2 gap - instrument failure'
  if gap_tm2 < 0:
    if ci[1] < 0:
      return ('PREMISE-INVERTED: free MORE legible than aware, CI<0 '
              '(mechanism-interesting; powered wave re-scoped)')
    return ('PREMISE-CONSISTENT (gap<0 within noise counts toward '
            'compression; CI includes 0)')
  ratio = gap_tm2 / DV3_GAP
  if ratio < 0.5:
    return 'PREMISE-STRONG'
  if ratio < 1.0:
    return 'PREMISE-CONSISTENT'
  return ('PREMISE-VIOLATED: TM2 legibility gap not compressed - P-F1 '
          'must be re-registered or dropped before the powered wave '
          'freezes')


def load_tm2(pattern):
  rows, excluded = {}, []
  for p in sorted(glob.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = TM2_RE.search(d.get('wm_run', os.path.basename(p)))
    assert m, p
    assert d.get('instrument') == 'tm2_legibility_probe_v1', p
    arm = m.group('arm')
    if arm == 'free':
      assert float(d['reward_coef']) == 0.0 and \
          float(d['value_coef']) == 0.0, (p, 'free-arm audit mismatch')
    else:
      assert float(d['reward_coef']) > 0.0, (p, 'aware-arm audit mismatch')
    key = (arm, m.group('side'), int(m.group('seed')))
    assert key not in rows, ('duplicate fit', key)
    # review #17: fold-count witness; nan-auroc fits exclude-disclosed
    assert int(d.get('n_folds', 0)) == N_FOLDS_EXPECT, (p, 'fold count')
    assert len(d['per_fold']) == N_FOLDS_EXPECT, (p, 'per_fold length')
    v = float(d['auroc'])
    if math.isnan(v):
      excluded.append((f'{arm}_s{m.group("side")}_seed{m.group("seed")}',
                       'degenerate labels (auroc nan)'))
      continue
    rows[key] = v
  return rows, excluded


def analyse(tm2_rows, excluded=()):
  aware = {k: v for k, v in tm2_rows.items() if k[0] == 'aware'}
  free = {k: v for k, v in tm2_rows.items() if k[0] == 'free'}
  # inventory gate re-checked AFTER exclusions (review #17); a failed
  # gate is the REGISTERED reported verdict, not a crash (rev-2 M2)
  for arm, d in (('aware', aware), ('free', free)):
    if len(d) < MIN_PER_ARM:
      return {'verdict': ('DIAGNOSTICS-BLOCKED: inventory gate - '
                          f'{arm} has {len(d)} < {MIN_PER_ARM} loadable '
                          'fits; powered-wave decision returns to the '
                          'user'),
              'n': {'aware': len(aware), 'free': len(free)},
              'excluded': list(excluded)}
    if {k[1] for k in d} != {'0', '1'}:
      return {'verdict': ('DIAGNOSTICS-BLOCKED: both-sides gate - '
                          f'{arm} arm is single-sided; powered-wave '
                          'decision returns to the user'),
              'n': {'aware': len(aware), 'free': len(free)},
              'excluded': list(excluded)}

  # review #20: side-balanced gap = mean of within-side gaps
  side_gaps = {}
  for side in '01':
    aw = [v for k, v in aware.items() if k[1] == side]
    fr = [v for k, v in free.items() if k[1] == side]
    side_gaps[side] = float(np.mean(aw) - np.mean(fr))
  gap_tm2 = float(np.mean(list(side_gaps.values())))
  # CI: house two-sample BCa on the unbalanced pooled gap (recorded
  # alongside; the registered POINT rule uses the side-balanced gap;
  # cluster = fit; seed-variance CI, n_data = 2 sides - disclosed)
  _, ci = bca_two_sample(list(aware.values()), list(free.values()))
  verdict = grade(gap_tm2, ci)
  return {
      'verdict': verdict,
      'gap_tm2_side_balanced': gap_tm2,
      'gap_tm2_pooled_ci': list(ci),
      'ratio': (gap_tm2 / DV3_GAP if math.isfinite(gap_tm2) else None),
      'side_gaps': side_gaps,
      'dv3_anchor': {'task_auroc': DV3_TASK_AUROC,
                     'apt_auroc': DV3_APT_AUROC, 'gap': DV3_GAP,
                     'source': DV3_SOURCE,
                     'note': 'EXECUTED read values - pinned, '
                             'value-aware disclosed'},
      'arm_means': {'aware': float(np.mean(list(aware.values()))),
                    'free': float(np.mean(list(free.values())))},
      'n': {'aware': len(aware), 'free': len(free)},
      'excluded': list(excluded),
      'panels': {f'{k[0]}_s{k[1]}_seed{k[2]}': v
                 for k, v in sorted(tm2_rows.items())},
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--tm2_glob')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.tm2_glob and args.output
  rows, excluded = load_tm2(args.tm2_glob)
  res = analyse(rows, excluded)
  res['prereg'] = 'PREREG_tm2_diag_20260811.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({k: res[k] for k in
                    ('verdict', 'ratio', 'gap_tm2_side_balanced',
                     'arm_means', 'excluded')}, indent=1))


def _mk(aware, free, jitter=0.02, rng_seed=0, seeds=range(1, 5)):
  rng = np.random.default_rng(rng_seed)
  tm2 = {}
  for arm, mu in (('aware', aware), ('free', free)):
    for side in '01':
      for seed in seeds:
        tm2[(arm, side, seed)] = mu + jitter * rng.standard_normal()
  return tm2


def selfcheck():
  import tempfile
  # gap ~0.08 vs pinned 0.5337 -> ratio ~.15 STRONG
  r = analyse(_mk(aware=0.78, free=0.70))
  assert r['verdict'] == 'PREMISE-STRONG', (r['verdict'], r['ratio'])
  # gap ~0.40 -> ratio ~.75 CONSISTENT
  r = analyse(_mk(aware=0.90, free=0.50))
  assert r['verdict'] == 'PREMISE-CONSISTENT', (r['verdict'], r['ratio'])
  # gap ~0.56 >= pinned -> VIOLATED
  r = analyse(_mk(aware=0.95, free=0.39))
  assert r['verdict'].startswith('PREMISE-VIOLATED'), r['verdict']
  # clearly inverted (CI<0)
  r = analyse(_mk(aware=0.50, free=0.80))
  assert r['verdict'].startswith('PREMISE-INVERTED'), r['verdict']
  # tiny negative gap within noise -> CONSISTENT (compression side)
  r = analyse(_mk(aware=0.70, free=0.73, jitter=0.04, rng_seed=2))
  assert r['gap_tm2_side_balanced'] < 0 < r['gap_tm2_pooled_ci'][1]
  assert r['verdict'].startswith('PREMISE-CONSISTENT'), \
      (r['verdict'], r['gap_tm2_side_balanced'])
  # side-balance: verify the point stat is the mean of within-side gaps
  tm2 = _mk(aware=0.8, free=0.6, jitter=0.0)
  tm2[('aware', '0', 9)] = 0.9      # unbalanced extra fit on side 0
  r = analyse(tm2)
  expect = np.mean([np.mean([0.8] * 4 + [0.9]) - 0.6, 0.8 - 0.6])
  assert abs(r['gap_tm2_side_balanced'] - expect) < 1e-12
  # inventory gate: too few free fits -> BLOCKED verdict (rev-2 M2)
  small = {k: v for k, v in _mk(0.8, 0.6).items()
           if not (k[0] == 'free' and k[2] > 2)}
  r = analyse(small)
  assert r['verdict'].startswith('DIAGNOSTICS-BLOCKED: inventory'), \
      r['verdict']
  # one-sided arm -> BLOCKED via the BOTH-SIDES leg specifically
  # (rev-2 M3: seeds 1-8 so the count gate cannot mask this leg)
  onesided = {k: v for k, v in _mk(0.8, 0.6,
                                   seeds=range(1, 9)).items()
              if not (k[0] == 'free' and k[1] == '1')}
  r = analyse(onesided)
  assert r['verdict'].startswith('DIAGNOSTICS-BLOCKED: both-sides'), \
      r['verdict']
  # loader round-trip (review #1 lesson): real files through load_tm2,
  # incl. audit gates, duplicate refusal, nan exclusion + gate re-check
  with tempfile.TemporaryDirectory() as td:
    def write(name, arm, auroc, rc=None):
      json.dump(dict(wm_run=name, instrument='tm2_legibility_probe_v1',
                     auroc=auroc, r2=0.1, per_fold=[0.1] * 4, n_folds=4,
                     reward_coef=(rc if rc is not None else
                                  (0.0 if arm == 'free' else 1.0)),
                     value_coef=0.0 if arm == 'free' else 1.0),
                open(os.path.join(td, name + '.json'), 'w'))
    for arm, mu in (('aware', 0.8), ('free', 0.7)):
      for side in '01':
        for seed in range(1, 5):
          write(f'tm2wm_finger_{arm}q1s{side}_seed{seed}', arm, mu)
    rows, exc = load_tm2(os.path.join(td, '*.json'))
    assert len(rows) == 16 and not exc
    r = analyse(rows, exc)
    assert r['verdict'] == 'PREMISE-STRONG', r['verdict']
    # nan-auroc fit -> excluded-disclosed, gate re-checked
    write('tm2wm_finger_freeq1s0_seed9', 'free', float('nan'))
    rows, exc = load_tm2(os.path.join(td, '*.json'))
    assert len(rows) == 16 and len(exc) == 1, (len(rows), exc)
    # free fit with nonzero reward_coef -> audit refusal
    write('tm2wm_finger_freeq1s1_seed9', 'free', 0.7, rc=0.5)
    try:
      load_tm2(os.path.join(td, '*.json'))
      raise SystemExit('expected free-arm audit refusal')
    except AssertionError:
      pass
  print('tm2_diag_read selfcheck PASS (strong/consistent/violated/'
        'inverted/noise-negative branches; side-balance verified; '
        'inventory+both-sides gates; loader round-trip w/ audit, '
        'nan-exclusion, duplicate-name guards)')


if __name__ == '__main__':
  main()
