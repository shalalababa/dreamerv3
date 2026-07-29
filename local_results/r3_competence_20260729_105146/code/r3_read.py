"""Frozen read for R3 — consumer-competence factorial (Paper 2).

Registered in prereg/PREREG_r3_competence_20260725.md and committed
BEFORE any R3 pilot or label exists. Labels come from the corrected
labeler (d1fix_20260724) with --oracle_all; estimands per labeled state:

  opportunity = max_m G[m] - G[m_now]      (from g_all, g_now)
  achieved    = G[m_real] - G[m_now]
  gap         = opportunity - achieved

Files: <labels_dir>/r3_{dom}_{dose}_seed{n}_{mat}.npz
  dom in {cup, finger, reacher}, dose in {e1, e4}, mat in {early, late}.
Cluster = training run (dom, dose, seed); both maturities of a run are
the same cluster. Registered grid: seeds 31-38 for cup+finger x e1+e4;
reacher cohort is all-or-nothing (conditional GO domain).

Amendment 1 (prereg/PREREG_r3_amend1_20260729.md, committed before the
one read): the frozen seed constant contradicted the registered grid
(1-8 vs the prereg's 31-38) and tripped check_grid before any
computation; seed literals corrected and value-blind integrity guards
added (unregistered-seed, finiteness, m_real bounds, n_states). No
decision rule, threshold, estimand, or bootstrap detail changed.

Primaries (run-clustered percentile bootstrap, B=10K, rng 0):
  P-R3a  pooled mean opportunity CI entirely > 0.2   (existence)
  P-R3b  pooled mean gap CI entirely > 0             (competence failure)
  P-R3c  per-run paired (late - early) achieved, pooled CI entirely > 0
         (maturity buys competence)

Floor policy: a (run, maturity) cell is a FLOOR cell iff every G in
g_all, g_now is 0. Floor cells are INCLUDED in all primaries
(conservative: they shrink opportunity and gap toward 0) and their
fraction is reported per domain; a domain with > 50% floor cells is
flagged floor-limited (disclosure only; primaries are pooled).

Usage:
  python -m analysis.r3_read --labels <dir> --output <dir>
  python -m analysis.r3_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
OPP_THRESHOLD = 0.2
FLOOR_LIMIT = 0.5
LABELER_VERSION = 'd1fix_20260724'
FILE_RE = re.compile(
    r'^r3_(cup|finger|reacher)_(e1|e4)_seed(\d+)_(early|late)\.npz$')
CORE_DOMAINS = ('cup', 'finger')
OPTIONAL_DOMAINS = ('reacher',)
EXPECT_SEEDS = tuple(range(31, 39))
DOSES = ('e1', 'e4')
MATS = ('early', 'late')


def cell_stats(g_all, g_now, m_real):
  g_all = np.asarray(g_all, float)
  g_now = np.asarray(g_now, float)
  m_real = np.asarray(m_real, int)
  opp = np.nanmax(g_all, 1) - g_now
  ach = g_all[np.arange(len(m_real)), m_real] - g_now
  floor = bool(np.all(g_all == 0) and np.all(g_now == 0))
  return dict(opp=float(np.nanmean(opp)), ach=float(np.nanmean(ach)),
              gap=float(np.nanmean(opp - ach)), floor=floor,
              n_states=int(len(g_now)))


def validate_arrays(name, g_all, g_now, m_real):
  """Amendment-1 guards: value-blind integrity checks per label file."""
  assert g_all.ndim == 2 and g_all.shape[0] == len(g_now) == len(m_real), (
      f'{name}: inconsistent array shapes '
      f'{g_all.shape}/{g_now.shape}/{m_real.shape}')
  assert np.isfinite(g_all).all() and np.isfinite(g_now).all(), (
      f'{name}: non-finite G values (oracle_all pass incomplete?)')
  assert m_real.min() >= 0 and m_real.max() < g_all.shape[1], (
      f'{name}: m_real outside candidate range [0,{g_all.shape[1]})')


def load_labels(labels_dir):
  cells = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = FILE_RE.match(name)
    if not m:
      continue
    dom, dose, seed, mat = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    z = np.load(path, allow_pickle=True)
    meta = str(z['meta'])
    assert LABELER_VERSION in meta, (
        f'{name}: labeler_version is not {LABELER_VERSION}')
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    validate_arrays(name, g_all, g_now, m_real)
    cells[(dom, dose, seed, mat)] = cell_stats(g_all, g_now, m_real)
  return cells


def check_grid(cells):
  bad = sorted({k[2] for k in cells} - set(EXPECT_SEEDS))
  assert not bad, f'unregistered seeds present: {bad}'
  n_states = sorted({v['n_states'] for v in cells.values()})
  assert n_states == [200], (
      f'n_states differs from the registered 200: {n_states}')
  doms = sorted({k[0] for k in cells})
  for dom in CORE_DOMAINS:
    assert dom in doms, f'core domain {dom} entirely absent'
  for dom in doms:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        have = [(dom, dose, seed, mat) in cells for mat in MATS]
        assert all(have) or not any(have), (
            f'{dom}/{dose}/seed{seed}: one maturity present without the '
            'other (all-or-nothing per run)')
        if dom in CORE_DOMAINS:
          assert all(have), f'core cell missing: {dom}/{dose}/seed{seed}'
  for dom in OPTIONAL_DOMAINS:
    n = len([k for k in cells if k[0] == dom])
    full = len(DOSES) * len(EXPECT_SEEDS) * len(MATS)
    assert n in (0, full), (
        f'optional domain {dom} partially present ({n}/{full} cells); '
        'the cohort is all-or-nothing')
  return doms


def _cluster_boot(per_cluster_vals, b=B_BOOT, seed=RNG_SEED):
  """per_cluster_vals: {cluster: [values]}; bootstrap of pooled mean."""
  keys = sorted(per_cluster_vals)
  rng = np.random.default_rng(seed)
  point = float(np.mean([v for k in keys for v in per_cluster_vals[k]]))
  n = len(keys)
  stats = np.empty(b)
  for i in range(b):
    pick = rng.integers(0, n, n)
    stats[i] = np.mean(
        [v for j in pick for v in per_cluster_vals[keys[j]]])
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=point, ci=[float(lo), float(hi)], n_clusters=n)


def analyze(cells):
  doms = check_grid(cells)
  runs = sorted({k[:3] for k in cells})

  def pooled(stat):
    return {r: [cells[r + (mat,)][stat] for mat in MATS] for r in runs}

  a = _cluster_boot(pooled('opp'))
  b = _cluster_boot(pooled('gap'))
  paired = {r: [cells[r + ('late',)]['ach'] - cells[r + ('early',)]['ach']]
            for r in runs}
  c = _cluster_boot(paired)
  fires = dict(
      p_r3a=a['ci'][0] > OPP_THRESHOLD,
      p_r3b=b['ci'][0] > 0,
      p_r3c=c['ci'][0] > 0)

  floors = {}
  for dom in doms:
    cf = [cells[k]['floor'] for k in cells if k[0] == dom]
    floors[dom] = dict(frac=float(np.mean(cf)), n_cells=len(cf),
                       floor_limited=bool(np.mean(cf) > FLOOR_LIMIT))

  def sub_boot(stat, sel):
    vals = {r: [cells[r + (mat,)][stat] for mat in MATS
                if r + (mat,) in cells]
            for r in runs if sel(r)}
    vals = {k: v for k, v in vals.items() if v}
    return _cluster_boot(vals) if vals else None

  secondary = dict(
      dose_effect_opp={dose: sub_boot('opp', lambda r, d=dose: r[1] == d)
                       for dose in DOSES},
      maturity_opp={mat: _cluster_boot(
          {r: [cells[r + (mat,)]['opp']] for r in runs}) for mat in MATS},
      per_domain={dom: dict(
          opp=sub_boot('opp', lambda r, d=dom: r[0] == d),
          ach=sub_boot('ach', lambda r, d=dom: r[0] == d),
          gap=sub_boot('gap', lambda r, d=dom: r[0] == d)) for dom in doms},
      floors=floors)

  if fires['p_r3a'] and fires['p_r3b']:
    verdict = ('CONSUMER-COMPETENCE FAILURE: opportunity exists (P-R3a) '
               'and is not harvested (P-R3b). '
               + ('Maturity buys competence (P-R3c fires) — achieved value '
                  'rises early→late.' if fires['p_r3c'] else
                  'P-R3c does not fire — competence does not detectably '
                  'improve with maturity at this range.'))
  elif not fires['p_r3a']:
    verdict = ('NO DETECTABLE OPPORTUNITY (P-R3a fails): the null is an '
               'information-content fact at these cells; competence claims '
               'unlicensed (registered branch).')
  else:
    verdict = ('OPPORTUNITY HARVESTED (P-R3a fires, P-R3b does not): '
               'achieved value tracks opportunity — the consumer is '
               'competent at these cells (registered branch).')
  return dict(primaries=dict(p_r3a_opportunity=a, p_r3b_gap=b,
                             p_r3c_maturity_ach=c),
              thresholds=dict(opp=OPP_THRESHOLD, floor_limit=FLOOR_LIMIT),
              fires=fires, secondary=secondary,
              domains=doms, n_runs=len(runs), verdict=verdict)


def read(args):
  cells = load_labels(args.labels)
  assert cells, f'no r3_* labels under {args.labels}'
  res = analyze(cells)
  res['labels_dir'] = os.path.abspath(args.labels)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'r3.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  for name, blk in res['primaries'].items():
    print(f"{name}: {blk['point']:+.3f} [{blk['ci'][0]:+.3f},"
          f"{blk['ci'][1]:+.3f}] n={blk['n_clusters']}")
  print('fires:', res['fires'])
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _synth_cells(opp_fn, ach_fn, doms=CORE_DOMAINS, floor_runs=()):
  rng = np.random.default_rng(7)
  cells = {}
  for dom in doms:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for mat in MATS:
          key = (dom, dose, seed, mat)
          if (dom, dose, seed) in floor_runs:
            cells[key] = dict(opp=0.0, ach=0.0, gap=0.0, floor=True,
                              n_states=200)
            continue
          o = opp_fn(dom, dose, seed, mat) + 0.05 * rng.standard_normal()
          ach = ach_fn(dom, dose, seed, mat) + 0.05 * rng.standard_normal()
          cells[key] = dict(opp=o, ach=ach, gap=o - ach, floor=False,
                            n_states=200)
  return cells


def selfcheck(args):
  # Branch 1: opportunity + failure + maturity effect.
  cells = _synth_cells(lambda d, e, s, m: 1.0,
                       lambda d, e, s, m: 0.1 + (0.4 if m == 'late' else 0))
  res = analyze(cells)
  assert res['fires'] == dict(p_r3a=True, p_r3b=True, p_r3c=True), res['fires']
  assert res['verdict'].startswith('CONSUMER-COMPETENCE FAILURE'), (
      res['verdict'])

  # Branch 2: no opportunity anywhere.
  cells = _synth_cells(lambda d, e, s, m: 0.02, lambda d, e, s, m: 0.0)
  res = analyze(cells)
  assert not res['fires']['p_r3a'] and res['verdict'].startswith(
      'NO DETECTABLE OPPORTUNITY'), res['verdict']

  # Branch 3: opportunity fully harvested.
  cells = _synth_cells(lambda d, e, s, m: 1.0, lambda d, e, s, m: 1.0)
  res = analyze(cells)
  assert res['fires']['p_r3a'] and not res['fires']['p_r3b']
  assert res['verdict'].startswith('OPPORTUNITY HARVESTED'), res['verdict']

  # Floor accounting: floors damp the primaries and are counted.
  floor_runs = {('finger', 'e1', s) for s in (31, 32, 33, 34, 35)} | {
      ('finger', 'e4', s) for s in (31, 32, 33, 34)}
  cells = _synth_cells(lambda d, e, s, m: 1.0, lambda d, e, s, m: 0.1,
                       floor_runs=floor_runs)
  res = analyze(cells)
  assert res['secondary']['floors']['finger']['floor_limited']
  assert not res['secondary']['floors']['cup']['floor_limited']

  # Optional reacher cohort: full cohort accepted, partial trips.
  cells = _synth_cells(lambda d, e, s, m: 1.0, lambda d, e, s, m: 0.1,
                       doms=CORE_DOMAINS + OPTIONAL_DOMAINS)
  res = analyze(cells)
  assert 'reacher' in res['domains'] and res['n_runs'] == 48
  bad = dict(cells)
  del bad[('reacher', 'e1', 33, 'late')]
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: partial reacher cohort not caught')
  except AssertionError:
    pass

  # Missing core cell and lone-maturity run must trip.
  cells = _synth_cells(lambda d, e, s, m: 1.0, lambda d, e, s, m: 0.1)
  bad = dict(cells)
  del bad[('cup', 'e1', 32, 'early')], bad[('cup', 'e1', 32, 'late')]
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: missing core run not caught')
  except AssertionError:
    pass
  bad = dict(cells)
  del bad[('cup', 'e1', 32, 'early')]
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: lone maturity not caught')
  except AssertionError:
    pass

  # Amendment-1 guards. Unregistered seed (even a complete run) trips.
  stray = dict(cells)
  for mat in MATS:
    stray[('cup', 'e1', 99, mat)] = dict(opp=1.0, ach=0.1, gap=0.9,
                                         floor=False, n_states=200)
  try:
    analyze(stray)
    raise SystemExit('selfcheck FAIL: unregistered seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e
  # Off-registration n_states trips.
  short = dict(cells)
  short[('cup', 'e1', 32, 'early')] = dict(
      short[('cup', 'e1', 32, 'early')], n_states=120)
  try:
    analyze(short)
    raise SystemExit('selfcheck FAIL: off-registration n_states not caught')
  except AssertionError as e:
    assert 'n_states' in str(e), e
  # Array-level guards: non-finite G, out-of-range m_real, shape mismatch.
  ok_g = np.ones((4, 3))
  ok_now = np.ones(4)
  ok_m = np.zeros(4, int)
  validate_arrays('ok', ok_g, ok_now, ok_m)
  for bad_args, tag in [
      ((np.where(np.eye(4, 3) > 0, np.nan, 1.0), ok_now, ok_m), 'nan g_all'),
      ((ok_g, np.array([1.0, np.inf, 1.0, 1.0]), ok_m), 'inf g_now'),
      ((ok_g, ok_now, np.array([0, -1, 0, 0])), 'negative m_real'),
      ((ok_g, ok_now, np.array([0, 3, 0, 0])), 'm_real past candidates'),
      ((ok_g, ok_now[:3], ok_m), 'shape mismatch'),
  ]:
    try:
      validate_arrays(tag, *bad_args)
      raise SystemExit(f'selfcheck FAIL: {tag} not caught')
    except AssertionError:
      pass

  # cell_stats floor detection + estimand identities on a hand case.
  g_all = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
  g_now = np.array([0.5, 1.0])
  m_real = np.array([2, 0])
  st = cell_stats(g_all, g_now, m_real)
  assert abs(st['opp'] - np.mean([1.5, 0.0])) < 1e-12
  assert abs(st['ach'] - np.mean([0.5, 0.0])) < 1e-12
  assert abs(st['gap'] - (st['opp'] - st['ach'])) < 1e-12 and not st['floor']
  assert cell_stats(np.zeros((2, 3)), np.zeros(2), np.array([0, 1]))['floor']

  print('selfcheck PASS: three verdict branches, floor accounting + '
        'domain flag, optional-cohort all-or-nothing, core completeness '
        'and lone-maturity trips, estimand identities, amendment-1 guards '
        '(unregistered seed, n_states, finiteness, m_real bounds, shapes)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='analysis_out/r3')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
