"""Frozen read for the R3 reacher third-domain replication (Paper 2).

Registered in prereg/PREREG_r3_reacher_20260729.md and committed BEFORE
any reacher R3 run exists. A NEW single-domain registration: the
original R3 read (artifacts/r3_competence_20260729/) executed with the
reacher conditional-GO cohort absent, so reacher is adjudicated here as
a fresh replication — the cup/finger primaries are NEVER recomputed.

Grid: reacher x {e1, e4} x seeds 31-38 x {early, late} = 32 cells,
16 run clusters. Estimands, dials, labeler (d1fix_20260724), floor
policy, and bootstrap machinery are imported from the frozen
analysis/r3_read.py (Amendment-1 version) unchanged.

Registered gates and rules:
  SMOKE GATE   the four reacher smoke files (e1/e4 x early/late) must
               exist in the labels dir — machine-checked here.
  FLOOR GATE   floor fraction > 50% => FLOOR-LIMITED: inconclusive,
               no adjudication of the primaries (sparse-task floors
               are an information-free outcome, registered as such).
  P-RRa  pooled opportunity CI entirely > 0.2   (same existence bar)
  P-RRb  pooled gap CI entirely > 0
  P-RRc  per-run paired (late - early) achieved, pooled CI > 0
Verdicts: REPLICATES (a+b fire; c scopes maturity) / NO-OPPORTUNITY
(a fails) / HARVESTED (a fires, b fails) / FLOOR-LIMITED.

Usage:
  python -m analysis.r3_reacher_read --labels <dir> --output <dir>
  python -m analysis.r3_reacher_read --selfcheck
"""

import argparse
import glob
import json
import os

import numpy as np

from analysis.r3_read import (
    B_BOOT, RNG_SEED, OPP_THRESHOLD, FLOOR_LIMIT, LABELER_VERSION,
    FILE_RE, EXPECT_SEEDS, DOSES, MATS, cell_stats, validate_arrays,
    _cluster_boot)

DOM = 'reacher'
SMOKE_KINDS = tuple((dose, mat) for dose in DOSES for mat in MATS)

# Pin the shared-frozen-reader import surface: a future amendment to
# r3_read must not silently change this reader's registered grid.
assert EXPECT_SEEDS == tuple(range(31, 39)), EXPECT_SEEDS


def check_smoke(labels_dir):
  missing = []
  for dose, mat in SMOKE_KINDS:
    path = os.path.join(labels_dir, f'r3_{DOM}_{dose}_seed31_{mat}_smoke.npz')
    if not os.path.exists(path):
      missing.append(os.path.basename(path))
      continue
    z = np.load(path, allow_pickle=True)
    assert LABELER_VERSION in str(z['meta']), (
        f'{os.path.basename(path)}: smoke labeler_version is not '
        f'{LABELER_VERSION}')
  assert not missing, (
      'registered reacher smoke gate not satisfied; missing: '
      + ', '.join(missing))


def load_labels(labels_dir):
  cells = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = FILE_RE.match(name)
    if not m or m.group(1) != DOM:
      continue
    dose, seed, mat = m.group(2), int(m.group(3)), m.group(4)
    z = np.load(path, allow_pickle=True)
    meta = str(z['meta'])
    assert LABELER_VERSION in meta, (
        f'{name}: labeler_version is not {LABELER_VERSION}')
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    validate_arrays(name, g_all, g_now, m_real)
    key = (dose, seed, mat)
    assert key not in cells, f'duplicate cell from {name}'
    cells[key] = cell_stats(g_all, g_now, m_real)
  return cells


def check_grid(cells):
  assert cells, 'no reacher label files found'
  bad = sorted({k[1] for k in cells} - set(EXPECT_SEEDS))
  assert not bad, f'unregistered seeds present: {bad}'
  n_states = sorted({v['n_states'] for v in cells.values()})
  assert n_states == [200], (
      f'n_states differs from the registered 200: {n_states}')
  for dose in DOSES:
    for seed in EXPECT_SEEDS:
      for mat in MATS:
        assert (dose, seed, mat) in cells, (
            f'cohort incomplete: reacher/{dose}/seed{seed}/{mat} missing '
            '(all-or-nothing)')
  assert len(cells) == len(DOSES) * len(EXPECT_SEEDS) * len(MATS), (
      'unexpected extra cells')


def analyze(cells):
  check_grid(cells)
  runs = sorted({k[:2] for k in cells})

  floor_frac = float(np.mean([cells[k]['floor'] for k in cells]))
  floors = dict(frac=floor_frac, n_cells=len(cells),
                floor_limited=bool(floor_frac > FLOOR_LIMIT))

  def pooled(stat):
    return {r: [cells[r + (mat,)][stat] for mat in MATS] for r in runs}

  a = _cluster_boot(pooled('opp'))
  b = _cluster_boot(pooled('gap'))
  paired = {r: [cells[r + ('late',)]['ach'] - cells[r + ('early',)]['ach']]
            for r in runs}
  c = _cluster_boot(paired)
  fires = dict(
      p_rra=a['ci'][0] > OPP_THRESHOLD,
      p_rrb=b['ci'][0] > 0,
      p_rrc=c['ci'][0] > 0)

  secondary = dict(
      dose_opp={dose: _cluster_boot(
          {r: [cells[r + (mat,)]['opp'] for mat in MATS]
           for r in runs if r[0] == dose}) for dose in DOSES},
      maturity_opp={mat: _cluster_boot(
          {r: [cells[r + (mat,)]['opp']] for r in runs}) for mat in MATS},
      ach=_cluster_boot(pooled('ach')),
      floors=floors)

  if floors['floor_limited']:
    verdict = ('FLOOR-LIMITED: > 50% of reacher cells are all-zero-G '
               'floors — inconclusive by the registered gate; primaries '
               'reported but NOT adjudicated; no further reacher compute '
               'without a redesign.')
    fires = dict(p_rra=None, p_rrb=None, p_rrc=None)
  elif fires['p_rra'] and fires['p_rrb']:
    verdict = ('REPLICATES: the opportunity/competence decomposition '
               'holds in a third domain — opportunity exists (P-RRa) and '
               'is not harvested (P-RRb). '
               + ('Maturity buys competence here (P-RRc fires).'
                  if fires['p_rrc'] else
                  'P-RRc does not fire — maturity does not close the '
                  'gap in reacher either.'))
  elif not fires['p_rra']:
    verdict = ('NO-OPPORTUNITY: no detectable oracle opportunity in '
               'reacher at these cells — a registered domain-scope '
               'statement; the cup/finger decomposition stands unchanged.')
  else:
    verdict = ('HARVESTED: opportunity exists and the reacher consumer '
               'harvests it — a registered competence-success domain; '
               'the cup/finger competence failure becomes domain-scoped.')
  return dict(primaries=dict(p_rra_opportunity=a, p_rrb_gap=b,
                             p_rrc_maturity_ach=c),
              thresholds=dict(opp=OPP_THRESHOLD, floor_limit=FLOOR_LIMIT),
              fires=fires, secondary=secondary,
              n_runs=len(runs), verdict=verdict)


def read(args):
  check_smoke(args.labels)
  cells = load_labels(args.labels)
  res = analyze(cells)
  res['labels_dir'] = os.path.abspath(args.labels)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'r3_reacher.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  for name, blk in res['primaries'].items():
    print(f"{name}: {blk['point']:+.3f} [{blk['ci'][0]:+.3f},"
          f"{blk['ci'][1]:+.3f}] n={blk['n_clusters']}")
  print('fires:', res['fires'])
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# selfcheck (deterministic constant fixtures; no RNG in the data)
# --------------------------------------------------------------------------

def _synth(opp, ach_early, ach_late, floor_runs=()):
  cells = {}
  for dose in DOSES:
    for seed in EXPECT_SEEDS:
      for mat in MATS:
        if (dose, seed) in floor_runs:
          cells[(dose, seed, mat)] = dict(opp=0.0, ach=0.0, gap=0.0,
                                          floor=True, n_states=200)
          continue
        ach = ach_late if mat == 'late' else ach_early
        cells[(dose, seed, mat)] = dict(opp=opp, ach=ach, gap=opp - ach,
                                        floor=False, n_states=200)
  return cells


def selfcheck(args):
  # REPLICATES + maturity null.
  res = analyze(_synth(1.0, 0.1, 0.1))
  assert res['fires'] == dict(p_rra=True, p_rrb=True, p_rrc=False)
  assert res['verdict'].startswith('REPLICATES'), res['verdict']
  # REPLICATES + maturity fires.
  res = analyze(_synth(1.0, 0.1, 0.5))
  assert res['fires']['p_rrc'] and 'Maturity buys' in res['verdict']
  # NO-OPPORTUNITY.
  res = analyze(_synth(0.05, 0.0, 0.0))
  assert res['verdict'].startswith('NO-OPPORTUNITY'), res['verdict']
  # HARVESTED.
  res = analyze(_synth(1.0, 1.0, 1.0))
  assert res['verdict'].startswith('HARVESTED'), res['verdict']
  # FLOOR-LIMITED: 9/16 runs floored (> 50% of cells), fires nulled.
  floor_runs = {('e1', s) for s in EXPECT_SEEDS[:5]} | {
      ('e4', s) for s in EXPECT_SEEDS[:4]}
  res = analyze(_synth(1.0, 0.1, 0.1, floor_runs=floor_runs))
  assert res['verdict'].startswith('FLOOR-LIMITED')
  assert res['fires'] == dict(p_rra=None, p_rrb=None, p_rrc=None)
  # Partial cohort trips.
  cells = _synth(1.0, 0.1, 0.1)
  del cells[('e1', EXPECT_SEEDS[0], 'late')]
  try:
    analyze(cells)
    raise SystemExit('selfcheck FAIL: partial cohort not caught')
  except AssertionError:
    pass
  # Stray seed trips.
  cells = _synth(1.0, 0.1, 0.1)
  for mat in MATS:
    cells[('e1', 99, mat)] = dict(opp=5.0, ach=0.0, gap=5.0, floor=False,
                                  n_states=200)
  try:
    analyze(cells)
    raise SystemExit('selfcheck FAIL: stray seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e
  # Off-registration n_states trips.
  cells = _synth(1.0, 0.1, 0.1)
  cells[('e1', EXPECT_SEEDS[0], 'early')] = dict(
      cells[('e1', EXPECT_SEEDS[0], 'early')], n_states=120)
  try:
    analyze(cells)
    raise SystemExit('selfcheck FAIL: n_states not caught')
  except AssertionError:
    pass
  # Smoke gate trips on an empty dir, and on wrong-labeler smoke files.
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    try:
      check_smoke(d)
      raise SystemExit('selfcheck FAIL: missing smoke not caught')
    except AssertionError as e:
      assert 'smoke gate' in str(e), e
    for dose in DOSES:
      for mat in MATS:
        np.savez(os.path.join(d, f'r3_{DOM}_{dose}_seed31_{mat}_smoke.npz'),
                 meta='{"labeler_version": "wrong"}')
    try:
      check_smoke(d)
      raise SystemExit('selfcheck FAIL: wrong-labeler smoke not caught')
    except AssertionError as e:
      assert 'labeler_version' in str(e), e
  # load_labels fixture leg (the file-level asserts must be reachable):
  # a valid file loads; a zero-padded seed duplicate trips; a wrong
  # labeler_version trips.
  def _write(d, name, version=LABELER_VERSION):
    np.savez(os.path.join(d, name),
             meta=f'{{"labeler_version": "{version}"}}',
             g_all=np.ones((200, 8), np.float32),
             g_now=np.ones(200), m_real=np.zeros(200, int))
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_reacher_e1_seed31_early.npz')
    got = load_labels(d)
    assert list(got) == [('e1', 31, 'early')] and got[('e1', 31, 'early')][
        'n_states'] == 200
    _write(d, 'r3_reacher_e1_seed031_early.npz')
    try:
      load_labels(d)
      raise SystemExit('selfcheck FAIL: duplicate cell not caught')
    except AssertionError as e:
      assert 'duplicate cell' in str(e), e
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_reacher_e1_seed31_early.npz', version='wrong')
    try:
      load_labels(d)
      raise SystemExit('selfcheck FAIL: wrong labeler_version not caught')
    except AssertionError as e:
      assert 'labeler_version' in str(e), e
  print('selfcheck PASS: four verdict branches incl. floor-limited gate, '
        'cohort/stray-seed/n_states/duplicate trips, smoke-gate '
        'missing + wrong-labeler trips')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='analysis_out/r3_reacher')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if not args.selfcheck and not args.labels:
    ap.error('--labels required (or --selfcheck)')
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
