"""Frozen read for the R3 candidate-count doubling test (Paper 2).

Registered in prereg/PREREG_r3_doubling_20260729.md and committed BEFORE
any M=16 label pass exists. Closes the residue of editorial objection 6
(continuous-action candidate-set approximation): theory T4 (addendum
Prop 'biassign') says the M-candidate opportunity is a conservative,
monotonically improving lower estimate — this test measures the 8->16
increment on the SAME checkpoints.

Design facts frozen here:
- M=16 passes relabel the 32 LATE cells (cup/finger x e1/e4 x seeds
  31-38) with --actions 16, all other dials identical to R3.
- Candidate sampling shares the agent RNG stream, so base trajectories
  DIVERGE from the M=8 runs after the first labeled state: state sets
  are NOT paired. All comparisons are CELL-level (same checkpoint, same
  state-sampling process), paired per run cell; per-state pairing is
  never computed.
- M=8 side = the ORIGINAL committed R3 late labels (never recomputed
  beyond cell_stats; the R3 primaries are untouched).

Registered rules (cluster = run cell, B=10K, default_rng(0)):
  SMOKE GATE  r3dbl_cup_e4_seed31_late_m16_smoke.npz must exist
              (--actions 16 has never run; machine-checked here).
  P-DBLa (monotone consistency): pooled paired [opp16 - opp8] CI
          entirely < 0 => ANOMALY (contradicts T4; audit before any
          use). Otherwise consistent.
  P-DBLb (materiality, adjudicated only on the consistent branch):
          CI upper bound < 0.6465 (= 50% of the committed R3 pooled
          gap point 1.29304688, truncated to 4 dp — truncation
          direction conservative) => TRUNCATION IMMATERIAL; else
          MATERIAL TRUNCATION DISCLOSED (the conservative direction is
          unaffected either way — opportunity remains a lower bound).
Secondaries (descriptive): gap16 pooled CI (re-fire check at M=16),
per-domain increments, [ach16 - ach8] (expected ~ 0).

Usage:
  python -m analysis.r3_doubling_read --labels8 <dir> --labels16 <dir> \
      --output <dir>
  python -m analysis.r3_doubling_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from analysis.r3_read import (
    LABELER_VERSION, EXPECT_SEEDS, DOSES, cell_stats, validate_arrays,
    _cluster_boot)

CORE_DOMAINS = ('cup', 'finger')
M8, M16 = 8, 16
# 50% of the committed R3 pooled gap: r3.json p_r3b_gap.point =
# 1.29304688 -> 0.64652344, truncated to 4 dp. Truncation direction is
# conservative (a SMALLER threshold makes IMMATERIAL harder to declare).
MATERIALITY = 0.6465

# Pin the shared-frozen-reader import surface (see r3_reacher_read).
assert EXPECT_SEEDS == tuple(range(31, 39)), EXPECT_SEEDS

# Registered dial identity, enforced from each file's meta on BOTH
# sides — a mis-dialed manual retry of one pass must trip, not bias.
DIALS = dict(states=200, horizon=100, label_every=25, seed=0,
             rollouts=16, ref_stride=5, mass_scale=1.0,
             behavior_checkpoint='')
FILE8_RE = re.compile(r'^r3_(cup|finger)_(e1|e4)_seed(\d+)_late\.npz$')
FILE16_RE = re.compile(
    r'^r3dbl_(cup|finger)_(e1|e4)_seed(\d+)_late_m16\.npz$')
SMOKE16 = 'r3dbl_cup_e4_seed31_late_m16_smoke.npz'


def check_smoke(labels16_dir):
  path = os.path.join(labels16_dir, SMOKE16)
  assert os.path.exists(path), (
      f'registered M=16 smoke gate not satisfied: {SMOKE16} missing')


def _load(labels_dir, file_re, m_expected):
  cells = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = file_re.match(name)
    if not m:
      continue
    dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
    z = np.load(path, allow_pickle=True)
    meta = str(z['meta'])
    assert LABELER_VERSION in meta, (
        f'{name}: labeler_version is not {LABELER_VERSION}')
    md = json.loads(meta)
    for k, v in DIALS.items():
      assert md.get(k) == v, (
          f'{name}: dial {k}={md.get(k)!r} != registered {v!r}')
    assert md.get('actions') == m_expected, (
        f"{name}: meta actions={md.get('actions')} != expected {m_expected}")
    assert md.get('train_seed') == seed, (
        f"{name}: meta train_seed={md.get('train_seed')} != filename {seed}")
    assert str(md.get('checkpoint', '')).endswith(
        f'r3_{dom}_{dose}_seed{seed}/ckpt'), (
        f"{name}: checkpoint {md.get('checkpoint')!r} is not the late "
        f'ckpt of r3_{dom}_{dose}_seed{seed}')
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    assert g_all.shape[1] == m_expected, (
        f'{name}: candidate count {g_all.shape[1]} != expected {m_expected}')
    validate_arrays(name, g_all, g_now, m_real)
    st = cell_stats(g_all, g_now, m_real)
    assert st['n_states'] == 200, (
        f'{name}: n_states {st["n_states"]} != registered 200')
    cells[(dom, dose, seed)] = st
  return cells


def check_grid(c8, c16):
  for tag, cells in (('M=8', c8), ('M=16', c16)):
    assert cells, f'no {tag} late label files found'
    bad = sorted({k[2] for k in cells} - set(EXPECT_SEEDS))
    assert not bad, f'{tag}: unregistered seeds present: {bad}'
    for dom in CORE_DOMAINS:
      for dose in DOSES:
        for seed in EXPECT_SEEDS:
          assert (dom, dose, seed) in cells, (
              f'{tag}: cell missing: {dom}/{dose}/seed{seed} '
              '(all-or-nothing)')
    assert len(cells) == len(CORE_DOMAINS) * len(DOSES) * len(EXPECT_SEEDS)


def analyze(c8, c16):
  check_grid(c8, c16)
  runs = sorted(c8)

  inc_opp = _cluster_boot({r: [c16[r]['opp'] - c8[r]['opp']] for r in runs})
  anomaly = inc_opp['ci'][1] < 0
  # P-DBLb is adjudicated only on the consistent branch (consequence
  # map): a quarantined ANOMALY outcome must not carry immaterial=True.
  immaterial = (not anomaly) and inc_opp['ci'][1] < MATERIALITY
  gap16 = _cluster_boot({r: [c16[r]['gap']] for r in runs})
  secondary = dict(
      gap16=gap16,
      gap16_refires=bool(gap16['ci'][0] > 0),
      inc_ach=_cluster_boot(
          {r: [c16[r]['ach'] - c8[r]['ach']] for r in runs}),
      per_domain_inc={dom: _cluster_boot(
          {r: [c16[r]['opp'] - c8[r]['opp']]
           for r in runs if r[0] == dom}) for dom in CORE_DOMAINS},
      opp8=_cluster_boot({r: [c8[r]['opp']] for r in runs}),
      opp16=_cluster_boot({r: [c16[r]['opp']] for r in runs}))

  if anomaly:
    verdict = ('ANOMALY: [opp16 - opp8] CI entirely negative — '
               'contradicts the T4 monotone direction; audit the M=16 '
               'pass before ANY use of these labels.')
  elif immaterial:
    verdict = ('CONSISTENT + TRUNCATION IMMATERIAL: doubling the '
               'candidate set moves pooled opportunity by less than 50% '
               'of the R3 gap — the 8-candidate estimate is an adequate '
               'conservative lower bound (objection-6 residue closed).')
  else:
    verdict = ('CONSISTENT + MATERIAL TRUNCATION DISCLOSED: the M=16 '
               'increment can exceed half the R3 gap — opportunity '
               'remains a lower bound (conservative direction '
               'unaffected), but the paper quantifies the truncation.')
  return dict(primary=dict(inc_opp=inc_opp),
              flags=dict(anomaly=bool(anomaly),
                         immaterial=bool(immaterial)),
              thresholds=dict(materiality=MATERIALITY),
              secondary=secondary, n_runs=len(runs), verdict=verdict)


def read(args):
  check_smoke(args.labels16)
  c8 = _load(args.labels8, FILE8_RE, M8)
  c16 = _load(args.labels16, FILE16_RE, M16)
  res = analyze(c8, c16)
  res['labels8'] = os.path.abspath(args.labels8)
  res['labels16'] = os.path.abspath(args.labels16)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'r3_doubling.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  blk = res['primary']['inc_opp']
  print(f"inc_opp (opp16-opp8): {blk['point']:+.3f} "
        f"[{blk['ci'][0]:+.3f},{blk['ci'][1]:+.3f}] n={blk['n_clusters']}")
  print('flags:', res['flags'])
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# selfcheck (deterministic constant fixtures)
# --------------------------------------------------------------------------

def _synth(opp8, opp16, ach=0.1):
  c8, c16 = {}, {}
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        c8[(dom, dose, seed)] = dict(opp=opp8, ach=ach, gap=opp8 - ach,
                                     floor=False, n_states=200)
        c16[(dom, dose, seed)] = dict(opp=opp16, ach=ach, gap=opp16 - ach,
                                      floor=False, n_states=200)
  return c8, c16


def selfcheck(args):
  # Immaterial: small positive increment.
  res = analyze(*_synth(1.0, 1.1))
  assert res['flags'] == dict(anomaly=False, immaterial=True)
  assert 'IMMATERIAL' in res['verdict']
  assert res['secondary']['gap16_refires']
  # Material: large increment.
  res = analyze(*_synth(1.0, 2.0))
  assert res['flags'] == dict(anomaly=False, immaterial=False)
  assert 'MATERIAL TRUNCATION' in res['verdict']
  # Anomaly: negative increment; immaterial must NOT co-fire.
  res = analyze(*_synth(1.0, 0.5))
  assert res['flags'] == dict(anomaly=True, immaterial=False)
  assert res['verdict'].startswith('ANOMALY')
  # Zero increment is consistent (not anomalous) and immaterial.
  res = analyze(*_synth(1.0, 1.0))
  assert res['flags'] == dict(anomaly=False, immaterial=True)
  # Missing cell trips.
  c8, c16 = _synth(1.0, 1.1)
  del c16[('cup', 'e1', EXPECT_SEEDS[0])]
  try:
    analyze(c8, c16)
    raise SystemExit('selfcheck FAIL: missing M=16 cell not caught')
  except AssertionError:
    pass
  # Stray seed trips.
  c8, c16 = _synth(1.0, 1.1)
  c16[('cup', 'e1', 99)] = dict(opp=9.0, ach=0.0, gap=9.0, floor=False,
                                n_states=200)
  try:
    analyze(c8, c16)
    raise SystemExit('selfcheck FAIL: stray seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e
  # Smoke gate trips on an empty dir.
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    try:
      check_smoke(d)
      raise SystemExit('selfcheck FAIL: missing smoke not caught')
    except AssertionError as e:
      assert 'smoke gate' in str(e), e
  # _load fixture leg: the file-level integrity asserts must be
  # reachable by this selfcheck (R3-lesson: constants their own
  # selfcheck never exercises are structurally unprotected).
  def _meta(**over):
    md = dict(states=200, horizon=100, label_every=25, seed=0,
              rollouts=16, ref_stride=5, mass_scale=1.0,
              behavior_checkpoint='', actions=8, train_seed=31,
              labeler_version=LABELER_VERSION,
              checkpoint='/x/r3_cup_e1_seed31/ckpt')
    md.update(over)
    return json.dumps(md)

  def _write(d, name, n=200, cols=8, **meta_over):
    np.savez(os.path.join(d, name), meta=_meta(**meta_over),
             g_all=np.ones((n, cols), np.float32), g_now=np.ones(n),
             m_real=np.zeros(n, int))

  def _expect_trip(d, file_re, m_expected, needle):
    try:
      _load(d, file_re, m_expected)
      raise SystemExit(f'selfcheck FAIL: {needle} not caught')
    except AssertionError as e:
      assert needle in str(e), e

  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_cup_e1_seed31_late.npz')
    got = _load(d, FILE8_RE, M8)
    assert list(got) == [('cup', 'e1', 31)]
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_cup_e1_seed31_late.npz', n=150, states=150)
    _expect_trip(d, FILE8_RE, M8, 'dial states')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3dbl_cup_e1_seed31_late_m16.npz', cols=8, actions=8)
    _expect_trip(d, FILE16_RE, M16, 'actions')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_cup_e1_seed31_late.npz', labeler_version='wrong')
    _expect_trip(d, FILE8_RE, M8, 'labeler_version')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_cup_e1_seed31_late.npz', horizon=50)
    _expect_trip(d, FILE8_RE, M8, 'dial horizon')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'r3_cup_e1_seed31_late.npz',
           checkpoint='/x/r3_cup_e1_seed31/ckpt_early')
    _expect_trip(d, FILE8_RE, M8, 'not the late')
  print('selfcheck PASS: immaterial/material/anomaly/zero branches '
        '(anomaly excludes immaterial), missing-cell + stray-seed + '
        'smoke-gate trips, _load fixture leg (dials, actions, labeler, '
        'checkpoint suffix)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels8')
  ap.add_argument('--labels16')
  ap.add_argument('--output', default='analysis_out/r3_doubling')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if not args.selfcheck and not (args.labels8 and args.labels16):
    ap.error('--labels8 and --labels16 required (or --selfcheck)')
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
