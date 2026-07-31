"""Frozen read for TM2-R3 — TD-MPC2 opportunity/competence cross-family
replication (Paper 2).

Registered in prereg/PREREG_tm2_competence_20260730.md and committed
BEFORE any TM2-R3 training run, smoke, or label exists. Labels come from
the ported TD-MPC2 labeler (tm2_oracle_20260730,
probing/tdmpc2_oracle_labels.py) with --oracle_all; estimands per
labeled state are IDENTICAL to the executed R3 read:

  opportunity = max_m G[m] - G[m_now]      (from g_all, g_now)
  achieved    = G[m_real] - G[m_now]
  gap         = opportunity - achieved

Files: <labels_dir>/tm2r3_{dom}_{dose}_seed{n}_{mat}.npz
  dom in {cup, finger}, dose in {e1, e4}, mat in {early, late},
  seeds 51-58 (disjoint from every prior cohort). Cluster = training
  run (dom, dose, seed); both maturities of a run share the cluster.
  Cohort is strict: all 64 cells or the read refuses.

Estimand machinery (cell_stats), integrity guards (validate_arrays),
bootstrap (_cluster_boot, B=10K, rng 0), the opportunity bar (0.2), and
the floor constant are IMPORTED from the frozen analysis/r3_read.py
with pin asserts — identical thresholds by construction.

Registered gates and rules:
  SMOKE GATE   machine-checked here: (i) the random-init API smoke
               tm2r3_apismoke_smoke.npz (meta smoke_random_init true)
               and (ii) the four dosed real smokes
               tm2r3_{cup,finger}_{e1,e4}_seed51_late_smoke.npz must
               exist in the labels dir, all at the pinned labeler,
               each a genuine NON-random-init pass of 5 states whose
               meta task/dose match the filename and whose checkpoint
               is the named run's seed-51 ckpt_late.pt.
  DIAL GUARD   per-file meta must carry the registered dials (states
               200, horizon 100, label_every 25, labeler seed 0,
               actions 8, ref_stride 5, mass_scale 1.0, no behavior
               checkpoint), task == the filename domain's task, dose
               == the filename dose (e4 => dim 32 scale 3.0), the
               training-audit echo (train_steps 100000,
               train_early_steps 25000, realized early/late snapshot
               steps present), train_seed == the filename seed,
               checkpoint == the named run's ckpt_{mat}.pt, and no
               smoke flag.
  FLOOR GATE   (reacher-style) any domain with > 50% all-zero-G floor
               cells => FLOOR-LIMITED: inconclusive, fires nulled to
               None, no adjudication. Floor cells are otherwise
               INCLUDED in all primaries (conservative).
  P-TM2a  pooled opportunity CI entirely > 0.2   (same existence bar)
  P-TM2b  pooled gap CI entirely > 0
  P-TM2c  per-run paired (late - early) achieved, pooled CI > 0
Verdicts: CROSS-FAMILY REPLICATION (a+b fire; c scopes maturity) /
OPPORTUNITY-ABSENT (a fails) / HARVESTED-BY-PLANNER (a fires, b fails
— the planner-consumer harvests; a registered family boundary for the
competence claim, paper-shaped either way) / FLOOR-LIMITED.

Usage:
  python -m analysis.tm2_r3_read --labels <dir> --output <dir>
  python -m analysis.tm2_r3_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from analysis.r3_read import (
    B_BOOT, RNG_SEED, OPP_THRESHOLD, FLOOR_LIMIT, DOSES, MATS,
    cell_stats, validate_arrays, _cluster_boot)

# Pin the shared-frozen-reader import surface: a future amendment to
# r3_read must not silently change this reader's registered rules.
assert B_BOOT == 10_000, B_BOOT
assert RNG_SEED == 0, RNG_SEED
assert OPP_THRESHOLD == 0.2, OPP_THRESHOLD
assert FLOOR_LIMIT == 0.5, FLOOR_LIMIT
assert DOSES == ('e1', 'e4'), DOSES
assert MATS == ('early', 'late'), MATS

LABELER_VERSION = 'tm2_oracle_20260730'
DOMAINS = ('cup', 'finger')
TASK_OF = dict(cup='dmc_cup_catch', finger='dmc_finger_turn_hard')
EXPECT_SEEDS = tuple(range(51, 59))
M_CANDS = 8
TRAIN_STEPS = 100_000       # registered training dial (nominal late)
EARLY_STEPS = 25_000        # registered training dial (nominal early)
SMOKE_STATES = 5            # registered dosed-smoke size
FILE_RE = re.compile(
    r'^tm2r3_(cup|finger)_(e1|e4)_seed(\d+)_(early|late)\.npz$')
APISMOKE = 'tm2r3_apismoke_smoke.npz'
SMOKE_KINDS = tuple((dom, dose) for dom in DOMAINS for dose in DOSES)

# Registered dial identity, enforced from each file's meta — a
# mis-dialed manual retry of one pass must trip, not silently bias.
DIALS = dict(states=200, horizon=100, label_every=25, seed=0,
             actions=M_CANDS, ref_stride=5, mass_scale=1.0,
             behavior_checkpoint='')
DOSE_DIMS = dict(e1=dict(dim=0), e4=dict(dim=32, scale=3.0))


def check_smoke(labels_dir):
  missing = []
  path = os.path.join(labels_dir, APISMOKE)
  if not os.path.exists(path):
    missing.append(APISMOKE)
  else:
    meta = json.loads(str(np.load(path, allow_pickle=True)['meta']))
    assert meta.get('labeler_version') == LABELER_VERSION, (
        f'{APISMOKE}: smoke labeler_version is not {LABELER_VERSION}')
    assert meta.get('smoke_random_init') is True, (
        f'{APISMOKE}: not a random-init API smoke')
  for dom, dose in SMOKE_KINDS:
    name = f'tm2r3_{dom}_{dose}_seed51_late_smoke.npz'
    p = os.path.join(labels_dir, name)
    if not os.path.exists(p):
      missing.append(name)
      continue
    meta = json.loads(str(np.load(p, allow_pickle=True)['meta']))
    assert meta.get('labeler_version') == LABELER_VERSION, (
        f'{name}: smoke labeler_version is not {LABELER_VERSION}')
    assert not meta.get('smoke_random_init'), (
        f'{name}: dosed real smoke carries the random-init flag')
    assert meta.get('task') == TASK_OF[dom], (
        f"{name}: smoke task {meta.get('task')!r} != {TASK_OF[dom]!r}")
    assert meta.get('dose') == dose, (
        f"{name}: smoke dose {meta.get('dose')!r} != filename dose {dose!r}")
    assert meta.get('states') == SMOKE_STATES, (
        f"{name}: smoke states {meta.get('states')!r} != registered "
        f'{SMOKE_STATES}')
    assert str(meta.get('checkpoint', '')).endswith(
        f'tm2r3_{dom}_{dose}_seed51/ckpt_late.pt'), (
        f"{name}: smoke checkpoint {meta.get('checkpoint')!r} is not the "
        f'seed-51 late snapshot of tm2r3_{dom}_{dose}')
  assert not missing, (
      'registered TM2-R3 smoke gate not satisfied; missing: '
      + ', '.join(missing))


def check_meta(name, md, dom, dose, seed, mat):
  assert md.get('labeler_version') == LABELER_VERSION, (
      f'{name}: labeler_version is not {LABELER_VERSION}')
  assert not md.get('smoke_random_init'), (
      f'{name}: smoke_random_init pass consumed as a real label')
  for k, v in DIALS.items():
    assert md.get(k) == v, (
        f'{name}: dial {k}={md.get(k)!r} != registered {v!r}')
  assert md.get('dose') == dose, (
      f"{name}: meta dose={md.get('dose')!r} != filename dose {dose!r}")
  dcfg = md.get('dose_config') or {}
  for k, v in DOSE_DIMS[dose].items():
    assert dcfg.get(k) == v, (
        f'{name}: dose_config {k}={dcfg.get(k)!r} != registered {v!r}')
  assert md.get('task') == TASK_OF[dom], (
      f"{name}: meta task={md.get('task')!r} != filename domain {dom!r} "
      f'({TASK_OF[dom]!r}) — domain identity must not rest on the run-dir '
      'name alone')
  assert md.get('train_steps') == TRAIN_STEPS, (
      f"{name}: train_steps={md.get('train_steps')!r} != registered "
      f'{TRAIN_STEPS}')
  assert md.get('train_early_steps') == EARLY_STEPS, (
      f"{name}: train_early_steps={md.get('train_early_steps')!r} != "
      f'registered {EARLY_STEPS}')
  for k in ('early_step_realized', 'late_step_realized'):
    assert isinstance(md.get(k), int), (
        f'{name}: {k}={md.get(k)!r} — training-audit echo missing (realized '
        'values themselves are FILL-slot territory, presence-pinned only)')
  assert md.get('train_seed') == seed, (
      f"{name}: meta train_seed={md.get('train_seed')} != filename {seed}")
  assert str(md.get('checkpoint', '')).endswith(
      f'tm2r3_{dom}_{dose}_seed{seed}/ckpt_{mat}.pt'), (
      f"{name}: checkpoint {md.get('checkpoint')!r} is not the {mat} "
      f'snapshot of tm2r3_{dom}_{dose}_seed{seed}')


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
    md = json.loads(str(z['meta']))
    check_meta(name, md, dom, dose, seed, mat)
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    assert g_all.shape[1] == M_CANDS, (
        f'{name}: candidate count {g_all.shape[1]} != registered {M_CANDS}')
    validate_arrays(name, g_all, g_now, m_real)
    key = (dom, dose, seed, mat)
    assert key not in cells, f'duplicate cell from {name}'
    cells[key] = cell_stats(g_all, g_now, m_real)
  return cells


def check_grid(cells):
  assert cells, 'no tm2r3_* label files found'
  bad = sorted({k[2] for k in cells} - set(EXPECT_SEEDS))
  assert not bad, f'unregistered seeds present: {bad}'
  n_states = sorted({v['n_states'] for v in cells.values()})
  assert n_states == [200], (
      f'n_states differs from the registered 200: {n_states}')
  for dom in DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for mat in MATS:
          assert (dom, dose, seed, mat) in cells, (
              f'cohort incomplete: {dom}/{dose}/seed{seed}/{mat} missing '
              '(all-or-nothing)')
  assert len(cells) == len(DOMAINS) * len(DOSES) * len(EXPECT_SEEDS) * \
      len(MATS), 'unexpected extra cells'


def analyze(cells):
  check_grid(cells)
  runs = sorted({k[:3] for k in cells})

  floors = {}
  for dom in DOMAINS:
    cf = [cells[k]['floor'] for k in cells if k[0] == dom]
    floors[dom] = dict(frac=float(np.mean(cf)), n_cells=len(cf),
                       floor_limited=bool(np.mean(cf) > FLOOR_LIMIT))
  floor_limited = any(floors[dom]['floor_limited'] for dom in DOMAINS)

  def pooled(stat):
    return {r: [cells[r + (mat,)][stat] for mat in MATS] for r in runs}

  a = _cluster_boot(pooled('opp'))
  b = _cluster_boot(pooled('gap'))
  paired = {r: [cells[r + ('late',)]['ach'] - cells[r + ('early',)]['ach']]
            for r in runs}
  c = _cluster_boot(paired)
  fires = dict(
      p_tm2a=a['ci'][0] > OPP_THRESHOLD,
      p_tm2b=b['ci'][0] > 0,
      p_tm2c=c['ci'][0] > 0)

  def sub_boot(stat, sel):
    vals = {r: [cells[r + (mat,)][stat] for mat in MATS]
            for r in runs if sel(r)}
    return _cluster_boot(vals) if vals else None

  secondary = dict(
      dose_effect_opp={dose: sub_boot('opp', lambda r, d=dose: r[1] == d)
                       for dose in DOSES},
      maturity_opp={mat: _cluster_boot(
          {r: [cells[r + (mat,)]['opp']] for r in runs}) for mat in MATS},
      ach=_cluster_boot(pooled('ach')),
      per_domain={dom: dict(
          opp=sub_boot('opp', lambda r, d=dom: r[0] == d),
          ach=sub_boot('ach', lambda r, d=dom: r[0] == d),
          gap=sub_boot('gap', lambda r, d=dom: r[0] == d))
          for dom in DOMAINS},
      floors=floors)

  if floor_limited:
    verdict = ('FLOOR-LIMITED: > 50% of a domain\'s cells are '
               'all-zero-G floors — inconclusive by the registered '
               'gate; primaries reported but NOT adjudicated; no '
               'further TM2-R3 compute without a redesign.')
    fires = dict(p_tm2a=None, p_tm2b=None, p_tm2c=None)
  elif fires['p_tm2a'] and fires['p_tm2b']:
    verdict = ('CROSS-FAMILY REPLICATION: the opportunity/competence '
               'decomposition holds in a second model family — '
               'opportunity exists (P-TM2a) and the TD-MPC2 planner '
               'does not harvest it (P-TM2b). '
               + ('Maturity buys competence here (P-TM2c fires).'
                  if fires['p_tm2c'] else
                  'P-TM2c does not fire — maturity does not close the '
                  'gap in this family either.'))
  elif not fires['p_tm2a']:
    verdict = ('OPPORTUNITY-ABSENT: no detectable oracle opportunity in '
               'the TD-MPC2 cells — an information-content fact about '
               'this family at these cells; the DreamerV3 decomposition '
               'stands unchanged with disclosed family scope '
               '(registered branch).')
  else:
    verdict = ('HARVESTED-BY-PLANNER: opportunity exists and the '
               'TD-MPC2 planner-consumer harvests it — the competence '
               'failure becomes consumer-class-scoped (one-step plug-in '
               'vs MPPI planner), the registered family boundary for '
               'the competence claim (paper-shaped either way).')
  return dict(primaries=dict(p_tm2a_opportunity=a, p_tm2b_gap=b,
                             p_tm2c_maturity_ach=c),
              thresholds=dict(opp=OPP_THRESHOLD, floor_limit=FLOOR_LIMIT),
              fires=fires, secondary=secondary,
              n_runs=len(runs), verdict=verdict)


def read(args):
  check_smoke(args.labels)
  cells = load_labels(args.labels)
  res = analyze(cells)
  res['labels_dir'] = os.path.abspath(args.labels)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'tm2_r3.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  for name, blk in res['primaries'].items():
    print(f"{name}: {blk['point']:+.3f} [{blk['ci'][0]:+.3f},"
          f"{blk['ci'][1]:+.3f}] n={blk['n_clusters']}")
  print('fires:', res['fires'])
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# selfcheck (deterministic constant fixtures; no torch/TDMPC2)
# --------------------------------------------------------------------------

def _synth(opp, ach_early, ach_late, floor_runs=()):
  cells = {}
  for dom in DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for mat in MATS:
          if (dom, dose, seed) in floor_runs:
            cells[(dom, dose, seed, mat)] = dict(
                opp=0.0, ach=0.0, gap=0.0, floor=True, n_states=200)
            continue
          ach = ach_late if mat == 'late' else ach_early
          cells[(dom, dose, seed, mat)] = dict(
              opp=opp, ach=ach, gap=opp - ach, floor=False, n_states=200)
  return cells


def _meta(**over):
  md = dict(states=200, horizon=100, label_every=25, seed=0, actions=8,
            ref_stride=5, mass_scale=1.0, behavior_checkpoint='',
            dose='e1', dose_config=dict(dim=0, scale=1.0),
            task='dmc_cup_catch', train_steps=100_000,
            train_early_steps=25_000, early_step_realized=25_000,
            late_step_realized=100_001,
            train_seed=51, labeler_version=LABELER_VERSION,
            smoke_random_init=False,
            checkpoint='/x/tm2r3_cup_e1_seed51/ckpt_early.pt')
  md.update(over)
  return json.dumps(md)


def _write(d, name, n=200, cols=8, **meta_over):
  np.savez(os.path.join(d, name), meta=_meta(**meta_over),
           g_all=np.ones((n, cols), np.float32), g_now=np.ones(n),
           m_real=np.zeros(n, int))


def selfcheck(args):
  # Verdict branch 1: CROSS-FAMILY REPLICATION + maturity null.
  res = analyze(_synth(1.0, 0.1, 0.1))
  assert res['fires'] == dict(p_tm2a=True, p_tm2b=True, p_tm2c=False)
  assert res['verdict'].startswith('CROSS-FAMILY REPLICATION'), res['verdict']
  # Branch 1b: maturity fires and is scoped in the verdict text.
  res = analyze(_synth(1.0, 0.1, 0.5))
  assert res['fires']['p_tm2c'] and 'Maturity buys' in res['verdict']
  # Branch 2: OPPORTUNITY-ABSENT.
  res = analyze(_synth(0.05, 0.0, 0.0))
  assert res['verdict'].startswith('OPPORTUNITY-ABSENT'), res['verdict']
  # Branch 3: HARVESTED-BY-PLANNER.
  res = analyze(_synth(1.0, 1.0, 1.0))
  assert res['fires']['p_tm2a'] and not res['fires']['p_tm2b']
  assert res['verdict'].startswith('HARVESTED-BY-PLANNER'), res['verdict']
  # Branch 4: FLOOR-LIMITED — 9/16 finger runs floored (> 50% of that
  # domain's cells); fires nulled to None, cup floors don't gate.
  floor_runs = {('finger', 'e1', s) for s in EXPECT_SEEDS[:5]} | {
      ('finger', 'e4', s) for s in EXPECT_SEEDS[:4]}
  res = analyze(_synth(1.0, 0.1, 0.1, floor_runs=floor_runs))
  assert res['verdict'].startswith('FLOOR-LIMITED'), res['verdict']
  assert res['fires'] == dict(p_tm2a=None, p_tm2b=None, p_tm2c=None)
  assert res['secondary']['floors']['finger']['floor_limited']
  assert not res['secondary']['floors']['cup']['floor_limited']

  # Grid trips: missing cell, lone maturity, stray seed, n_states.
  cells = _synth(1.0, 0.1, 0.1)
  bad = dict(cells)
  del bad[('cup', 'e1', 51, 'early')], bad[('cup', 'e1', 51, 'late')]
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: missing run not caught')
  except AssertionError:
    pass
  bad = dict(cells)
  del bad[('cup', 'e1', 51, 'early')]
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: lone maturity not caught')
  except AssertionError:
    pass
  stray = dict(cells)
  for mat in MATS:
    stray[('cup', 'e1', 99, mat)] = dict(opp=9.0, ach=0.0, gap=9.0,
                                         floor=False, n_states=200)
  try:
    analyze(stray)
    raise SystemExit('selfcheck FAIL: unregistered seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e
  short = dict(cells)
  short[('cup', 'e1', 51, 'early')] = dict(
      short[('cup', 'e1', 51, 'early')], n_states=120)
  try:
    analyze(short)
    raise SystemExit('selfcheck FAIL: off-registration n_states not caught')
  except AssertionError as e:
    assert 'n_states' in str(e), e

  # Smoke gate: empty dir trips; wrong-labeler smoke trips; an apismoke
  # without the random-init flag trips; the complete gate passes.
  import tempfile

  def _write_smokes(d, api_flag=True, version=LABELER_VERSION, **over):
    np.savez(os.path.join(d, APISMOKE), meta=json.dumps(dict(
        labeler_version=version, smoke_random_init=api_flag)))
    for dom, dose in SMOKE_KINDS:
      md = dict(labeler_version=version, smoke_random_init=False,
                task=TASK_OF[dom], dose=dose, states=SMOKE_STATES,
                checkpoint=f'/x/tm2r3_{dom}_{dose}_seed51/ckpt_late.pt')
      md.update(over)
      np.savez(os.path.join(d, f'tm2r3_{dom}_{dose}_seed51_late_smoke.npz'),
               meta=json.dumps(md))

  with tempfile.TemporaryDirectory() as d:
    try:
      check_smoke(d)
      raise SystemExit('selfcheck FAIL: missing smoke not caught')
    except AssertionError as e:
      assert 'smoke gate' in str(e), e
    _write_smokes(d)
    check_smoke(d)
  with tempfile.TemporaryDirectory() as d:
    _write_smokes(d, version='wrong')
    try:
      check_smoke(d)
      raise SystemExit('selfcheck FAIL: wrong-labeler smoke not caught')
    except AssertionError as e:
      assert 'labeler_version' in str(e), e
  with tempfile.TemporaryDirectory() as d:
    _write_smokes(d, api_flag=False)
    try:
      check_smoke(d)
      raise SystemExit('selfcheck FAIL: non-random-init apismoke not caught')
    except AssertionError as e:
      assert 'random-init' in str(e), e
  # Dosed-smoke authenticity legs: a random-init pass copied to a dosed
  # smoke name, a wrong-dose smoke, an off-size smoke, and a smoke run
  # against the wrong checkpoint must all trip.
  for over, needle in [
      (dict(smoke_random_init=True), 'random-init flag'),
      (dict(dose='e1'), 'smoke dose'),
      (dict(states=50), 'smoke states'),
      (dict(checkpoint='/x/tm2r3_cup_e1_seed51/ckpt_early.pt'),
       'late snapshot'),
      (dict(task='dmc_walker_walk'), 'smoke task'),
  ]:
    with tempfile.TemporaryDirectory() as d:
      _write_smokes(d, **over)
      try:
        check_smoke(d)
        raise SystemExit(f'selfcheck FAIL: dosed-smoke {needle} not caught')
      except AssertionError as e:
        assert needle in str(e), (needle, e)

  # load_labels fixture leg (file-level guards must be reachable):
  # a valid file loads; dial/dose/checkpoint/seed/version/smoke-flag
  # violations and zero-padded-seed duplicates trip.
  def _expect_trip(d, needle):
    try:
      load_labels(d)
      raise SystemExit(f'selfcheck FAIL: {needle} not caught')
    except AssertionError as e:
      assert needle in str(e), (needle, e)

  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz')
    got = load_labels(d)
    assert list(got) == [('cup', 'e1', 51, 'early')]
    assert got[('cup', 'e1', 51, 'early')]['n_states'] == 200
    _write(d, 'tm2r3_cup_e1_seed051_early.npz')
    _expect_trip(d, 'duplicate cell')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', labeler_version='wrong')
    _expect_trip(d, 'labeler_version')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', states=150, n=150)
    _expect_trip(d, 'dial states')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', horizon=50)
    _expect_trip(d, 'dial horizon')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e4_seed51_early.npz', dose='e4',
           dose_config=dict(dim=8, scale=1.0),
           checkpoint='/x/tm2r3_cup_e4_seed51/ckpt_early.pt')
    _expect_trip(d, 'dose_config')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e4_seed51_early.npz',
           checkpoint='/x/tm2r3_cup_e4_seed51/ckpt_early.pt')
    _expect_trip(d, 'meta dose')  # meta says e1, filename says e4
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_finger_e1_seed51_early.npz',
           checkpoint='/x/tm2r3_finger_e1_seed51/ckpt_early.pt')
    _expect_trip(d, 'meta task')  # cup-trained run binned as finger
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', train_steps=50_000)
    _expect_trip(d, 'train_steps')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', early_step_realized=None)
    _expect_trip(d, 'early_step_realized')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed52_early.npz')
    _expect_trip(d, 'train_seed')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_late.npz')
    _expect_trip(d, 'snapshot')  # early ckpt consumed as late
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', smoke_random_init=True)
    _expect_trip(d, 'smoke_random_init')
  with tempfile.TemporaryDirectory() as d:
    _write(d, 'tm2r3_cup_e1_seed51_early.npz', cols=16)
    _expect_trip(d, 'candidate count')

  # Estimand identities on a hand case (imported cell_stats).
  g_all = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
  g_now = np.array([0.5, 1.0])
  m_real = np.array([2, 0])
  st = cell_stats(g_all, g_now, m_real)
  assert abs(st['opp'] - np.mean([1.5, 0.0])) < 1e-12
  assert abs(st['ach'] - np.mean([0.5, 0.0])) < 1e-12
  assert abs(st['gap'] - (st['opp'] - st['ach'])) < 1e-12 and not st['floor']
  assert cell_stats(np.zeros((2, 3)), np.zeros(2), np.array([0, 1]))['floor']

  print('selfcheck PASS: four verdict branches incl. floor-limited gate '
        '(fires nulled), maturity scoping, grid trips (missing run, lone '
        'maturity, stray seed, n_states), smoke-gate legs (missing, '
        'wrong-labeler, non-random-init apismoke, complete-pass, dosed-'
        'smoke authenticity: random-init copy, dose, states, checkpoint, '
        'task), load_labels fixture legs (dials, dose identity, '
        'dose_config, task identity, training-audit echo, train_seed, '
        'checkpoint maturity, smoke flag, candidate count, duplicate), '
        'estimand identities, import pins')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='analysis_out/tm2_r3')
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
