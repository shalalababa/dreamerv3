"""Frozen read for the R3 cross-checkpoint consumer wave (Paper 2).

Registered in prereg/PREREG_r3_amend2_20260730.md (dated amendment to
PREREG_r3_competence_20260725.md, freeze commit 8b3380ce) and committed
BEFORE any cross-checkpoint label exists — --consumer_checkpoint has
never been run anywhere. Labels come from the xc-extended labeler
(version d1fix_20260724_xc1, EXACT pin: the base d1fix_20260724 string
is a substring of the xc string, so this reader matches the meta field
by equality, never by containment; the r3_* readers in turn never see
xc files because their filename regexes reject the xc_ prefix).

Files: <labels_dir>/xc_{dom}_{dose}_seed{s}_{evalmat}_h{headsmat}.npz
  dom in {cup, finger}, dose in {e1, e4}, seeds 31-38, evalmat = the
  evaluated checkpoint's maturity, headsmat = the maturity of the
  OVERLAID rew/con/valens heads. headsmat == evalmat is the CONTROL
  (still through the overlay code path); headsmat != evalmat is the
  swap. 4 passes per run x 32 runs = 128 files, all-or-nothing
  (registered existence gate). Labeler seed 1 (fresh states; contrasts
  are within-pass-pair only, never against the committed R3 labels).

Pairing guard (registered): the overlaid heads never influence the base
trajectory or the oracle rollouts (pol + restored counters + env RNG
only), so within a (run, eval side) pair episode/step must match
EXACTLY and g_all must match exactly (registered fallback: atol
PAIR_ATOL = 1e-5, disclosed as pair_tolerance_used) — any violation
QUARANTINES the read. m_now/m_real/qfull/g_now differ; that is the
chooser under study.

Estimand per state: achieved = g_all[arange, m_real] - g_now.
Primaries (run-clustered percentile bootstrap, B=10K, rng 0,
cluster = training run, 32 clusters):
  P-XC1 (late eval)  per-run mean of paired per-state
         [achieved(heads=early) - achieved(heads=late)];
         CI entirely < 0 => immature heads HURT on the late side.
  P-XC2 (early eval) per-run mean of paired per-state
         [achieved(heads=late) - achieved(heads=early)];
         CI entirely > 0 => mature heads RESCUE the early side.
Registered prediction (from the committed P-R3c null +0.014
[-0.177,+0.215]): both straddle 0 => HEADS-IRRELEVANT.

Verdicts: HEADS-IRRELEVANT / HEAD-DEFICIT (P-XC2 fires +) / HEAD-HARM
(P-XC1 fires -) / MIXED (both registered directions fire, or any CI
excursion in an unregistered direction) / QUARANTINE (pairing guard) /
SUBSTRATE-GONE (existence gate: driver marker, no adjudication).

Usage:
  python -m analysis.r3_xconsumer_read --labels <dir> --output <dir>
  python -m analysis.r3_xconsumer_read --pairgate <dir>   (value-blind
      pre-wave integrity gate: touches ONLY episode/step/g_all of the
      one registered full-dial pair; no estimand is ever computed)
  python -m analysis.r3_xconsumer_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from analysis.r3_read import (
    B_BOOT, RNG_SEED, LABELER_VERSION, EXPECT_SEEDS, DOSES, MATS,
    validate_arrays, _cluster_boot)

# Pin the shared-frozen-reader import surface: a future amendment to
# r3_read must not silently change this reader's registered grid or
# bootstrap machinery.
assert EXPECT_SEEDS == tuple(range(31, 39)), EXPECT_SEEDS
assert B_BOOT == 10_000 and RNG_SEED == 0, (B_BOOT, RNG_SEED)
assert LABELER_VERSION == 'd1fix_20260724', LABELER_VERSION
assert MATS == ('early', 'late') and DOSES == ('e1', 'e4'), (MATS, DOSES)

XC_VERSION = 'd1fix_20260724_xc1'
assert XC_VERSION == LABELER_VERSION + '_xc1', XC_VERSION
CONSUMER_REGEX = r'^(rew|con|valens\d+)/'
CORE_DOMAINS = ('cup', 'finger')
N_STATES = 200
N_CANDS = 8
PAIR_ATOL = 1e-5
FILE_XC = re.compile(
    r'^xc_(cup|finger)_(e1|e4)_seed(\d+)_(early|late)_h(early|late)\.npz$')
# Registered dial identity, enforced from every file's meta — a
# mis-dialed manual retry must trip, not bias.
DIALS = dict(states=200, horizon=100, label_every=25, seed=1,
             actions=8, rollouts=16, ref_stride=5, mass_scale=1.0,
             behavior_checkpoint='', consumer_regex=CONSUMER_REGEX,
             max_steps=1000)
# Registered smoke gate: 4 dosed smokes (5 states) on cup e4 seed31,
# covering both swap directions and both controls.
SMOKES = tuple(f'xc_cup_e4_seed31_{mat}_h{h}_smoke.npz'
               for mat in MATS for h in MATS)
# Registered pairing gate: ONE full-dial pair, run and machine-checked
# via --pairgate BEFORE the wave.
PAIRGATE_CELL = ('cup', 'e1', 31, 'late')
SUBSTRATE_MARKER = 'SUBSTRATE_GONE'


def per_state_achieved(g_all, g_now, m_real):
  """achieved = G[m_real] - G[m_now], per labeled state."""
  g_all = np.asarray(g_all, float)
  return g_all[np.arange(len(m_real)), np.asarray(m_real, int)] - np.asarray(
      g_now, float)


def _assert_ckpt(name, field, md, run, mat):
  want = f'{run}/ckpt_early' if mat == 'early' else f'{run}/ckpt'
  got = str(md.get(field, '')).rstrip('/')
  assert got.endswith(want), (
      f'{name}: {field} {got!r} does not end with {want!r}')


def check_smoke(labels_dir):
  missing = []
  for smoke in SMOKES:
    path = os.path.join(labels_dir, smoke)
    if not os.path.exists(path):
      missing.append(smoke)
      continue
    z = np.load(path, allow_pickle=True)
    md = json.loads(str(z['meta']))
    assert md.get('labeler_version') == XC_VERSION, (
        f'{smoke}: smoke labeler_version {md.get("labeler_version")!r} '
        f'!= {XC_VERSION!r}')
    # Authenticity, not just existence: each smoke must be the CELL its
    # filename claims (eval ckpt + consumer ckpt suffixes) and 5 states
    # — four copies of one control would otherwise pass the gate and
    # the wave would launch with the swap direction never smoked.
    sm = re.match(
        r'^xc_(cup|finger)_(e1|e4)_seed(\d+)_(early|late)_h(early|late)'
        r'_smoke\.npz$', smoke)
    run = f'r3_{sm.group(1)}_{sm.group(2)}_seed{int(sm.group(3))}'
    _assert_ckpt(smoke, 'checkpoint', md, run, sm.group(4))
    _assert_ckpt(smoke, 'consumer_checkpoint', md, run, sm.group(5))
    assert md.get('states') == 5, (
        f'{smoke}: smoke states={md.get("states")!r} != 5')
  assert not missing, (
      'registered xc smoke gate not satisfied; missing: '
      + ', '.join(missing))


def load_xc(labels_dir):
  files = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = FILE_XC.match(name)
    if not m:
      continue
    dom, dose = m.group(1), m.group(2)
    seed, emat, hmat = int(m.group(3)), m.group(4), m.group(5)
    z = np.load(path, allow_pickle=True)
    md = json.loads(str(z['meta']))
    assert md.get('labeler_version') == XC_VERSION, (
        f'{name}: labeler_version {md.get("labeler_version")!r} != '
        f'{XC_VERSION!r} (exact pin; the base d1fix string alone is '
        'NOT an xc label)')
    for k, v in DIALS.items():
      assert md.get(k) == v, (
          f'{name}: dial {k}={md.get(k)!r} != registered {v!r}')
    assert md.get('train_seed') == seed, (
        f"{name}: meta train_seed={md.get('train_seed')} != filename {seed}")
    run = f'r3_{dom}_{dose}_seed{seed}'
    _assert_ckpt(name, 'checkpoint', md, run, emat)
    _assert_ckpt(name, 'consumer_checkpoint', md, run, hmat)
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    m_now = np.asarray(z['m_now'], int)
    validate_arrays(name, g_all, g_now, m_real)
    assert g_all.shape == (N_STATES, N_CANDS), (
        f'{name}: g_all shape {g_all.shape} != registered '
        f'({N_STATES}, {N_CANDS})')
    assert m_now.min() >= 0 and m_now.max() < N_CANDS, f'{name}: m_now range'
    # oracle_all schema identity: g_now must be g_all at the plug-in pick
    assert np.allclose(g_now, g_all[np.arange(N_STATES), m_now],
                       atol=1e-5), (
        f'{name}: g_now != g_all[m_now] (oracle_all identity violated)')
    key = (dom, dose, seed, emat, hmat)
    assert key not in files, f'duplicate cell from {name}'
    files[key] = dict(
        ach=per_state_achieved(g_all, g_now, m_real),
        g_all=g_all, m_real=m_real, m_now=m_now,
        episode=np.asarray(z['episode'], int),
        step=np.asarray(z['step'], int),
        opp=float(np.mean(np.max(g_all, 1) - g_now)),
        maxg=float(np.mean(np.max(g_all, 1))))
  return files


def check_grid(files):
  bad = sorted({k[2] for k in files} - set(EXPECT_SEEDS))
  assert not bad, f'unregistered seeds present: {bad}'
  missing = []
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for emat in MATS:
          for hmat in MATS:
            if (dom, dose, seed, emat, hmat) not in files:
              missing.append(f'{dom}/{dose}/seed{seed}/{emat}/h{hmat}')
  assert not missing, (
      'registered wave incomplete (all-or-nothing existence gate); '
      f'{len(missing)} passes missing, e.g. ' + ', '.join(missing[:6]))
  assert len(files) == 128, f'unexpected extra cells ({len(files)})'
  return sorted({k[:3] for k in files})


def _swap(mat):
  return 'early' if mat == 'late' else 'late'


def pair_violations(files, runs):
  """Value-blind pairing guard over ALL pairs: within (run, eval side),
  control (heads=eval) and swapped passes must be state-identical."""
  viols, tol_used = [], []
  for r in runs:
    for emat in MATS:
      c = files[r + (emat, emat)]
      s = files[r + (emat, _swap(emat))]
      tag = '/'.join(map(str, r)) + f'/{emat}'
      if not (np.array_equal(c['episode'], s['episode'])
              and np.array_equal(c['step'], s['step'])):
        viols.append(tag + ': episode/step mismatch')
        continue
      if np.array_equal(c['g_all'], s['g_all']):
        continue
      if np.allclose(c['g_all'], s['g_all'], rtol=0, atol=PAIR_ATOL):
        tol_used.append(tag)
      else:
        viols.append(
            tag + f': g_all mismatch beyond PAIR_ATOL={PAIR_ATOL}')
  return viols, tol_used


def analyze(files):
  runs = check_grid(files)
  viols, tol_used = pair_violations(files, runs)
  if viols:
    return dict(
        primaries=None,
        fires=dict(p_xc1_harm=None, p_xc2_deficit=None),
        pair_violations=viols, pair_tolerance_used=tol_used,
        n_runs=len(runs),
        verdict=('QUARANTINE: pairing guard violated — control and '
                 'swapped passes are not state-identical for: '
                 + '; '.join(viols[:6])
                 + (' ...' if len(viols) > 6 else '')
                 + '. No adjudication; audit the label passes '
                 '(counter restore / env divergence) before ANY use.'))

  def side_diffs(emat):
    # per-run mean of PER-STATE paired [achieved(swapped)-achieved(ctrl)]
    return {r: [float(np.mean(files[r + (emat, _swap(emat))]['ach']
                              - files[r + (emat, emat)]['ach']))]
            for r in runs}

  xc1 = _cluster_boot(side_diffs('late'))    # heads=early minus heads=late
  xc2 = _cluster_boot(side_diffs('early'))   # heads=late minus heads=early
  fires = dict(p_xc1_harm=bool(xc1['ci'][1] < 0),
               p_xc2_deficit=bool(xc2['ci'][0] > 0))
  off = dict(xc1_pos=bool(xc1['ci'][0] > 0), xc2_neg=bool(xc2['ci'][1] < 0))

  def disag_rate(field, emat):
    return float(np.mean([
        np.mean(files[r + (emat, _swap(emat))][field]
                != files[r + (emat, emat)][field]) for r in runs]))

  secondary = dict(
      per_domain={dom: dict(
          xc1=_cluster_boot({r: v for r, v in side_diffs('late').items()
                             if r[0] == dom}),
          xc2=_cluster_boot({r: v for r, v in side_diffs('early').items()
                             if r[0] == dom})) for dom in CORE_DOMAINS},
      chooser_disagreement={
          f'{emat}_{field}': disag_rate(field, emat)
          for emat in MATS for field in ('m_real', 'm_now')},
      opportunity_level={
          f'{emat}_h{hmat}': _cluster_boot(
              {r: [files[r + (emat, hmat)]['opp']] for r in runs})
          for emat in MATS for hmat in MATS},
      # redundant sanity on the guard: per-pair mean max-G is identical
      pair_maxg_max_absdiff=float(max(
          abs(files[r + (emat, emat)]['maxg']
              - files[r + (emat, _swap(emat))]['maxg'])
          for r in runs for emat in MATS)))

  if fires['p_xc1_harm'] and fires['p_xc2_deficit']:
    verdict = ('MIXED: both registered directions fire — immature heads '
               'hurt the late side AND mature heads rescue the early '
               'side; head knowledge matters on both sides. No headline '
               'claim beyond the two CIs; interpretation deferred.')
  elif off['xc1_pos'] or off['xc2_neg']:
    verdict = ('MIXED: a CI excursion in an unregistered direction ('
               + ('XC1 entirely positive' if off['xc1_pos'] else '')
               + (' and ' if off['xc1_pos'] and off['xc2_neg'] else '')
               + ('XC2 entirely negative' if off['xc2_neg'] else '')
               + ') — off the registered map; disclosed, no headline '
               'claim, interpretation deferred.')
  elif fires['p_xc2_deficit']:
    verdict = ('HEAD-DEFICIT: mature heads rescue achieved value on the '
               'early side (P-XC2 fires +) — part of the competence '
               'deficit lives in the value heads. Registered lower-bound '
               'caveat: the overlay reads mismatched features, which '
               'biases toward null, so this fire is a LOWER bound on '
               'head-knowledge transfer.')
  elif fires['p_xc1_harm']:
    verdict = ('HEAD-HARM: immature heads reduce achieved value on the '
               'late side (P-XC1 fires -) — late-side competence '
               'detectably depends on head knowledge.')
  else:
    verdict = ('HEADS-IRRELEVANT: both paired contrasts straddle 0 — '
               'the competence deficit does not detectably live in the '
               'value heads (the registered prediction, from the '
               'committed P-R3c null); the representational account is '
               'strengthened (support must be legible cross-link).')
  return dict(
      primaries=dict(p_xc1_late_eval=xc1, p_xc2_early_eval=xc2),
      fires=fires, off_direction=off,
      pair_violations=[], pair_tolerance_used=tol_used,
      thresholds=dict(pair_atol=PAIR_ATOL),
      secondary=secondary, n_runs=len(runs), verdict=verdict)


def read(args):
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'r3_xconsumer.json')
  marker = os.path.join(args.labels, SUBSTRATE_MARKER)
  if os.path.exists(marker):
    res = dict(
        primaries=None,
        fires=dict(p_xc1_harm=None, p_xc2_deficit=None),
        substrate_marker=os.path.abspath(marker),
        verdict=('SUBSTRATE-GONE: the registered existence gate failed '
                 '— the 32 R3 training run dirs are not all present, '
                 'and the wave is all-or-nothing. Recorded as the '
                 'registered outcome; no labels, no adjudication.'))
    with open(out, 'w') as f:
      json.dump(res, f, indent=2)
    print(res['verdict'])
    print(f'-> {out}')
    return
  check_smoke(args.labels)
  files = load_xc(args.labels)
  assert files, f'no xc_* labels under {args.labels}'
  res = analyze(files)
  res['labels_dir'] = os.path.abspath(args.labels)
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  if res['primaries']:
    for name, blk in res['primaries'].items():
      print(f"{name}: {blk['point']:+.3f} [{blk['ci'][0]:+.3f},"
            f"{blk['ci'][1]:+.3f}] n={blk['n_clusters']}")
  print('fires:', res['fires'])
  print(res['verdict'])
  print(f'-> {out}')


def pairgate(labels_dir):
  """Registered pre-wave integrity gate on the ONE full-dial pair
  (PAIRGATE_CELL: control + swapped). VALUE-BLIND by construction:
  only meta version, episode, step, and g_all are ever read —
  g_now/m_real/m_now/qfull are never loaded and no achieved value,
  mean, or any other estimand is computed here."""
  dom, dose, seed, emat = PAIRGATE_CELL
  names = [f'xc_{dom}_{dose}_seed{seed}_{emat}_h{emat}.npz',
           f'xc_{dom}_{dose}_seed{seed}_{emat}_h{_swap(emat)}.npz']
  arrs = []
  for n in names:
    path = os.path.join(labels_dir, n)
    assert os.path.exists(path), f'pairgate file missing: {n}'
    z = np.load(path, allow_pickle=True)
    md = json.loads(str(z['meta']))
    assert md.get('labeler_version') == XC_VERSION, (
        f'{n}: labeler_version {md.get("labeler_version")!r} != '
        f'{XC_VERSION!r}')
    arrs.append((np.asarray(z['episode']), np.asarray(z['step']),
                 np.asarray(z['g_all'], float)))
  (e1, s1, g1), (e2, s2, g2) = arrs
  assert np.isfinite(g1).all() and np.isfinite(g2).all(), (
      'pairgate: non-finite g_all (oracle_all pass incomplete?)')
  ok_idx = np.array_equal(e1, e2) and np.array_equal(s1, s2)
  exact = np.array_equal(g1, g2)
  ok_g = exact or np.allclose(g1, g2, rtol=0, atol=PAIR_ATOL)
  if not (ok_idx and ok_g):
    print('PAIRGATE QUARANTINE: control and swapped passes of '
          f'{PAIRGATE_CELL} are not state-identical '
          f'(episode/step ok={ok_idx}, g_all ok={ok_g}). Do NOT run '
          'the wave; audit the overlay counter restore first.')
    raise SystemExit(1)
  print('PAIRGATE OK: episode/step exact, g_all '
        + ('exact' if exact else f'within registered atol {PAIR_ATOL}')
        + ' (value-blind: only episode/step/g_all touched, no estimand '
        'computed)')


# --------------------------------------------------------------------------
# selfcheck (synthetic fixtures only; no real label path is ever named)
# --------------------------------------------------------------------------

G_ROW = np.round(np.arange(N_CANDS) * 0.1, 6)  # G[m] = 0.1*m, all states


def _mk_file(m_real_val, m_now_val=0, g_bump=0.0, episode=None, step=None):
  g_all = np.tile(G_ROW, (N_STATES, 1)) + g_bump
  m_real = np.full(N_STATES, m_real_val, int)
  m_now = np.full(N_STATES, m_now_val, int)
  g_now = g_all[np.arange(N_STATES), m_now]
  episode = np.zeros(N_STATES, int) if episode is None else episode
  step = (np.arange(N_STATES) * 25 + 25) if step is None else step
  return dict(ach=per_state_achieved(g_all, g_now, m_real),
              g_all=g_all, m_real=m_real, m_now=m_now,
              episode=episode, step=step,
              opp=float(np.mean(np.max(g_all, 1) - g_now)),
              maxg=float(np.mean(np.max(g_all, 1))))


def _synth_files(late_swap_m=3, early_swap_m=3, ctrl_m=3):
  """Control passes pick candidate ctrl_m; swapped passes pick the
  side's swap candidate. achieved = 0.1*m (m_now=0), so the per-state
  paired diff is 0.1*(swap_m - ctrl_m) exactly."""
  files = {}
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for emat in MATS:
          for hmat in MATS:
            if hmat == emat:
              mr = ctrl_m
            else:
              mr = late_swap_m if emat == 'late' else early_swap_m
            files[(dom, dose, seed, emat, hmat)] = _mk_file(mr)
  return files


def _write_npz(d, name, blk, version=XC_VERSION, **meta_over):
  dom, dose, seed, emat, hmat = None, None, None, None, None
  m = FILE_XC.match(name.replace('_smoke.npz', '.npz'))
  if m:
    dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
    emat, hmat = m.group(4), m.group(5)
  run = f'/x/r3_{dom}_{dose}_seed{seed}'
  md = dict(DIALS, train_seed=seed, labeler_version=version,
            checkpoint=run + ('/ckpt_early' if emat == 'early' else '/ckpt'),
            consumer_checkpoint=run + (
                '/ckpt_early' if hmat == 'early' else '/ckpt'))
  md.update(meta_over)
  g_now = blk['g_all'][np.arange(len(blk['m_now'])), blk['m_now']]
  np.savez(os.path.join(d, name), meta=json.dumps(md),
           g_all=blk['g_all'].astype(np.float32), g_now=g_now,
           m_real=blk['m_real'], m_now=blk['m_now'],
           episode=blk['episode'], step=blk['step'])


def selfcheck(args):
  import tempfile

  # Estimand identity on a hand case: G=[[0,2,1],[1,1,1]], m_now=[0,2],
  # m_real=[2,0] => g_now=[0,1], achieved=[1-0, 1-1]=[1, 0].
  g = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
  ach = per_state_achieved(g, np.array([0.0, 1.0]), np.array([2, 0]))
  assert np.allclose(ach, [1.0, 0.0]), ach

  # Verdict branch 1: HEADS-IRRELEVANT (all diffs exactly 0).
  res = analyze(_synth_files())
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=False)
  assert res['verdict'].startswith('HEADS-IRRELEVANT'), res['verdict']
  assert res['pair_violations'] == [] and res['pair_tolerance_used'] == []
  assert res['secondary']['pair_maxg_max_absdiff'] == 0.0
  assert res['secondary']['chooser_disagreement']['late_m_real'] == 0.0

  # Branch 2: HEAD-DEFICIT (early swap picks a better candidate: +0.2).
  res = analyze(_synth_files(early_swap_m=5))
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=True)
  assert res['verdict'].startswith('HEAD-DEFICIT'), res['verdict']
  blk = res['primaries']['p_xc2_early_eval']
  assert abs(blk['point'] - 0.2) < 1e-9, blk
  assert res['secondary']['chooser_disagreement']['early_m_real'] == 1.0

  # Branch 3: HEAD-HARM (late swap picks a worse candidate: -0.2).
  res = analyze(_synth_files(late_swap_m=1))
  assert res['fires'] == dict(p_xc1_harm=True, p_xc2_deficit=False)
  assert res['verdict'].startswith('HEAD-HARM'), res['verdict']
  assert abs(res['primaries']['p_xc1_late_eval']['point'] + 0.2) < 1e-9

  # Branch 4: MIXED, both registered directions fire.
  res = analyze(_synth_files(late_swap_m=1, early_swap_m=5))
  assert res['fires'] == dict(p_xc1_harm=True, p_xc2_deficit=True)
  assert res['verdict'].startswith('MIXED: both'), res['verdict']

  # Branch 5: MIXED, off-registered direction (late swap HELPS: XC1 > 0).
  res = analyze(_synth_files(late_swap_m=5))
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=False)
  assert res['off_direction']['xc1_pos']
  assert res['verdict'].startswith('MIXED: a CI excursion'), res['verdict']

  # Branch 6: QUARANTINE on a g_all pair mismatch beyond tolerance.
  files = _synth_files()
  key = ('cup', 'e1', 31, 'late', 'early')
  files[key] = _mk_file(3, g_bump=1.0)
  res = analyze(files)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert res['fires'] == dict(p_xc1_harm=None, p_xc2_deficit=None)
  assert any('cup/e1/31/late' in v for v in res['pair_violations'])
  # ... and on an episode/step mismatch.
  files = _synth_files()
  files[key] = _mk_file(3, step=np.arange(N_STATES) * 25 + 26)
  assert analyze(files)['verdict'].startswith('QUARANTINE')
  # Tolerance fallback: sub-atol g_all wobble adjudicates + discloses.
  files = _synth_files()
  files[key] = _mk_file(3, g_bump=1e-6)
  res = analyze(files)
  assert res['verdict'].startswith('HEADS-IRRELEVANT')
  assert res['pair_tolerance_used'] == ['cup/e1/31/late'], (
      res['pair_tolerance_used'])

  # Grid guards: missing side / partial run / unregistered seed / extras.
  files = _synth_files()
  del files[('cup', 'e1', 31, 'late', 'early')]
  try:
    analyze(files)
    raise SystemExit('selfcheck FAIL: missing swapped side not caught')
  except AssertionError as e:
    assert 'all-or-nothing' in str(e), e
  files = _synth_files()
  for hmat in MATS:
    del files[('finger', 'e4', 38, 'early', hmat)]
  try:
    analyze(files)
    raise SystemExit('selfcheck FAIL: partial run not caught')
  except AssertionError:
    pass
  files = _synth_files()
  files[('cup', 'e1', 99, 'late', 'early')] = _mk_file(3)
  try:
    analyze(files)
    raise SystemExit('selfcheck FAIL: unregistered seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e

  # File-level tempdir-npz fixture legs: full end-to-end read on a
  # complete synthetic wave (marker absent, smokes present), then the
  # load-time guards.
  ns = argparse.Namespace
  with tempfile.TemporaryDirectory() as d:
    labels, outdir = os.path.join(d, 'labels'), os.path.join(d, 'out')
    os.makedirs(labels)
    for key, blk in _synth_files(early_swap_m=5).items():
      dom, dose, seed, emat, hmat = key
      _write_npz(labels, f'xc_{dom}_{dose}_seed{seed}_{emat}_h{hmat}.npz',
                 blk)
    # smoke gate must trip before any smoke exists
    try:
      read(ns(labels=labels, output=outdir))
      raise SystemExit('selfcheck FAIL: missing smoke gate not caught')
    except AssertionError as e:
      assert 'smoke gate' in str(e), e
    for smoke in SMOKES:
      _write_npz(labels, smoke, _mk_file(3), states=5)
    read(ns(labels=labels, output=outdir))
    with open(os.path.join(outdir, 'r3_xconsumer.json')) as f:
      res = json.load(f)
    assert res['verdict'].startswith('HEAD-DEFICIT'), res['verdict']
    assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=True)
    # smoke-authenticity trips: a control-copy smoke (consumer suffix
    # contradicting its filename cell) and an over-length smoke must
    # both trip the gate — existence alone is not authenticity.
    _write_npz(labels, SMOKES[1], _mk_file(3), states=5,
               consumer_checkpoint='/x/r3_cup_e4_seed31/ckpt_early'
               if '_hlate' in SMOKES[1] else '/x/r3_cup_e4_seed31/ckpt')
    try:
      check_smoke(labels)
      raise SystemExit('selfcheck FAIL: control-copy smoke not caught')
    except AssertionError as e:
      assert 'consumer_checkpoint' in str(e), e
    _write_npz(labels, SMOKES[1], _mk_file(3), states=5)  # restore
    _write_npz(labels, SMOKES[0], _mk_file(3), states=200)
    try:
      check_smoke(labels)
      raise SystemExit('selfcheck FAIL: states=200 smoke not caught')
    except AssertionError as e:
      assert 'states' in str(e), e
    _write_npz(labels, SMOKES[0], _mk_file(3), states=5)  # restore
    # pairgate leg on the same tempdir (control vs swapped share g_all)
    pairgate(labels)
    # SUBSTRATE-GONE marker short-circuits everything
    open(os.path.join(labels, SUBSTRATE_MARKER), 'w').close()
    read(ns(labels=labels, output=outdir))
    with open(os.path.join(outdir, 'r3_xconsumer.json')) as f:
      assert json.load(f)['verdict'].startswith('SUBSTRATE-GONE')

  def _expect_load_trip(d, needle):
    try:
      load_xc(d)
      raise SystemExit(f'selfcheck FAIL: {needle} not caught')
    except AssertionError as e:
      assert needle in str(e), e

  with tempfile.TemporaryDirectory() as d:
    # a valid file loads and keys correctly
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3))
    got = load_xc(d)
    assert list(got) == [('cup', 'e1', 31, 'late', 'early')]
    assert abs(got[('cup', 'e1', 31, 'late', 'early')]['ach'][0]
               - 0.3) < 1e-6
  with tempfile.TemporaryDirectory() as d:
    # the base d1fix version alone must be REJECTED (exact pin — the
    # substring the r3 readers use would otherwise let it through)
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3),
               version=LABELER_VERSION)
    _expect_load_trip(d, 'exact pin')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3), seed=0)
    _expect_load_trip(d, 'dial seed')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3),
               consumer_regex='^(rew)/')
    _expect_load_trip(d, 'dial consumer_regex')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3),
               consumer_checkpoint='/x/r3_cup_e1_seed31/ckpt')
    _expect_load_trip(d, 'consumer_checkpoint')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3),
               checkpoint='/x/r3_cup_e1_seed31/ckpt_early')
    _expect_load_trip(d, 'checkpoint')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', _mk_file(3),
               train_seed=32)
    _expect_load_trip(d, 'train_seed')
  with tempfile.TemporaryDirectory() as d:
    blk = _mk_file(3)
    blk = dict(blk, g_all=blk['g_all'].copy())
    blk['g_all'][0, 0] = np.nan
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', blk)
    _expect_load_trip(d, 'non-finite')
  with tempfile.TemporaryDirectory() as d:
    # g_now != g_all[m_now] (oracle_all identity) trips
    blk = _mk_file(3)
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz', blk)
    z = dict(np.load(os.path.join(d, 'xc_cup_e1_seed31_late_hearly.npz'),
                     allow_pickle=True))
    z['g_now'] = np.asarray(z['g_now']) + 0.5
    np.savez(os.path.join(d, 'xc_cup_e1_seed31_late_hearly.npz'), **z)
    _expect_load_trip(d, 'oracle_all identity')

  # pairgate trips: missing file, then beyond-tolerance mismatch.
  with tempfile.TemporaryDirectory() as d:
    try:
      pairgate(d)
      raise SystemExit('selfcheck FAIL: pairgate missing file not caught')
    except AssertionError as e:
      assert 'pairgate file missing' in str(e), e
    _write_npz(d, 'xc_cup_e1_seed31_late_hlate.npz', _mk_file(3))
    _write_npz(d, 'xc_cup_e1_seed31_late_hearly.npz',
               _mk_file(3, g_bump=1.0))
    try:
      pairgate(d)
      raise SystemExit('selfcheck FAIL: pairgate mismatch not caught')
    except SystemExit as e:
      assert e.code == 1, e

  print('selfcheck PASS: estimand identity, six verdict branches '
        '(heads-irrelevant, head-deficit, head-harm, mixed-both, '
        'mixed-off-direction, quarantine x2 + tolerance fallback, '
        'substrate-gone), grid trips (missing side, partial run, '
        'unregistered seed), end-to-end tempdir wave read + smoke-gate '
        'trip + smoke-authenticity trips (control-copy suffix, '
        'states=5), load guards (exact version pin vs base substring, '
        'dials incl. max_steps, consumer regex/ckpt suffixes, '
        'train_seed, finiteness, oracle_all identity), pairgate '
        'ok/missing/mismatch')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='analysis_out/r3_xconsumer')
  ap.add_argument('--pairgate', metavar='LABELS_DIR')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  elif args.pairgate:
    pairgate(args.pairgate)
  elif args.labels:
    read(args)
  else:
    ap.error('--labels, --pairgate, or --selfcheck required')


if __name__ == '__main__':
  main()
