"""Frozen read for the R3 cross-checkpoint consumer wave (Paper 2).

AMENDED 2026-08-02 by prereg/PREREG_r3_xc_amend1_20260802.md (parent
registration prereg/PREREG_r3_amend2_20260730.md; parent reader NEVER
EXECUTED — no xc wave label exists). The parent's two-pass design
paired a control pass against a swapped pass across separate labeler
invocations with a g_all atol-1e-5 gate; the pre-wave determinism
probe (artifacts/r3_xc_detprobe_20260801/) showed cross-invocation
g_all drift up to 95.0 with exact state identity, so that pairing is
UNMEASURABLE on this substrate and the wave was never submitted. The
amended wave computes BOTH choosers inside ONE labeler invocation
(--dual_chooser, labeler version d1fix_20260724_xc2, EXACT pin):
control (eval-own heads: m_now/m_real) and shadow (overlaid consumer
heads: m_now_x/m_real_x) on the same states, the same candidate set,
and one g_all — no float is ever compared across invocations.

Files: <labels_dir>/xc_{dom}_{dose}_seed{s}_{evalmat}_dual.npz
  dom in {cup, finger}, dose in {e1, e4}, seeds 31-38, evalmat = the
  evaluated checkpoint's maturity; the consumer checkpoint is REQUIRED
  to be the OTHER maturity of the same run (enforced from meta path
  suffixes — a control-configured dual file cannot enter the wave;
  the labeler separately refuses byte-identical head sets). 2 sides
  x 32 runs = 64 files, all-or-nothing (registered existence gate).
  Labeler seed 1 (fresh states; every contrast is within-pass).

Estimand per state (within one pass, one g_all):
  achieved(ctrl)   = g_all[m_real]   - g_all[m_now]
  achieved(shadow) = g_all[m_real_x] - g_all[m_now_x]
  d_pair = achieved(shadow) - achieved(ctrl)
Primaries (run-clustered percentile bootstrap, B=10K, rng 0,
cluster = training run, 32 clusters) — SAME quantities as the parent,
now realized within-pass:
  P-XC1 (late eval)  per-run mean d_pair with consumer=early;
         CI entirely < 0 => immature heads HURT on the late side.
  P-XC2 (early eval) per-run mean d_pair with consumer=late;
         CI entirely > 0 => mature heads RESCUE the early side.
Registered prediction (from the committed P-R3c null +0.014
[-0.177,+0.215]): both straddle 0 => HEADS-IRRELEVANT.

Verdicts: HEADS-IRRELEVANT / HEAD-DEFICIT (P-XC2 fires +) / HEAD-HARM
(P-XC1 fires -) / MIXED (both registered directions fire, or any CI
excursion in an unregistered direction) / QUARANTINE (within-pass
integrity violated: g_now != g_all[m_now], candidate drift beyond the
labeler's DUAL_CANDS_ATOL, chooser index out of range, or non-finite
label content) / SUBSTRATE-GONE (existence gate: driver marker).

Usage:
  python -m analysis.r3_xconsumer_read --labels <dir> --output <dir>
  python -m analysis.r3_xconsumer_read --dualgate <dir>   (value-blind
      pre-wave integrity gate on the ONE registered full-dial dual
      pass: meta pins, finiteness, identities, chooser bounds,
      candidate-drift bound; NO estimand — d_pair and every mean of
      achieved are never computed)
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

XC_VERSION = 'd1fix_20260724_xc2'
assert XC_VERSION == LABELER_VERSION + '_xc2', XC_VERSION
CONSUMER_REGEX = r'^(rew|con|valens\d+)/'
DUAL_CANDS_ATOL = 1e-4
# Machine pin, not a comment: the reader's bound must be the labeler's
# (d0.oracle_labels has no module-level jax import, so this is free).
from d0.oracle_labels import DUAL_CANDS_ATOL as _LABELER_ATOL
assert DUAL_CANDS_ATOL == _LABELER_ATOL, (DUAL_CANDS_ATOL, _LABELER_ATOL)
CORE_DOMAINS = ('cup', 'finger')
N_STATES = 200
N_CANDS = 8
FILE_XC2 = re.compile(
    r'^xc_(cup|finger)_(e1|e4)_seed(\d+)_(early|late)_dual\.npz$')
# Registered dial identity, enforced from every file's meta — a
# mis-dialed manual retry must trip, not bias.
DIALS = dict(states=200, horizon=100, label_every=25, seed=1,
             actions=8, rollouts=16, ref_stride=5, mass_scale=1.0,
             behavior_checkpoint='', consumer_regex=CONSUMER_REGEX,
             max_steps=1000, dual_chooser=True,
             dual_cands_atol=DUAL_CANDS_ATOL,
             dual_allow_identical=False)
# Registered smoke gate: 2 dual smokes (5 states) on cup e4 seed31 —
# one per eval side; each dual pass smokes BOTH choosers at once, so
# two files cover both swap directions and both controls.
SMOKES = tuple(f'xc_cup_e4_seed31_{mat}_dual_smoke.npz' for mat in MATS)
# Registered pre-wave gate: ONE full-dial dual pass, machine-checked
# via --dualgate BEFORE the remaining 63.
DUALGATE_CELL = ('cup', 'e1', 31, 'late')
SUBSTRATE_MARKER = 'SUBSTRATE_GONE'


def _swap(mat):
  return 'early' if mat == 'late' else 'late'


def per_state_d_pair(g_all, m_real, m_now, m_real_x, m_now_x):
  """[g_all[m_real_x]-g_all[m_now_x]] - [g_all[m_real]-g_all[m_now]]
  per labeled state, all indices from ONE pass, one g_all."""
  g_all = np.asarray(g_all, float)
  idx = np.arange(len(m_real))
  ach_c = g_all[idx, np.asarray(m_real, int)] \
      - g_all[idx, np.asarray(m_now, int)]
  ach_x = g_all[idx, np.asarray(m_real_x, int)] \
      - g_all[idx, np.asarray(m_now_x, int)]
  return ach_x - ach_c


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
    # filename claims (eval ckpt + OTHER-maturity consumer) and 5
    # states — two copies of one side would otherwise pass the gate
    # and the wave would launch with one swap direction never smoked.
    sm = re.match(
        r'^xc_(cup|finger)_(e1|e4)_seed(\d+)_(early|late)'
        r'_dual_smoke\.npz$', smoke)
    run = f'r3_{sm.group(1)}_{sm.group(2)}_seed{int(sm.group(3))}'
    _assert_ckpt(smoke, 'checkpoint', md, run, sm.group(4))
    _assert_ckpt(smoke, 'consumer_checkpoint', md, run,
                 _swap(sm.group(4)))
    assert md.get('states') == 5, (
        f'{smoke}: smoke states={md.get("states")!r} != 5')
    assert 'm_real_x' in z and 'm_now_x' in z, (
        f'{smoke}: dual smoke lacks shadow-chooser arrays')
  assert not missing, (
      'registered xc dual smoke gate not satisfied; missing: '
      + ', '.join(missing))


def load_xc(labels_dir):
  files = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = FILE_XC2.match(name)
    if not m:
      continue
    dom, dose = m.group(1), m.group(2)
    seed, emat = int(m.group(3)), m.group(4)
    z = np.load(path, allow_pickle=True)
    md = json.loads(str(z['meta']))
    assert md.get('labeler_version') == XC_VERSION, (
        f'{name}: labeler_version {md.get("labeler_version")!r} != '
        f'{XC_VERSION!r} (exact pin; neither the base d1fix string nor '
        'a two-pass _xc1 file is an amended dual label)')
    for k, v in DIALS.items():
      assert md.get(k) == v, (
          f'{name}: dial {k}={md.get(k)!r} != registered {v!r}')
    assert md.get('train_seed') == seed, (
        f"{name}: meta train_seed={md.get('train_seed')} != filename {seed}")
    run = f'r3_{dom}_{dose}_seed{seed}'
    _assert_ckpt(name, 'checkpoint', md, run, emat)
    # The consumer MUST be the other maturity of the same run: a
    # control-configured dual file (consumer == eval) cannot enter.
    _assert_ckpt(name, 'consumer_checkpoint', md, run, _swap(emat))
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    m_now = np.asarray(z['m_now'], int)
    m_real_x = np.asarray(z['m_real_x'], int)
    m_now_x = np.asarray(z['m_now_x'], int)
    validate_arrays(name, g_all, g_now, m_real)
    assert g_all.shape == (N_STATES, N_CANDS), (
        f'{name}: g_all shape {g_all.shape} != registered '
        f'({N_STATES}, {N_CANDS})')
    scores_x = np.asarray(z['real_scores_x'], float)
    qfull_x = np.asarray(z['qfull_x'], float)
    assert scores_x.shape == (N_STATES, N_CANDS), (
        f'{name}: real_scores_x shape {scores_x.shape}')
    assert qfull_x.shape[0] == N_STATES and qfull_x.shape[-1] == N_CANDS, (
        f'{name}: qfull_x shape {qfull_x.shape}')
    key = (dom, dose, seed, emat)
    assert key not in files, f'duplicate cell from {name}'
    files[key] = dict(
        g_all=g_all, g_now=g_now,
        m_real=m_real, m_now=m_now, m_real_x=m_real_x, m_now_x=m_now_x,
        cands_max_absdiff=float(np.max(np.asarray(
            z['cands_max_absdiff'], float))),
        scores_finite=bool(np.isfinite(scores_x).all()
                           and np.isfinite(qfull_x).all()),
        opp=float(np.mean(np.max(g_all, 1) - g_now)))
  return files


def check_grid(files):
  bad = sorted({k[2] for k in files} - set(EXPECT_SEEDS))
  assert not bad, f'unregistered seeds present: {bad}'
  missing = []
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for emat in MATS:
          if (dom, dose, seed, emat) not in files:
            missing.append(f'{dom}/{dose}/seed{seed}/{emat}')
  assert not missing, (
      'registered wave incomplete (all-or-nothing existence gate); '
      f'{len(missing)} passes missing, e.g. ' + ', '.join(missing[:6]))
  assert len(files) == 64, f'unexpected extra cells ({len(files)})'
  return sorted({k[:3] for k in files})


def integrity_violations(files):
  """Value-blind within-pass integrity class (registered QUARANTINE
  triggers): oracle_all g_now identity, candidate-drift bound, shadow
  chooser index range, finiteness of shadow chooser content."""
  viols = []
  for key in sorted(files):
    blk = files[key]
    tag = '/'.join(map(str, key))
    if not np.allclose(blk['g_now'],
                       blk['g_all'][np.arange(len(blk['m_now'])),
                                    blk['m_now']], atol=1e-5):
      viols.append(tag + ': g_now != g_all[m_now] (oracle_all identity)')
    if blk['cands_max_absdiff'] > DUAL_CANDS_ATOL:
      viols.append(tag + f": candidate drift {blk['cands_max_absdiff']:.3e}"
                   f' > {DUAL_CANDS_ATOL}')
    for f in ('m_real', 'm_now', 'm_real_x', 'm_now_x'):
      if blk[f].min() < 0 or blk[f].max() >= N_CANDS:
        viols.append(tag + f': {f} outside candidate range')
    if not blk['scores_finite']:
      viols.append(tag + ': non-finite shadow chooser content '
                   '(qfull_x/real_scores_x)')
  return viols


def analyze(files):
  runs = check_grid(files)
  viols = integrity_violations(files)
  if viols:
    return dict(
        primaries=None,
        fires=dict(p_xc1_harm=None, p_xc2_deficit=None),
        integrity_violations=viols, n_runs=len(runs),
        verdict=('QUARANTINE: within-pass integrity violated for: '
                 + '; '.join(viols[:6])
                 + (' ...' if len(viols) > 6 else '')
                 + '. No adjudication; audit the dual label passes '
                 'before ANY use.'))

  def side_diffs(emat):
    # per-run mean of PER-STATE within-pass d_pair on this eval side
    return {r: [float(np.mean(per_state_d_pair(
        files[r + (emat,)]['g_all'], files[r + (emat,)]['m_real'],
        files[r + (emat,)]['m_now'], files[r + (emat,)]['m_real_x'],
        files[r + (emat,)]['m_now_x'])))] for r in runs}

  xc1 = _cluster_boot(side_diffs('late'))    # consumer=early on late eval
  xc2 = _cluster_boot(side_diffs('early'))   # consumer=late on early eval
  fires = dict(p_xc1_harm=bool(xc1['ci'][1] < 0),
               p_xc2_deficit=bool(xc2['ci'][0] > 0))
  off = dict(xc1_pos=bool(xc1['ci'][0] > 0), xc2_neg=bool(xc2['ci'][1] < 0))

  def disag_rate(field, emat):
    return float(np.mean([
        np.mean(files[r + (emat,)][field + '_x']
                != files[r + (emat,)][field]) for r in runs]))

  disag = {f'{emat}_{field}': disag_rate(field, emat)
           for emat in MATS for field in ('m_real', 'm_now')}
  secondary = dict(
      per_domain={dom: dict(
          xc1=_cluster_boot({r: v for r, v in side_diffs('late').items()
                             if r[0] == dom}),
          xc2=_cluster_boot({r: v for r, v in side_diffs('early').items()
                             if r[0] == dom})) for dom in CORE_DOMAINS},
      chooser_disagreement=disag,
      opportunity_level={
          emat: _cluster_boot({r: [files[r + (emat,)]['opp']]
                               for r in runs}) for emat in MATS},
      cands_max_absdiff_max=float(max(
          files[r + (emat,)]['cands_max_absdiff']
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
  if all(v == 0.0 for v in disag.values()):
    # Non-decisional honesty disclosure: byte-level chooser agreement
    # everywhere is stronger than a genuine null needs (the labeler's
    # identical-heads refusal + witness read-back should make this
    # unreachable) — flag for audit rather than silently reporting.
    verdict += (' [SUSPECT-NO-OP DISCLOSURE: shadow and control '
                'choosers agree on every state of every pass — audit '
                'the head-swap wiring before publishing this outcome.]')
  return dict(
      primaries=dict(p_xc1_late_eval=xc1, p_xc2_early_eval=xc2),
      fires=fires, off_direction=off,
      integrity_violations=[],
      thresholds=dict(dual_cands_atol=DUAL_CANDS_ATOL),
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
  assert files, f'no xc_*_dual labels under {args.labels}'
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


def dualgate(labels_dir):
  """Registered pre-wave integrity gate on the ONE full-dial dual pass
  (DUALGATE_CELL). VALUE-BLIND by construction: meta pins, array
  finiteness, the oracle_all identity, chooser index bounds, and the
  candidate-drift bound are checked; d_pair, achieved, and every mean
  of them are NEVER computed here."""
  dom, dose, seed, emat = DUALGATE_CELL
  name = f'xc_{dom}_{dose}_seed{seed}_{emat}_dual.npz'
  path = os.path.join(labels_dir, name)
  assert os.path.exists(path), f'dualgate file missing: {name}'
  z = np.load(path, allow_pickle=True)
  md = json.loads(str(z['meta']))
  problems = []
  if md.get('labeler_version') != XC_VERSION:
    problems.append(f"labeler_version {md.get('labeler_version')!r}")
  for k, v in DIALS.items():
    if md.get(k) != v:
      problems.append(f'dial {k}={md.get(k)!r} != {v!r}')
  run = f'r3_{dom}_{dose}_seed{seed}'
  for field, mat in (('checkpoint', emat),
                     ('consumer_checkpoint', _swap(emat))):
    want = f'{run}/ckpt_early' if mat == 'early' else f'{run}/ckpt'
    if not str(md.get(field, '')).rstrip('/').endswith(want):
      problems.append(f'{field} suffix != {want}')
  g_all = np.asarray(z['g_all'], float)
  g_now = np.asarray(z['g_now'], float)
  if not (np.isfinite(g_all).all() and np.isfinite(g_now).all()
          and np.isfinite(np.asarray(z['qfull_x'], float)).all()
          and np.isfinite(np.asarray(z['real_scores_x'], float)).all()):
    problems.append('non-finite label content')
  if g_all.shape != (N_STATES, N_CANDS):
    problems.append(f'g_all shape {g_all.shape}')
  m_now = np.asarray(z['m_now'], int)
  if not np.allclose(g_now, g_all[np.arange(len(m_now)), m_now],
                     atol=1e-5):
    problems.append('g_now != g_all[m_now]')
  for f in ('m_real', 'm_now', 'm_real_x', 'm_now_x'):
    arr = np.asarray(z[f], int)
    if arr.min() < 0 or arr.max() >= N_CANDS:
      problems.append(f'{f} outside candidate range')
  cdiff = float(np.max(np.asarray(z['cands_max_absdiff'], float)))
  if cdiff > DUAL_CANDS_ATOL:
    problems.append(f'candidate drift {cdiff:.3e} > {DUAL_CANDS_ATOL}')
  if problems:
    print('DUALGATE FAIL for ' + name + ': ' + '; '.join(problems)
          + '. Do NOT run the remaining passes; audit the dual labeler '
          'first.')
    raise SystemExit(1)
  print(f'DUALGATE OK: {name} — meta pins, finiteness, oracle_all '
        f'identity, chooser bounds, candidate drift {cdiff:.3e} <= '
        f'{DUAL_CANDS_ATOL} (value-blind: no estimand computed)')


# --------------------------------------------------------------------------
# selfcheck (synthetic fixtures only; no real label path is ever named)
# --------------------------------------------------------------------------

G_ROW = np.round(np.arange(N_CANDS) * 0.1, 6)  # G[m] = 0.1*m, all states


def _mk_file(m_real_val, m_now_val=0, m_real_x_val=None, m_now_x_val=None,
             g_bump=0.0, episode=None, step=None, cdiff=0.0,
             g_now_break=0.0, scores_val=1.0):
  g_all = np.tile(G_ROW, (N_STATES, 1)) + g_bump
  m_real = np.full(N_STATES, m_real_val, int)
  m_now = np.full(N_STATES, m_now_val, int)
  m_real_x = np.full(N_STATES, m_real_val if m_real_x_val is None
                     else m_real_x_val, int)
  m_now_x = np.full(N_STATES, m_now_val if m_now_x_val is None
                    else m_now_x_val, int)
  g_now = g_all[np.arange(N_STATES), m_now] + g_now_break
  episode = np.zeros(N_STATES, int) if episode is None else episode
  step = (np.arange(N_STATES) * 25 + 25) if step is None else step
  return dict(g_all=g_all, g_now=g_now,
              m_real=m_real, m_now=m_now,
              m_real_x=m_real_x, m_now_x=m_now_x,
              cands_max_absdiff=float(cdiff),
              scores_finite=bool(np.isfinite(scores_val)),
              _scores_val=float(scores_val),
              episode=episode, step=step,
              opp=float(np.mean(np.max(g_all, 1) - g_now)))


def _synth_files(late_x_m=3, early_x_m=3, ctrl_m=3, **kw):
  """Control choosers pick (m_real=ctrl_m, m_now=0) in every pass; the
  shadow chooser picks the side's x candidate (m_now_x=0). d_pair is
  exactly 0.1*(x_m - ctrl_m) per state on that side."""
  files = {}
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for emat in MATS:
          xm = late_x_m if emat == 'late' else early_x_m
          files[(dom, dose, seed, emat)] = _mk_file(
              ctrl_m, m_real_x_val=xm, **kw)
  return files


def _write_npz(d, name, blk, version=XC_VERSION, **meta_over):
  m = FILE_XC2.match(name.replace('_dual_smoke.npz', '_dual.npz'))
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  emat = m.group(4)
  run = f'/x/r3_{dom}_{dose}_seed{seed}'
  hmat = _swap(emat)
  md = dict(DIALS, train_seed=seed, labeler_version=version,
            checkpoint=run + ('/ckpt_early' if emat == 'early' else '/ckpt'),
            consumer_checkpoint=run + (
                '/ckpt_early' if hmat == 'early' else '/ckpt'))
  md.update(meta_over)
  n = len(blk['m_now'])
  sv = blk.get('_scores_val', 1.0)
  np.savez(os.path.join(d, name), meta=json.dumps(md),
           g_all=blk['g_all'].astype(np.float32), g_now=blk['g_now'],
           m_real=blk['m_real'], m_now=blk['m_now'],
           m_real_x=blk['m_real_x'], m_now_x=blk['m_now_x'],
           qfull_x=np.full((n, 2, N_CANDS), sv, np.float32),
           real_scores_x=np.full((n, N_CANDS), sv, np.float32),
           cands_max_absdiff=np.full(n, blk['cands_max_absdiff'],
                                     np.float32),
           episode=blk['episode'], step=blk['step'])


def selfcheck(args):
  import tempfile

  # Estimand identity on a hand case: G=[[0,2,1],[1,3,1]],
  # ctrl (m_real=1, m_now=0) -> ach_c=[2, 2];
  # shadow (m_real_x=2, m_now_x=1) -> ach_x=[1-2, 1-3]=[-1, -2];
  # d_pair = [-3, -4].
  g = np.array([[0.0, 2.0, 1.0], [1.0, 3.0, 1.0]])
  d = per_state_d_pair(g, np.array([1, 1]), np.array([0, 0]),
                       np.array([2, 2]), np.array([1, 1]))
  assert np.allclose(d, [-3.0, -4.0]), d

  # Verdict branch 1: HEADS-IRRELEVANT (all diffs exactly 0).
  res = analyze(_synth_files())
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=False)
  assert res['verdict'].startswith('HEADS-IRRELEVANT'), res['verdict']
  assert res['integrity_violations'] == []
  # ... exact-zero disagreement everywhere carries the SUSPECT-NO-OP
  # disclosure (non-decisional honesty valve)
  assert 'SUSPECT-NO-OP' in res['verdict']
  assert res['secondary']['chooser_disagreement']['late_m_real'] == 0.0
  assert analyze(_synth_files()) == res          # determinism

  # Branch 2: HEAD-DEFICIT (early shadow picks a better candidate).
  res = analyze(_synth_files(early_x_m=5))
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=True)
  assert res['verdict'].startswith('HEAD-DEFICIT'), res['verdict']
  blk = res['primaries']['p_xc2_early_eval']
  assert abs(blk['point'] - 0.2) < 1e-9, blk
  assert res['secondary']['chooser_disagreement']['early_m_real'] == 1.0
  assert 'SUSPECT-NO-OP' not in res['verdict']

  # Branch 3: HEAD-HARM (late shadow picks a worse candidate).
  res = analyze(_synth_files(late_x_m=1))
  assert res['fires'] == dict(p_xc1_harm=True, p_xc2_deficit=False)
  assert res['verdict'].startswith('HEAD-HARM'), res['verdict']
  assert abs(res['primaries']['p_xc1_late_eval']['point'] + 0.2) < 1e-9

  # Branch 4: MIXED, both registered directions fire.
  res = analyze(_synth_files(late_x_m=1, early_x_m=5))
  assert res['fires'] == dict(p_xc1_harm=True, p_xc2_deficit=True)
  assert res['verdict'].startswith('MIXED: both'), res['verdict']

  # Branch 5: MIXED, off-registered direction (late shadow HELPS).
  res = analyze(_synth_files(late_x_m=5))
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=False)
  assert res['off_direction']['xc1_pos']
  assert res['verdict'].startswith('MIXED: a CI excursion'), res['verdict']

  # m_now_x term is load-bearing: shadow plugin flip alone moves d_pair
  # by -0.1*(m_now_x) even with m_real_x == m_real.
  files = _synth_files()
  key = ('cup', 'e1', 31, 'late')
  files[key] = _mk_file(3, m_real_x_val=3, m_now_x_val=2)
  res = analyze(files)
  one = per_state_d_pair(files[key]['g_all'], files[key]['m_real'],
                         files[key]['m_now'], files[key]['m_real_x'],
                         files[key]['m_now_x'])
  assert np.allclose(one, -0.2), one[:3]   # ach_x = 0.3-0.2 vs ach_c 0.3
  assert res['primaries']['p_xc1_late_eval']['point'] < 0
  # Reviewer M1: the fire rules are CI-ENDPOINT rules. One negative run
  # among 32 gives a nondegenerate xc1 CI = [<0, 0.0]; the registered
  # rule (ci[1] < 0) must NOT fire — an endpoint typo (ci[0] < 0)
  # would.
  blk = res['primaries']['p_xc1_late_eval']
  assert blk['ci'][0] < 0 <= blk['ci'][1], blk
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=False)
  # ... mirrored on the early side: one positive run among 32 gives
  # xc2 CI = [0.0, >0]; the registered rule (ci[0] > 0) must NOT fire.
  files = _synth_files()
  files[('cup', 'e1', 31, 'early')] = _mk_file(3, m_real_x_val=5)
  res = analyze(files)
  blk = res['primaries']['p_xc2_early_eval']
  assert blk['point'] > 0 and blk['ci'][0] <= 0 < blk['ci'][1], blk
  assert res['fires'] == dict(p_xc1_harm=False, p_xc2_deficit=False)

  # QUARANTINE class: g_now identity break / candidate drift / range /
  # non-finite shadow scores.
  for kw, frag in ((dict(g_now_break=0.5), 'g_now != g_all[m_now]'),
                   (dict(cdiff=1e-3), 'candidate drift'),
                   (dict(m_now_x_val=N_CANDS), 'outside candidate range'),
                   (dict(scores_val=np.nan), 'non-finite')):
    files = _synth_files()
    files[key] = _mk_file(3, **kw)
    res = analyze(files)
    assert res['verdict'].startswith('QUARANTINE'), (kw, res['verdict'])
    assert res['fires'] == dict(p_xc1_harm=None, p_xc2_deficit=None)
    assert any(frag in v for v in res['integrity_violations']), (
        frag, res['integrity_violations'])

  # Grid guards: missing side / unregistered seed.
  files = _synth_files()
  del files[('cup', 'e1', 31, 'late')]
  try:
    analyze(files)
    raise SystemExit('selfcheck FAIL: missing side not caught')
  except AssertionError as e:
    assert 'all-or-nothing' in str(e), e
  files = _synth_files()
  files[('cup', 'e1', 99, 'late')] = _mk_file(3)
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
    for key, blk in _synth_files(early_x_m=5).items():
      dom, dose, seed, emat = key
      _write_npz(labels, f'xc_{dom}_{dose}_seed{seed}_{emat}_dual.npz',
                 blk)
    # smoke gate must trip before any smoke exists
    try:
      read(ns(labels=labels, output=outdir))
      raise SystemExit('selfcheck FAIL: missing smoke gate not caught')
    except AssertionError as e:
      assert 'smoke gate' in str(e), e
    for smoke in SMOKES:
      blk = _mk_file(3)
      small = {k: (v[:5] if isinstance(v, np.ndarray) else v)
               for k, v in blk.items()}
      _write_npz(labels, smoke, small, states=5)
    read(ns(labels=labels, output=outdir))
    with open(os.path.join(outdir, 'r3_xconsumer.json')) as f:
      res = json.load(f)
    assert res['verdict'].startswith('HEAD-DEFICIT'), res['verdict']
    assert abs(res['primaries']['p_xc2_early_eval']['point'] - 0.2) < 1e-6
    # dualgate on the wave's registered cell passes, value-blind
    dualgate(labels)
    # version pin: a two-pass _xc1 file or a plain d1fix file must trip
    for badver in ('d1fix_20260724_xc1', 'd1fix_20260724'):
      bad = os.path.join(d, 'badver')
      os.makedirs(bad, exist_ok=True)
      _write_npz(bad, 'xc_cup_e1_seed31_late_dual.npz', _mk_file(3),
                 version=badver)
      try:
        load_xc(bad)
        raise SystemExit(f'selfcheck FAIL: version {badver} not caught')
      except AssertionError as e:
        assert 'labeler_version' in str(e), e
    # dial pin: wrong labeler seed trips
    bad = os.path.join(d, 'badseed')
    os.makedirs(bad)
    _write_npz(bad, 'xc_cup_e1_seed31_late_dual.npz', _mk_file(3), seed=0)
    try:
      load_xc(bad)
      raise SystemExit('selfcheck FAIL: mis-dialed seed not caught')
    except AssertionError as e:
      assert 'dial' in str(e), e
    # missing dual_chooser meta (a non-dual overlay pass renamed) trips
    bad = os.path.join(d, 'baddual')
    os.makedirs(bad)
    _write_npz(bad, 'xc_cup_e1_seed31_late_dual.npz', _mk_file(3),
               dual_chooser=None)
    try:
      load_xc(bad)
      raise SystemExit('selfcheck FAIL: missing dual_chooser not caught')
    except AssertionError as e:
      assert 'dual_chooser' in str(e), e
    # consumer-suffix pin: consumer == eval maturity (control config)
    bad = os.path.join(d, 'badcons')
    os.makedirs(bad)
    _write_npz(bad, 'xc_cup_e1_seed31_late_dual.npz', _mk_file(3),
               consumer_checkpoint='/x/r3_cup_e1_seed31/ckpt')
    try:
      load_xc(bad)
      raise SystemExit('selfcheck FAIL: control-config consumer not caught')
    except AssertionError as e:
      assert 'consumer_checkpoint' in str(e), e
    # smoke authenticity: wrong state count trips
    bad_smoke = os.path.join(d, 'badsmoke')
    os.makedirs(bad_smoke)
    for key, blk in _synth_files().items():
      dom, dose, seed, emat = key
      _write_npz(bad_smoke,
                 f'xc_{dom}_{dose}_seed{seed}_{emat}_dual.npz', blk)
    for smoke in SMOKES:
      _write_npz(bad_smoke, smoke, _mk_file(3), states=7)
    try:
      read(ns(labels=bad_smoke, output=outdir))
      raise SystemExit('selfcheck FAIL: smoke state count not caught')
    except AssertionError as e:
      assert 'states' in str(e), e
    # dualgate trips on candidate drift (value-blind fail path)
    badgate = os.path.join(d, 'badgate')
    os.makedirs(badgate)
    _write_npz(badgate, 'xc_cup_e1_seed31_late_dual.npz',
               _mk_file(3, cdiff=1e-2))
    try:
      dualgate(badgate)
      raise SystemExit('selfcheck FAIL: dualgate drift not caught')
    except SystemExit as e:
      assert e.code == 1, e
    # substrate marker short-circuits to the registered outcome
    marker_dir = os.path.join(d, 'gone')
    os.makedirs(marker_dir)
    open(os.path.join(marker_dir, SUBSTRATE_MARKER), 'w').close()
    read(ns(labels=marker_dir, output=os.path.join(d, 'gone_out')))
    with open(os.path.join(d, 'gone_out', 'r3_xconsumer.json')) as f:
      res = json.load(f)
    assert res['verdict'].startswith('SUBSTRATE-GONE'), res['verdict']

  print('selfcheck PASS (Amendment 1, dual-chooser): estimand identity '
        'closed-form (incl. the m_now_x term), five verdict branches + '
        'off-direction map, SUSPECT-NO-OP disclosure on exact-zero '
        'disagreement, QUARANTINE on g_now identity / candidate drift / '
        'index range / non-finite shadow content, all-or-nothing 64-grid '
        '+ unregistered-seed guards, e2e npz read with closed-form '
        'primary, exact _xc2 version pin (plain d1fix and two-pass _xc1 '
        'both trip), dial + dual_chooser meta + control-config consumer '
        'pins, dual smoke authenticity, value-blind dualgate pass/fail, '
        'substrate marker, determinism)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='analysis_out/r3_xconsumer')
  ap.add_argument('--dualgate', default='',
                  help='run the value-blind pre-wave integrity gate on '
                       'the registered full-dial dual pass in this '
                       'labels dir, then exit')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  elif args.dualgate:
    dualgate(args.dualgate)
  else:
    assert args.labels, '--labels required (or --selfcheck/--dualgate)'
    read(args)


if __name__ == '__main__':
  main()
