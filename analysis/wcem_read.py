"""Frozen read for the WCEM consumer wave (anti-harvest SEED cell).

Registered under PREREG_p2_cem_consumer_20260821.md (ThreeAxes item 10:
"the sharpest generality test of anti-harvest... a cheaper first cell
than the seed's current one"). ONE execution; EXECUTION IS SEQUENCED
BEHIND the Wave-1 read (the slate's calibrate -> attribute -> substrate
order).

Question: is the anti-harvest fact (TM2's planner below its own prior
mode, S3/S4) a property of that planner, or of MPC consumption per se?
A CEM planner over the FROZEN dv3 RSSM — a model family whose actor
consumer showed NO anti-harvest — answers it with no new training.

Files (labeler `--cem_consumer --w1_repeats 8`, version *_wcem):
  wcem_{cup|finger}_{e1|e4}_seed{31,32}_late.npz            (8 cells)
  wcemdup_finger_e1_seed31_late.npz, wcemdup_cup_e4_seed32_late.npz (2)

PRIMARY — CRN-paired consumer contrast (cluster = cell, n=8;
permutation-primary + BCa):
  d_cell = mean_s mean_r [g_cem_rep - g_actor_rep]
(the CEM branch and the actor branch share every repeat mark, snapshot,
and follower schedule discipline — the pairing is by construction).

PLANNER-COMPETENCE GATE (review finding 1 — the degenerate-CEM failure
mode, a null plan earning ~0 return, would otherwise be READ AS the
flagship anti-harvest branch, the exact construct-invalidity shape that
struck TM2-R3): per cell, mean_s [cem_score - q_best] — the CEM plan's
own model-predicted H-step value against the best one-step candidate Q
(same heads, same valnorm convention, directly comparable). If the
pooled mean is < 0 the planner failed its own objective and the wave
returns NO-CALL-PLANNER-INCOMPETENT: no adjudication, diagnostics
reported. The gate is conservative AGAINST the flagship (a false
NO-CALL costs the claim; it can never license it).

Branch map (adjudicated only when the competence gate passes):
  CEM-ANTI-HARVESTS           d fires negative: planning through the
                              model HURTS on dv3 too — anti-harvest
                              generalizes across families/planners
  CEM-OUTPERFORMS-ACTOR       d fires positive: anti-harvest is
                              planner/family-specific
  CEM-ACTOR-INDISTINGUISHABLE no fire; realized MDE80 printed
  NO-CALL-PLANNER-INCOMPETENT competence gate failed

SECONDARY (descriptive): CEM harvest vs the candidate mean on ODD
repeats (the exact P-W1b estimator slices — review finding 3).

Usage:
  python -m analysis.wcem_read --labels <dir> --output <dir> \
      --after_wave1 <path to the landed Wave-1 read json>
  python -m analysis.wcem_read --selfcheck
"""

import argparse
import glob
import json
import os
import re
import zlib

import numpy as np

from analysis import w1_read

ALPHA = 0.05
EXPECT_ENV_SEED = 20260824
EXPECT_REPEATS = 8
EXPECT_STATES = 200
EXPECT_HORIZON = 100
EXPECT_CELLS = 8
CEM_PINS = dict(iters=4, samples=64, horizon=6, elites=8, std=0.5)
VERSION_SUFFIX = '_wcem'
FILE_RE = re.compile(r'^wcem_(cup|finger)_(e1|e4)_seed(3[12])_late\.npz$')
DUP_RE = re.compile(r'^wcemdup_(cup|finger)_(e1|e4)_seed(3[12])_late\.npz$')
DUP_CELLS = ('wcemdup_finger_e1_seed31_late.npz',
             'wcemdup_cup_e4_seed32_late.npz')


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


def load_cell(path, dup=False):
  name = os.path.basename(path)
  z = np.load(path, allow_pickle=True)
  meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
  version = str(meta.get('labeler_version', ''))
  if not version.endswith(VERSION_SUFFIX):
    refuse(f'{name}: labeler_version {version!r} != *{VERSION_SUFFIX}')
  cem = meta.get('cem', {})
  for k, want in CEM_PINS.items():
    got = cem.get(k, None)
    if got is None or float(got) != float(want):
      refuse(f'{name}: cem.{k} {got} != registered {want}')
  w1 = meta.get('w1', {})
  for got, want, label in (
      (int(meta.get('env_seed', -1)), EXPECT_ENV_SEED, 'env_seed'),
      (int(w1.get('repeats', -1)), EXPECT_REPEATS, 'repeats'),
      (int(meta.get('states', -1)), EXPECT_STATES, 'states'),
      (int(meta.get('horizon', -1)), EXPECT_HORIZON, 'horizon')):
    if got != want:
      refuse(f'{name}: {label} {got} != registered {want}')
  m = (DUP_RE if dup else FILE_RE).match(name)
  rid = str(np.asarray(z['run_id']).reshape(-1)[0])
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  if not (dom in rid and dose in rid and f'seed{seed}' in rid):
    refuse(f'{name}: filename ({dom},{dose},{seed}) not in run_id {rid!r}')
  if not w1.get('cem'):
    refuse(f'{name}: meta.w1.cem not set — not a WCEM pass')
  e = dict(
      g=np.asarray(z['g_all_rep'], float),
      ga=np.asarray(z['g_actor_rep'], float),
      gc=np.asarray(z['g_cem_rep'], float),
      gw=np.asarray(z['g_wit_rep'], float),
      cem_act=np.asarray(z['cem_act'], float),
      cem_score=np.asarray(z['cem_score'], float),
      q_best=np.asarray(z['q_best'], float),
      rr=np.asarray(z['r_real'], float),
      dup=np.asarray(z['dup_cand'], bool))
  S, R, M = e['g'].shape
  if (S, R) != (EXPECT_STATES, EXPECT_REPEATS):
    refuse(f'{name}: g_all_rep shape {e["g"].shape}')
  if e['gc'].shape != (S, R) or e['ga'].shape != (S, R):
    refuse(f'{name}: consumer branch shapes {e["gc"].shape} '
           f'{e["ga"].shape}')
  for key in ('g', 'ga', 'gc', 'gw', 'cem_act', 'cem_score', 'q_best',
              'rr'):
    if not np.isfinite(e[key]).all():
      refuse(f'{name}: non-finite {key}')
  if float(np.max(np.abs(e['cem_act']))) > 1.0 + 1e-5:
    refuse(f'{name}: cem_act outside the action box')
  # CRN witness (review finding 6): candidate-0 re-run bit-identical
  if not np.array_equal(e['gw'], e['g'][:, :, 0]):
    refuse(f'{name}: CRN witness g_wit_rep != g_all_rep[:,:,0] — '
           'within-repeat CRN broken (INSTRUMENT-INVALID)')
  if dup and not e['dup'].all():
    refuse(f'{name}: dup file without dup rows')
  if not dup and e['dup'].any():
    refuse(f'{name}: dup rows in a main file')
  return (dom, dose, seed), e


def load_all(labels_dir):
  cells, dups = {}, []
  for path in sorted(glob.glob(os.path.join(labels_dir, 'wcem*.npz'))):
    name = os.path.basename(path)
    if FILE_RE.match(name):
      key, e = load_cell(path)
      if key in cells:
        refuse(f'duplicate cell {key}')
      cells[key] = e
    elif DUP_RE.match(name):
      if name not in DUP_CELLS:
        refuse(f'{name}: dup pass not one of the registered {DUP_CELLS}')
      dups.append((name, load_cell(path, dup=True)[1]))
    else:
      refuse(f'unregistered wcem file {name}')
  if len(cells) != EXPECT_CELLS:
    refuse(f'{len(cells)} cells != registered {EXPECT_CELLS}')
  if len(dups) != len(DUP_CELLS):
    refuse(f'{len(dups)} dup passes != registered {len(DUP_CELLS)}')
  for name, e in dups:
    if not np.all(e['g'] == e['g'][:, :, :1]):
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: candidate branches '
             'of identical actions differ')
    if float(np.max(np.ptp(e['rr'], axis=1))) != 0.0:
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: r_real varies '
             'across identical candidates')
  return cells


def _stat(vals, cl, b=None):
  b = b or w1_read.B_BOOT
  point, ci = w1_read.bca(np.asarray(vals, float), cl, b=b)
  return dict(point=point, ci=list(ci),
              perm_p=w1_read.perm_p(vals, b=b), n_cells=len(vals))


def run_read(labels_dir, output, after_wave1=None, b=None):
  # sequencing gate (review finding 4): execution is registered as
  # conditional on the Wave-1 read having landed
  if not (after_wave1 and os.path.isfile(after_wave1)):
    refuse('Wave-1 read not landed — pass --after_wave1 <path to the '
           'executed p2_wave1 read json> (PREREG_p2_cem_consumer '
           'sequences this wave behind it)')
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, 'wcem_read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  cells = load_all(labels_dir)
  cl = ['%s_%s_%d' % k for k in cells]
  d = [float(np.mean(e['gc'] - e['ga'])) for e in cells.values()]
  harv = [float(np.mean(e['gc'][:, 1::2].mean(1)
                        - e['g'][:, 1::2].mean(1).mean(1)))
          for e in cells.values()]
  comp = [float(np.mean(e['cem_score'] - e['q_best']))
          for e in cells.values()]
  blk = _stat(d, cl, b=b)
  fires_pos = blk['ci'][0] > 0 and blk['perm_p'] < ALPHA
  fires_neg = blk['ci'][1] < 0 and blk['perm_p'] < ALPHA
  sd = float(np.std(d, ddof=1))
  competent = float(np.mean(comp)) >= 0.0
  out = dict(
      n_cells=len(cells),
      cem_minus_actor=blk,
      cem_harvest_vs_candidates=_stat(harv, cl, b=b),
      planner_competence=dict(
          per_cell=comp, pooled=float(np.mean(comp)),
          gate='PASS' if competent else 'FAIL'),
      realized_mde80=float(2.8 * sd / np.sqrt(len(d))),
      verdict=('NO-CALL-PLANNER-INCOMPETENT' if not competent else
               'CEM-ANTI-HARVESTS' if fires_neg else
               'CEM-OUTPERFORMS-ACTOR' if fires_pos else
               'CEM-ACTOR-INDISTINGUISHABLE'))
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _write_cell(tmp, name, edge=0.0, comp=0.5, dup=False, run_id=None,
                meta_over=None, cem_over=None, break_witness=False):
  S, R, M, A = EXPECT_STATES, EXPECT_REPEATS, 8, 2
  m = (DUP_RE if dup else FILE_RE).match(name)
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  rng = np.random.default_rng(zlib.crc32(name.encode()))
  g = rng.normal(0, 1.0, (S, R, M)).astype(np.float32)
  if dup:
    g = np.repeat(g[:, :, :1], M, 2)
  ga = rng.normal(0, 1.0, (S, R)).astype(np.float32)
  gc = (ga + edge + rng.normal(0, 0.3, (S, R))).astype(np.float32)
  gw = g[:, :, 0].copy()
  if break_witness:
    gw = gw + 1.0
  q_best = rng.normal(3.0, 0.2, S).astype(np.float32)
  cem_score = (q_best + comp
               + rng.normal(0, 0.1, S)).astype(np.float32)
  rr = np.zeros((S, M), np.float32)
  cem = dict(CEM_PINS)
  cem.update(cem_over or {})
  meta = dict(labeler_version='d1fix_20260724_wcem',
              env_seed=EXPECT_ENV_SEED, states=S, horizon=EXPECT_HORIZON,
              w1=dict(repeats=R, cem=True), cem=cem)
  meta.update(meta_over or {})
  np.savez_compressed(
      os.path.join(tmp, name), meta=json.dumps(meta),
      g_all_rep=g, g_actor_rep=ga, g_cem_rep=gc, g_wit_rep=gw,
      cem_act=rng.uniform(-1, 1, (S, A)).astype(np.float32),
      cem_score=cem_score, q_best=q_best,
      r_real=rr, m_now=np.zeros(S, np.int64),
      dup_cand=np.full(S, dup),
      run_id=np.array([run_id or f'r3_{dom}_{dose}_seed{seed}'] * S))


def _write_suite(tmp, edge=0.0, comp=0.5):
  for dom in ('cup', 'finger'):
    for dose in ('e1', 'e4'):
      for seed in (31, 32):
        _write_cell(tmp, f'wcem_{dom}_{dose}_seed{seed}_late.npz',
                    edge=edge, comp=comp)
  for name in DUP_CELLS:
    _write_cell(tmp, name, dup=True)


def selfcheck():
  import tempfile

  B = 500

  def _w1json(tmp):
    p = os.path.join(tmp, 'p2wave1_read.json')
    with open(p, 'w') as f:
      f.write('{}')
    return p

  for edge, want in ((-0.5, 'CEM-ANTI-HARVESTS'),
                     (0.5, 'CEM-OUTPERFORMS-ACTOR'),
                     (0.0, 'CEM-ACTOR-INDISTINGUISHABLE')):
    with tempfile.TemporaryDirectory() as tmp:
      _write_suite(tmp, edge=edge)
      out = run_read(tmp, os.path.join(tmp, 'out'),
                     after_wave1=_w1json(tmp), b=B)
      assert out['verdict'] == want, (edge, out['verdict'])
      assert out['planner_competence']['gate'] == 'PASS'
  # planner-competence gate (review finding 1): an incompetent CEM
  # forces NO-CALL even when the contrast would fire the flagship
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp, edge=-0.5, comp=-0.5)
    out = run_read(tmp, os.path.join(tmp, 'out'),
                   after_wave1=_w1json(tmp), b=B)
    assert out['verdict'] == 'NO-CALL-PLANNER-INCOMPETENT', out['verdict']
    assert out['planner_competence']['gate'] == 'FAIL'
  # sequencing gate (review finding 4)
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    try:
      run_read(tmp, os.path.join(tmp, 'out'), after_wave1=None, b=B)
      raise AssertionError('sequencing gate failed to trip')
    except SystemExit as e:
      assert 'Wave-1 read not landed' in str(e), e
  # ONE-read guard
  with tempfile.TemporaryDirectory() as tmp:
    outdir = os.path.join(tmp, 'out')
    os.makedirs(outdir)
    open(os.path.join(outdir, 'wcem_read.json'), 'w').close()
    try:
      run_read(tmp, outdir, after_wave1=_w1json(tmp), b=B)
      raise AssertionError('ONE-read guard failed to trip')
    except SystemExit as e:
      assert 'ONE read execution' in str(e), e

  def expect_refusal(marker, **kw):
    with tempfile.TemporaryDirectory() as tmp:
      _write_suite(tmp)
      _write_cell(tmp, 'wcem_cup_e1_seed31_late.npz', **kw)
      try:
        run_read(tmp, os.path.join(tmp, 'out'),
                 after_wave1=_w1json(tmp), b=B)
        raise AssertionError(f'refusal not tripped: {marker}')
      except SystemExit as e:
        assert marker in str(e), (marker, str(e))

  expect_refusal('cem.iters', cem_over=dict(iters=1))
  expect_refusal('env_seed', meta_over=dict(env_seed=1))
  expect_refusal('labeler_version',
                 meta_over=dict(labeler_version='d1fix_20260724_w1'))
  expect_refusal('run_id', run_id='r3_finger_e1_seed31')
  expect_refusal('CRN witness', break_witness=True)
  expect_refusal('w1.cem', meta_over=dict(w1=dict(repeats=8)))
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    os.remove(os.path.join(tmp, 'wcem_cup_e1_seed31_late.npz'))
    try:
      run_read(tmp, os.path.join(tmp, 'out'),
               after_wave1=_w1json(tmp), b=B)
      raise AssertionError('cell-count refusal failed to trip')
    except SystemExit as e:
      assert 'cells != registered' in str(e), e
  # dup violation
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    fix = os.path.join(tmp, DUP_CELLS[0])
    z = dict(np.load(fix, allow_pickle=True))
    g = np.array(z['g_all_rep'])
    g[:, :, 1] += 1.0
    z['g_all_rep'] = g
    np.savez_compressed(fix, **z)
    try:
      run_read(tmp, os.path.join(tmp, 'out'),
               after_wave1=_w1json(tmp), b=B)
      raise AssertionError('dup violation failed to trip')
    except SystemExit as e:
      assert 'DUPLICATE-NULL VIOLATION' in str(e), e
  print('SELFCHECK PASS (wcem_read: edge -0.5/+0.5/0 -> ANTI-HARVESTS/'
        'OUTPERFORMS/INDISTINGUISHABLE; incompetent CEM forces NO-CALL '
        'even on a firing contrast; sequencing gate; ONE-read guard; '
        'refusals: cem pins, env_seed, version, run_id identity, CRN '
        'witness, w1.cem pin, cell count, duplicate-null violation)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels')
  p.add_argument('--output')
  p.add_argument('--after_wave1',
                 help='path to the executed p2_wave1 read json (the '
                      'registered sequencing gate)')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.labels and args.output):
    p.error('--labels and --output required (or --selfcheck)')
  run_read(args.labels, args.output, after_wave1=args.after_wave1)


if __name__ == '__main__':
  main()
