"""Frozen read for the WDC diverse-candidate wave (Paper 2).

Registered under PREREG_p2_dcand_20260821.md. ONE execution. The board's
acknowledged gap (ThreeAxes §5): the flatness verdict is a property of
(environment x trained policy x candidate generator) — converged actors
sample near-duplicate candidates, so part of the measured flatness may
be policy convergence, not substrate poverty. This wave separates them.

Files (labeler `--dcand_uniform 4`, version *_wdc):
  wdc_{cup|finger}_{e1|e4}_seed{31..34}_late.npz          (16 cells)
  wdcdup_finger_e1_seed31_late.npz, wdcdup_cup_e4_seed34_late.npz (2)

Per cell: g_all_rep (S,R,M) for the M=8 policy candidates; g_dcand_rep
(S,R,U) for U=4 uniform candidates under the SAME repeat marks; g_wit_rep
(S,R) candidate-0 CRN witness; dcands (S,U,A).

Estimators (cluster = cell, n=16; permutation-primary + BCa via the
frozen w1_read machinery):

  CEILING (per candidate set): per state, the ddof=1 variance-components
  estimator (the corrected form from ThreeAxes §2.2 — unbiased at
  planted truth):
    S2(s) = Var_m[mean_r g] - (1/R) * mean_m Var_r[g]
  cell ceiling = sqrt(max(mean_s S2, 0))  [G units].
  Sets: P = the 8 policy candidates; D = P + the 4 uniform candidates.

  PRIMARY: delta_ceiling = ceiling_D - ceiling_P, paired per cell.
  NOISE FLOOR (review 5.1): sqrt(max(.,0)) gives the ceiling a positive
  floor (~0.09 at realized dispersions); a matched per-cell null floor
  (candidate axis shuffled within each repeat, pinned rng) is reported
  and the above-bar conjunct is adjudicated NET of it. The sign-flip
  perm_p is suppressed on the two ceiling blocks (non-negative by
  construction, so it is structurally minimal — review 5.2).
  SECONDARY (descriptive): split-opportunity vs the CANDIDATE-0
  baseline (candidate 0 is identical in both sets, so the baseline
  cancels exactly in the paired difference; this is NOT the m_now
  baseline W1 uses) on each set, paired difference.

Branch map (BAR = 0.20, the corrected substrate ceiling bar):
  FLATNESS-IS-POLICY          delta fires positive AND pooled ceiling_D
                              net of its noise floor > BAR (the
                              substrate holds resolvable value the
                              policy generator hides)
  DIVERSITY-RAISES-SUBCEILING delta fires positive, net ceiling_D <= BAR
  GENERATOR-INSENSITIVE       delta does not fire (substrate-poverty leg
                              strengthened)
  CEILING-FALLS               delta fires negative (mechanism-anomalous;
                              reported, licenses nothing)

Usage:
  python -m analysis.wdc_read --labels <dir> --output <dir>
  python -m analysis.wdc_read --selfcheck
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
BAR = 0.20
EXPECT_ENV_SEED = 20260822
EXPECT_REPEATS = 8
EXPECT_STATES = 200
EXPECT_HORIZON = 100
EXPECT_LABEL_EVERY = 25
EXPECT_M = 8
EXPECT_UNIFORM = 4
EXPECT_CELLS = 16
# Review 5.1: sqrt(max(.,0)) gives the ceiling a POSITIVE noise floor
# (~0.09 pooled at the realized within-repeat dispersions — roughly
# half the BAR). The floor is estimated per cell by shuffling the
# candidate axis independently within each repeat (destroys any true
# between-candidate signal, preserves the noise law) and the above-bar
# conjunct is adjudicated NET of it.
FLOOR_SHUFFLES = 20
FLOOR_SEED = 20260826
VERSION_SUFFIX = '_wdc'
FILE_RE = re.compile(r'^wdc_(cup|finger)_(e1|e4)_seed(3[1-4])_late\.npz$')
DUP_RE = re.compile(r'^wdcdup_(cup|finger)_(e1|e4)_seed(3[1-4])_late\.npz$')
DUP_CELLS = ('wdcdup_finger_e1_seed31_late.npz',
             'wdcdup_cup_e4_seed34_late.npz')


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


def load_cell(path, dup=False):
  name = os.path.basename(path)
  z = np.load(path, allow_pickle=True)
  meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
  version = str(meta.get('labeler_version', ''))
  if not version.endswith(VERSION_SUFFIX):
    refuse(f'{name}: labeler_version {version!r} != *{VERSION_SUFFIX}')
  w1 = meta.get('w1', {})
  for got, want, label in (
      (int(meta.get('env_seed', -1)), EXPECT_ENV_SEED, 'env_seed'),
      (int(w1.get('repeats', -1)), EXPECT_REPEATS, 'repeats'),
      (int(w1.get('dcand_uniform', -1)), EXPECT_UNIFORM, 'dcand_uniform'),
      (int(meta.get('states', -1)), EXPECT_STATES, 'states'),
      (int(meta.get('horizon', -1)), EXPECT_HORIZON, 'horizon'),
      (int(meta.get('label_every', -1)), EXPECT_LABEL_EVERY,
       'label_every')):
    if got != want:
      refuse(f'{name}: {label} {got} != registered {want}')
  m = (DUP_RE if dup else FILE_RE).match(name)
  rid = str(np.asarray(z['run_id']).reshape(-1)[0])
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  if not (dom in rid and dose in rid and f'seed{seed}' in rid):
    refuse(f'{name}: filename ({dom},{dose},{seed}) not in run_id {rid!r}')
  e = dict(
      g=np.asarray(z['g_all_rep'], float),
      gd=np.asarray(z['g_dcand_rep'], float),
      gw=np.asarray(z['g_wit_rep'], float),
      dcands=np.asarray(z['dcands'], np.float32),
      cands=np.asarray(z['cands'], np.float32),
      mn=np.asarray(z['m_now'], int),
      ep=np.asarray(z['episode'], int),
      st=np.asarray(z['step'], int),
      dup=np.asarray(z['dup_cand'], bool))
  S, R, M = e['g'].shape
  if (S, R, M) != (EXPECT_STATES, EXPECT_REPEATS, EXPECT_M):
    refuse(f'{name}: g_all_rep shape {e["g"].shape} != registered '
           f'({EXPECT_STATES}, {EXPECT_REPEATS}, {EXPECT_M})')
  if e['gd'].shape != (S, R, EXPECT_UNIFORM):
    refuse(f'{name}: g_dcand_rep shape {e["gd"].shape}')
  for arr_name in ('g', 'gd', 'gw', 'dcands', 'cands'):
    if not np.isfinite(e[arr_name]).all():
      refuse(f'{name}: non-finite values in {arr_name}')
  # CRN witness: candidate-0 re-run bit-identical
  if not np.array_equal(e['gw'], e['g'][:, :, 0]):
    refuse(f'{name}: CRN witness g_wit_rep != g_all_rep[:,:,0] — '
           'within-repeat CRN broken (INSTRUMENT-INVALID)')
  # pinned-stream reproducibility of the uniform candidates
  if dup:
    if not e['dup'].all():
      refuse(f'{name}: dup file without dup rows')
    if not np.all(e['dcands'] == e['cands'][:, :1]):
      refuse(f'{name}: dup uniform set != plug-in candidate copies')
  else:
    if e['dup'].any():
      refuse(f'{name}: dup rows in a main file')
    A = e['dcands'].shape[2]
    for s in range(S):
      drng = np.random.default_rng(
          [EXPECT_ENV_SEED, int(e['ep'][s]), int(e['st'][s])])
      want = drng.uniform(-1.0, 1.0,
                          (EXPECT_UNIFORM, A)).astype(np.float32)
      if not np.array_equal(e['dcands'][s], want):
        refuse(f'{name}: dcands not reproducible from the pinned stream '
               f'at state {s} — wiring/provenance error')
  return (dom, dose, seed), e


def load_all(labels_dir):
  cells, dups, seen = {}, [], set()
  for path in sorted(glob.glob(os.path.join(labels_dir, 'wdc*.npz'))):
    name = os.path.basename(path)
    if FILE_RE.match(name):
      key, e = load_cell(path)
      if key in cells:
        refuse(f'duplicate cell {key}')
      cells[key] = e
      seen.add(name)
    elif DUP_RE.match(name):
      if name not in DUP_CELLS:
        refuse(f'{name}: dup pass not one of the registered {DUP_CELLS}')
      dups.append(load_cell(path, dup=True)[1])
      seen.add(name)
    else:
      refuse(f'unregistered wdc file {name}')
  if len(cells) != EXPECT_CELLS:
    refuse(f'{len(cells)} cells != registered {EXPECT_CELLS}')
  if len(dups) != len(DUP_CELLS):
    refuse(f'{len(dups)} dup passes != registered {len(DUP_CELLS)}')
  for e in dups:
    if float(np.max(np.abs(e['gd'] - e['g'][:, :, :1]))) != 0.0:
      refuse('DUPLICATE-NULL VIOLATION: diverse branches of identical '
             'actions differ')
    if not np.all(e['g'] == e['g'][:, :, :1]):
      refuse('DUPLICATE-NULL VIOLATION: policy branches differ')
  return cells


def _ceiling(garr):
  """garr (S, R, M) -> per-cell ceiling in G units (ddof=1 corrected
  variance-components estimator)."""
  S, R, M = garr.shape
  rep_means = garr.mean(1)                       # (S, M)
  between = rep_means.var(1, ddof=1)             # (S,)
  within = garr.var(1, ddof=1).mean(1)           # (S,) mean_m Var_r
  s2 = between - within / R
  return float(np.sqrt(max(float(np.mean(s2)), 0.0)))


def _ceiling_floor(garr, rng, shuffles=FLOOR_SHUFFLES):
  """Matched null floor of the ceiling estimator (review 5.1): permute
  the candidate axis independently within each repeat — any true
  between-candidate signal is destroyed, the noise law is preserved —
  and average the resulting ceiling over pinned shuffles."""
  S, R, M = garr.shape
  vals = []
  for _ in range(shuffles):
    shuffled = np.empty_like(garr)
    for r in range(R):
      perm = rng.permuted(
          np.broadcast_to(np.arange(M), (S, M)), axis=1)
      shuffled[:, r] = np.take_along_axis(garr[:, r], perm, axis=1)
    vals.append(_ceiling(shuffled))
  return float(np.mean(vals))


def _split_opp_c0(garr):
  """Split-selected opportunity vs the candidate-0 baseline (candidate
  0 is identical in both sets, so the baseline cancels exactly in the
  paired D - P difference; review 5.4 renamed this from 'mode' — it is
  NOT the m_now baseline W1 uses)."""
  even = garr[:, 0::2].mean(1)
  odd = garr[:, 1::2].mean(1)
  sel = even.argmax(1)
  i = np.arange(len(sel))
  return float(np.mean(odd[i, sel] - odd[i, 0]))


def _stat(vals, cl, b=None):
  b = b or w1_read.B_BOOT
  point, ci = w1_read.bca(np.asarray(vals, float), cl, b=b)
  return dict(point=point, ci=list(ci),
              perm_p=w1_read.perm_p(vals, b=b), n_cells=len(vals))


def run_read(labels_dir, output, b=None):
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, 'wdc_read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  cells = load_all(labels_dir)
  cl = ['%s_%s_%d' % k for k in cells]
  floor_rng = np.random.default_rng(FLOOR_SEED)
  ceil_p, ceil_d, delta, opp_diff = [], [], [], []
  floor_p, floor_d = [], []
  for e in cells.values():
    gP = e['g']
    gD = np.concatenate([e['g'], e['gd']], axis=2)
    cp, cd = _ceiling(gP), _ceiling(gD)
    ceil_p.append(cp)
    ceil_d.append(cd)
    floor_p.append(_ceiling_floor(gP, floor_rng))
    floor_d.append(_ceiling_floor(gD, floor_rng))
    delta.append(cd - cp)
    opp_diff.append(_split_opp_c0(gD) - _split_opp_c0(gP))
  ceil_p_blk = _stat(ceil_p, cl, b=b)
  ceil_d_blk = _stat(ceil_d, cl, b=b)
  # Review 5.2: a sign-flip perm p on a non-negative-by-construction
  # quantity is structurally minimal and would read as "ceiling > 0" —
  # suppressed on the two ceiling blocks (delta keeps its perm).
  ceil_p_blk['perm_p'] = None
  ceil_d_blk['perm_p'] = None
  out = dict(
      n_cells=len(cells), bar=BAR,
      ceiling_policy=ceil_p_blk,
      ceiling_diverse=ceil_d_blk,
      ceiling_floor_policy=float(np.mean(floor_p)),
      ceiling_floor_diverse=float(np.mean(floor_d)),
      delta_ceiling=_stat(delta, cl, b=b),
      opp_c0_diff=_stat(opp_diff, cl, b=b))
  blk = out['delta_ceiling']
  fires_pos = blk['ci'][0] > 0 and blk['perm_p'] < ALPHA
  fires_neg = blk['ci'][1] < 0 and blk['perm_p'] < ALPHA
  # above-bar adjudicated NET of the matched noise floor (review 5.1)
  above_bar = (out['ceiling_diverse']['point']
               - out['ceiling_floor_diverse']) > BAR
  out['verdict'] = (
      'FLATNESS-IS-POLICY' if (fires_pos and above_bar) else
      'DIVERSITY-RAISES-SUBCEILING' if fires_pos else
      'CEILING-FALLS' if fires_neg else
      'GENERATOR-INSENSITIVE')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _write_cell(tmp, name, uni_spread=0.0, pol_spread=0.0, dup=False,
                run_id=None, meta_over=None, break_witness=False,
                break_dcands=False, noise_sd=1.0, nan_g=False,
                m_override=None):
  S, R, M, U, A = (EXPECT_STATES, EXPECT_REPEATS,
                   m_override or EXPECT_M, EXPECT_UNIFORM, 2)
  m = (DUP_RE if dup else FILE_RE).match(name)
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  rng = np.random.default_rng(zlib.crc32(name.encode()))
  ep = np.zeros(S, int)
  st = np.arange(S, dtype=int) * 25 + 25
  mu_p = rng.normal(0, pol_spread, (S, M)) if pol_spread else np.zeros((S, M))
  mu_u = rng.normal(0, uni_spread, (S, U)) if uni_spread else np.zeros((S, U))
  g = (mu_p[:, None]
       + rng.normal(0, noise_sd, (S, R, M))).astype(np.float32)
  gd = (mu_u[:, None]
        + rng.normal(0, noise_sd, (S, R, U))).astype(np.float32)
  if nan_g:
    g[0, 0, 0] = np.nan
  cands = rng.uniform(-1, 1, (S, M, A)).astype(np.float32)
  dcands = np.stack([
      np.random.default_rng([EXPECT_ENV_SEED, int(ep[s]), int(st[s])])
      .uniform(-1.0, 1.0, (U, A)).astype(np.float32) for s in range(S)])
  if dup:
    g = np.repeat(g[:, :, :1], M, 2)
    gd = np.repeat(g[:, :, :1], U, 2)
    dcands = np.repeat(cands[:, :1], U, 1)
  gw = g[:, :, 0].copy()
  if break_witness:
    gw = gw + 1.0
  if break_dcands:
    dcands = dcands + 0.5
  meta = dict(labeler_version='d1fix_20260724_wdc',
              env_seed=EXPECT_ENV_SEED, states=S, horizon=EXPECT_HORIZON,
              label_every=EXPECT_LABEL_EVERY,
              w1=dict(repeats=R, dcand_uniform=U,
                      env_seed=EXPECT_ENV_SEED))
  meta.update(meta_over or {})
  np.savez_compressed(
      os.path.join(tmp, name), meta=json.dumps(meta),
      g_all_rep=g, g_dcand_rep=gd, g_wit_rep=gw, dcands=dcands,
      cands=cands, m_now=np.zeros(S, np.int64),
      episode=ep, step=st, dup_cand=np.full(S, dup),
      run_id=np.array([run_id or f'r3_{dom}_{dose}_seed{seed}'] * S))


def _write_suite(tmp, uni_spread=0.0, pol_spread=0.0):
  for dom in ('cup', 'finger'):
    for dose in ('e1', 'e4'):
      for seed in range(31, 35):
        _write_cell(tmp, f'wdc_{dom}_{dose}_seed{seed}_late.npz',
                    uni_spread=uni_spread, pol_spread=pol_spread)
  for name in DUP_CELLS:
    _write_cell(tmp, name, dup=True)


def selfcheck():
  import tempfile
  B = 500

  # ceiling estimator calibration: recovers planted truth (ddof=1)
  rng = np.random.default_rng(0)
  for truth in (0.0, 0.5, 2.0):
    est = np.mean([_ceiling(
        rng.normal(0, truth, (400, 1, 12)) +
        rng.normal(0, 1.0, (400, 8, 12))) for _ in range(20)])
    assert abs(est - truth) < 0.12, (truth, est)

  # floor calibration (review 5.1): on pure noise the matched floor
  # tracks the noise-induced ceiling
  garr = rng.normal(0, 5.0, (200, 8, 12))
  c = _ceiling(garr)
  fl = _ceiling_floor(garr, np.random.default_rng(1), shuffles=8)
  assert c > 0.03 and abs(c - fl) < 0.06, (c, fl)

  # noise-only -> GENERATOR-INSENSITIVE; floors ~ ceilings; ceiling
  # perm_p suppressed
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    out = run_read(tmp, os.path.join(tmp, 'out'), b=B)
    assert out['verdict'] == 'GENERATOR-INSENSITIVE', out['verdict']
    assert abs(out['ceiling_floor_diverse']
               - out['ceiling_diverse']['point']) < 0.05, out
    assert out['ceiling_policy']['perm_p'] is None

  # large uniform-only spread -> FLATNESS-IS-POLICY net of the floor
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp, uni_spread=1.5)
    out = run_read(tmp, os.path.join(tmp, 'out'), b=B)
    assert out['verdict'] == 'FLATNESS-IS-POLICY', out
    assert (out['ceiling_diverse']['point']
            - out['ceiling_floor_diverse']) > BAR

  # ONE-read guard
  with tempfile.TemporaryDirectory() as tmp:
    outdir = os.path.join(tmp, 'out')
    os.makedirs(outdir)
    open(os.path.join(outdir, 'wdc_read.json'), 'w').close()
    try:
      run_read(tmp, outdir, b=B)
      raise AssertionError('ONE-read guard failed to trip')
    except SystemExit as e:
      assert 'ONE read execution' in str(e), e

  # refusal legs
  def expect_refusal(marker, **kw):
    with tempfile.TemporaryDirectory() as tmp:
      _write_suite(tmp)
      _write_cell(tmp, 'wdc_cup_e1_seed31_late.npz', **kw)
      try:
        run_read(tmp, os.path.join(tmp, 'out'), b=B)
        raise AssertionError(f'refusal not tripped: {marker}')
      except SystemExit as e:
        assert marker in str(e), (marker, str(e))

  expect_refusal('CRN witness', break_witness=True)
  expect_refusal('pinned stream', break_dcands=True)
  expect_refusal('non-finite', nan_g=True)
  expect_refusal('g_all_rep shape', m_override=6)
  expect_refusal('label_every', meta_over=dict(label_every=1))
  expect_refusal('env_seed', meta_over=dict(env_seed=1))
  expect_refusal('dcand_uniform',
                 meta_over=dict(w1=dict(repeats=8, dcand_uniform=2,
                                        env_seed=EXPECT_ENV_SEED)))
  expect_refusal('run_id', run_id='r3_finger_e1_seed31')
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    os.remove(os.path.join(tmp, 'wdc_cup_e1_seed31_late.npz'))
    try:
      run_read(tmp, os.path.join(tmp, 'out'), b=B)
      raise AssertionError('cell-count refusal failed to trip')
    except SystemExit as e:
      assert 'cells != registered' in str(e), e
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    _write_cell(tmp, 'wdcdup_cup_e1_seed32_late.npz', dup=True)
    try:
      run_read(tmp, os.path.join(tmp, 'out'), b=B)
      raise AssertionError('unregistered-dup refusal failed to trip')
    except SystemExit as e:
      assert 'not one of the registered' in str(e), e

  print('SELFCHECK PASS (wdc_read: ceiling estimator recovers planted '
        'truth 0/.5/2; matched noise floor tracks the noise ceiling; '
        'noise -> GENERATOR-INSENSITIVE w/ floor~ceiling + suppressed '
        'ceiling perm_p; planted uniform spread -> FLATNESS-IS-POLICY '
        'net of floor; ONE-read guard; refusals: witness, dcands '
        'stream, non-finite, M pin, label_every pin, env_seed, '
        'dcand_uniform, run_id identity, cell count, unregistered dup)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.labels and args.output):
    p.error('--labels and --output required (or --selfcheck)')
  run_read(args.labels, args.output)


if __name__ == '__main__':
  main()
