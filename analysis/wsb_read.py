"""Frozen read for the Stage B certified-injection wave (Paper 2).

Registered under PREREG_p2_stageb_injection_20260821.md (ThreeAxes item
8). ONE execution. The in-vivo positive control: a known reward bonus
delta * clip(a[0]) is stamped by the RewardStamp wrapper onto EXACTLY the
first step of every restored branch (arm-on-restore), so every
candidate's G and the probe's r_real carry the planted per-candidate
value

    planted(s, m) = DELTA * clip(cands[s, m, 0], -1, 1)

exactly once, computable by this reader as an EXACT referee. The frozen
critic knows nothing of the stamp — the designed dissociation is
probe-harvests / consumer-does-not.

Files (labeler `--stamp_delta 2.0 --w1_repeats 8`, version *_w1sb):
  wsb_{cup|finger}_{e1|e4}_seed{31,32}_late.npz             (8 cells)
  wsbdup_finger_e1_seed31_late.npz, wsbdup_cup_e4_seed32_late.npz (2)

Estimators (cluster = cell, n=8; permutation-primary + BCa via the
frozen w1_read machinery):

  PRIMARY — recovery slope: per cell, regress the within-state-centered
  all-repeat candidate means hbar(s, m) on the centered planted values
  p(s, m) (pooled OLS through the origin). The chain recovers the
  planted signal iff the slope is 1.
  SECONDARY — in-vivo fire: split-selected opportunity (m_now baseline)
  on the stamped arrays (n=8; low-powered, reported not adjudicated).
  CONSUMER-BLIND witness: delta * clip(actor_act[0]) - mean_m planted —
  the frozen consumer cannot tilt toward the stamp; a positive fire here
  is a wiring alarm, not a finding.

Admissibility (registered BEFORE data): median over states of
sd_m(planted) must be >= SPREAD_MIN = 0.10, else NO-CALL-UNDERSPREAD
(candidate convergence made the plant degenerate; the wave re-registers
with a diversified plant rather than reading noise).

Branch map:
  IN-VIVO-CALIBRATED   slope CI excludes 0 AND covers 1
  IN-VIVO-BIASED       slope CI excludes 0, does not cover 1
  CHAIN-BLIND          slope CI includes 0 (instrument cannot see a
                       real planted signal in vivo — the strongest
                       possible indictment of the chain)
  NO-CALL-UNDERSPREAD  admissibility gate failed (no adjudication)

Usage:
  python -m analysis.wsb_read --labels <dir> --output <dir>
  python -m analysis.wsb_read --selfcheck
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
DELTA = 2.0
# Raised 0.10 -> 0.20 at build review (finding 5.7 optional fix): an
# admitted plant must DOMINATE the archived substrate ceiling (0.2026),
# not merely exist. Archived-analogue median spread ~0.75, so NO-CALL
# stays unlikely.
SPREAD_MIN = 0.20
EXPECT_ENV_SEED = 20260823
EXPECT_REPEATS = 8
EXPECT_STATES = 200
EXPECT_HORIZON = 100
EXPECT_LABEL_EVERY = 25
EXPECT_M = 8
EXPECT_CELLS = 8
# Review B1: the stamp count is an exact identity of the W1 protocol —
# per labeled state: 2 determinism probes + M op_real probe steps +
# R*(M+1) branch first-steps + 1 base-trajectory resume step.
EXPECT_STAMPED = EXPECT_STATES * (
    3 + EXPECT_M + EXPECT_REPEATS * (EXPECT_M + 1))
STAMP_KIND = 'first_step_dim0_clip'
VERSION_SUFFIX = '_w1sb'
FILE_RE = re.compile(r'^wsb_(cup|finger)_(e1|e4)_seed(3[12])_late\.npz$')
DUP_RE = re.compile(r'^wsbdup_(cup|finger)_(e1|e4)_seed(3[12])_late\.npz$')
DUP_CELLS = ('wsbdup_finger_e1_seed31_late.npz',
             'wsbdup_cup_e4_seed32_late.npz')


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


def load_cell(path, dup=False):
  name = os.path.basename(path)
  z = np.load(path, allow_pickle=True)
  meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
  version = str(meta.get('labeler_version', ''))
  if not version.endswith(VERSION_SUFFIX):
    refuse(f'{name}: labeler_version {version!r} != *{VERSION_SUFFIX}')
  stamp = meta.get('stamp', {})
  if float(stamp.get('delta', -1)) != DELTA:
    refuse(f'{name}: stamp delta {stamp.get("delta")} != registered '
           f'{DELTA}')
  if str(stamp.get('kind', '')) != STAMP_KIND:
    refuse(f'{name}: stamp kind {stamp.get("kind")!r} != {STAMP_KIND!r}')
  if int(stamp.get('stamped_steps', -1)) != EXPECT_STAMPED:
    refuse(f'{name}: stamped_steps {stamp.get("stamped_steps")} != the '
           f'exact protocol identity {EXPECT_STAMPED} '
           f'(= S*(3 + M + R*(M+1))) — end-to-end stamp wiring broken '
           '(review B1)')
  w1 = meta.get('w1', {})
  for got, want, label in (
      (int(meta.get('env_seed', -1)), EXPECT_ENV_SEED, 'env_seed'),
      (int(w1.get('repeats', -1)), EXPECT_REPEATS, 'repeats'),
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
      cands=np.asarray(z['cands'], float),
      actor=np.asarray(z['actor_act'], float),
      rr=np.asarray(z['r_real'], float),
      rs=np.asarray(z['real_scores'], float),
      mn=np.asarray(z['m_now'], int),
      dup=np.asarray(z['dup_cand'], bool))
  S, R, M = e['g'].shape
  if (S, R, M) != (EXPECT_STATES, EXPECT_REPEATS, EXPECT_M):
    refuse(f'{name}: g_all_rep shape {e["g"].shape} != registered '
           f'({EXPECT_STATES}, {EXPECT_REPEATS}, {EXPECT_M})')
  for key in ('g', 'cands', 'actor', 'rr', 'rs'):
    if not np.isfinite(e[key]).all():
      refuse(f'{name}: non-finite {key} (review B2: a NaN would '
             'silently disable the admissibility gate)')
  if dup and not e['dup'].all():
    refuse(f'{name}: dup file without dup rows')
  if not dup and e['dup'].any():
    refuse(f'{name}: dup rows in a main file')
  return (dom, dose, seed), e


def load_all(labels_dir):
  cells, dups = {}, []
  for path in sorted(glob.glob(os.path.join(labels_dir, 'wsb*.npz'))):
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
      refuse(f'unregistered wsb file {name}')
  if len(cells) != EXPECT_CELLS:
    refuse(f'{len(cells)} cells != registered {EXPECT_CELLS}')
  if len(dups) != len(DUP_CELLS):
    refuse(f'{len(dups)} dup passes != registered {len(DUP_CELLS)}')
  for name, e in dups:
    if not np.all(e['g'] == e['g'][:, :, :1]):
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: stamped branches of '
             'identical actions differ — CRN or stamp broken')
    if float(np.max(np.ptp(e['rr'], axis=1))) != 0.0:
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: r_real varies across '
             'identical candidates')
  return cells


def _planted(e):
  return DELTA * np.clip(e['cands'][:, :, 0], -1.0, 1.0)   # (S, M)


def _slope_on(e, mat):
  """Pooled within-state-centered OLS slope of an (S, M) matrix on the
  planted values."""
  p = _planted(e)
  hc = mat - mat.mean(1, keepdims=True)
  pc = p - p.mean(1, keepdims=True)
  denom = float(np.sum(pc * pc))
  if denom <= 0:
    return np.nan
  return float(np.sum(hc * pc) / denom)


def _cell_slope(e):
  return _slope_on(e, e['g'].mean(1))


def _cell_probe_slope(e):
  """Review 5.6: the probe's r_real against the planted values — an
  essentially confound-free referee for whether the stamp REACHED the
  env (the archived analogue substrate has candidate-constant r_real),
  separating 'stamp never landed' from 'G accounting destroys it'
  when the primary reads CHAIN-BLIND."""
  return _slope_on(e, e['rr'])


def _cell_split_opp(e):
  even = e['g'][:, 0::2].mean(1)
  odd = e['g'][:, 1::2].mean(1)
  sel = even.argmax(1)
  i = np.arange(len(sel))
  return float(np.mean(odd[i, sel] - odd[i, e['mn']]))


def _cell_consumer_tilt(e):
  p = _planted(e)
  actor_p = DELTA * np.clip(e['actor'][:, 0], -1.0, 1.0)
  return float(np.mean(actor_p - p.mean(1)))


def _stat(vals, cl, b=None):
  b = b or w1_read.B_BOOT
  point, ci = w1_read.bca(np.asarray(vals, float), cl, b=b)
  return dict(point=point, ci=list(ci),
              perm_p=w1_read.perm_p(vals, b=b), n_cells=len(vals))


def run_read(labels_dir, output, b=None):
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, 'wsb_read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  cells = load_all(labels_dir)
  cl = ['%s_%s_%d' % k for k in cells]
  spreads = [float(np.median(np.std(_planted(e), axis=1, ddof=1)))
             for e in cells.values()]
  realized_spread = float(np.median(spreads))
  if not np.isfinite(realized_spread):
    refuse('realized planted spread is non-finite — the admissibility '
           'gate cannot adjudicate (review B2)')
  out = dict(n_cells=len(cells), delta=DELTA,
             realized_planted_spread_median=realized_spread,
             spread_min=SPREAD_MIN)
  if realized_spread < SPREAD_MIN:
    # Review 5.7: NOTHING else is computed or emitted on this branch —
    # the registered remedy (re-register with a diversified plant on
    # the same cells) must stay value-blind.
    out['verdict'] = 'NO-CALL-UNDERSPREAD'
  else:
    slope = _stat([_cell_slope(e) for e in cells.values()], cl, b=b)
    out['recovery_slope'] = slope
    # Disclosure (review 5.5): the estimand is formally 1 + beta where
    # beta is the substrate's own action-dim-0 value gradient; measured
    # pre-freeze on the archived analogue cells at -0.007 +/- 0.024.
    out['slope_note'] = 'estimand = 1 + substrate beta (disclosed ~0)'
    lo, hi = slope['ci']
    excl0 = lo > 0 or hi < 0
    cov1 = lo <= 1.0 <= hi
    out['verdict'] = ('IN-VIVO-CALIBRATED' if (excl0 and cov1) else
                      'IN-VIVO-BIASED' if excl0 else 'CHAIN-BLIND')
    # witnesses (adjudicate nothing; review 5.6 probe-slope localizer)
    out['probe_slope'] = _stat(
        [_cell_probe_slope(e) for e in cells.values()], cl, b=b)
    out['split_opp_stamped'] = _stat(
        [_cell_split_opp(e) for e in cells.values()], cl, b=b)
    tilt = _stat([_cell_consumer_tilt(e) for e in cells.values()],
                 cl, b=b)
    tilt['alarm'] = bool(tilt['ci'][0] > 0 and tilt['perm_p'] < ALPHA)
    out['consumer_tilt'] = tilt
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _write_cell(tmp, name, gain=1.0, cand_spread=0.5, dup=False,
                run_id=None, meta_over=None, stamp_over=None,
                nan_cands=False, m_override=None):
  S, R, M, A = (EXPECT_STATES, EXPECT_REPEATS,
                m_override or EXPECT_M, 2)
  m = (DUP_RE if dup else FILE_RE).match(name)
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  rng = np.random.default_rng(zlib.crc32(name.encode()))
  cands = rng.uniform(-cand_spread, cand_spread,
                      (S, M, A)).astype(np.float32)
  if dup:
    cands = np.repeat(cands[:, :1], M, 1)
  planted = DELTA * np.clip(cands[:, :, 0], -1, 1)
  g = (gain * planted[:, None, :] +
       rng.normal(0, 1.0, (S, R, M))).astype(np.float32)
  if dup:
    g = np.repeat(g[:, :, :1], M, 2)
  actor = rng.uniform(-cand_spread, cand_spread, (S, A)).astype(np.float32)
  rr = planted + rng.normal(0, 0.05, (S, M))
  if dup:
    rr = np.repeat(rr[:, :1], M, 1)
  if nan_cands:
    cands[0, 0, 0] = np.nan
  stamp = dict(delta=DELTA, kind=STAMP_KIND,
               stamped_steps=EXPECT_STAMPED)
  stamp.update(stamp_over or {})
  meta = dict(labeler_version='d1fix_20260724_w1sb',
              env_seed=EXPECT_ENV_SEED, states=S, horizon=EXPECT_HORIZON,
              label_every=EXPECT_LABEL_EVERY,
              w1=dict(repeats=R), stamp=stamp)
  meta.update(meta_over or {})
  np.savez_compressed(
      os.path.join(tmp, name), meta=json.dumps(meta),
      g_all_rep=g, cands=cands, actor_act=actor,
      r_real=rr.astype(np.float32),
      real_scores=rr.astype(np.float32),
      m_now=np.zeros(S, np.int64), dup_cand=np.full(S, dup),
      run_id=np.array([run_id or f'r3_{dom}_{dose}_seed{seed}'] * S))


def _write_suite(tmp, gain=1.0, cand_spread=0.5):
  for dom in ('cup', 'finger'):
    for dose in ('e1', 'e4'):
      for seed in (31, 32):
        _write_cell(tmp, f'wsb_{dom}_{dose}_seed{seed}_late.npz',
                    gain=gain, cand_spread=cand_spread)
  for name in DUP_CELLS:
    _write_cell(tmp, name, dup=True)


def selfcheck():
  import tempfile
  B = 500
  for gain, want in ((1.0, 'IN-VIVO-CALIBRATED'),
                     (0.4, 'IN-VIVO-BIASED'),
                     (0.0, 'CHAIN-BLIND')):
    with tempfile.TemporaryDirectory() as tmp:
      _write_suite(tmp, gain=gain)
      out = run_read(tmp, os.path.join(tmp, 'out'), b=B)
      assert out['verdict'] == want, (gain, out['verdict'])
      assert not out['consumer_tilt']['alarm'], out['consumer_tilt']
      # probe-slope localizer (review 5.6): rr = planted + small noise
      # in the fixture, so the probe slope must read ~1 regardless of
      # the G gain — CHAIN-BLIND with probe_slope~1 localizes to G
      # accounting
      assert abs(out['probe_slope']['point'] - 1.0) < 0.1, (
          out['probe_slope'])
  # underspread admissibility: NOTHING else emitted (review 5.7)
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp, cand_spread=0.01)
    out = run_read(tmp, os.path.join(tmp, 'out'), b=B)
    assert out['verdict'] == 'NO-CALL-UNDERSPREAD', out['verdict']
    for key in ('recovery_slope', 'split_opp_stamped', 'consumer_tilt',
                'probe_slope'):
      assert key not in out, key
  # ONE-read guard
  with tempfile.TemporaryDirectory() as tmp:
    outdir = os.path.join(tmp, 'out')
    os.makedirs(outdir)
    open(os.path.join(outdir, 'wsb_read.json'), 'w').close()
    try:
      run_read(tmp, outdir, b=B)
      raise AssertionError('ONE-read guard failed to trip')
    except SystemExit as e:
      assert 'ONE read execution' in str(e), e

  # refusal legs
  def expect_refusal(marker, **kw):
    with tempfile.TemporaryDirectory() as tmp:
      _write_suite(tmp)
      _write_cell(tmp, 'wsb_cup_e1_seed31_late.npz', **kw)
      try:
        run_read(tmp, os.path.join(tmp, 'out'), b=B)
        raise AssertionError(f'refusal not tripped: {marker}')
      except SystemExit as e:
        assert marker in str(e), (marker, str(e))

  expect_refusal('stamp delta', stamp_over=dict(delta=1.0))
  expect_refusal('exact protocol identity',
                 stamp_over=dict(stamped_steps=1))
  expect_refusal('non-finite cands', nan_cands=True)
  expect_refusal('g_all_rep shape', m_override=4)
  expect_refusal('label_every', meta_over=dict(label_every=1))
  expect_refusal('env_seed', meta_over=dict(env_seed=1))
  expect_refusal('labeler_version',
                 meta_over=dict(labeler_version='d1fix_20260724_w1'))
  expect_refusal('run_id', run_id='r3_finger_e1_seed31')
  # dup violation: differing "identical" branches must refuse
  with tempfile.TemporaryDirectory() as tmp:
    _write_suite(tmp)
    name_fix = os.path.join(tmp, DUP_CELLS[0])
    z = dict(np.load(name_fix, allow_pickle=True))
    g = np.array(z['g_all_rep'])
    g[:, :, 1] += 1.0
    z['g_all_rep'] = g
    np.savez_compressed(name_fix, **z)
    try:
      run_read(tmp, os.path.join(tmp, 'out'), b=B)
      raise AssertionError('dup violation failed to trip')
    except SystemExit as e:
      assert 'DUPLICATE-NULL VIOLATION' in str(e), e

  print('SELFCHECK PASS (wsb_read: gain 1/.4/0 -> CALIBRATED/BIASED/'
        'BLIND with probe-slope~1 localizer; underspread NO-CALL emits '
        'nothing else; ONE-read guard; refusals: delta pin, exact '
        'stamped_steps identity, non-finite cands, M pin, label_every '
        'pin, env_seed, version, run_id identity, duplicate-null '
        'violation)')


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
