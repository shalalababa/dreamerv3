"""Frozen reader for the task-arm dose-response wave (#21).

Registration: PREREG_dose_task_20260810.md (rules implemented verbatim;
reviewer findings 1-12 applied pre-freeze). ONE execution, plain
`python -m analysis.dose_task_read` (never -O: gates are asserts).

Inputs:
  --auc_csv       NEW collated adaptation csv (adaptation_auc schema);
                  modes ax1td0..ax1td3, finger domain, milestone 500000,
                  8 seeds each.
  --e4_csv        E4 collate csv (stratified_error collate) for the 32
                  td fits (P-DR3 descriptive).
  --dose_manifest the dose grid's manifest.json (build-dose output);
                  realized occupancies read from levels[*].occ_recomputed
                  (never hand-transcribed); each must be within +-0.02 of
                  nominal {0.054, 0.131, 0.249, 0.323} or the read
                  REFUSES. Manifest sha256 recorded in the output.
  --fit_counters  json {run_name: {"update": U, "total": T}}; U != T or
                  a missing fit REFUSES the read.
  --ckpt_steps    json {run_name: step}; every consumed fit must be at
                  500000 (second witness, 10-Aug dual-witness rule).
  --leak_overlap  OPTIONAL json of per-level probeset-episode overlap
                  counts (D1-leak disclosure, recorded verbatim; the dose
                  buffers predate the --holdout guard).
  --output        artifact dir.

Selfcheck: python -m analysis.dose_task_read --selfcheck
"""

import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np

B_BOOT = 10_000
B_PERM_SHAPE = 1_000     # registered shape-null calibration draws
RNG_SEED = 0
LEVELS = ('0', '1', '2', '3')
NOMINAL_F = {'0': 0.054, '1': 0.131, '2': 0.249, '3': 0.323}
F_TOL = 0.02
MILESTONE = 500_000
SEEDS_PER_CELL = 8
MODE_PREFIX = 'ax1td'
WM_PREFIX = 'ax1wm_finger_td'
MEMBER_BAR = 1.5
MEMBER_ANCHOR = 0.827
MDE_NOTE = ('interior-contrast MDE ~ 69 at n=8v8 (highfr pooled-sd '
            'basis); interior nulls are weak evidence. G1 power at the '
            'same basis: ~0.73 for a true +100, ~0.50 at +69 - a G1 '
            'failure is only interpretable AT THIS POWER.')


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def _ndtri(p):
  return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _ndtr(z):
  return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sample(a, b, rng_seed=RNG_SEED, b_boot=B_BOOT):
  """diff = mean(a) - mean(b); BCa 95% CI + two-sided permutation p
  (house form, as analysis/swave_read.two_sample). One run per seed per
  cell, so plain bootstrap == seed-cluster bootstrap here (disclosed)."""
  a = np.asarray(a, float)
  b = np.asarray(b, float)
  rng = np.random.default_rng(rng_seed)
  diff = float(a.mean() - b.mean())
  boots = (a[rng.integers(0, len(a), (b_boot, len(a)))].mean(1)
           - b[rng.integers(0, len(b), (b_boot, len(b)))].mean(1))
  prop = float(np.mean(boots < diff))
  prop = min(max(prop, 1.0 / (b_boot + 1)), 1.0 - 1.0 / (b_boot + 1))
  z0 = _ndtri(prop)
  pooled = np.concatenate([a, b])
  jack = []
  for i in range(len(pooled)):
    keep = np.ones(len(pooled), bool)
    keep[i] = False
    aa = pooled[:len(a)][keep[:len(a)]]
    bb = pooled[len(a):][keep[len(a):]]
    jack.append(aa.mean() - bb.mean())
  jack = np.asarray(jack)
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0

  def q(alpha_pt):
    z = _ndtri(alpha_pt)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    return float(np.percentile(boots, 100.0 * min(max(_ndtr(adj), 0.0),
                                                  1.0)))
  ci = (q(0.025), q(0.975))
  obs = abs(diff)
  count = 0
  for _ in range(b_boot):
    perm = rng.permutation(pooled)
    if abs(perm[:len(a)].mean() - perm[len(a):].mean()) >= obs - 1e-15:
      count += 1
  p = float((count + 1) / (b_boot + 1))
  return dict(diff=diff, ci=list(ci), perm_p=p, n=[len(a), len(b)])


def bh_survivors(pvals, q=0.05):
  items = sorted(pvals.items(), key=lambda kv: kv[1])
  m = len(items)
  thresh = 0
  for i, (k, p) in enumerate(items, 1):
    if p <= q * i / m:
      thresh = i
  return {k for i, (k, p) in enumerate(items, 1) if i <= thresh}


def activation_from_contrasts(contrasts):
  """P-DR1 wording map incl. the NON-MONOTONE branch (review #3).
  A survivor needs BH-significance AND a CI excluding 0 (conjunction)."""
  surv = bh_survivors({l: c['perm_p'] for l, c in contrasts.items()})
  fired = {l for l in surv
           if not (contrasts[l]['ci'][0] <= 0.0 <= contrasts[l]['ci'][1])}
  if '1' in fired and '2' in fired:
    act = 'EARLY (effect present by the 0.131 cell)'
  elif '2' in fired:
    act = 'MID (onset between 0.131 and 0.249)'
  elif '1' in fired:
    act = ('NON-MONOTONE (only the 0.131 contrast survives; no '
           'activation wording licensed)')
  else:
    act = 'LATE (no interior survivor; threshold-like above 0.249)'
  return sorted(fired), act


# --------------------------------------------------------------------------
# P-DR2 shape machinery (review #1/#2: fold-internal cut selection +
# permutation-calibrated decisiveness)
# --------------------------------------------------------------------------

def _ols2_predict(x_tr, y_tr, x_te):
  """Exact 2-parameter OLS y = a + b*x via normal equations."""
  xm, ym = x_tr.mean(), y_tr.mean()
  vx = float(np.sum((x_tr - xm) ** 2))
  bcoef = float(np.sum((x_tr - xm) * (y_tr - ym)) / vx) if vx > 0 else 0.0
  return ym + bcoef * (x_te - xm)


def _cv_mses(y, f, seeds, cuts):
  """Leave-one-seed-out CV MSE for the three forms. The step cut is
  chosen INSIDE each outer fold by an inner leave-one-seed-out CV over
  the training seeds only (review #2 - no selection leakage)."""
  uniq = sorted(set(seeds.tolist()))
  errs = {'linear': [], 'log': [], 'step': []}
  fold_cuts = []
  for s in uniq:
    tr, te = seeds != s, seeds == s
    errs['linear'].extend(
        ((y[te] - _ols2_predict(f[tr], y[tr], f[te])) ** 2).tolist())
    errs['log'].extend(
        ((y[te] - _ols2_predict(np.log(f[tr]), y[tr],
                                np.log(f[te]))) ** 2).tolist())
    inner = []
    tr_seeds = seeds[tr]
    for c in cuts:
      ie = []
      for s2 in sorted(set(tr_seeds.tolist())):
        itr = tr & (seeds != s2)
        ite = tr & (seeds == s2)
        ie.extend(((y[ite] - _ols2_predict(
            (f[itr] >= c).astype(float), y[itr],
            (f[ite] >= c).astype(float))) ** 2).tolist())
      inner.append(float(np.mean(ie)))
    # ties resolve to the SMALLEST cut (deterministic, documented)
    c_s = cuts[int(np.argmin(inner))]
    fold_cuts.append(c_s)
    errs['step'].extend(((y[te] - _ols2_predict(
        (f[tr] >= c_s).astype(float), y[tr],
        (f[te] >= c_s).astype(float))) ** 2).tolist())
  return {k: float(np.mean(v)) for k, v in errs.items()}, fold_cuts


def _margin(mses):
  ranked = sorted(mses.items(), key=lambda kv: kv[1])
  (w, wm), (r, rm) = ranked[0], ranked[1]
  return (w, r, float(1.0 - wm / rm) if rm > 0 else 0.0)


def cv_shape(rows, f_by_level, b_perm=B_PERM_SHAPE, rng_seed=RNG_SEED):
  y = np.asarray([r['auc'] for r in rows], float)
  f = np.asarray([f_by_level[r['level']] for r in rows], float)
  seeds = np.asarray([r['seed'] for r in rows])
  gaps = sorted(set(f_by_level.values()))
  cuts = [(gaps[i] + gaps[i + 1]) / 2.0 for i in range(len(gaps) - 1)]
  mses, fold_cuts = _cv_mses(y, f, seeds, cuts)
  winner, runner, margin = _margin(mses)
  # permutation null: permute the level labels WITHIN each seed
  # (destroys shape, preserves seed effects), full procedure re-run.
  rng = np.random.default_rng(rng_seed)
  null_margins = np.empty(b_perm)
  uniq = sorted(set(seeds.tolist()))
  for i in range(b_perm):
    yp = y.copy()
    for s in uniq:
      idx = np.where(seeds == s)[0]
      yp[idx] = yp[rng.permutation(idx)]
    m_p, _ = _cv_mses(yp, f, seeds, cuts)
    null_margins[i] = _margin(m_p)[2]
  crit = float(np.percentile(null_margins, 95.0))
  decisive = margin > crit
  vals, cnts = np.unique(np.asarray(fold_cuts), return_counts=True)
  best_cut = round(float(vals[int(np.argmax(cnts))]), 4)
  if not decisive:
    verdict = 'TIE-UNDISCRIMINATED: no shape wording licensed'
  elif winner == 'step':
    verdict = (f'THRESHOLD-AT-{best_cut} (on this grid realization): '
               'step form wins the calibrated CV comparison')
  elif winner == 'log':
    verdict = ('GRADED-LOG (on this grid realization): log-occupancy '
               'form wins (coheres with the archived log-interaction)')
  else:
    verdict = ('GRADED-LINEAR (on this grid realization): '
               'linear-occupancy form wins')
  return {'cv_mse': mses, 'fold_cuts': [float(c) for c in fold_cuts],
          'best_cut': best_cut, 'winner': winner, 'runner_up': runner,
          'margin': margin, 'null_margin_crit95': crit,
          'b_perm': b_perm, 'decisive': bool(decisive), 'verdict': verdict,
          'df_note': 'step cut chosen inside each training fold (inner '
                     'LOSO); decisiveness calibrated against the '
                     'within-seed label-permutation null'}


def analyse(auc_rows, f_by_level, e4_rows=None, b_perm=B_PERM_SHAPE):
  cells = {l: [r for r in auc_rows if r['level'] == l] for l in LEVELS}
  for l in LEVELS:
    assert len(cells[l]) >= 6, (l, len(cells[l]),
                                'cell below the n=6 refusal bar')
  res = {'n_per_cell': {l: len(cells[l]) for l in LEVELS},
         'cell_means': {l: float(np.mean([r['auc'] for r in cells[l]]))
                        for l in LEVELS},
         'occupancies': f_by_level, 'mde_note': MDE_NOTE}
  g1 = two_sample([r['auc'] for r in cells['3']],
                  [r['auc'] for r in cells['0']])
  g1_pass = g1['ci'][0] > 0.0 and g1['perm_p'] < 0.05
  res['g1_replication'] = dict(**g1, gate_pass=bool(g1_pass))
  contrasts = {l: two_sample([r['auc'] for r in cells[l]],
                             [r['auc'] for r in cells['0']])
               for l in ('1', '2')}
  fired, act = activation_from_contrasts(contrasts)
  res['p_dr1'] = {'contrasts': contrasts, 'bh_survivors': fired,
                  'activation': act}
  res['p_dr2'] = cv_shape(auc_rows, f_by_level, b_perm=b_perm)
  if not g1_pass:
    res['verdict'] = ('GRID-UNINFORMATIVE: the known occupancy effect '
                      'does not reproduce on dose-constructed buffers '
                      'AT THIS POWER (G1 power note in mde_note); '
                      'P-DR1/P-DR2 reported with zero decision weight')
  else:
    res['verdict'] = res['p_dr2']['verdict'] + ' | activation: ' + act
  if e4_rows is not None:
    memb = {}
    for l in LEVELS:
      vals = [r['rew_nll_in'] for r in e4_rows if r['level'] == l]
      memb[l] = dict(mean=float(np.mean(vals)) if vals else None,
                     n=len(vals),
                     in_band=(bool(np.mean(vals) <= MEMBER_BAR)
                              if vals else None))
    res['p_dr3_membership'] = dict(
        per_level=memb, bar=MEMBER_BAR, anchor=MEMBER_ANCHOR,
        note='descriptive, never adjudicated; the dose buffers predate '
             'the --holdout guard - D1-leak overlap disclosed in '
             'leak_overlap when supplied')
  return res


# --------------------------------------------------------------------------
# loading (fail-closed)
# --------------------------------------------------------------------------

def load_auc(path):
  rows, excluded_qc = [], []
  with open(path) as fh:
    for r in csv.DictReader(fh):
      if not r['mode'].startswith(MODE_PREFIX):
        continue
      if r['domain'] != 'finger':          # fail-closed (review #15)
        continue
      if int(float(r['milestone'])) != MILESTONE:   # review #6
        continue
      level = r['mode'][len(MODE_PREFIX):]
      assert level in LEVELS, r['mode']
      row = dict(level=level, seed=int(r['seed']),
                 auc=float(r['auc100k']),
                 n_ep=int(float(r['n_ep_100k'])))
      if str(r.get('qc_pass', 'True')) not in ('True', '1', 'true'):
        excluded_qc.append((level, row['seed']))
        continue
      rows.append(row)
  # duplicate assert BEFORE the modal filter (review #6)
  seen = set()
  for r in rows:
    key = (r['level'], r['seed'])
    assert key not in seen, ('duplicate row', key)
    seen.add(key)
  n_ep = [r['n_ep'] for r in rows]
  modal = int(np.bincount(n_ep).argmax()) if n_ep else 0
  kept = [r for r in rows if r['n_ep'] == modal]
  excluded = [(r['level'], r['seed']) for r in rows if r['n_ep'] != modal]
  return kept, modal, excluded, excluded_qc


def load_e4(path):
  rows = []
  with open(path) as fh:
    for r in csv.DictReader(fh):
      if int(float(r['horizon'])) != 0:
        continue
      rid = r['run_id']
      if WM_PREFIX not in rid:
        continue
      level = rid.split(WM_PREFIX)[1].split('_')[0]
      assert level in LEVELS, rid
      rows.append(dict(level=level,
                       seed=int(rid.rsplit('seed', 1)[1]),
                       rew_nll_in=float(r['rew_nll_in'])))
  return rows


def load_occupancies(manifest_path):
  """Realized occupancies straight from the build-dose manifest
  (review #7 - never hand-transcribed); returns (f_by_level, sha256)."""
  blob = open(manifest_path, 'rb').read()
  sha = hashlib.sha256(blob).hexdigest()
  man = json.loads(blob)
  f = {}
  for rep in man['levels']:
    l = str(rep['level'])
    assert l in LEVELS, rep
    v = float(rep['occ_recomputed'])
    assert abs(v - NOMINAL_F[l]) <= F_TOL, (
        l, v, NOMINAL_F[l], 'occupancy outside the registered gate')
    f[l] = v
  assert set(f) == set(LEVELS), sorted(f)
  return f, sha


def check_counters(counters):
  bad = []
  for l in LEVELS:
    for s in range(1, SEEDS_PER_CELL + 1):
      name = f'{WM_PREFIX}{l}_seed{s}'
      c = counters.get(name)
      if c is None or int(c['update']) != int(c['total']):
        bad.append((name, c))
  assert not bad, ('fit-counter refusal', bad)


def check_ckpt_steps(steps):
  """Second witness (review #4): the checkpoint actually present per fit
  dir must be at MILESTONE for every consumed fit."""
  bad = []
  for l in LEVELS:
    for s in range(1, SEEDS_PER_CELL + 1):
      name = f'{WM_PREFIX}{l}_seed{s}'
      v = steps.get(name)
      if v is None or int(v) != MILESTONE:
        bad.append((name, v))
  assert not bad, ('checkpoint-step refusal', bad)


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

SC_PERM = 200   # selfcheck-only calibration draws (procedure unchanged)


def _rows(mk, noise=8.0, rng=None):
  rng = rng or np.random.default_rng(3)
  rows = []
  for l in LEVELS:
    for s in range(1, 9):
      rows.append(dict(level=l, seed=s,
                       auc=mk(NOMINAL_F[l]) + rng.normal(0, noise)))
  return rows


def _fake_contrast(diff, ci, p):
  return dict(diff=diff, ci=list(ci), perm_p=p, n=[8, 8])


def selfcheck():
  import tempfile
  f = dict(NOMINAL_F)
  # planted shapes recovered decisively under the CALIBRATED procedure
  r = analyse(_rows(lambda x: 100 + 60 * math.log(x)), f, b_perm=SC_PERM)
  assert r['g1_replication']['gate_pass'], r['g1_replication']
  assert r['p_dr2']['winner'] == 'log' and r['p_dr2']['decisive'], \
      r['p_dr2']
  r = analyse(_rows(lambda x: 100 + 400 * x), f, b_perm=SC_PERM)
  assert r['p_dr2']['winner'] == 'linear' and r['p_dr2']['decisive'], \
      r['p_dr2']
  r = analyse(_rows(lambda x: 100 + (120 if x >= 0.19 else 0)), f,
              b_perm=SC_PERM)
  assert r['p_dr2']['winner'] == 'step' and r['p_dr2']['decisive'], \
      r['p_dr2']
  assert abs(r['p_dr2']['best_cut'] - 0.19) < 0.01, r['p_dr2']
  assert 'MID' in r['p_dr1']['activation'], r['p_dr1']
  # NULL leg (review #1): flat grid at the wave's own noise basis must
  # NOT be decisive - the false-THRESHOLD path the review killed.
  r = analyse(_rows(lambda x: 100.0, noise=77.18,
                    rng=np.random.default_rng(7)), f, b_perm=SC_PERM)
  assert r['p_dr2']['decisive'] is False, r['p_dr2']
  assert 'TIE' in r['p_dr2']['verdict']
  # flat grid also fails G1 -> GRID-UNINFORMATIVE
  r = analyse(_rows(lambda x: 100.0), f, b_perm=SC_PERM)
  assert not r['g1_replication']['gate_pass']
  assert r['verdict'].startswith('GRID-UNINFORMATIVE')
  # G1 masking legs (review #5i/ii): each conjunct must bind alone.
  base = _rows(lambda x: 100 + 400 * x)
  g1ref = analyse(base, f, b_perm=SC_PERM)['g1_replication']
  assert g1ref['gate_pass']
  fake_g1 = dict(g1ref)
  fake_g1_pass = lambda g: g['ci'][0] > 0.0 and g['perm_p'] < 0.05
  fake_g1.update(ci=[-1.0, 200.0], perm_p=0.001)
  assert not fake_g1_pass(fake_g1), 'CI leg must bind alone'
  fake_g1.update(ci=[10.0, 200.0], perm_p=0.20)
  assert not fake_g1_pass(fake_g1), 'p leg must bind alone'
  # ... and through the real path: signal with one wild level-0 cell so
  # p can pass while the CI straddles is hard to force determinantly;
  # instead verify the reader's own conjunction expression directly on
  # the computed dict (same expression as analyse()).
  g = g1ref
  assert (g['ci'][0] > 0.0 and g['perm_p'] < 0.05) == g['gate_pass']
  # activation unit tests (reviews #3/#5iv) on handcrafted contrasts:
  fired, act = activation_from_contrasts({
      '1': _fake_contrast(50, (10, 90), 0.001),
      '2': _fake_contrast(60, (20, 100), 0.001)})
  assert 'EARLY' in act and fired == ['1', '2']
  fired, act = activation_from_contrasts({
      '1': _fake_contrast(5, (-30, 40), 0.6),
      '2': _fake_contrast(60, (20, 100), 0.001)})
  assert 'MID' in act
  fired, act = activation_from_contrasts({
      '1': _fake_contrast(50, (10, 90), 0.001),
      '2': _fake_contrast(5, (-30, 40), 0.6)})
  assert 'NON-MONOTONE' in act, act
  # CI-and-BH conjunction: BH-significant p but straddling CI -> no fire
  fired, act = activation_from_contrasts({
      '1': _fake_contrast(50, (-5, 90), 0.001),
      '2': _fake_contrast(5, (-30, 40), 0.6)})
  assert fired == [] and 'LATE' in act, (fired, act)
  # BH /m mutant killer (review #5iii): p=.04 with m=2 -> threshold for
  # rank 1 is .025 -> must NOT survive (the q*i mutant would pass it).
  assert bh_survivors({'1': 0.04, '2': 0.6}) == set(), 'BH /m mutant'
  assert bh_survivors({'1': 0.02, '2': 0.6}) == {'1'}
  assert bh_survivors({'1': 0.02, '2': 0.049}) == {'1', '2'}, 'BH rank-up'
  # counter gates: missing name AND update!=total on a complete set
  try:
    check_counters({f'{WM_PREFIX}{l}_seed{s}': {'update': 500000,
                                                'total': 500000}
                    for l in LEVELS for s in range(1, 9)
                    if not (l == '2' and s == 5)})
    raise SystemExit('selfcheck FAIL: missing-fit not refused')
  except AssertionError:
    pass
  full = {f'{WM_PREFIX}{l}_seed{s}': {'update': 500000, 'total': 500000}
          for l in LEVELS for s in range(1, 9)}
  check_counters(full)
  full[f'{WM_PREFIX}1_seed3'] = {'update': 224000, 'total': 500000}
  try:
    check_counters(full)
    raise SystemExit('selfcheck FAIL: truncated counter not refused')
  except AssertionError:
    pass
  # ckpt-step witness
  steps = {f'{WM_PREFIX}{l}_seed{s}': 500000
           for l in LEVELS for s in range(1, 9)}
  check_ckpt_steps(steps)
  steps[f'{WM_PREFIX}3_seed8'] = 450000
  try:
    check_ckpt_steps(steps)
    raise SystemExit('selfcheck FAIL: stale ckpt step not refused')
  except AssertionError:
    pass
  # loader round-trips (review #5v) via temp csvs
  with tempfile.TemporaryDirectory() as td:
    ap = os.path.join(td, 'auc.csv')
    with open(ap, 'w', newline='') as fh:
      w = csv.writer(fh)
      w.writerow(['run_id', 'mode', 'domain', 'seed', 'milestone',
                  'auc100k', 'n_ep_100k', 'qc_pass'])
      for l in LEVELS:
        for s in range(1, 9):
          w.writerow([f'adapt_ax1td{l}_finger_seed{s}_ckpt500000',
                      f'ax1td{l}', 'finger', s, 500000,
                      100 + 10 * int(l) + s, 40, 'True'])
      # foreign mode, foreign domain, qc-fail, sub-modal, off-milestone
      w.writerow(['x', 'ax1swr11q1s0', 'finger', 1, 500000, 999, 40,
                  'True'])
      w.writerow(['x', 'ax1td0', 'cup', 1, 500000, 999, 40, 'True'])
      w.writerow(['adapt_ax1td0_finger_seed9_ckpt500000', 'ax1td0',
                  'finger', 9, 500000, 999, 40, 'False'])
      w.writerow(['adapt_ax1td1_finger_seed9_ckpt500000', 'ax1td1',
                  'finger', 9, 500000, 999, 13, 'True'])
      w.writerow(['adapt_ax1td2_finger_seed1_ckpt450000', 'ax1td2',
                  'finger', 1, 450000, 999, 40, 'True'])
    rows, modal, ex_sub, ex_qc = load_auc(ap)
    assert len(rows) == 32 and modal == 40, (len(rows), modal)
    assert ex_sub == [('1', 9)] and ex_qc == [('0', 9)], (ex_sub, ex_qc)
    assert not any(r['auc'] == 999 for r in rows), 'leak through filters'
    # duplicate at the SAME milestone must refuse (pre-modal, review #6)
    with open(ap, 'a', newline='') as fh:
      csv.writer(fh).writerow(
          ['adapt_ax1td0_finger_seed1_ckpt500000b', 'ax1td0', 'finger',
           1, 500000, 999, 13, 'True'])
    try:
      load_auc(ap)
      raise SystemExit('selfcheck FAIL: same-milestone duplicate kept')
    except AssertionError:
      pass
    ep = os.path.join(td, 'e4.csv')
    with open(ep, 'w', newline='') as fh:
      w = csv.writer(fh)
      w.writerow(['run_id', 'horizon', 'rew_nll_in'])
      for l in LEVELS:
        for s in range(1, 9):
          for h in (0, 1, 5, 20):
            w.writerow([f'ax1wm_finger_td{l}_seed{s}', h,
                        (0.5 if l != '0' else 3.0) + 0.01 * h])
    e4 = load_e4(ep)
    assert len(e4) == 32, len(e4)
    out = analyse(rows, f, e4_rows=e4, b_perm=SC_PERM)
    m = out['p_dr3_membership']['per_level']
    assert m['0']['in_band'] is False and m['3']['in_band'] is True, m
    assert m['0']['n'] == 8
    # occupancy gate through the manifest loader
    mp = os.path.join(td, 'manifest.json')
    json.dump({'levels': [
        {'level': int(l), 'occ_recomputed': NOMINAL_F[l]}
        for l in LEVELS]}, open(mp, 'w'))
    fl, sha = load_occupancies(mp)
    assert fl == NOMINAL_F and len(sha) == 64
    json.dump({'levels': [
        {'level': int(l),
         'occ_recomputed': (0.30 if l == '2' else NOMINAL_F[l])}
        for l in LEVELS]}, open(mp, 'w'))
    try:
      load_occupancies(mp)
      raise SystemExit('selfcheck FAIL: occupancy gate not tripped')
    except AssertionError:
      pass
  print('dose_task_read selfcheck PASS (planted log/linear/step '
        'recovered decisively under the CALIBRATED procedure w/ '
        'fold-internal cuts; flat grid at wave-noise NOT decisive; '
        'GRID-UNINFORMATIVE path; NON-MONOTONE + CI-and-BH conjunction '
        'unit-tested; BH /m + rank mutants killed; counter/ckpt-step/'
        'occupancy/duplicate/milestone/qc/sub-modal gates all trip; '
        'loader round-trips green)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--auc_csv')
  p.add_argument('--e4_csv')
  p.add_argument('--dose_manifest')
  p.add_argument('--fit_counters')
  p.add_argument('--ckpt_steps')
  p.add_argument('--leak_overlap')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not all([args.auc_csv, args.e4_csv, args.dose_manifest,
              args.fit_counters, args.ckpt_steps, args.output]):
    p.error('all inputs + --output required (or --selfcheck)')
  f, manifest_sha = load_occupancies(args.dose_manifest)
  check_counters(json.load(open(args.fit_counters)))
  check_ckpt_steps(json.load(open(args.ckpt_steps)))
  auc_rows, modal, excluded, excluded_qc = load_auc(args.auc_csv)
  e4_rows = load_e4(args.e4_csv)
  res = analyse(auc_rows, f, e4_rows)
  res['modal_n_ep'] = modal
  res['excluded_submodal'] = excluded
  res['excluded_qc'] = excluded_qc
  res['dose_manifest_sha256'] = manifest_sha
  if args.leak_overlap:
    res['leak_overlap'] = json.load(open(args.leak_overlap))
  res['prereg'] = 'PREREG_dose_task_20260810.md'
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'dose_task.json')
  with open(out, 'w') as fh:
    json.dump(res, fh, indent=1, sort_keys=True)
  summary = {'verdict': res['verdict'], 'g1': res['g1_replication']}
  if res['verdict'].startswith('GRID-UNINFORMATIVE'):
    summary['p_dr2_ZERO_WEIGHT'] = res['p_dr2']
  else:
    summary['p_dr2'] = res['p_dr2']
  print(json.dumps(summary, indent=1))
  print(f'-> {out}')


if __name__ == '__main__':
  main()
