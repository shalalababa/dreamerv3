"""Frozen read for the s×w_r wave (Paper 1).

Registered: PREREG_swave_theory_correction_20260808.md (constraints) +
PREREG_swave_wave_20260809.md (design + decision rules — implemented
verbatim here). ONE execution.

Inputs:
  --auc_csv       NEW collated adaptation csv (adaptation_auc schema:
                  run_id, mode, domain, seed, milestone, auc100k,
                  n_ep_100k, qc_pass). Domain-filtered to finger (the
                  08-08 B1 multi-domain lesson).
  --e4_true       E4 collate csv for TRUE-label passes — must contain
                  ONLY the s=1 cells (swr11/swr12/swr13, swv11).
  --e4_s245       E4 collate csv for the scale-2.45 own-label passes —
                  ONLY swr21/swr22/swr23.
  --e4_s446       E4 collate csv for the scale-4.46 own-label passes —
                  ONLY swr31/swr32/swr33, swv31.
  --fit_counters  json {run_name: {"update": U, "total": T}} dumped
                  from OFFLINE_FIT_PROGRESS at collate time; any
                  U != T or missing fit REFUSES the read (08-07
                  standing rule).

Cells: mode = ax1sw{r|v}<si><wi>q1s0; si -> s in {1, 2.45, 4.46},
wi -> w_r in {1, 10, 100}. rgo grid = 9 cells, vgo leg = swv11/swv31.
8 seeds/cell registered; qc/sub-modal exclusions disclosed, cell
refuses below n=6.

Rules (run-clustered BCa 95% B=10K rng 0 + permutation p; fires need
both):
  P-SW1  rew_nll_in(h0, own-label) [swr31] - [swr11] : CI < 0 & p<.05
  P-SW2  rew_nll_in(h0, true)      [swr13] - [swr11] : CI < 0 & p<.05
  P-SW3  auc100k: each of the 8 non-baseline rgo cells vs swr11;
         BH(q=.05) across the 8 two-sided permutation p's; a contrast
         is a survivor iff BH-significant AND its CI excludes 0
  P-SW4  registered dissociation map (verdict wording from SW1-3)
  P-SW5  rew_nll_in(h0, own-label) [swv31] - [swv11] : CI < 0 & p<.05

Usage:
  python -m analysis.swave_read --auc_csv ... --e4_true ... \
      --e4_s245 ... --e4_s446 ... --fit_counters ... --output <dir>
  python -m analysis.swave_read --selfcheck
"""

import argparse
import csv
import json
import math
import os
import re

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
MODE_RE = re.compile(r'^ax1sw(?P<arm>[rv])(?P<si>[123])(?P<wi>[123])q1s0$')
WM_RE = re.compile(
    r'^ax1wm_finger_sw(?P<arm>[rv])(?P<si>[123])(?P<wi>[123])q1s0'
    r'_seed(?P<seed>\d+)$')
RGO_CELLS = [f'r{si}{wi}' for si in '123' for wi in '123']
VGO_CELLS = ['v11', 'v31']
BASELINE = 'r11'
CONTRAST_CELLS = [c for c in RGO_CELLS if c != BASELINE]
E4_SCOPE = dict(true={'r11', 'r12', 'r13', 'v11'},
                s245={'r21', 'r22', 'r23'},
                s446={'r31', 'r32', 'r33', 'v31'})
SEEDS_PER_CELL = 8
MIN_CELL_N = 6
MILESTONE = '500000'
# Batch-review F1: symexp_twohot achievable-NLL floors per rewarded
# frame depend on where symlog(s) falls between bins (spacing 20/127) —
# a DETERMINISTIC cross-scale offset in the fire direction. Registered
# constants (entropy of the two-hot weights at symlog(s)):
#   s=1: ln2/Δ frac .4013 -> 0.6735 ; s=2.45: ln3.45/Δ frac .8638 ->
#   0.3981 ; s=4.46: ln5.46/Δ frac .7787 -> 0.5286. Zero labels sit
#   exactly on a bin (floor 0), so the per-run floor contribution is
#   FLOOR[s] * n_in_rew/(n_in_rew + n_in_norew). All NLL rules run on
#   floor-CORRECTED values.
TWOHOT_FLOOR = {'1': 0.6735, '2': 0.3981, '3': 0.5286}
OVERRIDE_EXPECT = dict(true=None, s245=('scale', 2.45),
                       s446=('scale', 4.46))


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
  """diff = mean(a) - mean(b); BCa 95% CI + two-sided permutation p."""
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
  # permutation
  obs = abs(diff)
  count = 0
  for _ in range(b_boot):
    perm = rng.permutation(pooled)
    if abs(perm[:len(a)].mean() - perm[len(a):].mean()) >= obs - 1e-15:
      count += 1
  p = float((count + 1) / (b_boot + 1))
  return dict(diff=diff, ci=list(ci), perm_p=p, n=[len(a), len(b)])


def bh_survivors(pvals, q=0.05):
  """Benjamini-Hochberg: returns the boolean survivor mask."""
  pvals = np.asarray(pvals, float)
  order = np.argsort(pvals)
  m = len(pvals)
  thresh = q * (np.arange(1, m + 1)) / m
  passed = pvals[order] <= thresh
  k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
  mask = np.zeros(m, bool)
  mask[order[:k]] = True
  return mask


def load_auc(path):
  """Cell -> seed-ordered auc100k values. Guards: finger-only (B1),
  milestone pin, duplicate fatal, qc excluded+counted, modal-n_ep
  sub-window excluded+counted, exactly SEEDS_PER_CELL rows pre-QC."""
  rows = list(csv.DictReader(open(path)))
  cells, pre_counts = {}, {}
  excl = dict(qc=0, subwindow=0)
  neps = [int(float(r['n_ep_100k'])) for r in rows
          if r.get('domain') == 'finger' and MODE_RE.match(r['mode'])
          and r['milestone'] == MILESTONE]
  modal = max(set(neps), key=neps.count) if neps else 0
  seen = set()
  for r in rows:
    if r.get('domain') != 'finger' or r['milestone'] != MILESTONE:
      continue
    m = MODE_RE.match(r['mode'])
    if not m:
      continue
    cell = m.group('arm') + m.group('si') + m.group('wi')
    key = (cell, int(r['seed']))
    assert key not in seen, f'duplicate row {key} in {path}'
    seen.add(key)
    pre_counts[cell] = pre_counts.get(cell, 0) + 1
    if r['qc_pass'] not in ('1', 'True', 'true'):
      excl['qc'] += 1
      continue
    if int(float(r['n_ep_100k'])) < modal:
      excl['subwindow'] += 1
      continue
    cells.setdefault(cell, []).append(float(r['auc100k']))
  for cell, n in pre_counts.items():
    assert n == SEEDS_PER_CELL, (
        f'{cell}: {n} rows pre-QC != registered {SEEDS_PER_CELL}')
  for cell in RGO_CELLS + VGO_CELLS:
    assert cell in pre_counts, f'missing cell {cell} in {path}'
    assert len(cells.get(cell, [])) >= MIN_CELL_N, (
        f'{cell}: {len(cells.get(cell, []))} usable seeds < {MIN_CELL_N}'
        ' — cell refuses')
  return cells, excl, modal


def load_e4(path, scope_key):
  """Cell -> floor-CORRECTED rew_nll_in values at h=0 (F1). Scope-
  enforced two ways: run names per csv, AND the collate's override
  provenance columns (F4) — own-vs-true discipline is structural, not
  trusted."""
  rows = list(csv.DictReader(open(path)))
  cells = {}
  seen = set()
  expect_ov = OVERRIDE_EXPECT[scope_key]
  for r in rows:
    if int(r['horizon']) != 0:
      continue
    m = WM_RE.match(r['run_id'])
    if not m:
      continue
    cell = m.group('arm') + m.group('si') + m.group('wi')
    assert cell in E4_SCOPE[scope_key], (
        f'{r["run_id"]} in the {scope_key} csv: cell {cell} is outside '
        f'the registered scope {sorted(E4_SCOPE[scope_key])} — '
        'own/true-label discipline violated, read refuses')
    # F4: label provenance machine check per row.
    kind = (r.get('override_kind') or '').strip()
    scale = (r.get('override_scale') or '').strip()
    if expect_ov is None:
      assert not kind and not scale, (
          f'{r["run_id"]} in the true csv carries an override '
          f'({kind!r}, {scale!r}) — read refuses')
    else:
      assert kind == expect_ov[0] and scale and (
          abs(float(scale) - expect_ov[1]) < 1e-9), (
          f'{r["run_id"]} in the {scope_key} csv: override '
          f'({kind!r}, {scale!r}) != registered {expect_ov} — '
          'read refuses')
    key = (cell, int(m.group('seed')))
    assert key not in seen, f'duplicate {key} in {path}'
    seen.add(key)
    n_rew = float(r['n_in_rew'])
    n_norew = float(r['n_in_norew'])
    frac = n_rew / max(n_rew + n_norew, 1.0)
    corr = float(r['rew_nll_in']) - TWOHOT_FLOOR[m.group('si')] * frac
    cells.setdefault(cell, []).append(corr)
  for cell in sorted(E4_SCOPE[scope_key]):
    assert len(cells.get(cell, [])) >= MIN_CELL_N, (
        f'{cell}: {len(cells.get(cell, []))} e4 rows < {MIN_CELL_N} '
        f'in {path}')
  return cells


def check_counters(path):
  data = json.load(open(path))
  expect = [f'ax1wm_finger_sw{c[0]}{c[1]}{c[2]}q1s0_seed{k}'
            for c in RGO_CELLS + VGO_CELLS
            for k in range(1, SEEDS_PER_CELL + 1)]
  for run in expect:
    assert run in data, f'fit counter missing for {run} — read refuses'
    u, t = int(data[run]['update']), int(data[run]['total'])
    assert u == t, (f'{run}: fit counter {u}/{t} — truncated fit, '
                    'read refuses (08-07 standing rule)')
  return len(expect)


def run_read(args):
  auc, excl, modal = load_auc(args.auc_csv)
  e4 = {}
  e4.update(load_e4(args.e4_true, 'true'))
  e4.update(load_e4(args.e4_s245, 's245'))
  e4.update(load_e4(args.e4_s446, 's446'))
  n_counters = check_counters(args.fit_counters)

  out = dict(modal_n_ep=modal, exclusions=excl, n_fit_counters=n_counters)
  sw1 = two_sample(e4['r31'], e4['r11'])
  out['p_sw1_s_membership'] = sw1
  out['p_sw1_fires'] = bool(sw1['ci'][1] < 0 and sw1['perm_p'] < 0.05)
  sw2 = two_sample(e4['r13'], e4['r11'])
  out['p_sw2_wr_membership'] = sw2
  out['p_sw2_fires'] = bool(sw2['ci'][1] < 0 and sw2['perm_p'] < 0.05)

  contrasts = {}
  pvals = []
  for cell in CONTRAST_CELLS:
    st = two_sample(auc[cell], auc[BASELINE], rng_seed=RNG_SEED)
    contrasts[cell] = st
    pvals.append(st['perm_p'])
  mask = bh_survivors(pvals)
  survivors = [c for c, keep, st in
               zip(CONTRAST_CELLS, mask, contrasts.values())
               if keep and (st['ci'][0] > 0 or st['ci'][1] < 0)]
  out['p_sw3_contrasts'] = contrasts
  out['p_sw3_bh_survivors'] = survivors

  sw5 = two_sample(e4['v31'], e4['v11'])
  out['p_sw5_vgo'] = sw5
  out['p_sw5_fires'] = bool(sw5['ci'][1] < 0 and sw5['perm_p'] < 0.05)

  membership = out['p_sw1_fires'] or out['p_sw2_fires']
  behav = bool(survivors)
  # Precedence (registered): the SPECIFIC s-inert-while-w_r-moves
  # branch outranks the generic membership branch — it is the sharper
  # mechanism claim (label ENERGY is not the mechanism).
  if (not out['p_sw1_fires']) and out['p_sw2_fires'] and not behav:
    verdict = ('ENERGY-IS-NOT-THE-MECHANISM: s inert while w_r moves — '
               'rewarded-frame COUNT account (feeds Paper-3 D3)')
  elif membership and not behav:
    verdict = ('INCLUSION-NOT-USEFULNESS-NEW-AXIS: membership moves '
               '(s and/or w_r) with no behavioral response')
  elif behav and not membership:
    verdict = ('BEHAVIOR-WITHOUT-MEMBERSHIP: registered anomaly — '
               'disclosed, no headline')
  elif membership and behav:
    verdict = 'BOTH-MOVE: dose-responsive axis, full map in contrasts'
  else:
    verdict = ('BOTH-AXES-INERT: corrected a^2-term refuted in the '
               'tested range (registered honest theory negative)')
  out['p_sw4_verdict'] = verdict

  # descriptive s-shape (no decision weight): which regressor tracks
  s_vals = np.array([1.0, 2.45, 4.46])
  nll = np.array([np.mean(e4['r11']), np.mean(e4['r21']),
                  np.mean(e4['r31'])])
  for name, x in (('ln1p_sq', np.log1p(s_vals) ** 2),
                  ('raw_sq', s_vals ** 2)):
    X = np.stack([np.ones(3), x], 1)
    beta, res, *_ = np.linalg.lstsq(X, nll, rcond=None)
    ss_tot = float(np.sum((nll - nll.mean()) ** 2))
    ss_res = float(res[0]) if len(res) else float(
        np.sum((nll - X @ beta) ** 2))
    out[f'shape_r2_{name}'] = (1.0 - ss_res / ss_tot) if ss_tot else None

  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, 'swave.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _write_fixture(tmp, auc_shift=None, e4_shift=None, seeds=8,
                   contaminate=False, dup=False, wrong_scope=False,
                   short_counter=False, e4_frac=0.3, e4_noise=0.15,
                   bad_override=False):
  rng = np.random.default_rng(3)
  auc_shift = auc_shift or {}
  e4_shift = e4_shift or {}
  auc_rows = []
  for cell in RGO_CELLS + VGO_CELLS:
    for k in range(1, seeds + 1):
      auc_rows.append(dict(
          run_id=f'adapt_ax1sw{cell}q1s0_finger_seed{k}_ckpt500000',
          mode=f'ax1sw{cell}q1s0', domain='finger', seed=k,
          milestone=MILESTONE,
          auc100k=round(100 + auc_shift.get(cell, 0.0)
                        + rng.normal(0, 8), 3),
          n_ep_100k=40, qc_pass=1))
  if contaminate:
    auc_rows.append(dict(run_id='adapt_ax1swr11q1s0_cup_seed1_ckpt500000',
                         mode='ax1swr11q1s0', domain='cup', seed=1,
                         milestone=MILESTONE, auc100k=600.0,
                         n_ep_100k=40, qc_pass=1))
  if dup:
    auc_rows.append(dict(auc_rows[0]))
  auc_path = os.path.join(tmp, 'auc.csv')
  with open(auc_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(auc_rows[0]))
    w.writeheader()
    w.writerows(auc_rows)

  e4_paths = {}
  ov_cols = dict(true=dict(override_kind='', override_scale=''),
                 s245=dict(override_kind='scale', override_scale=2.45),
                 s446=dict(override_kind='scale', override_scale=4.46))
  if bad_override:
    ov_cols['s245'] = dict(override_kind='scale', override_scale=4.46)
  for scope, cells in E4_SCOPE.items():
    rows = []
    for cell in sorted(cells):
      si = cell[1]
      for k in range(1, seeds + 1):
        # raw value = floor component (frac * FLOOR[si]) + true
        # component + noise: the reader must recover the true
        # component (F1 floor correction).
        raw = (TWOHOT_FLOOR[si] * e4_frac + 1.2
               + e4_shift.get(cell, 0.0) + rng.normal(0, e4_noise))
        rows.append(dict(
            run_id=f'ax1wm_finger_sw{cell}q1s0_seed{k}',
            horizon=0, rew_nll_in=round(raw, 5),
            n_in_rew=int(1000 * e4_frac),
            n_in_norew=int(1000 * (1 - e4_frac)),
            **ov_cols[scope]))
    if wrong_scope and scope == 'true':
      rows.append(dict(run_id='ax1wm_finger_swr21q1s0_seed1', horizon=0,
                       rew_nll_in=1.0, n_in_rew=300, n_in_norew=700,
                       override_kind='', override_scale=''))
    p = os.path.join(tmp, f'e4_{scope}.csv')
    with open(p, 'w', newline='') as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0]))
      w.writeheader()
      w.writerows(rows)
    e4_paths[scope] = p

  counters = {f'ax1wm_finger_sw{c}q1s0_seed{k}':
              dict(update=500000, total=500000)
              for c in RGO_CELLS + VGO_CELLS
              for k in range(1, seeds + 1)}
  if short_counter:
    counters['ax1wm_finger_swr22q1s0_seed3'] = dict(update=311_000,
                                                    total=500000)
  cpath = os.path.join(tmp, 'counters.json')
  json.dump(counters, open(cpath, 'w'))

  class A:
    pass
  a = A()
  a.auc_csv = auc_path
  a.e4_true, a.e4_s245, a.e4_s446 = (e4_paths['true'], e4_paths['s245'],
                                     e4_paths['s446'])
  a.fit_counters = cpath
  a.output = os.path.join(tmp, 'out')
  return a


def selfcheck():
  import tempfile
  # 1) null: nothing fires, verdict = both-axes-inert
  with tempfile.TemporaryDirectory() as tmp:
    out = run_read(_write_fixture(tmp))
    assert not out['p_sw1_fires'] and not out['p_sw2_fires']
    assert not out['p_sw3_bh_survivors']
    assert not out['p_sw5_fires']
    assert out['p_sw4_verdict'].startswith('BOTH-AXES-INERT'), out
  # 2) planted s-membership (own-label NLL drop at swr31): SW1 fires,
  # verdict = inclusion-not-usefulness
  with tempfile.TemporaryDirectory() as tmp:
    out = run_read(_write_fixture(tmp, e4_shift={'r31': -0.8}))
    assert out['p_sw1_fires'] and not out['p_sw2_fires']
    assert out['p_sw4_verdict'].startswith('INCLUSION-NOT-USEFULNESS')
  # 3) planted w_r membership only: energy-not-mechanism branch
  with tempfile.TemporaryDirectory() as tmp:
    out = run_read(_write_fixture(tmp, e4_shift={'r13': -0.8}))
    assert out['p_sw2_fires'] and not out['p_sw1_fires']
    assert out['p_sw4_verdict'].startswith('ENERGY-IS-NOT'), out
  # 4) planted behavior in exactly two cells: BH survivors exact;
  # with membership also planted -> BOTH-MOVE
  with tempfile.TemporaryDirectory() as tmp:
    out = run_read(_write_fixture(
        tmp, auc_shift={'r32': 40.0, 'r23': 40.0},
        e4_shift={'r31': -0.8}))
    assert set(out['p_sw3_bh_survivors']) == {'r32', 'r23'}, out
    assert out['p_sw4_verdict'].startswith('BOTH-MOVE')
  # 5) vgo repair fires independently
  with tempfile.TemporaryDirectory() as tmp:
    out = run_read(_write_fixture(tmp, e4_shift={'v31': -0.8}))
    assert out['p_sw5_fires'] and not out['p_sw1_fires']
  # 5b) FLOOR-KILL (F1): identical fits whose raw NLLs differ ONLY by
  # the twohot floor (frac=1, tiny noise) must NOT fire after
  # correction — and the uncorrected contrast WOULD have (mutant
  # regression: floor diff r31-r11 = -0.145 at ~6 sigma here)
  with tempfile.TemporaryDirectory() as tmp:
    a = _write_fixture(tmp, e4_frac=1.0, e4_noise=0.05)
    out = run_read(a)
    assert not out['p_sw1_fires'] and not out['p_sw5_fires'], out
    raw31 = [float(r['rew_nll_in']) for r in
             csv.DictReader(open(a.e4_s446))
             if r['run_id'].startswith('ax1wm_finger_swr31')]
    raw11 = [float(r['rew_nll_in']) for r in
             csv.DictReader(open(a.e4_true))
             if r['run_id'].startswith('ax1wm_finger_swr11')]
    naive = two_sample(raw31, raw11)
    assert naive['ci'][1] < 0 and naive['perm_p'] < 0.05, (
        'floor-kill fixture lost its bite', naive)
  # 5c) behav-only verdict branch (M3): behavioral response with no
  # membership -> registered anomaly wording
  with tempfile.TemporaryDirectory() as tmp:
    out = run_read(_write_fixture(tmp, auc_shift={'r32': 40.0}))
    assert out['p_sw4_verdict'].startswith('BEHAVIOR-WITHOUT'), out
  # 6) guards: cup contamination ignored (values unchanged vs clean)
  with tempfile.TemporaryDirectory() as tmp:
    clean = run_read(_write_fixture(tmp))
  with tempfile.TemporaryDirectory() as tmp:
    cont = run_read(_write_fixture(tmp, contaminate=True))
    assert (cont['p_sw3_contrasts']['r12']['diff']
            == clean['p_sw3_contrasts']['r12']['diff'])
  # 7) duplicate row fatal
  import contextlib
  for kw, msg in ((dict(dup=True), 'duplicate'),
                  (dict(wrong_scope=True), 'scope'),
                  (dict(short_counter=True), 'counter'),
                  (dict(bad_override=True), 'override provenance'),
                  (dict(seeds=5), 'rows pre-QC')):
    with tempfile.TemporaryDirectory() as tmp:
      try:
        with contextlib.redirect_stdout(__import__('io').StringIO()):
          run_read(_write_fixture(tmp, **kw))
        raise SystemExit(f'selfcheck FAIL: guard not tripped: {kw}')
      except AssertionError:
        pass
  print('SELFCHECK PASS (swave_read: null no-fire; planted SW1/SW2/'
        'SW3-BH-exact/SW5 fires; ALL FIVE verdict branches exercised '
        'incl. behav-only; FLOOR-KILL leg — floor-only difference '
        'no-fires corrected while the naive form fires; cup-'
        'contamination inert; duplicate/scope/override-provenance/'
        'counter/row-count guards trip)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--auc_csv')
  p.add_argument('--e4_true')
  p.add_argument('--e4_s245')
  p.add_argument('--e4_s446')
  p.add_argument('--fit_counters')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  need = [args.auc_csv, args.e4_true, args.e4_s245, args.e4_s446,
          args.fit_counters, args.output]
  if not all(need):
    p.error('all inputs + --output required (or --selfcheck)')
  run_read(args)


if __name__ == '__main__':
  main()
