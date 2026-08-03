"""Frozen reader for the recon-detached pixel arm (rde wave).

Registered in prereg/PREREG_rde_pixel_20260802.md and committed BEFORE any
rde run exists. The arm: pixel WM fits identical to the X2 task cells
except reconstruction is detached from the trunk (agent.recon_grad False:
the decoder trains on stop-gradient latents, so enc+dyn are shaped by
reward/value/con/dyn losses only) — the complement of the pe arm, which
removed the competition but froze reward-free features. Registered
predictions: P-RD1 inclusion restored at the representation level, P-RD2
frozen-readout adaptation lifts off the committed X2 floor, P-RD3
(conditional) the occupancy split re-engages.

New vs the pe reader — the fire branch carries two extra conjuncts
(post-review, both anti-masquerade):
1. TRIVIAL-FLOOR DOMINANCE: with a trainable trunk, the reward head can
   regress to the stratum-marginal, whose in-regime NLL sits below the
   2.0-nat swamping bar. NLL in [floor, 2.0) is at-or-worse-than the
   best constant predictor, so the fire requires the CI entirely BELOW
   the floor; otherwise NOT-BETTER-THAN-CONSTANT (no inclusion claim).
2. LATENT-ALIVE WITNESS: E4 `deter_std` (across-state std of h=0
   posterior deter features) must clear ALIVE_FRAC x the median of the
   fpxpx calibration pass for EVERY rde fit; a numeric fire without it
   reads DEGENERATE-MASQUERADE.
Under PARTIAL-RELIEF and NO-RELIEF the witness is a registered
disclosure (collapse-qualified), never decisional; a significantly
negative P-RD2 and a NO-RELIEF-with-lift cell each carry their own
registered disclosure clauses.

Usage:
  python -m analysis.rde_pixel_read --auc <canonical auc.csv> \
      --e4 <e4_fingerpx_v1_rde csv> --calib <e4_fingerpx_v1cal_fpxpx csv> \
      --output <dir>
  python -m analysis.rde_pixel_read selfcheck
"""

import argparse
import csv
import json
import os
import re

import numpy as np

from analysis.pixel_swamping_read import seed_cluster_ci
from analysis.unfrozen_stress_read import (
    load_cells, _boot, B_BOOT, RNG_SEED, SEEDS, DOMAIN, MILESTONE)

# Frozen-registration pins (fail loudly if the imported conventions drift).
assert SEEDS == tuple(range(1, 9)), SEEDS
assert DOMAIN == 'finger' and MILESTONE == '500000'
assert B_BOOT == 10_000 and RNG_SEED == 0

# Adapt cells (canonical csv modes) and the committed X2 task baseline.
RDE_MODES = dict(s0='ax1rdepxq1ms0', s1='ax1rdepxq1ms1')
X2_BASE_MODES = dict(s0='ax1pxpxq1ms0', s1='ax1pxpxq1ms1')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X2_BASELINE = os.path.join(_REPO, 'artifacts/pixel_x2_20260724/auc.csv')

# E4 fit-run naming and the committed swamping anchors
# (artifacts/pixel_swamping_20260725/pixel_swamping.json: task-arm
# rew_nll_in h=0 23.34 [19.41, 27.29]; trivial two-hot floor 0.6713 —
# same frozen probeset fingerpx_v1, so the floor transports).
RDE_FIT_RE = re.compile(r'^ax1wm_finger_rdepxq1ms([01])_seed(\d+)$')
CALIB_FIT_RE = re.compile(r'^ax1wm_finger_fpxpxq1ms([01])_seed(\d+)$')
# Full-precision committed values (pixel_swamping.json), not rounded —
# "entirely below the committed lower bound" means exactly that.
SW_BASE = dict(point=23.338872993169502,
               lo=19.409267325402837, hi=27.285688932142705)
THRESH_SWAMP = 2.0   # registered swamping bar (PREREG_pixel_swamping_20260724)
THRESH_MEMBER = 1.5  # membership descriptor bar (never decisional)
# TRIVIAL_FLOOR is DECISIONAL in this reader (unlike pe): it is the NLL of
# the best CONSTANT reward prediction on the probeset's in-regime frames,
# so the fire branch requires the CI to sit entirely BELOW it — a head
# that cannot beat a constant carries no state information and sub-2.0
# NLL alone is the marginal-head masquerade, not inclusion.
TRIVIAL_FLOOR = 0.671345852414428
# Latent-alive witness: every rde fit's deter_std must clear ALIVE_FRAC x
# median(calibration deter_std). Collapse is orders of magnitude, so 0.10
# is generous to genuine-but-recon-poor latents. CALIB_MIN guards the
# instrument itself: a near-zero calibration median means the witness
# cannot discriminate and the read REFUSES rather than adjudicates.
ALIVE_FRAC = 0.10
CALIB_MIN = 1e-4


def load_e4(path):
  """rde fit rows at h=0 -> ({seed: [side nll]}, {(side, seed): deter_std})."""
  by_seed = {}
  dstd = {}
  seen = set()
  with open(path) as f:
    for r in csv.DictReader(f):
      m = RDE_FIT_RE.match(r['run_id'])
      if not m:
        continue
      if int(r['horizon']) != 0:
        continue
      side, seed = int(m.group(1)), int(m.group(2))
      assert seed in SEEDS, f'unregistered seed {seed} in {r["run_id"]}'
      key = (side, seed)
      assert key not in seen, f'duplicate e4 row for side{side} seed{seed}'
      seen.add(key)
      assert r.get('reward_aware') in ('1', 'True', 'true'), (
          f'{r["run_id"]}: reward_aware={r.get("reward_aware")} — rde fits '
          'must be task-mode (reward_aware gate changed?)')
      v = float(r['rew_nll_in'])
      assert np.isfinite(v), f'{r["run_id"]}: non-finite rew_nll_in'
      d = r.get('deter_std')
      assert d not in (None, ''), (
          f'{r["run_id"]}: deter_std column missing — E4 ran with '
          'pre-witness stratified_error?')
      d = float(d)
      assert np.isfinite(d) and d >= 0, f'{r["run_id"]}: deter_std={d}'
      dstd[key] = d
      by_seed.setdefault(seed, []).append(v)
  missing = [s for s in SEEDS if len(by_seed.get(s, [])) != 2]
  assert not missing, (
      f'incomplete e4 grid — seeds without both sides at h=0: {missing}')
  return by_seed, dstd


def load_calib(path):
  """fpxpx calibration rows at h=0 -> alive threshold (with guards)."""
  vals = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = CALIB_FIT_RE.match(r['run_id'])
      if not m:
        continue
      if int(r['horizon']) != 0:
        continue
      side, seed = int(m.group(1)), int(m.group(2))
      assert seed in SEEDS, f'unregistered calib seed {seed}'
      key = (side, seed)
      assert key not in vals, f'duplicate calib row {key}'
      assert r.get('reward_aware') in ('0', 'False', 'false'), (
          f'{r["run_id"]}: calibration must be the reward-free fpxpx fits '
          f'(reward_aware={r.get("reward_aware")})')
      d = r.get('deter_std')
      assert d not in (None, ''), f'{r["run_id"]}: calib deter_std missing'
      d = float(d)
      assert np.isfinite(d) and d >= 0, f'{r["run_id"]}: deter_std={d}'
      vals[key] = d
  missing = [(sd, s) for sd in (0, 1) for s in SEEDS if (sd, s) not in vals]
  assert not missing, f'incomplete calibration grid: missing {missing}'
  med = float(np.median(list(vals.values())))
  assert med > CALIB_MIN, (
      f'calibration median deter_std {med} <= {CALIB_MIN} — witness cannot '
      'discriminate; instrument refuses (registered refusal, not a verdict)')
  return dict(values={f's{k[0]}_seed{k[1]}': v for k, v in vals.items()},
              median=med, threshold=ALIVE_FRAC * med)


def analyze(auc_csv, e4_csv, calib_csv):
  rng = np.random.default_rng(RNG_SEED)

  # Latent-alive witness (calibration first: its guards are instrument
  # gates and must trip before any estimand is computed).
  calib = load_calib(calib_csv)
  nll, dstd = load_e4(e4_csv)
  thr = calib['threshold']
  dead = sorted([f's{sd}_seed{s}' for (sd, s), v in dstd.items() if v < thr])
  alive_all = not dead

  # P-RD1: representation level (b/seed passed explicitly so the pinned
  # constants here govern, not pixel_swamping_read's module defaults).
  # The fire branch has THREE conjuncts, checked most-fundamental first:
  # sub-bar NLL alone is compatible with a marginal head (NLL in
  # [TRIVIAL_FLOOR, 2.0) means at-or-worse-than-constant), so inclusion
  # requires beating the constant (CI entirely below the floor) AND the
  # latent-alive witness. Neither extra conjunct touches the
  # PARTIAL/NO-RELIEF ladder, which stays P-PE1-comparable.
  point, lo, hi = seed_cluster_ci(nll, b=B_BOOT, seed=RNG_SEED)
  if hi < THRESH_SWAMP:
    if not hi < TRIVIAL_FLOOR:
      rd1 = 'NOT-BETTER-THAN-CONSTANT'
    elif not alive_all:
      rd1 = 'DEGENERATE-MASQUERADE'
    else:
      rd1 = 'INCLUSION-RESTORED'
  elif hi < SW_BASE['lo']:
    rd1 = 'PARTIAL-RELIEF'
  else:
    rd1 = 'NO-RELIEF'

  # P-RD2: behavioral lift vs the committed X2 task cells (paired per
  # seed and cell, mean across cells — the U4 relief estimator).
  assert_no_dup_rows(auc_csv, RDE_MODES)
  assert_no_dup_rows(X2_BASELINE, X2_BASE_MODES)
  cells = load_cells(auc_csv, RDE_MODES)
  base = load_cells(X2_BASELINE, X2_BASE_MODES)
  per_seed = np.mean([cells[k] - base[k] for k in ('s0', 's1')], axis=0)
  lift = _boot(per_seed, rng)
  lifts = lift['ci'][0] > 0
  neg = lift['ci'][1] < 0  # significantly BELOW the X2 floor

  # P-RD3 (conditional secondary): occupancy split within the rde arm.
  split = _boot(cells['s1'] - cells['s0'], rng)
  if lifts:
    rd3 = 'FIRES' if split['ci'][0] > 0 else 'null'
  else:
    rd3 = 'premise-idle'

  return dict(
      p_rd1=dict(point=point, ci=[lo, hi], verdict=rd1,
                 baseline=SW_BASE, thresh_swamp=THRESH_SWAMP,
                 member_band=(hi <= THRESH_MEMBER),
                 trivial_floor=TRIVIAL_FLOOR),
      p_rd2=dict(**lift, fires=bool(lifts), negative=bool(neg)),
      p_rd3=dict(**split, verdict=rd3),
      alive_witness=dict(
          threshold=thr, calib_median=calib['median'],
          alive_frac=ALIVE_FRAC, dead_fits=dead, alive_all=alive_all,
          deter_std={f's{k[0]}_seed{k[1]}': v for k, v in sorted(dstd.items())},
          calib=calib['values']),
      cell_means={k: float(np.mean(v)) for k, v in cells.items()},
      baseline_cell_means={k: float(np.mean(v)) for k, v in base.items()},
      verdict=_verdict(rd1, lifts, neg, dead),
      auc_csv=os.path.abspath(auc_csv), e4_csv=os.path.abspath(e4_csv),
      calib_csv=os.path.abspath(calib_csv), baseline_csv=X2_BASELINE)


def assert_no_dup_rows(path, modes):
  """auc-side duplicate guard (load_cells silently last-write-wins)."""
  seen = set()
  targets = set(modes.values())
  with open(path) as f:
    for r in csv.DictReader(f):
      if r['mode'] not in targets or r.get('domain', DOMAIN) != DOMAIN:
        continue
      if r.get('milestone', MILESTONE) != MILESTONE:
        continue
      key = (r['mode'], int(r['seed']))
      assert key not in seen, f'duplicate auc row for {key} in {path}'
      seen.add(key)


def _verdict(rd1, lifts, neg, dead):
  # Registered disclosure clauses, appended wherever they apply — none of
  # them changes the decision, but no cell is silently absorbed into a
  # narrative it does not support.
  qual = ('' if not dead else
          ' COLLAPSE-QUALIFIED: latent-alive witness failed for ' +
          ', '.join(dead) + ' — signal-poverty vs optimization-collapse '
          'not separable for those cells (registered disclosure).')
  negq = ('' if not neg else
          ' P-RD2 NEGATIVE: the rde arm sits significantly BELOW the X2 '
          'floor — outside registered accounts, reported as-is.')
  if rd1 == 'INCLUSION-RESTORED' and lifts:
    return ('COMPETITION-CONFIRMED: detaching reconstruction from the '
            'trunk restores reward inclusion AND unlocks transfer — the '
            'pixel boundary is a causal consequence of objective '
            'composition, the strongest form of the legibility headline.')
  if rd1 == 'INCLUSION-RESTORED' and not lifts:
    return ('INCLUDED-BUT-USELESS AT PIXEL: inclusion restored without '
            'behavioral lift — membership is necessary, not sufficient, '
            'in the pixel regime too (stamping-pattern analogue).' + negq)
  if rd1 == 'NOT-BETTER-THAN-CONSTANT':
    return ('NOT-BETTER-THAN-CONSTANT: rew-NLL clears the 2.0-nat '
            'swamping bar but the CI does not sit entirely below the '
            'trivial constant-predictor floor (' + f'{TRIVIAL_FLOOR:.4f}'
            + ') — the head is at-or-worse-than-constant on in-regime '
            'states, which is the marginal-head masquerade, not '
            'inclusion. No inclusion claim; the latent-alive panel '
            'discriminates collapsed-features vs alive-features-with-'
            'marginal-head descriptively.' + qual + negq)
  if rd1 == 'DEGENERATE-MASQUERADE':
    return ('DEGENERATE-MASQUERADE: the rew-NLL criterion fired but the '
            'latent-alive witness failed (' + ', '.join(dead) + ') — a '
            'collapsed trunk fits the stratum-marginal, which is not '
            'inclusion. No inclusion claim; wave uninformative on '
            'P-RD1.' + negq)
  if rd1 == 'PARTIAL-RELIEF':
    return ('PARTIAL-RELIEF: rew-NLL leaves the measured swamping range '
            'but not the swamping regime; directional support only, no '
            'headline change; behavioral leg reported as measured.'
            + qual + negq)
  base = ('NO-RELIEF: removing reconstruction from the trunk competition '
          'does not restore reward legibility — with the pe arm, both '
          'directions of the swamping lever have now failed; the causal '
          'competition account is retired and the pixel boundary stands '
          'as a modality-level scope fact.')
  if lifts:
    base = ('NO-RELIEF (representation) WITH BEHAVIORAL LIFT: rew-NLL '
            'stays in the swamping regime but P-RD2 fired — transfer '
            'without measurable inclusion is outside registered '
            'accounts; the retirement of the competition account applies '
            'to the representation claim only, and the behavioral cell '
            'is reported as-is for follow-up registration.')
  return base + qual + negq


def selfcheck(_args=None):
  import tempfile
  tmp = tempfile.mkdtemp()

  def write_auc(path, rde_by_cell, base_by_cell):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'mode', 'domain', 'seed', 'milestone',
                  'auc100k', 'qc_pass'])
      for cells, prefix in ((rde_by_cell, 'adapt_rde'),
                            (base_by_cell, 'adapt_x2')):
        for mode, vals in cells.items():
          for s, v in zip(SEEDS, vals):
            w.writerow([f'{prefix}_{mode}_seed{s}', mode, 'finger', s,
                        '500000', v, 1])

  def write_e4(path, nll_by_cell, aware='1', dstd=1.0, dstd_map=None):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'horizon', 'rew_nll_in', 'reward_aware',
                  'deter_std'])
      for side in (0, 1):
        for s, v in zip(SEEDS, nll_by_cell[side]):
          d = dstd_map.get((side, s), dstd) if dstd_map else dstd
          w.writerow([f'ax1wm_finger_rdepxq1ms{side}_seed{s}', 0, v,
                      aware, d])
          w.writerow([f'ax1wm_finger_rdepxq1ms{side}_seed{s}', 5, v + 1,
                      aware, d])

  def write_calib(path, dstd=1.0, aware='0', drop=None, dstd_map=None):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'horizon', 'nll_all', 'reward_aware',
                  'deter_std'])
      for side in (0, 1):
        for s in SEEDS:
          if drop and (side, s) == drop:
            continue
          d = dstd_map.get((side, s), dstd) if dstd_map else dstd
          w.writerow([f'ax1wm_finger_fpxpxq1ms{side}_seed{s}', 0, 30.0,
                      aware, d])
          w.writerow([f'ax1wm_finger_fpxpxq1ms{side}_seed{s}', 5, 33.0,
                      aware, d])

  base = {m: [78 + i for i in range(8)] for m in X2_BASE_MODES.values()}
  # sub_nll: genuine fire — CI entirely below the trivial floor 0.6713.
  sub_nll = {0: [0.32 + 0.02 * i for i in range(8)],
             1: [0.30 + 0.02 * i for i in range(8)]}
  # floor_nll: sub-2.0 but NOT sub-floor — the marginal-head masquerade
  # region [floor, 2.0) that must NEVER fire (B2 fixture; this was the
  # pre-review Case-1 fixture, asserted then as a fire).
  floor_nll = {0: [1.1 + 0.02 * i for i in range(8)],
               1: [1.0 + 0.02 * i for i in range(8)]}
  # straddle_nll: per-seed means alternate ~1.2/~2.8 so the CI straddles
  # the 2.0 bar (kills an hi->lo mutant of the fire criterion).
  straddle_nll = {0: [1.2, 2.8] * 4, 1: [1.2, 2.8] * 4}
  # band_nll: CI entirely inside (19.41, 23.34) — NO-RELIEF, kills a
  # lower-bound->point mutant of the PARTIAL criterion.
  band_nll = {0: [20.5 + 0.1 * i for i in range(8)],
              1: [21.0 + 0.1 * i for i in range(8)]}
  hi_nll = {0: [21 + i for i in range(8)], 1: [22 + i for i in range(8)]}
  mid_nll = {0: [8 + 0.1 * i for i in range(8)],
             1: [9 + 0.1 * i for i in range(8)]}

  auc = os.path.join(tmp, 'auc.csv')
  e4 = os.path.join(tmp, 'e4.csv')
  cal = os.path.join(tmp, 'calib.csv')
  write_calib(cal)

  # Case 1: full confirmation + RD3 fires (alive witness passing).
  rde = {RDE_MODES['s0']: [140 + i for i in range(8)],
         RDE_MODES['s1']: [220 + i for i in range(8)]}
  write_auc(auc, rde, base); write_e4(e4, sub_nll)
  global X2_BASELINE
  real_baseline = X2_BASELINE
  X2_BASELINE = auc
  try:
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'INCLUSION-RESTORED', r['p_rd1']
    assert r['p_rd1']['member_band'] is True
    assert r['alive_witness']['alive_all'] is True
    assert r['p_rd2']['fires'] and r['p_rd3']['verdict'] == 'FIRES'
    assert r['verdict'].startswith('COMPETITION-CONFIRMED')

    # Case 1b: same NLLs but one collapsed fit -> DEGENERATE-MASQUERADE
    # (the anti-masquerade guard is the load-bearing new piece).
    write_e4(e4, sub_nll, dstd_map={(1, 3): 0.05})  # 0.05 < 0.10*median(1.0)
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'DEGENERATE-MASQUERADE', r['p_rd1']
    assert r['alive_witness']['dead_fits'] == ['s1_seed3']
    assert r['verdict'].startswith('DEGENERATE-MASQUERADE')

    # Case 1c: borderline-alive (just above threshold) stays restored.
    write_e4(e4, sub_nll, dstd_map={(1, 3): 0.11})
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'INCLUSION-RESTORED', r['p_rd1']

    # Case 1d (B2): sub-2.0 but not sub-floor, witness healthy — the
    # marginal-head masquerade region must NOT fire.
    write_e4(e4, floor_nll)
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'NOT-BETTER-THAN-CONSTANT', r['p_rd1']
    assert r['verdict'].startswith('NOT-BETTER-THAN-CONSTANT')

    # Case 1e (skewed calibration): median-thr 0.10 vs mean-thr 0.719 —
    # rde dstd 0.5 is alive under the registered median rule only.
    write_calib(cal, dstd_map={(0, 1): 100.0})
    write_e4(e4, sub_nll, dstd=0.5)
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'INCLUSION-RESTORED', r['p_rd1']
    assert abs(r['alive_witness']['calib_median'] - 1.0) < 1e-12
    write_calib(cal)  # restore

    # Case 2: inclusion restored, no lift (stamping analogue) + RD3 idle.
    rde2 = {RDE_MODES['s0']: base[X2_BASE_MODES['s0']],
            RDE_MODES['s1']: base[X2_BASE_MODES['s1']]}
    write_auc(auc, rde2, base); write_e4(e4, sub_nll)
    r = analyze(auc, e4, cal)
    assert not r['p_rd2']['fires'] and r['p_rd3']['verdict'] == 'premise-idle'
    assert r['verdict'].startswith('INCLUDED-BUT-USELESS')

    # Case 2b (M4): lift point positive but CI spans 0 — must not fire.
    rde3 = {RDE_MODES['s0']: [base[X2_BASE_MODES['s0']][i]
                              + (12 if i % 2 == 0 else -8) for i in range(8)],
            RDE_MODES['s1']: [base[X2_BASE_MODES['s1']][i]
                              + (12 if i % 2 == 0 else -8) for i in range(8)]}
    write_auc(auc, rde3, base)
    r = analyze(auc, e4, cal)
    assert r['p_rd2']['point'] > 0 and not r['p_rd2']['fires'], r['p_rd2']
    assert not r['verdict'].startswith('COMPETITION-CONFIRMED')

    # Case 2c (M2): significantly negative lift — disclosed, not absorbed.
    rden = {m: [v - 59 for v in base[b]] for m, b in
            zip(RDE_MODES.values(), X2_BASE_MODES.values())}
    write_auc(auc, rden, base)
    r = analyze(auc, e4, cal)
    assert r['p_rd2']['negative'] is True
    assert 'P-RD2 NEGATIVE' in r['verdict'], r['verdict']

    # Case 3: partial relief (restore a lifting auc first).
    write_auc(auc, rde, base); write_e4(e4, mid_nll)
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'PARTIAL-RELIEF', r['p_rd1']
    assert r['verdict'].startswith('PARTIAL-RELIEF')

    # Case 3b (M4): CI straddling the 2.0 bar is PARTIAL, never a fire.
    write_e4(e4, straddle_nll)
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['ci'][0] < THRESH_SWAMP < r['p_rd1']['ci'][1], r['p_rd1']
    assert r['p_rd1']['verdict'] == 'PARTIAL-RELIEF', r['p_rd1']

    # Case 3c (M1): partial relief under collapse carries the qualifier.
    write_e4(e4, mid_nll, dstd_map={(0, 4): 0.0})
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'PARTIAL-RELIEF'
    assert 'COLLAPSE-QUALIFIED' in r['verdict'], r['verdict']

    # Case 4: no relief (alive) — clean retirement, no qualifier. The
    # auc still lifts here, so the M3 branch text must engage.
    write_e4(e4, hi_nll)
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'NO-RELIEF'
    assert r['verdict'].startswith('NO-RELIEF (representation) WITH '
                                   'BEHAVIORAL LIFT'), r['verdict']
    assert 'COLLAPSE-QUALIFIED' not in r['verdict']

    # Case 4b (M4): CI inside (19.41, 23.34) — NO-RELIEF, not partial.
    write_auc(auc, rde2, base); write_e4(e4, band_nll)
    r = analyze(auc, e4, cal)
    assert SW_BASE['lo'] < r['p_rd1']['ci'][0] \
        and r['p_rd1']['ci'][1] < SW_BASE['point'], r['p_rd1']
    assert r['p_rd1']['verdict'] == 'NO-RELIEF', r['p_rd1']
    assert r['verdict'].startswith('NO-RELIEF: '), r['verdict']

    # Case 4c: no relief with a dead fit — verdict unchanged, qualifier on.
    write_e4(e4, hi_nll, dstd_map={(0, 2): 0.0})
    r = analyze(auc, e4, cal)
    assert r['p_rd1']['verdict'] == 'NO-RELIEF'
    assert 'COLLAPSE-QUALIFIED' in r['verdict']
    assert r['alive_witness']['dead_fits'] == ['s0_seed2']

    # Guard trips.
    def trips(fn):
      try:
        fn()
      except AssertionError:
        return True
      return False

    def missing_side():
      with open(e4, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'horizon', 'rew_nll_in', 'reward_aware',
                    'deter_std'])
        for s, v in zip(SEEDS[:-1], sub_nll[0][:7]):
          w.writerow([f'ax1wm_finger_rdepxq1ms0_seed{s}', 0, v, '1', 1.0])
      load_e4(e4)
    assert trips(missing_side), 'missing-side must trip'

    write_e4(e4, sub_nll, aware='0')
    assert trips(lambda: load_e4(e4)), 'reward_aware=0 must trip'

    def dup():
      write_e4(e4, sub_nll)
      with open(e4, 'a', newline='') as f:
        csv.writer(f).writerow(
            ['ax1wm_finger_rdepxq1ms0_seed1', 0, 1.0, '1', 1.0])
      load_e4(e4)
    assert trips(dup), 'duplicate must trip'

    def nan_nll():
      write_e4(e4, {0: [float('nan')] * 8, 1: sub_nll[1]})
      load_e4(e4)
    assert trips(nan_nll), 'nan must trip'

    def no_dstd_col():
      with open(e4, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'horizon', 'rew_nll_in', 'reward_aware'])
        for side in (0, 1):
          for s, v in zip(SEEDS, sub_nll[side]):
            w.writerow([f'ax1wm_finger_rdepxq1ms{side}_seed{s}', 0, v, '1'])
      load_e4(e4)
    assert trips(no_dstd_col), 'missing deter_std column must trip'

    def calib_missing():
      write_calib(cal, drop=(1, 5))
      load_calib(cal)
    assert trips(calib_missing), 'incomplete calibration must trip'

    def calib_flat():
      write_calib(cal, dstd=1e-6)
      load_calib(cal)
    assert trips(calib_flat), 'flat calibration must REFUSE'

    def calib_aware():
      write_calib(cal, aware='1')
      load_calib(cal)
    assert trips(calib_aware), 'reward-aware calibration rows must trip'
    write_calib(cal)  # restore

    def bad_qc():
      write_auc(auc, rde, base)
      rows = list(csv.reader(open(auc)))
      rows[1][-1] = '0'
      with open(auc, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
      load_cells(auc, RDE_MODES)
    assert trips(bad_qc), 'qc fail must trip'

    def dup_auc():
      write_auc(auc, rde, base)
      with open(auc, 'a', newline='') as f:
        csv.writer(f).writerow(
            ['adapt_rde_dup', RDE_MODES['s0'], 'finger', 1, '500000',
             99.0, 1])
      assert_no_dup_rows(auc, RDE_MODES)
    assert trips(dup_auc), 'duplicate auc row must trip'
  finally:
    X2_BASELINE = real_baseline

  print('selfcheck PASS: confirmed/masquerade/borderline/'
        'not-better-than-constant/skewed-calib/included-useless/'
        'span0-lift/negative-lift/partial(+straddle+collapse-qualified)/'
        'no-relief(+band+lift+collapse-qualified) branches + RD3 '
        'fires/idle + guard trips (missing-side, reward_aware, '
        'e4-duplicate, nan, missing-deter_std, calib-incomplete, '
        'calib-refusal, calib-aware, qc, auc-duplicate)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', nargs='?', choices=('selfcheck',))
  ap.add_argument('--auc')
  ap.add_argument('--e4')
  ap.add_argument('--calib')
  ap.add_argument('--output', default='analysis_out/rde_pixel')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck(args)
    return
  if not (args.auc and args.e4 and args.calib):
    ap.error('--auc, --e4 and --calib required (or selfcheck)')
  res = analyze(args.auc, args.e4, args.calib)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'rde_pixel.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  print(json.dumps(res, indent=2))
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
