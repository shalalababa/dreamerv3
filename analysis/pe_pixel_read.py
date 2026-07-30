"""Frozen reader for the pretrained-encoder pixel arm (PE wave).

Registered in prereg/PREREG_pe_pixel_20260730.md and committed BEFORE any
pe run exists. The arm: stage-2 pixel WM fits that load ONLY the encoder
from the seed/side-matched reward-free pixel fit (regex '^enc/'), freeze
it (agent.frozen_enc), and train dyn+heads task-mode (full) on the same
buffer — removing reconstruction from the encoder's gradient competition.
Registered predictions: P-PE1 inclusion restored at the representation
level (in-regime rew-NLL leaves the swamping regime), P-PE2 frozen-readout
adaptation lifts off the committed X2 floor, P-PE3 (conditional) the
occupancy split re-engages.

Usage:
  python -m analysis.pe_pixel_read --auc <canonical auc.csv> \
      --e4 <e4_fingerpx_pe csv> --output <dir>
  python -m analysis.pe_pixel_read selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

from analysis.pixel_swamping_read import seed_cluster_ci
from analysis.unfrozen_stress_read import (
    load_cells, _boot, B_BOOT, RNG_SEED, SEEDS, DOMAIN, MILESTONE)

# Frozen-registration pins (fail loudly if the imported conventions drift).
assert SEEDS == tuple(range(1, 9)), SEEDS
assert DOMAIN == 'finger' and MILESTONE == '500000'
assert B_BOOT == 10_000 and RNG_SEED == 0

# Adapt cells (canonical csv modes) and the committed X2 task baseline.
PE_MODES = dict(s0='ax1pepxq1ms0', s1='ax1pepxq1ms1')
X2_BASE_MODES = dict(s0='ax1pxpxq1ms0', s1='ax1pxpxq1ms1')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X2_BASELINE = os.path.join(_REPO, 'artifacts/pixel_x2_20260724/auc.csv')

# E4 fit-run naming and the committed swamping anchors
# (artifacts/pixel_swamping_20260725/pixel_swamping.json: task-arm
# rew_nll_in h=0 23.34 [19.41, 27.29]; trivial two-hot floor 0.6713 —
# same frozen probeset fingerpx_v1, so the floor transports).
import re
PE_FIT_RE = re.compile(r'^ax1wm_finger_pepxq1ms([01])_seed(\d+)$')
SW_BASE = dict(point=23.34, lo=19.41, hi=27.29)
THRESH_SWAMP = 2.0   # registered swamping bar (PREREG_pixel_swamping_20260724)
THRESH_MEMBER = 1.5  # membership descriptor bar (never decisional)
TRIVIAL_FLOOR = 0.671345852414428


def load_e4(path):
  """pe fit rows at h=0 -> {seed: [side values]} with integrity guards."""
  by_seed = {}
  seen = set()
  with open(path) as f:
    for r in csv.DictReader(f):
      m = PE_FIT_RE.match(r['run_id'])
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
          f'{r["run_id"]}: reward_aware={r.get("reward_aware")} — pe fits '
          'must be task-mode (reward_aware gate changed?)')
      v = float(r['rew_nll_in'])
      assert np.isfinite(v), f'{r["run_id"]}: non-finite rew_nll_in'
      by_seed.setdefault(seed, []).append(v)
  missing = [s for s in SEEDS if len(by_seed.get(s, [])) != 2]
  assert not missing, (
      f'incomplete e4 grid — seeds without both sides at h=0: {missing}')
  return by_seed


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


def analyze(auc_csv, e4_csv):
  rng = np.random.default_rng(RNG_SEED)

  # P-PE1: representation level (b/seed passed explicitly so the pinned
  # constants here govern, not pixel_swamping_read's module defaults).
  nll = load_e4(e4_csv)
  point, lo, hi = seed_cluster_ci(nll, b=B_BOOT, seed=RNG_SEED)
  if hi < THRESH_SWAMP:
    pe1 = 'INCLUSION-RESTORED'
  elif hi < SW_BASE['lo']:
    pe1 = 'PARTIAL-RELIEF'
  else:
    pe1 = 'NO-RELIEF'

  # P-PE2: behavioral lift vs the committed X2 task cells (paired per
  # seed and cell, mean across cells — the U4 relief estimator).
  assert_no_dup_rows(auc_csv, PE_MODES)
  assert_no_dup_rows(X2_BASELINE, X2_BASE_MODES)
  cells = load_cells(auc_csv, PE_MODES)
  base = load_cells(X2_BASELINE, X2_BASE_MODES)
  per_seed = np.mean([cells[k] - base[k] for k in ('s0', 's1')], axis=0)
  lift = _boot(per_seed, rng)
  lifts = lift['ci'][0] > 0

  # P-PE3 (conditional secondary): occupancy split within the pe arm.
  split = _boot(cells['s1'] - cells['s0'], rng)
  if lifts:
    pe3 = 'FIRES' if split['ci'][0] > 0 else 'null'
  else:
    pe3 = 'premise-idle'

  return dict(
      p_pe1=dict(point=point, ci=[lo, hi], verdict=pe1,
                 baseline=SW_BASE, thresh_swamp=THRESH_SWAMP,
                 member_band=(hi < THRESH_MEMBER),
                 trivial_floor=TRIVIAL_FLOOR),
      p_pe2=dict(**lift, fires=bool(lifts)),
      p_pe3=dict(**split, verdict=pe3),
      cell_means={k: float(np.mean(v)) for k, v in cells.items()},
      baseline_cell_means={k: float(np.mean(v)) for k, v in base.items()},
      verdict=_verdict(pe1, lifts),
      auc_csv=os.path.abspath(auc_csv), e4_csv=os.path.abspath(e4_csv),
      baseline_csv=X2_BASELINE)


def _verdict(pe1, lifts):
  if pe1 == 'INCLUSION-RESTORED' and lifts:
    return ('PREDICTION-CONFIRMED: removing reconstruction from the '
            'encoder competition restores reward inclusion AND unlocks '
            'transfer — the pixel boundary converts to confirmed structure.')
  if pe1 == 'INCLUSION-RESTORED' and not lifts:
    return ('INCLUDED-BUT-USELESS AT PIXEL: inclusion restored without '
            'behavioral lift — membership is necessary, not sufficient, '
            'in the pixel regime too (stamping-pattern analogue).')
  if pe1 == 'PARTIAL-RELIEF':
    return ('PARTIAL-RELIEF: rew-NLL leaves the measured swamping range '
            'but not the swamping regime; directional support only, no '
            'headline change; behavioral leg reported as measured.')
  return ('NO-RELIEF: the encoder-competition lever fails its first '
          'intervention — registered honest negative; the swamping '
          'account needs revision before further pixel arms.')


def selfcheck(_args=None):
  import tempfile
  tmp = tempfile.mkdtemp()

  def write_auc(path, pe_by_cell, base_by_cell):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'mode', 'domain', 'seed', 'milestone',
                  'auc100k', 'qc_pass'])
      for cells, prefix in ((pe_by_cell, 'adapt_pe'), (base_by_cell, 'adapt_x2')):
        for mode, vals in cells.items():
          for s, v in zip(SEEDS, vals):
            w.writerow([f'{prefix}_{mode}_seed{s}', mode, 'finger', s,
                        '500000', v, 1])

  def write_e4(path, nll_by_cell, aware='1'):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'horizon', 'rew_nll_in', 'reward_aware'])
      for side in (0, 1):
        for s, v in zip(SEEDS, nll_by_cell[side]):
          w.writerow([f'ax1wm_finger_pepxq1ms{side}_seed{s}', 0, v, aware])
          w.writerow([f'ax1wm_finger_pepxq1ms{side}_seed{s}', 5, v + 1, aware])

  base = {m: [78 + i for i in range(8)] for m in X2_BASE_MODES.values()}
  lo_nll = {0: [1.1 + 0.02 * i for i in range(8)],
            1: [1.0 + 0.02 * i for i in range(8)]}
  hi_nll = {0: [21 + i for i in range(8)], 1: [22 + i for i in range(8)]}
  mid_nll = {0: [8 + 0.1 * i for i in range(8)], 1: [9 + 0.1 * i for i in range(8)]}

  auc = os.path.join(tmp, 'auc.csv')
  e4 = os.path.join(tmp, 'e4.csv')

  # Case 1: full confirmation + PE3 fires.
  pe = {PE_MODES['s0']: [140 + i for i in range(8)],
        PE_MODES['s1']: [220 + i for i in range(8)]}
  write_auc(auc, pe, base); write_e4(e4, lo_nll)
  global X2_BASELINE
  real_baseline = X2_BASELINE
  X2_BASELINE = auc
  try:
    r = analyze(auc, e4)
    assert r['p_pe1']['verdict'] == 'INCLUSION-RESTORED', r['p_pe1']
    assert r['p_pe1']['member_band'] is True
    assert r['p_pe2']['fires'] and r['p_pe3']['verdict'] == 'FIRES'
    assert r['verdict'].startswith('PREDICTION-CONFIRMED')

    # Case 2: inclusion restored, no lift (stamping analogue) + PE3 idle.
    pe2 = {PE_MODES['s0']: base[X2_BASE_MODES['s0']],
           PE_MODES['s1']: base[X2_BASE_MODES['s1']]}
    write_auc(auc, pe2, base); write_e4(e4, lo_nll)
    r = analyze(auc, e4)
    assert not r['p_pe2']['fires'] and r['p_pe3']['verdict'] == 'premise-idle'
    assert r['verdict'].startswith('INCLUDED-BUT-USELESS')

    # Case 3: partial relief.
    write_e4(e4, mid_nll)
    r = analyze(auc, e4)
    assert r['p_pe1']['verdict'] == 'PARTIAL-RELIEF', r['p_pe1']
    assert r['verdict'].startswith('PARTIAL-RELIEF')

    # Case 4: no relief.
    write_e4(e4, hi_nll)
    r = analyze(auc, e4)
    assert r['p_pe1']['verdict'] == 'NO-RELIEF'
    assert r['verdict'].startswith('NO-RELIEF')

    # Guard trips.
    def trips(fn):
      try:
        fn()
      except AssertionError:
        return True
      return False

    bad = dict(lo_nll); bad[0] = bad[0][:-1]  # drop one side value

    def missing_side():
      with open(e4, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'horizon', 'rew_nll_in', 'reward_aware'])
        for s, v in zip(SEEDS[:-1], bad[0][:7]):
          w.writerow([f'ax1wm_finger_pepxq1ms0_seed{s}', 0, v, '1'])
      load_e4(e4)
    assert trips(missing_side), 'missing-side must trip'

    write_e4(e4, lo_nll, aware='0')
    assert trips(lambda: load_e4(e4)), 'reward_aware=0 must trip'

    def dup():
      write_e4(e4, lo_nll)
      with open(e4, 'a', newline='') as f:
        csv.writer(f).writerow(['ax1wm_finger_pepxq1ms0_seed1', 0, 1.0, '1'])
      load_e4(e4)
    assert trips(dup), 'duplicate must trip'

    def nan_nll():
      write_e4(e4, {0: [float('nan')] * 8, 1: lo_nll[1]})
      load_e4(e4)
    assert trips(nan_nll), 'nan must trip'

    def bad_qc():
      write_auc(auc, pe, base)
      rows = list(csv.reader(open(auc)))
      rows[1][-1] = '0'
      with open(auc, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
      load_cells(auc, PE_MODES)
    assert trips(bad_qc), 'qc fail must trip'

    def dup_auc():
      write_auc(auc, pe, base)
      with open(auc, 'a', newline='') as f:
        csv.writer(f).writerow(
            ['adapt_pe_dup', PE_MODES['s0'], 'finger', 1, '500000', 99.0, 1])
      assert_no_dup_rows(auc, PE_MODES)
    assert trips(dup_auc), 'duplicate auc row must trip'
  finally:
    X2_BASELINE = real_baseline

  print('selfcheck PASS: confirmed/included-useless/partial/no-relief '
        'branches + PE3 fires/idle + guard trips '
        '(missing-side, reward_aware, e4-duplicate, nan, qc, auc-duplicate)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', nargs='?', choices=('selfcheck',))
  ap.add_argument('--auc')
  ap.add_argument('--e4')
  ap.add_argument('--output', default='analysis_out/pe_pixel')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck(args)
    return
  if not (args.auc and args.e4):
    ap.error('--auc and --e4 required (or selfcheck)')
  res = analyze(args.auc, args.e4)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'pe_pixel.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  print(json.dumps({k: res[k] for k in ('p_pe1', 'p_pe2', 'p_pe3', 'verdict')},
                   indent=2))
  print('wrote', out)


if __name__ == '__main__':
  main()
