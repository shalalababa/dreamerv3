"""Frozen read for the high-f_R falling-limb wave.

Registered in prereg/PREREG_highfr_wave_20260802.md, adjudicating the
frozen theory predictions PREREG_highfr_theory_20260730 (P-HF1 falling
limb, P-HF2 support-starvation dissociation, P-HF3 weak interior
maximum, monotone-rise refutation). Committed BEFORE any curated
high-f_R buffer, fit, or adaptation exists.

Cells (canonical AUC csv modes, task arm, finger):
  hi-f side1  ax1hfq1fs1  seeds 1-8   (curated f_R target 0.80; PRIMARY)
  hi-f side0  ax1hfq1fs0  seeds 1-8   (curated f_R target 0.60)
  comparator  ax1v2q1v200s1 seeds 1-12 (natural f=0.3232, matched volume;
                                        frozen rows, volume_repl bundle)
  grid low    ax1v2q1v200s0 seeds 7-12 (f=0.054, matched volume; frozen)

PRIMARY (sole confirmatory, P-HF1): mean[hi-f side1] - mean[v200s1],
AUC100k, two-sample seed-cluster bootstrap (B=10K, default_rng(0));
FIRES iff CI entirely < 0. Registered branches: monotone-rise
refutation (cell means nondecreasing in f AND side1-v200s1 CI entirely
> 0 => the breadth-patch trade-off form is REFUTED, revision required);
saturation (primary straddles 0 AND diversity_pr does not separate =>
compatible, uninformative).

P-HF2 (registered secondary): side1 task fits' in-regime reward-NLL at
horizon 0 (frozen E4 machinery, finger_v1 probe set), seed-cluster CI:
membership intact iff CI entirely <= 1.5 nats; inclusion-effect
alternative iff CI entirely >= 2.0 (the pixel-swamping registered
constants, same substrate/anchors); else indeterminate. Dissociation
holds iff membership intact AND diversity_pr(side1) < 9.230033291492585
(the frozen v200s1 instrument value).

P-HF3 (registered secondary, weak form): over the matched-200-episode
grid {v200s0, v200s1, included hi-f cells}, the max cell-mean AUC100k
is NOT at the highest-f cell.

Validity gates: spectral_v1_1_20260724 on both curated-side jsons;
side1 f_rewarded in [0.60, 0.85] (else the read REFUSES — the wave
should have halted pre-fit); side0 f_rewarded in [0.55, 0.65] AND
side1-side0 separation >= 0.08, else side0 is EXCLUDED (disclosed,
never fatal); all-or-nothing 8-seed cohorts on included hi-f cells;
qc_pass on every consumed row; comparator cohorts exactly 12 / 6.

Usage:
  python -m analysis.highfr_read --auc <fresh auc.csv> \
      --auc_frozen <volume_repl bundle auc.csv> \
      --spectral_lo <q1f/spectral_side0.json> \
      --spectral_hi <q1f/spectral_side1.json> \
      --e4 <e4_finger_v1_hf.csv> --output <dir>
  python -m analysis.highfr_read --selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

B_BOOT = 10_000
# Dedicated rng seed per registered statistic (draw order independence).
RNG_PRIMARY, RNG_LO, RNG_NLL_HI, RNG_NLL_LO = 0, 1, 2, 3

MODE_HI = 'ax1hfq1fs1'
MODE_LO = 'ax1hfq1fs0'
MODE_V200S1 = 'ax1v2q1v200s1'
MODE_V200S0 = 'ax1v2q1v200s0'
WM_HI = 'ax1wm_finger_hfq1fs1_seed'
WM_LO = 'ax1wm_finger_hfq1fs0_seed'
SEEDS_HF = tuple(range(1, 9))
SEEDS_V200S1 = tuple(range(1, 13))
SEEDS_V200S0 = tuple(range(7, 13))

MEASURE_VERSION = 'spectral_v1_1_20260724'
F_HI_RANGE = (0.60, 0.85)
F_LO_RANGE = (0.55, 0.65)
MIN_SEP = 0.08
REF_F = 0.32317182817182816       # v200s1, frozen instrument value
REF_DIV = 9.230033291492585       # v200s1 diversity_pr, frozen
F_V200S0 = 0.054                  # v200s0 grid coordinate — the SEARCH
                                  # occupancy from the volume-buffer build
                                  # (no instrument json exists for v200s0;
                                  # on this regime the instrument tracks
                                  # search occ to ~1e-3: v200s1 .3232/.323)
NLL_MEMBER_MAX = 1.5              # pixel-swamping registered constants
NLL_NONMEMBER_MIN = 2.0


def load_auc(path, modes):
  """{mode: {seed: auc}} over qc-passing registered rows, plus
  {mode: [seeds]} of qc failures — the CALLER decides fatality
  (side0 qc failures exclude side0; anywhere else they abort)."""
  cells = {m: {} for m in modes}
  qc_fail = {m: [] for m in modes}
  with open(path) as f:
    for r in csv.DictReader(f):
      if (r['mode'] not in cells or r['domain'] != 'finger'
          or int(r['milestone']) != 500000):
        continue
      if r['qc_pass'] in ('1', 'True', 'true'):
        cells[r['mode']][int(r['seed'])] = float(r['auc100k'])
      else:
        qc_fail[r['mode']].append(int(r['seed']))
  return cells, qc_fail


def load_spectral(path, expected_side):
  with open(path) as f:
    m = json.load(f)
  assert m.get('version') == MEASURE_VERSION, (
      f"{path}: version {m.get('version')!r} != {MEASURE_VERSION}")
  assert m.get('n_episodes') == 200, m.get('n_episodes')
  assert m.get('side') == expected_side and m.get('domain') == 'finger', (
      f"{path}: side/domain {m.get('side')!r}/{m.get('domain')!r} != "
      f"{expected_side!r}/'finger' — swapped or foreign instrument json")
  return dict(f=float(m['f_rewarded']), div=float(m['diversity_pr']),
              spectrum_pr=(float(m['spectrum_pr'])
                           if m.get('spectrum_pr') is not None else None),
              tag=m.get('tag'))


def check_ref(path):
  with open(path) as f:
    m = json.load(f)
  assert m.get('version') == MEASURE_VERSION
  assert abs(float(m['f_rewarded']) - REF_F) < 1e-9, m['f_rewarded']
  assert abs(float(m['diversity_pr']) - REF_DIV) < 1e-9, m['diversity_pr']


def load_e4(path, prefix, seeds, required):
  vals = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      if r['run_id'].startswith(prefix) and int(r['horizon']) == 0:
        assert r['reward_aware'] in ('1', 'True', 'true'), (
            f"{r['run_id']}: reward_aware={r['reward_aware']}")
        seed = int(r['run_id'][len(prefix):])
        vals[seed] = float(r['rew_nll_in'])
  if required:
    missing = [s for s in seeds if s not in vals]
    assert not missing, f'{prefix}*: missing E4 h0 rows for seeds {missing}'
  return vals


def _two_sample(a, b, rng):
  a, b = np.asarray(a, float), np.asarray(b, float)
  point = float(a.mean() - b.mean())
  n, m = len(a), len(b)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = (a[rng.integers(0, n, n)].mean()
                - b[rng.integers(0, m, m)].mean())
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=point, ci=[float(lo), float(hi)], n=[n, m])


def _one_sample(a, rng):
  a = np.asarray(a, float)
  n = len(a)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = a[rng.integers(0, n, n)].mean()
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=float(a.mean()), ci=[float(lo), float(hi)], n=n)


def analyze(cells, spec_lo, spec_hi, nll_hi_vals, nll_lo_vals,
            qc_fail=None):
  qc_fail = qc_fail or {}
  # ---- validity gates -------------------------------------------------
  assert F_HI_RANGE[0] <= spec_hi['f'] <= F_HI_RANGE[1], (
      f"side1 f_rewarded {spec_hi['f']:.4f} outside {F_HI_RANGE} — "
      'invalid wave, the read refuses (dated amendment required)')
  # side0 is NEVER load-bearing (registered): instrument-gate misses,
  # qc failures, and incomplete cohorts all EXCLUDE it with a recorded
  # reason instead of aborting the read.
  lo_included = (F_LO_RANGE[0] <= spec_lo['f'] <= F_LO_RANGE[1]
                 and spec_hi['f'] - spec_lo['f'] >= MIN_SEP)
  lo_reason = None if lo_included else 'instrument gate (range/separation)'
  if lo_included and qc_fail.get(MODE_LO):
    lo_included = False
    lo_reason = f'qc-failed side0 seeds {sorted(qc_fail[MODE_LO])}'
  if lo_included:
    missing = [s for s in SEEDS_HF if s not in cells.get(MODE_LO, {})]
    if missing:
      lo_included = False
      lo_reason = f'incomplete side0 cohort: missing seeds {missing}'
  gates = dict(f_hi=spec_hi['f'], f_lo=spec_lo['f'],
               separation=round(spec_hi['f'] - spec_lo['f'], 6),
               lo_included=bool(lo_included), lo_excluded_reason=lo_reason,
               f_hi_range=list(F_HI_RANGE), f_lo_range=list(F_LO_RANGE),
               min_sep=MIN_SEP)

  # ---- cohorts (all-or-nothing; fatal everywhere except side0) -------
  sel = lambda m, seeds: [cells[m][s] for s in seeds]
  for mode, seeds in ((MODE_HI, SEEDS_HF), (MODE_V200S1, SEEDS_V200S1),
                      (MODE_V200S0, SEEDS_V200S0)):
    assert not qc_fail.get(mode), (
        f'{mode}: qc-failed rows for seeds {sorted(qc_fail[mode])}')
    missing = [s for s in seeds if s not in cells[mode]]
    assert not missing, f'{mode}: missing seeds {missing}'

  # ---- registered statistics -----------------------------------------
  primary = _two_sample(sel(MODE_HI, SEEDS_HF),
                        sel(MODE_V200S1, SEEDS_V200S1),
                        np.random.default_rng(RNG_PRIMARY))
  fires = primary['ci'][1] < 0
  secondary_lo = (_two_sample(sel(MODE_LO, SEEDS_HF),
                              sel(MODE_V200S1, SEEDS_V200S1),
                              np.random.default_rng(RNG_LO))
                  if lo_included else None)

  mean = lambda m, seeds: float(np.mean(sel(m, seeds)))
  hi_mean = mean(MODE_HI, SEEDS_HF)
  v200s1_mean = mean(MODE_V200S1, SEEDS_V200S1)

  # monotone-rise refutation branch (frozen failure mode)
  chain = [(REF_F, v200s1_mean)]
  if lo_included:
    chain.append((spec_lo['f'], mean(MODE_LO, SEEDS_HF)))
  chain.append((spec_hi['f'], hi_mean))
  nondecreasing = all(chain[i + 1][1] >= chain[i][1]
                      for i in range(len(chain) - 1))
  refutation = bool(nondecreasing and primary['ci'][0] > 0)

  # P-HF3 weak interior maximum over the matched-volume grid
  grid = [(F_V200S0, mean(MODE_V200S0, SEEDS_V200S0)),
          (REF_F, v200s1_mean)]
  if lo_included:
    grid.append((spec_lo['f'], mean(MODE_LO, SEEDS_HF)))
  grid.append((spec_hi['f'], hi_mean))
  argmax_f = max(grid, key=lambda t: t[1])[0]
  phf3_holds = bool(argmax_f != max(f for f, _ in grid))

  # P-HF2 dissociation
  nll = _one_sample([nll_hi_vals[s] for s in SEEDS_HF],
                    np.random.default_rng(RNG_NLL_HI))
  membership = ('member' if nll['ci'][1] <= NLL_MEMBER_MAX else
                'nonmember' if nll['ci'][0] >= NLL_NONMEMBER_MIN else
                'indeterminate')
  diversity_lower = bool(spec_hi['div'] < REF_DIV)
  dissociation = bool(membership == 'member' and diversity_lower)
  inclusion_reading = bool(membership == 'nonmember')
  nll_lo = (_one_sample([nll_lo_vals[s] for s in SEEDS_HF],
                        np.random.default_rng(RNG_NLL_LO))
            if nll_lo_vals and all(s in nll_lo_vals for s in SEEDS_HF)
            else None)
  spectrum_pr_hi = spec_hi.get('spectrum_pr')
  spectrum_pr_lo = spec_lo.get('spectrum_pr')

  straddles = primary['ci'][0] < 0 < primary['ci'][1]
  saturation = bool(straddles and not diversity_lower)

  verdict = []
  if fires:
    verdict.append(
        'P-HF1 FIRES: transfer from the curated high-f_R buffer is worse '
        'than from the natural f=0.32 buffer at matched volume — the '
        'falling limb is real.')
  elif refutation:
    verdict.append(
        'MONOTONE RISE: transfer keeps rising through the curated high-f_R '
        'range — the breadth-patch trade-off form is REFUTED (frozen '
        'consequence: revision required, not annotation).')
  else:
    verdict.append('P-HF1 does NOT fire: the high-f_R contrast is not '
                   'entirely negative.')
  if saturation:
    verdict.append('Saturation reading (registered): primary straddles 0 '
                   'and diversity_pr does not separate — compatible, '
                   'uninformative.')
  if dissociation:
    verdict.append('P-HF2 DISSOCIATION: membership intact (rew-NLL CI '
                   f'<= {NLL_MEMBER_MAX}) with lower diversity_pr — the '
                   'deficit is support starvation with inclusion intact.')
  elif inclusion_reading:
    verdict.append('P-HF2 ALTERNATIVE: the high-f_R trunk leaves the '
                   'membership band — inclusion effect; the breadth '
                   'trade-off account gains no support here.')
  else:
    verdict.append(f'P-HF2 {membership}/diversity_lower={diversity_lower}: '
                   'dissociation not adjudicated.')
  verdict.append(f'P-HF3 max-not-at-highest-f (weak form) '
                 f'{"HOLDS" if phf3_holds else "FAILS"} '
                 f'(argmax at f={argmax_f:.4f}).')

  return dict(
      gates=gates, primary_hi_minus_v200s1=primary, fires=bool(fires),
      secondary_lo_minus_v200s1=secondary_lo,
      monotone_refutation=dict(chain=[[round(f, 6), round(m, 4)]
                                      for f, m in chain],
                               nondecreasing=bool(nondecreasing),
                               fires=refutation),
      saturation=saturation,
      phf2=dict(nll_hi=nll, membership=membership,
                member_max=NLL_MEMBER_MAX, nonmember_min=NLL_NONMEMBER_MIN,
                diversity_pr_hi=spec_hi['div'], diversity_pr_lo=spec_lo['div'],
                diversity_ref=REF_DIV, diversity_lower=diversity_lower,
                dissociation=dissociation,
                inclusion_reading=inclusion_reading,
                nll_lo_descriptive=nll_lo,
                spectrum_pr_hi_descriptive=spectrum_pr_hi,
                spectrum_pr_lo_descriptive=spectrum_pr_lo),
      phf3=dict(grid=[[round(f, 6), round(m, 4)] for f, m in grid],
                argmax_f=round(argmax_f, 6), holds=phf3_holds),
      cell_means={m: {str(s): v for s, v in sorted(cells[m].items())}
                  for m in cells},
      verdict=' '.join(verdict))


def read(args):
  cells_new, qf_new = load_auc(args.auc, (MODE_HI, MODE_LO))
  cells_frozen, qf_frozen = load_auc(args.auc_frozen,
                                     (MODE_V200S1, MODE_V200S0))
  cells = {**cells_new, **cells_frozen}
  qc_fail = {**qf_new, **qf_frozen}
  check_ref(args.spectral_ref)
  spec_lo = load_spectral(args.spectral_lo, 'lo')
  spec_hi = load_spectral(args.spectral_hi, 'hi')
  nll_hi = load_e4(args.e4, WM_HI, SEEDS_HF, required=True)
  nll_lo = load_e4(args.e4, WM_LO, SEEDS_HF, required=False)
  res = analyze(cells, spec_lo, spec_hi, nll_hi, nll_lo, qc_fail=qc_fail)
  res['inputs'] = dict(auc=os.path.abspath(args.auc),
                       auc_frozen=os.path.abspath(args.auc_frozen),
                       spectral_lo=os.path.abspath(args.spectral_lo),
                       spectral_hi=os.path.abspath(args.spectral_hi),
                       spectral_ref=os.path.abspath(args.spectral_ref),
                       e4=os.path.abspath(args.e4))
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'highfr.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  p = res['primary_hi_minus_v200s1']
  print(f"PRIMARY hi-f - v200s1: {p['point']:+.1f} "
        f"CI=[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] fires={res['fires']}")
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _mk(hi=100.0, lo=150.0, v1=204.0, v0=120.0, sd=25.0, f_hi=0.79,
        f_lo=0.61, div_hi=7.0, div_lo=8.2, nll=1.2, nll_sd=0.05, seed=5):
  rng = np.random.default_rng(seed)
  cells = {MODE_HI: {}, MODE_LO: {}, MODE_V200S1: {}, MODE_V200S0: {}}
  for s in SEEDS_HF:
    cells[MODE_HI][s] = hi + sd * rng.standard_normal()
    cells[MODE_LO][s] = lo + sd * rng.standard_normal()
  for s in SEEDS_V200S1:
    cells[MODE_V200S1][s] = v1 + sd * rng.standard_normal()
  for s in SEEDS_V200S0:
    cells[MODE_V200S0][s] = v0 + sd * rng.standard_normal()
  spec_lo = dict(f=f_lo, div=div_lo, tag='q1f_side0')
  spec_hi = dict(f=f_hi, div=div_hi, tag='q1f_side1')
  nll_hi = {s: nll + nll_sd * rng.standard_normal() for s in SEEDS_HF}
  nll_lo = {s: nll + nll_sd * rng.standard_normal() for s in SEEDS_HF}
  return cells, spec_lo, spec_hi, nll_hi, nll_lo


def _write_fixture(tmp, cells, spec_lo, spec_hi, nll_hi, nll_lo):
  rows = ['run_id,mode,domain,seed,milestone,auc100k,qc_pass']
  frozen = ['run_id,mode,domain,seed,milestone,auc100k,qc_pass']
  for m, seedmap in cells.items():
    dst = frozen if m in (MODE_V200S1, MODE_V200S0) else rows
    for s, v in seedmap.items():
      dst.append(f'r,{m},finger,{s},500000,{v},1')
  paths = {}
  for name, content in (('auc.csv', '\n'.join(rows)),
                        ('auc_frozen.csv', '\n'.join(frozen))):
    paths[name] = os.path.join(tmp, name)
    with open(paths[name], 'w') as f:
      f.write(content)
  for name, side, spec in (('lo.json', 'lo', spec_lo),
                           ('hi.json', 'hi', spec_hi)):
    paths[name] = os.path.join(tmp, name)
    with open(paths[name], 'w') as f:
      json.dump(dict(version=MEASURE_VERSION, n_episodes=200,
                     side=side, domain='finger',
                     f_rewarded=spec['f'], diversity_pr=spec['div'],
                     spectrum_pr=40.0, tag=spec['tag']), f)
  paths['ref.json'] = os.path.join(tmp, 'ref.json')
  with open(paths['ref.json'], 'w') as f:
    json.dump(dict(version=MEASURE_VERSION, n_episodes=200,
                   f_rewarded=REF_F, diversity_pr=REF_DIV,
                   tag='volume_q1v200_s1'), f)
  e4 = ['run_id,horizon,rew_nll_in,reward_aware']
  for prefix, vals in ((WM_HI, nll_hi), (WM_LO, nll_lo)):
    for s, v in vals.items():
      for h in (0, 1, 5):
        e4.append(f'{prefix}{s},{h},{v if h == 0 else v + 1.0},1')
  paths['e4.csv'] = os.path.join(tmp, 'e4.csv')
  with open(paths['e4.csv'], 'w') as f:
    f.write('\n'.join(e4))
  return paths


def _trip(fn):
  try:
    fn()
  except AssertionError:
    return True
  return False


def selfcheck(args):
  # Branch 1: falling limb fires + dissociation + interior max
  cells, sl, sh, nh, nl = _mk()
  res = analyze(cells, sl, sh, nh, nl)
  assert res['fires'] and 'P-HF1 FIRES' in res['verdict']
  assert res['phf2']['dissociation'] and 'DISSOCIATION' in res['verdict']
  assert res['phf3']['holds'] and res['phf3']['argmax_f'] == round(REF_F, 6)
  assert not res['monotone_refutation']['fires'] and not res['saturation']
  assert res['gates']['lo_included']
  assert res['secondary_lo_minus_v200s1']['point'] < 0
  assert analyze(cells, sl, sh, nh, nl) == res      # determinism

  # Branch 2: straddle + diversity does not separate => saturation
  cells, sl, sh, nh, nl = _mk(hi=200.0, div_hi=10.5)
  res = analyze(cells, sl, sh, nh, nl)
  assert not res['fires'] and res['saturation']
  assert 'Saturation' in res['verdict']

  # Branch 3: monotone rise refutes (primary CI entirely > 0)
  cells, sl, sh, nh, nl = _mk(hi=330.0, lo=260.0, sd=10.0)
  res = analyze(cells, sl, sh, nh, nl)
  assert res['monotone_refutation']['fires'] and not res['fires']
  assert 'REFUTED' in res['verdict']
  assert not res['phf3']['holds']                   # max at highest f

  # Branch 4: nonmember NLL => inclusion reading, no dissociation
  cells, sl, sh, nh, nl = _mk(nll=2.4)
  res = analyze(cells, sl, sh, nh, nl)
  assert res['phf2']['inclusion_reading'] and not res['phf2']['dissociation']
  assert 'ALTERNATIVE' in res['verdict']

  # Branch 5: indeterminate NLL band
  cells, sl, sh, nh, nl = _mk(nll=1.7, nll_sd=0.01)
  res = analyze(cells, sl, sh, nh, nl)
  assert res['phf2']['membership'] == 'indeterminate'
  assert not res['phf2']['dissociation']

  # Branch 5b (reviewer M1a): membership must be a CI rule, not a point
  # rule — point on the member side of 1.5 with CI crossing it, and
  # point past 2.0 with CI crossing back.
  cells, sl, sh, nh, nl = _mk()
  spread = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8]      # mean 1.4
  nh = {s: v for s, v in zip(SEEDS_HF, spread)}
  res = analyze(cells, sl, sh, nh, nl)
  assert res['phf2']['nll_hi']['point'] < NLL_MEMBER_MAX
  assert res['phf2']['nll_hi']['ci'][1] > NLL_MEMBER_MAX  # fixture sanity
  assert res['phf2']['membership'] == 'indeterminate'
  nh = {s: v + 0.7 for s, v in zip(SEEDS_HF, spread)}     # mean 2.1
  res = analyze(cells, sl, sh, nh, nl)
  assert res['phf2']['nll_hi']['point'] >= NLL_NONMEMBER_MIN
  assert res['phf2']['nll_hi']['ci'][0] < NLL_NONMEMBER_MIN
  assert res['phf2']['membership'] == 'indeterminate'

  # Branch 3b (reviewer M1b): a side0 dip must kill the monotone chain
  # even when the primary CI is entirely positive.
  cells, sl, sh, nh, nl = _mk(hi=330.0, lo=120.0, sd=10.0)
  res = analyze(cells, sl, sh, nh, nl)
  assert res['primary_hi_minus_v200s1']['ci'][0] > 0
  assert not res['monotone_refutation']['nondecreasing']
  assert not res['monotone_refutation']['fires']

  # Branch 3c (reviewer M1c): nondecreasing chain + straddling primary
  # must NOT refute (refutation needs CI entirely > 0, not upper > 0).
  cells, sl, sh, nh, nl = _mk()
  base = np.linspace(-1.5, 1.5, 8)
  cells[MODE_HI] = {s: 209.0 + 40 * b for s, b in zip(SEEDS_HF, base)}
  cells[MODE_LO] = {s: 206.0 + 5 * b for s, b in zip(SEEDS_HF, base)}
  cells[MODE_V200S1] = {s: 204.0 + 5 * b for s, b
                        in zip(SEEDS_V200S1, np.linspace(-1.5, 1.5, 12))}
  res = analyze(cells, sl, sh, nh, nl)
  assert res['monotone_refutation']['nondecreasing']
  p = res['primary_hi_minus_v200s1']
  assert p['ci'][0] < 0 < p['ci'][1], p
  assert not res['monotone_refutation']['fires']

  # Reviewer NIT1: diversity_pr exactly equal to the reference is NOT
  # "lower" — with a straddling primary this is the saturation branch.
  cells, sl, sh, nh, nl = _mk(hi=200.0, div_hi=REF_DIV)
  res = analyze(cells, sl, sh, nh, nl)
  assert not res['phf2']['diversity_lower'] and res['saturation']

  # Gate: side1 f out of range refuses the read
  cells, sl, sh, nh, nl = _mk(f_hi=0.50)
  assert _trip(lambda: analyze(cells, sl, sh, nh, nl))
  cells, sl, sh, nh, nl = _mk(f_hi=0.90)
  assert _trip(lambda: analyze(cells, sl, sh, nh, nl))

  # Gate: side0 out of range or under-separated => excluded, not fatal
  for kw in (dict(f_lo=0.70), dict(f_lo=0.62, f_hi=0.65)):
    cells, sl, sh, nh, nl = _mk(**kw)
    res = analyze(cells, sl, sh, nh, nl)
    assert not res['gates']['lo_included']
    assert res['secondary_lo_minus_v200s1'] is None
    assert len(res['phf3']['grid']) == 3
    del cells[MODE_LO]                    # excluded cell may lack rows
    res2 = analyze(cells | {MODE_LO: {}}, sl, sh, nh, nl)
    assert res2['gates'] == res['gates']

  # Cohort rules (reviewer B2): side0 data problems EXCLUDE side0 with
  # a recorded reason — they never abort; everywhere else they abort.
  cells, sl, sh, nh, nl = _mk()
  del cells[MODE_HI][3]
  assert _trip(lambda: analyze(cells, sl, sh, nh, nl))
  cells, sl, sh, nh, nl = _mk()
  del cells[MODE_LO][5]
  res = analyze(cells, sl, sh, nh, nl)
  assert not res['gates']['lo_included']
  assert 'incomplete' in res['gates']['lo_excluded_reason']
  assert res['fires'] and res['secondary_lo_minus_v200s1'] is None
  cells, sl, sh, nh, nl = _mk()
  res = analyze(cells, sl, sh, nh, nl, qc_fail={MODE_LO: [2]})
  assert not res['gates']['lo_included']
  assert 'qc-failed' in res['gates']['lo_excluded_reason']
  assert res['fires']
  cells, sl, sh, nh, nl = _mk()
  assert _trip(lambda: analyze(cells, sl, sh, nh, nl,
                               qc_fail={MODE_HI: [4]}))
  assert _trip(lambda: analyze(cells, sl, sh, nh, nl,
                               qc_fail={MODE_V200S0: [9]}))
  cells, sl, sh, nh, nl = _mk()
  del cells[MODE_V200S1][12]
  assert _trip(lambda: analyze(cells, sl, sh, nh, nl))

  # Loader chain through files: qc gate, ref pin, e4 requirements
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    cells, sl, sh, nh, nl = _mk()
    paths = _write_fixture(tmp, cells, sl, sh, nh, nl)
    new_cells, new_qf = load_auc(paths['auc.csv'], (MODE_HI, MODE_LO))
    frz_cells, frz_qf = load_auc(paths['auc_frozen.csv'],
                                 (MODE_V200S1, MODE_V200S0))
    loaded = {**new_cells, **frz_cells}
    qf = {**new_qf, **frz_qf}
    assert loaded[MODE_HI] == {s: cells[MODE_HI][s] for s in SEEDS_HF}
    assert not any(qf.values())
    check_ref(paths['ref.json'])
    e4h = load_e4(paths['e4.csv'], WM_HI, SEEDS_HF, required=True)
    assert all(abs(e4h[s] - nh[s]) < 1e-9 for s in SEEDS_HF)  # h0 only
    res_files = analyze(loaded, load_spectral(paths['lo.json'], 'lo'),
                        load_spectral(paths['hi.json'], 'hi'), e4h,
                        load_e4(paths['e4.csv'], WM_LO, SEEDS_HF,
                                required=False), qc_fail=qf)
    assert res_files['fires']
    assert res_files['phf2']['spectrum_pr_hi_descriptive'] == 40.0
    # swapped --spectral_lo/--spectral_hi must trip (reviewer M3)
    assert _trip(lambda: load_spectral(paths['hi.json'], 'lo'))
    assert _trip(lambda: load_spectral(paths['lo.json'], 'hi'))
    # qc_pass=0 on a hi row -> loader reports, analyze aborts
    with open(paths['auc.csv']) as f:
      bad = '\n'.join(l[:-1] + '0' if f',{MODE_HI},finger,4,' in l else l
                      for l in f.read().splitlines())
    with open(os.path.join(tmp, 'bad.csv'), 'w') as f:
      f.write(bad)
    bad_cells, bad_qf = load_auc(os.path.join(tmp, 'bad.csv'),
                                 (MODE_HI, MODE_LO))
    assert bad_qf[MODE_HI] == [4] and 4 not in bad_cells[MODE_HI]
    assert _trip(lambda: analyze(
        {**bad_cells, **frz_cells}, sl, sh, nh, nl,
        qc_fail={**bad_qf, **frz_qf}))
    # qc_pass=0 on a side0 row -> exclusion, primary unaffected (B2)
    with open(paths['auc.csv']) as f:
      bad = '\n'.join(l[:-1] + '0' if f',{MODE_LO},finger,2,' in l else l
                      for l in f.read().splitlines())
    with open(os.path.join(tmp, 'bad_lo.csv'), 'w') as f:
      f.write(bad)
    bl_cells, bl_qf = load_auc(os.path.join(tmp, 'bad_lo.csv'),
                               (MODE_HI, MODE_LO))
    res_bl = analyze({**bl_cells, **frz_cells}, sl, sh, nh, nl,
                     qc_fail={**bl_qf, **frz_qf})
    assert not res_bl['gates']['lo_included'] and res_bl['fires']
    assert 'qc-failed' in res_bl['gates']['lo_excluded_reason']
    # tampered ref json must trip
    with open(paths['ref.json']) as f:
      ref = json.load(f)
    ref['diversity_pr'] = 9.0
    with open(os.path.join(tmp, 'badref.json'), 'w') as f:
      json.dump(ref, f)
    assert _trip(lambda: check_ref(os.path.join(tmp, 'badref.json')))
    # missing e4 seed trips when required
    e4rows = open(paths['e4.csv']).read().splitlines()
    e4rows = [r for r in e4rows if not r.startswith(f'{WM_HI}7,0,')]
    with open(os.path.join(tmp, 'bade4.csv'), 'w') as f:
      f.write('\n'.join(e4rows))
    assert _trip(lambda: load_e4(os.path.join(tmp, 'bade4.csv'), WM_HI,
                                 SEEDS_HF, required=True))
    # reward_aware=0 row trips
    e4rows = open(paths['e4.csv']).read().splitlines()
    e4rows = [r.replace(',1.2', ',1.2').rsplit(',', 1)[0] + ',0'
              if r.startswith(f'{WM_HI}2,0,') else r for r in e4rows]
    with open(os.path.join(tmp, 'bade4b.csv'), 'w') as f:
      f.write('\n'.join(e4rows))
    assert _trip(lambda: load_e4(os.path.join(tmp, 'bade4b.csv'), WM_HI,
                                 SEEDS_HF, required=True))
    # wrong spectral version trips
    with open(paths['hi.json']) as f:
      spec = json.load(f)
    spec['version'] = 'spectral_v1_20260724'
    with open(os.path.join(tmp, 'badspec.json'), 'w') as f:
      json.dump(spec, f)
    assert _trip(lambda: load_spectral(os.path.join(tmp, 'badspec.json'),
                                       'hi'))
  print('selfcheck PASS: fire/dissociation/interior branches, saturation, '
        'monotone-rise refutation, nonmember alternative, indeterminate '
        'band (CI-not-point, both thresholds), side0-dip kills the chain, '
        'straddle-not-refute, diversity-equality edge, side1 gate refusal, '
        'side0 exclusion non-fatal (instrument/qc/incomplete, reasons '
        'recorded, grid shrinks), hi/comparator cohort + qc aborts, loader '
        'chain (domain+milestone filter, qc split, frozen ref pin, e4 h0 + '
        'reward_aware + missing-seed, spectral version + side/domain pin), '
        'determinism')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--auc_frozen')
  ap.add_argument('--spectral_lo')
  ap.add_argument('--spectral_hi')
  ap.add_argument('--spectral_ref',
                  default='artifacts/volume_repl_20260729/'
                          'diversity_q1v200_s1.json')
  ap.add_argument('--e4')
  ap.add_argument('--output', default='analysis_out/highfr')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
