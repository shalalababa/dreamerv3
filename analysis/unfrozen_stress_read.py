"""Frozen readers for the U1-U4 unfrozen stress waves.

Registered in prereg/PREREG_unfrozen_u{1,2,3,4}_20260727.md and
committed BEFORE any uz* run exists. Motivation: the frozen-protocol
audit (research_notes/Audit_FrozenProtocol_20260726.md) — a frozen
null cannot license "useless as initialization"; each wave re-runs a
frozen-null-bearing contrast with the proven `unfrozen_readout`
config, everything else identical to its frozen counterpart.

Shared conventions: AUC100k, domain finger, milestone 500000, seeds
1-8, qc asserted; run-clustered percentile bootstrap B=10K,
default_rng(0); decisional statistics live ENTIRELY within the
unfrozen wave; frozen-vs-unfrozen level changes are registered
DESCRIPTORS against committed artifact csvs (never decisional).

Subcommands (one frozen read each):
  u1  apt+task unfrozen 2x2      modes ax1uzt / ax1uzf  q1s{0,1}
  u2  stamped unfrozen           modes ax1uz{srd0,srd1,sid} q1s{0,1}
  u3  orthogonal unfrozen        modes ax1uzog{rgo,sgb} q1s{0,1}
  u4  pixel unfrozen 2x2         modes ax1uz{,f}pxpxq1m s{0,1}

Usage:
  python -m analysis.unfrozen_stress_read u1 --auc <csv> --output <dir>
  python -m analysis.unfrozen_stress_read selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
SEEDS = tuple(range(1, 9))
DOMAIN = 'finger'
MILESTONE = '500000'
U3_FLOOR = 20.0
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINES = {
    'u1': 'artifacts/scaling_optionb_20260726/auc.csv',
    'u2': 'artifacts/stamping_20260718/auc.csv',
    'u3': 'artifacts/orthogonal_obj_20260725/auc.csv',
    'u4': 'artifacts/pixel_x2_20260724/auc.csv',
}

MODES = {
    'u1': dict(t0='ax1uztq1s0', t1='ax1uztq1s1',
               f0='ax1uzfq1s0', f1='ax1uzfq1s1'),
    'u2': dict(srd0s0='ax1uzsrd0q1s0', srd0s1='ax1uzsrd0q1s1',
               srd1s0='ax1uzsrd1q1s0', srd1s1='ax1uzsrd1q1s1',
               sids0='ax1uzsidq1s0', sids1='ax1uzsidq1s1'),
    'u3': dict(rgo0='ax1uzogrgoq1s0', rgo1='ax1uzogrgoq1s1',
               sgb0='ax1uzogsgbq1s0', sgb1='ax1uzogsgbq1s1'),
    'u4': dict(t0='ax1uzpxpxq1ms0', t1='ax1uzpxpxq1ms1',
               f0='ax1uzfpxpxq1ms0', f1='ax1uzfpxpxq1ms1'),
}

BASE_MODES = {
    'u1': dict(t0='ax1q1s0', t1='ax1q1s1', f0='ax1fq1s0', f1='ax1fq1s1'),
    'u2': dict(srd0s0='ax1srd0q1s0', srd0s1='ax1srd0q1s1',
               srd1s0='ax1srd1q1s0', srd1s1='ax1srd1q1s1',
               sids0='ax1sidq1s0', sids1='ax1sidq1s1'),
    'u3': dict(rgo0='ax1ogrgoq1s0', rgo1='ax1ogrgoq1s1',
               sgb0='ax1ogsgbq1s0', sgb1='ax1ogsgbq1s1'),
    'u4': dict(t0='ax1pxpxq1ms0', t1='ax1pxpxq1ms1',
               f0='ax1fpxpxq1ms0', f1='ax1fpxpxq1ms1'),
}


def load_cells(path, modes, require=True):
  cells = {k: {} for k in modes}
  inv = {}
  for k, m in modes.items():
    inv.setdefault(m, []).append(k)
  with open(path) as f:
    for r in csv.DictReader(f):
      if r.get('domain', DOMAIN) != DOMAIN:
        continue
      if r.get('milestone', MILESTONE) != MILESTONE:
        continue
      s = int(r['seed'])
      if s not in SEEDS:
        continue
      for k in inv.get(r['mode'], []):
        assert r.get('qc_pass', '1') in ('1', 'True', 'true'), (
            f"{r['mode']} seed {s}: qc_pass={r.get('qc_pass')}")
        cells[k][s] = float(r['auc100k'])
  if require:
    for k in cells:
      missing = [s for s in SEEDS if s not in cells[k]]
      assert not missing, f'{k} ({modes[k]}): missing seeds {missing}'
  return {k: np.array([v[s] for s in SEEDS]) if len(v) == len(SEEDS)
          else v for k, v in cells.items()}


def _boot(deltas, rng):
  d = np.asarray(deltas, float)
  n = len(d)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = d[rng.integers(0, n, n)].mean()
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=float(d.mean()), ci=[float(lo), float(hi)], n=n,
              pos=int((d > 0).sum()),
              per_seed=[float(x) for x in d])


def _level_changes(cells, base, keys, rng):
  out = {}
  for k in keys:
    if isinstance(base.get(k), np.ndarray):
      out[k] = _boot(cells[k] - base[k], rng)
    else:
      out[k] = None
  return out


# ---------------------------------------------------------------------------
# U1 — apt+task unfrozen 2x2 (the decisive reward-free-null stress)
# ---------------------------------------------------------------------------

def analyze_u1(cells, base=None):
  rng = np.random.default_rng(RNG_SEED)
  inter = _boot((cells['t1'] - cells['t0']) - (cells['f1'] - cells['f0']),
                rng)
  bapt = _boot(cells['f1'] - cells['f0'], rng)
  a_fires = inter['ci'][0] > 0
  b_flips = bapt['ci'][0] > 0
  levels = (_level_changes(cells, base, ('t0', 't1', 'f0', 'f1'), rng)
            if base else None)
  if a_fires and not b_flips:
    verdict = ('PROTOCOL-ROBUST: the interaction persists unfrozen AND '
               'the reward-free occupancy null holds under fine-tuning — '
               'the strongest outcome: reward-free hi-occupancy support '
               'is not merely frozen-illegible, it is not an '
               'initialization advantage either. The headline and the '
               'null family generalize beyond the frozen protocol.')
  elif b_flips and a_fires:
    verdict = ('PARTIAL FLIP: the apt occupancy benefit EXPRESSES under '
               'fine-tuning while the interaction still fires — '
               'reward-free claims rescope to frozen-readout; legibility '
               'remains the stronger axis but is no longer the only one.')
  elif b_flips:
    verdict = ('FLIP: the frozen apt null was protocol-bound and the '
               'unfrozen interaction does not separate the arms — '
               '"the benefit exists only under reward-capable '
               'objectives" must be withdrawn; legibility demotes to a '
               'property of the frozen-readout measurement (registered '
               'consequence: headline rescope).')
  else:
    verdict = ('INTERACTION DOES NOT FIRE UNFROZEN (apt null holds): '
               'the interaction is registered as frozen-protocol-scoped; '
               'the null family stands. Report both protocols side by '
               'side; no claim withdrawal (the frozen claims were '
               'protocol-scoped by registration).')
  if bapt['ci'][1] < 0:
    verdict = ('NEGATIVE B_apt — outside registered accounts (apt '
               'occupancy actively harmful unfrozen): audited and '
               'reported as-is, no adjudication beyond the registered '
               'no-flip. ' + verdict)
  return dict(p_u1a_interaction=inter, p_u1b_apt_occupancy=bapt,
              fires_a=bool(a_fires), flips_b=bool(b_flips),
              level_changes_vs_frozen=levels,
              discriminator_note=(
                  'apt level change vs the disclosed proprio anchors '
                  '(rgo +235.2 / sgb +30.4, unfrozen calibration): '
                  'sgb-band gain => needed features ABSENT from apt fits '
                  '(competition/exclusion); rgo-band gain => '
                  'present-but-illegible. Descriptive only.'),
              verdict=verdict)


# ---------------------------------------------------------------------------
# U2 — stamped unfrozen (P-B1 / inclusion-not-usefulness stress)
# ---------------------------------------------------------------------------

def analyze_u2(cells, base=None):
  rng = np.random.default_rng(RNG_SEED)
  b_srd = 0.5 * ((cells['srd0s1'] - cells['srd0s0'])
                 + (cells['srd1s1'] - cells['srd1s0']))
  b_sid = cells['sids1'] - cells['sids0']
  primary = _boot(b_srd - b_sid, rng)
  flips = primary['ci'][0] > 0
  levels = (_level_changes(cells, base, tuple(MODES['u2']), rng)
            if base else None)
  if flips:
    verdict = ('FLIP: fine-tuning repurposes the installed stamp '
               '([B_srd − B_sid] > 0 unfrozen) — P-B1 and the '
               '"inclusion ≠ usefulness" claim rescope to '
               'frozen-readout; the UNREAL-refutation framing must be '
               'weakened as registered.')
  elif primary['ci'][1] < 0:
    verdict = ('NEGATIVE [B_srd − B_sid] — outside registered accounts '
               '(the stamp actively interferes unfrozen): audited and '
               'reported as-is; the no-flip decision stands but the '
               '"inert" description does not apply.')
  else:
    verdict = ('PROTOCOL-ROBUST: the installed stamp is inert even '
               'under full fine-tuning — "inclusion ≠ usefulness" '
               'upgrades from a frozen-readout fact to a two-protocol '
               'fact; the UNREAL-style refutation strengthens '
               '(aux-stamped directions are not even a useful '
               'initialization).')
  return dict(p_u2_srd_minus_sid=primary, flips=bool(flips),
              b_srd=_boot(b_srd, rng), b_sid=_boot(b_sid, rng),
              level_changes_vs_frozen=levels, verdict=verdict)


# ---------------------------------------------------------------------------
# U3 — orthogonal-objective unfrozen (objective-specificity stress)
# ---------------------------------------------------------------------------

def analyze_u3(cells, base=None):
  rng = np.random.default_rng(RNG_SEED)
  pooled_level = float(np.mean([cells[k].mean() for k in
                                ('rgo0', 'rgo1', 'sgb0', 'sgb1')]))
  d = 0.5 * ((cells['rgo1'] - cells['sgb1'])
             + (cells['rgo0'] - cells['sgb0']))
  primary = _boot(d, rng)
  floor_uninformative = pooled_level < U3_FLOOR
  flips = (not floor_uninformative) and primary['ci'][0] > 0
  levels = (_level_changes(cells, base, tuple(MODES['u3']), rng)
            if base else None)
  if floor_uninformative:
    verdict = (f'FLOOR — uninformative: pooled unfrozen orthogonal level '
               f'{pooled_level:.1f} < {U3_FLOOR}; the spin objective was '
               'not reachable even unfrozen; no adjudication (registered '
               'guard, mirrors the frozen wave).')
  elif flips:
    verdict = ('FLIP: the rgo advantage EXPRESSES on the orthogonal '
               'objective under fine-tuning — "objective-specific '
               'support" rescopes to frozen-readout; the band\'s '
               'orthogonal leg is re-annotated as protocol-scoped '
               '(registered consequence).')
  elif primary['ci'][1] < 0:
    verdict = ('NEGATIVE rgo − sgb on the orthogonal objective — '
               'outside registered accounts (sgb beats rgo under spin '
               'fine-tuning): audited and reported as-is; the no-flip '
               'decision stands but the null description does not '
               'apply.')
  else:
    verdict = ('PROTOCOL-ROBUST: rgo − sgb stays null on the orthogonal '
               'objective even unfrozen (above floor) — '
               'objective-specificity holds as an initialization fact, '
               'not just a frozen-feature fact; the strict headline '
               'reading strengthens.')
  return dict(p_u3_orth_effect=primary, pooled_level=pooled_level,
              floor_uninformative=bool(floor_uninformative),
              flips=bool(flips),
              side_simples=dict(
                  s0=_boot(cells['rgo0'] - cells['sgb0'], rng),
                  s1=_boot(cells['rgo1'] - cells['sgb1'], rng)),
              level_changes_vs_frozen=levels, verdict=verdict)


# ---------------------------------------------------------------------------
# U4 — pixel unfrozen 2x2 (floor relief + swamping-derived prediction)
# ---------------------------------------------------------------------------

def analyze_u4(cells, base=None):
  rng = np.random.default_rng(RNG_SEED)
  inter = _boot((cells['t1'] - cells['t0']) - (cells['f1'] - cells['f0']),
                rng)
  relief = None
  lifted = None
  if base and all(isinstance(base.get(k), np.ndarray) for k in
                  ('t0', 't1', 'f0', 'f1')):
    per_seed = np.mean([cells[k] - base[k]
                        for k in ('t0', 't1', 'f0', 'f1')], axis=0)
    relief = _boot(per_seed, rng)
    lifted = relief['ci'][0] > 0
  # Fail-safe: an uncomputable guard censors (never adjudicate P-U4b
  # unless P-U4a demonstrably fired) — review finding 27 Jul.
  censored = (lifted is not True)
  fires_b = (not censored) and inter['ci'][0] > 0
  if censored:
    verdict = ('FLOOR-CENSORED — uninformative: unfrozen adaptation did '
               'not lift the pixel cells off the frozen floor (P-U4a '
               'does not fire), so the interaction read cannot express; '
               'reported, no adjudication (registered guard).')
  elif fires_b:
    verdict = ('FLIP: the pixel interaction EXPRESSES under fine-tuning '
               '— the pixel boundary rescopes from "predicted structure '
               '(g > G2)" toward a frozen-readout construct, and the '
               'swamping account needs an initialization-geometry '
               'amendment (registered consequence; P-SW1 itself — a '
               'fit-time fact — is untouched).')
  else:
    verdict = ('PROTOCOL-ROBUST (the P-SW1-derived prediction): levels '
               'lift under fine-tuning but NO differential rescue — with '
               'no reward information in the trunk (P-SW1), unfrozen '
               'adaptation must re-carve from scratch identically across '
               'cells; the pixel boundary and the swamping mechanism '
               'hold in both protocols.')
  return dict(p_u4a_floor_relief=relief, lifted=lifted,
              p_u4b_interaction=inter, censored=bool(censored),
              fires_b=bool(fires_b),
              cell_means={k: float(v.mean()) for k, v in cells.items()},
              verdict=verdict)


ANALYZERS = dict(u1=analyze_u1, u2=analyze_u2, u3=analyze_u3, u4=analyze_u4)


def read(args):
  cells = load_cells(args.auc, MODES[args.wave])
  bpath = args.baseline or os.path.join(_REPO, BASELINES[args.wave])
  if not os.path.exists(bpath):
    raise SystemExit(f'{args.wave}: registered frozen baseline csv not '
                     f'found at {bpath} — the registered descriptors '
                     '(and, for u4, the censoring guard) cannot run')
  # require=True: every baseline cell must be complete at seeds 1-8 —
  # an incomplete/wrong baseline must never silently drop registered
  # descriptors or (u4) the censoring guard.
  base = load_cells(bpath, BASE_MODES[args.wave], require=True)
  res = ANALYZERS[args.wave](cells, base)
  res['auc_csv'] = os.path.abspath(args.auc)
  res['baseline_csv'] = os.path.abspath(bpath) if base else None
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, f'unfrozen_{args.wave}.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  for k, v in res.items():
    if isinstance(v, dict) and 'ci' in v:
      print(f"{k}: {v['point']:+.1f} CI=[{v['ci'][0]:+.1f},"
            f"{v['ci'][1]:+.1f}] pos={v['pos']}/{v['n']}")
  print(res['verdict'])
  print(f'-> {out}')


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
  cols = ['run_id', 'mode', 'domain', 'seed', 'milestone', 'auc100k',
          'qc_pass']
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)


def _rows(modes, level_fn, seeds=SEEDS):
  # Deterministic fixtures: constant per-cell levels make every branch
  # exact (bootstrap of constant deltas collapses to the point value).
  rows = []
  for key, mode in modes.items():
    for s in seeds:
      rows.append(dict(run_id=f'adapt_{mode}_finger_seed{s}_ckpt500000',
                       mode=mode, domain=DOMAIN, seed=s,
                       milestone=MILESTONE,
                       auc100k=round(level_fn(key, s), 3),
                       qc_pass=1))
  return rows


def selfcheck(args):
  import tempfile
  tmp = tempfile.mkdtemp()
  p = lambda n: os.path.join(tmp, n)

  # U1 branches: robust (interaction fires, apt null) and flip.
  lv = dict(t0=100, t1=260, f0=80, f1=80)
  _write_csv(p('u1.csv'), _rows(MODES['u1'], lambda k, s: lv[k])
             + _rows(dict(x='ax1q1s0'), lambda k, s: 999))  # frozen row
  cells = load_cells(p('u1.csv'), MODES['u1'])
  assert all(abs(cells[k].mean() - lv[k]) < 10 for k in lv), \
      'frozen-mode row leaked into U1 cells'
  res = analyze_u1(cells)
  assert res['fires_a'] and not res['flips_b']
  assert res['verdict'].startswith('PROTOCOL-ROBUST')
  lv2 = dict(t0=100, t1=260, f0=80, f1=200)
  _write_csv(p('u1b.csv'), _rows(MODES['u1'], lambda k, s: lv2[k]))
  res = analyze_u1(load_cells(p('u1b.csv'), MODES['u1']))
  assert res['flips_b'] and res['verdict'].startswith('PARTIAL FLIP')

  # U2 branches.
  lvr = dict(srd0s0=90, srd0s1=200, srd1s0=95, srd1s1=210,
             sids0=100, sids1=105)
  _write_csv(p('u2.csv'), _rows(MODES['u2'], lambda k, s: lvr[k]))
  res = analyze_u2(load_cells(p('u2.csv'), MODES['u2']))
  assert res['flips'] and res['verdict'].startswith('FLIP')
  lvn = {k: 100 + (5 if k.endswith('s1') else 0) for k in MODES['u2']}
  _write_csv(p('u2n.csv'), _rows(MODES['u2'], lambda k, s: lvn[k]))
  res = analyze_u2(load_cells(p('u2n.csv'), MODES['u2']))
  assert not res['flips'] and res['verdict'].startswith('PROTOCOL-ROBUST')
  # Negative-CI branch: significantly negative contrast must NOT be
  # narrated as an inert-stamp null (review finding 27 Jul).
  lvneg = dict(srd0s0=200, srd0s1=100, srd1s0=200, srd1s1=100,
               sids0=100, sids1=200)
  _write_csv(p('u2neg.csv'), _rows(MODES['u2'], lambda k, s: lvneg[k]))
  res = analyze_u2(load_cells(p('u2neg.csv'), MODES['u2']))
  assert not res['flips'] and res['verdict'].startswith('NEGATIVE')

  # U3 branches incl. floor gate.
  lvf = dict(rgo0=250, rgo1=300, sgb0=100, sgb1=120)
  _write_csv(p('u3.csv'), _rows(MODES['u3'], lambda k, s: lvf[k]))
  res = analyze_u3(load_cells(p('u3.csv'), MODES['u3']))
  assert res['flips'] and res['verdict'].startswith('FLIP')
  lvn3 = dict(rgo0=150, rgo1=160, sgb0=150, sgb1=160)
  _write_csv(p('u3n.csv'), _rows(MODES['u3'], lambda k, s: lvn3[k]))
  res = analyze_u3(load_cells(p('u3n.csv'), MODES['u3']))
  assert not res['flips'] and res['verdict'].startswith('PROTOCOL-ROBUST')
  lvfl = dict(rgo0=5, rgo1=6, sgb0=5, sgb1=5)
  _write_csv(p('u3f.csv'), _rows(MODES['u3'], lambda k, s: lvfl[k]))
  res = analyze_u3(load_cells(p('u3f.csv'), MODES['u3']))
  assert res['floor_uninformative'] and res['verdict'].startswith('FLOOR')

  # U4 branches: censored / robust / flip (needs baseline).
  basepx = dict(t0=77, t1=83, f0=81, f1=83)
  _write_csv(p('u4base.csv'), _rows(BASE_MODES['u4'],
                                    lambda k, s: basepx[k]))
  bcells = load_cells(p('u4base.csv'), BASE_MODES['u4'], require=False)
  _write_csv(p('u4c.csv'), _rows(MODES['u4'], lambda k, s: basepx[k]))
  res = analyze_u4(load_cells(p('u4c.csv'), MODES['u4']), bcells)
  assert res['censored'] and res['verdict'].startswith('FLOOR-CENSORED')
  lifted = dict(t0=200, t1=210, f0=195, f1=205)
  _write_csv(p('u4r.csv'), _rows(MODES['u4'], lambda k, s: lifted[k]))
  res = analyze_u4(load_cells(p('u4r.csv'), MODES['u4']), bcells)
  assert res['lifted'] and not res['fires_b']
  assert res['verdict'].startswith('PROTOCOL-ROBUST')
  flip = dict(t0=150, t1=380, f0=180, f1=190)
  _write_csv(p('u4f.csv'), _rows(MODES['u4'], lambda k, s: flip[k]))
  res = analyze_u4(load_cells(p('u4f.csv'), MODES['u4']), bcells)
  assert res['fires_b'] and res['verdict'].startswith('FLIP')
  # Fail-safe guard: an uncomputable P-U4a (incomplete/absent baseline
  # cells) must CENSOR, never adjudicate P-U4b (review finding 27 Jul).
  res = analyze_u4(load_cells(p('u4f.csv'), MODES['u4']), {})
  assert res['censored'] and not res['fires_b'], (
      'uncomputable guard must censor')
  # And read()'s require=True path: a baseline missing one seed trips.
  bad = _rows(BASE_MODES['u4'], lambda k, s: basepx[k])
  bad = [r for r in bad if not (r['mode'] == 'ax1pxpxq1ms0'
                                and r['seed'] == 5)]
  _write_csv(p('u4badbase.csv'), bad)
  try:
    load_cells(p('u4badbase.csv'), BASE_MODES['u4'], require=True)
    raise SystemExit('selfcheck FAIL: incomplete baseline not caught')
  except AssertionError:
    pass

  # Missing cell trips; qc trips.
  rows = _rows(MODES['u1'], lambda k, s: 100)
  rows = [r for r in rows if not (r['mode'] == 'ax1uzfq1s1'
                                  and r['seed'] == 4)]
  _write_csv(p('miss.csv'), rows)
  try:
    load_cells(p('miss.csv'), MODES['u1'])
    raise SystemExit('selfcheck FAIL: missing seed not caught')
  except AssertionError:
    pass
  rows = _rows(MODES['u1'], lambda k, s: 100)
  rows[0]['qc_pass'] = 0
  _write_csv(p('qc.csv'), rows)
  try:
    load_cells(p('qc.csv'), MODES['u1'])
    raise SystemExit('selfcheck FAIL: qc fail not caught')
  except AssertionError:
    pass

  print('selfcheck PASS: u1 robust/partial-flip, u2 flip/robust/negative, '
        'u3 flip/robust/floor, u4 censored/robust/flip + fail-safe guard '
        '+ incomplete-baseline trip, frozen-mode isolation, missing-seed '
        'and qc trips')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('wave', nargs='?', choices=('u1', 'u2', 'u3', 'u4',
                                              'selfcheck'))
  ap.add_argument('--auc')
  ap.add_argument('--baseline', default='')
  ap.add_argument('--output', default='analysis_out/unfrozen_stress')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck or args.wave == 'selfcheck':
    selfcheck(args)
  else:
    assert args.wave and args.auc, 'need <wave> --auc <csv>'
    read(args)


if __name__ == '__main__':
  main()
