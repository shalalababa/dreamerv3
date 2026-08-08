"""Frozen readers for the four 2026-08-08 review-resolution waves.

One file, four subcommands (registered in the corresponding preregs):
  randominit -> PREREG_randominit_control_20260808.md   (P-RI1/P-RI2)
  external   -> PREREG_inprotocol_external_20260808.md  (P-X1/P-X2)
  orthpos    -> PREREG_orthogonal_poscontrol_20260808.md (P-OP1)
  tm2mpc     -> PREREG_tm2_mpcfalse_20260808.md         (P-MF1/P-MF2)

Shared frozen conventions (08-08 standing rules baked in):
  * AUC100k rows from adaptation_auc csvs; qc_pass must be 1; rows with
    n_ep_100k below the csv's modal count are EXCLUDED and listed
    (sub-modal snapshot guard); duplicate (mode, seed, milestone) rows are
    fatal.
  * Primary test = exact permutation (sign-flip for paired contrasts,
    group-label shuffle for unpaired; 100,000 draws, rng(0), two-sided).
  * Percentile bootstrap CI (B = 10,000, rng(0)) is REPORTED alongside,
    never decisional alone. BCa + Welch t reported as sensitivity.

Selfcheck: `python -m analysis.review_waves_read selfcheck` — synthetic
fixtures crossing the registered verdict branches (incl. multi-domain
contamination, sub-modal/duplicate/qc guards, underpowered bound).
"""

import argparse
import collections
import csv
import json
import os

import numpy as np
from scipy import stats as sps

B_BOOT = 10_000
N_PERM = 100_000
RANDOM_FLOOR_BAND = (53.284, 113.682)   # mean +/- 2 sd of the 5-seed floor


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------

def load_rows(paths, modes, milestones=None, domain='finger',
              expect_ep=None):
  """mode -> {seed: auc100k}; enforces qc, modal-n_ep, duplicate guards.

  `paths` may be one csv path or a list. Within ONE file a duplicated
  (mode, seed, milestone) is fatal; ACROSS files the max-n_ep row wins
  (the D14 complete-bundle policy, explicit). No single archived csv
  carries every baseline population (verified 2026-08-08: ax1fq1s* lives
  in the volume_repl csv, scratch in the U1 csv), so the baseline input
  is registered as a LIST of archived csvs.
  The modal-n_ep guard is applied PER MODE (new adapt-only waves and
  archived baselines legitimately differ in episode cadence)."""
  if isinstance(paths, str):
    paths = [paths]
  best = {}
  found_any = False
  excluded = []
  for path in paths:
    # 2026-08-08 batch-review B1: the canonical csvs are MULTI-DOMAIN
    # (ax1fq1s0 seed1 appears for finger, cup AND synth) — without a
    # domain filter the duplicate guard is fatal on every archived csv
    # and, with a weaker guard, cup values (~600 AUC) would silently
    # pool into finger populations.
    rows = [r for r in csv.DictReader(open(path))
            if r['mode'] in modes and r.get('domain', domain) == domain and
            (milestones is None or r['milestone'] in milestones)]
    seen = collections.Counter(
        (r['mode'], r['seed'], r['milestone']) for r in rows)
    dups = [k for k, c in seen.items() if c > 1]
    assert not dups, f'{path}: duplicate rows: {dups}'
    for r in rows:
      found_any = True
      if r['qc_pass'] not in ('1', 'True', 'true'):
        # batch-review m11: exclude + disclose, never crash the read
        excluded.append((r['run_id'], 'qc_fail', r['qc_pass']))
        continue
      key = (r['mode'], int(r['seed']), r['milestone'])
      n = int(r['n_ep_100k'])
      if key not in best or n > best[key][1]:
        best[key] = (r['run_id'], n, float(r['auc100k']))
  if not found_any:
    raise SystemExit(f'{paths}: no rows for modes {sorted(modes)} '
                     f'domain {domain}')
  # batch-review m10: one mode at two milestones must not silently
  # collapse per-seed — fatal, decide the milestone upstream.
  ms_by_mode = collections.defaultdict(set)
  for (mode, seed, ms) in best:
    ms_by_mode[mode].add(ms)
  multi = {m: sorted(v) for m, v in ms_by_mode.items() if len(v) > 1}
  assert not multi, f'multiple milestones per mode: {multi}'
  per_mode_neps = collections.defaultdict(collections.Counter)
  for (mode, seed, ms), (rid, n, v) in best.items():
    per_mode_neps[mode][n] += 1
  modal = {m: c.most_common(1)[0][0] for m, c in per_mode_neps.items()}
  if expect_ep is not None:
    # batch-review m12: under majority truncation the modal count IS the
    # truncated count — pin the expectation where the protocol fixes it.
    bad = {m: c for m, c in modal.items() if c != expect_ep}
    assert not bad, (f'modal n_ep != expected {expect_ep}: {bad} — '
                     f'majority-truncation or protocol drift')
  out = collections.defaultdict(dict)
  for (mode, seed, ms), (rid, n, v) in sorted(best.items()):
    if n < modal[mode]:
      excluded.append((rid, n, modal[mode]))
      continue
    out[mode][seed] = v
  return dict(out), modal, excluded


def perm_unpaired(a, b, n=N_PERM):
  a, b = np.asarray(a, float), np.asarray(b, float)
  obs = abs(a.mean() - b.mean())
  pool = np.concatenate([a, b])
  rng = np.random.default_rng(0)
  cnt = 0
  for _ in range(n):
    rng.shuffle(pool)
    if abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs - 1e-12:
      cnt += 1
  return (cnt + 1) / (n + 1)


def perm_signflip(d):
  d = np.asarray(d, float)
  m = len(d)
  obs = abs(d.mean())
  if m <= 20:
    cnt = 0
    for mask in range(2 ** m):
      s = np.array([1 if (mask >> i) & 1 else -1 for i in range(m)])
      if abs((d * s).mean()) >= obs - 1e-12:
        cnt += 1
    return cnt / 2 ** m
  rng = np.random.default_rng(0)
  s = rng.choice([-1, 1], (N_PERM, m))
  return float((np.abs((d * s).mean(1)) >= obs - 1e-12).mean())


def boot_ci_unpaired(a, b):
  a, b = np.asarray(a, float), np.asarray(b, float)
  rng = np.random.default_rng(0)
  da = a[rng.integers(0, len(a), (B_BOOT, len(a)))].mean(1)
  db = b[rng.integers(0, len(b), (B_BOOT, len(b)))].mean(1)
  d = da - db
  return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_ci_paired(d):
  d = np.asarray(d, float)
  rng = np.random.default_rng(0)
  m = d[rng.integers(0, len(d), (B_BOOT, len(d)))].mean(1)
  return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def contrast_unpaired(a, b):
  return dict(
      n=[len(a), len(b)], mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
      diff=float(np.mean(a) - np.mean(b)),
      perm_p=perm_unpaired(a, b), boot_ci=boot_ci_unpaired(a, b),
      welch_p=float(sps.ttest_ind(a, b, equal_var=False).pvalue))


def contrast_paired(d):
  d = np.asarray(d, float)
  return dict(n=len(d), mean=float(d.mean()), pos=int((d > 0).sum()),
              perm_p=perm_signflip(d), boot_ci=boot_ci_paired(d),
              t_p=float(sps.ttest_1samp(d, 0).pvalue))


def emit(out, output):
  os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
  with open(output, 'w') as f:
    json.dump(out, f, indent=2)
  print(json.dumps({k: v for k, v in out.items()
                    if k in ('verdict', 'excluded_submodal')}, indent=1))
  print(f'-> {output}')


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_randominit(args):
  rif, modal_r, exc_r = load_rows(args.auc, {args.rif_mode})
  hist, modal_h, exc_h = load_rows(
      args.baseline_auc, {'ax1fq1s0', 'ax1fq1s1', 'scratch'})
  rif_v = list(rif[args.rif_mode].values())
  assert len(rif_v) == 8, f'need 8 rif runs, have {len(rif_v)}'
  apt = list(hist['ax1fq1s0'].values()) + list(hist['ax1fq1s1'].values())
  scratch = list(hist['scratch'].values())
  p_ri1 = contrast_unpaired(rif_v, apt)
  fires_diff = p_ri1['perm_p'] < 0.05
  lo, hi = RANDOM_FLOOR_BAND
  in_band = lo <= np.mean(rif_v) <= hi
  # batch-review M5: the wording upgrade needs perm-ns AND a magnitude
  # bound (|diff| <= 15.1 = one floor-sd), else ns is just low power.
  if not fires_diff and abs(p_ri1['diff']) <= 15.1:
    verdict = ('P-RI1: rif ~ apt (perm ns AND |diff| <= 15.1) => '
               'reward-free frozen transfer is indistinguishable from an '
               'untrained network; null family upgrades to no-learning form')
  elif not fires_diff:
    verdict = ('P-RI1: perm ns but |diff| > 15.1 => UNDERPOWERED — no '
               'wording upgrade; report the CI and stop')
  elif p_ri1['diff'] < 0:
    verdict = ('P-RI1: rif BELOW apt => apt features carry some frozen '
               'value; null-family wording stays "no benefit of '
               'composition"')
  else:
    verdict = ('P-RI1: rif ABOVE apt => frozen negative transfer of apt '
               'features (not predicted; report as-is, adversarial review '
               'before external use)')
  emit(dict(prereg='PREREG_randominit_control_20260808.md',
            p_ri1=p_ri1, p_ri2_floor_band=dict(band=[lo, hi],
                                               mean=float(np.mean(rif_v)),
                                               inside=bool(in_band)),
            scratch_mean=float(np.mean(scratch)),
            excluded_submodal=exc_r + exc_h, verdict=verdict), args.output)


def cmd_external(args):
  new, _, exc = load_rows(args.auc, {args.frozen_mode, args.unfrozen_mode})
  hist, _, exc_h = load_rows(
      args.baseline_auc, {'ax1fq1s0', 'ax1fq1s1', 'scratch'})
  fo = list(new[args.frozen_mode].values())
  uo = list(new[args.unfrozen_mode].values())
  assert len(fo) == len(uo) == 5, (len(fo), len(uo))
  apt = list(hist['ax1fq1s0'].values()) + list(hist['ax1fq1s1'].values())
  scratch = list(hist['scratch'].values())
  p_x1 = contrast_unpaired(fo, apt)
  p_x2 = contrast_unpaired(uo, scratch)
  v1 = ('P-X1: p2e-online frozen ~ apt => the apt arm fairly represents '
        'the family in-protocol' if p_x1['perm_p'] >= 0.05 else
        'P-X1: p2e-online frozen differs from apt => scope the reward-free '
        'null to offline fits (family-representation correction)')
  v2 = ('P-X2: p2e-online unfrozen below scratch => C-R2 extends to the '
        'canonical external method' if (p_x2['perm_p'] < 0.05 and
                                        p_x2['diff'] < 0) else
        'P-X2: not below scratch => C-R2 stays scoped to offline apt fits')
  emit(dict(prereg='PREREG_inprotocol_external_20260808.md',
            p_x1=p_x1, p_x2=p_x2,
            floor_band=RANDOM_FLOOR_BAND,
            excluded_submodal=exc + exc_h, verdict=f'{v1}; {v2}'),
       args.output)


def cmd_orthpos(args):
  modes = {f'{args.prefix}rgoq1s0', f'{args.prefix}rgoq1s1',
           f'{args.prefix}sgbq1s0', f'{args.prefix}sgbq1s1'}
  cells, _, exc = load_rows(args.auc, modes)
  # batch-review M8: the floor must be TURN_EASY's own, measured by the
  # wave's 5 random-policy runs (mode rndte) — the turn_hard band does
  # not transport (0.07 vs 0.03 target).
  fl, _, exc_f = load_rows(args.floor_auc, {args.floor_mode})
  fv = list(fl[args.floor_mode].values())
  assert len(fv) >= 5, f'need >= 5 floor runs, have {len(fv)}'
  f_lo = float(np.mean(fv) - 2 * np.std(fv, ddof=1))
  f_hi = float(np.mean(fv) + 2 * np.std(fv, ddof=1))
  pairs = []
  levels = []
  for side in ('s0', 's1'):
    r = cells[f'{args.prefix}rgoq1{side}']
    s = cells[f'{args.prefix}sgbq1{side}']
    common = sorted(set(r) & set(s))
    pairs += [r[k] - s[k] for k in common]
    levels += list(r.values()) + list(s.values())
  p_op1 = contrast_paired(np.asarray(pairs))
  pooled = float(np.mean(levels))
  sig = p_op1['perm_p'] < 0.05
  if sig and p_op1['mean'] > 0:
    verdict = ('P-OP1 FIRES => positive control passes: trunk differences '
               'detectable on the adjacent objective; the spin null is '
               'informative about orthogonality (U3 strengthened)')
  elif sig:
    verdict = ('P-OP1 SIGNIFICANTLY NEGATIVE (sgb > rgo on turn_easy) — '
               'not a registered branch: report as-is, adversarial review '
               'before any use; no U3 consequence taken')
  elif pooled <= f_hi:
    verdict = ('P-OP1 null AND pooled level at/below the measured '
               'turn_easy random-floor band => floor-confound confirmed: '
               'the orthogonal null is downgraded to unadjudicated '
               '(registered consequence)')
  else:
    verdict = ('P-OP1 null with levels above the measured turn_easy floor '
               'band => specificity narrower than "objective-specific" — '
               'sharpening; adversarial review before external use')
  emit(dict(prereg='PREREG_orthogonal_poscontrol_20260808.md',
            p_op1=p_op1, pooled_level=pooled,
            floor_band_turn_easy=[f_lo, f_hi],
            floor_runs=len(fv),
            excluded_submodal=exc + exc_f, verdict=verdict),
       args.output)


def cmd_tm2mpc(args):
  modes = {f'{args.prefix}awareq1s0', f'{args.prefix}awareq1s1',
           f'{args.prefix}freeq1s0', f'{args.prefix}freeq1s1'}
  cells, _, exc = load_rows(args.auc, modes)
  def delta(arm):
    a = cells[f'{args.prefix}{arm}q1s1']
    b = cells[f'{args.prefix}{arm}q1s0']
    common = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in common]), common
  d_aw, ca = delta('aware')
  d_fr, cf = delta('free')
  common = sorted(set(ca) & set(cf))
  # batch-review M6: the registered primary is n=16 seed-paired — all four
  # cells x 16 fit seeds (64 adapts). Fewer pairs = a different wave.
  assert len(common) == 16, (
      f'P-MF1 registered at n=16 pairs; have {len(common)} '
      f'(aware {len(ca)}, free {len(cf)})')
  inter = np.array([d_aw[ca.index(k)] - d_fr[cf.index(k)] for k in common])
  p_mf1 = contrast_paired(inter)
  p_mf2 = contrast_paired(d_aw)
  fires = p_mf1['perm_p'] < 0.05 and p_mf1['mean'] > 0
  verdict = (('P-MF1 FIRES => the interaction exists without the planner: '
              'family boundary re-attributed to the adaptation mechanism '
              '(wording change registered)') if fires else
             ('P-MF1 null => boundary survives the readout de-confound; '
              'family wording stands, one confound retired'))
  emit(dict(prereg='PREREG_tm2_mpcfalse_20260808.md',
            p_mf1=p_mf1, p_mf2_aware_simple=p_mf2,
            excluded_submodal=exc, verdict=verdict), args.output)


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
  cols = ['run_id', 'mode', 'domain', 'seed', 'milestone', 'auc50k',
          'n_ep_50k', 'auc100k', 'n_ep_100k', 'auc125k', 'n_ep_125k',
          'final10', 'qc_pass']
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
      base = dict(domain='finger', milestone='0', auc50k='1', n_ep_50k='48',
                  auc125k='1', n_ep_125k='112', final10='1', qc_pass='1',
                  n_ep_100k='96')
      base.update(r)
      base.setdefault('run_id',
                      f"adapt_{base['mode']}_finger_seed{base['seed']}"
                      f"_ckpt{base['milestone']}")
      w.writerow(base)


def cmd_selfcheck(args):
  import tempfile
  rng = np.random.default_rng(0)
  with tempfile.TemporaryDirectory() as tmp:
    # fixtures: apt ~ N(81, 15) n=32; scratch ~ N(148, 30) n=8
    hist = ([dict(mode='ax1fq1s0', seed=str(i), milestone='500000',
                  auc100k=f'{81 + 12 * rng.standard_normal():.3f}')
             for i in range(1, 17)] +
            [dict(mode='ax1fq1s1', seed=str(i), milestone='500000',
                  auc100k=f'{81 + 12 * rng.standard_normal():.3f}')
             for i in range(1, 17)] +
            [dict(mode='scratch', seed=str(i),
                  auc100k=f'{148 + 20 * rng.standard_normal():.3f}')
             for i in range(1, 9)])
    hist_csv = os.path.join(tmp, 'hist.csv')
    _write_csv(hist_csv, hist)

    # --- multi-domain contamination fixture (batch-review B1/m13) --------
    # the SAME (mode, seed) at another domain must neither trip the dup
    # guard nor pool into the finger population.
    hist_md = hist + [dict(mode='ax1fq1s0', seed=str(i), milestone='500000',
                           domain='cup', auc100k='600.0')
                      for i in range(1, 17)]
    hist_md_csv = os.path.join(tmp, 'hist_md.csv')
    _write_csv(hist_md_csv, hist_md)

    # --- randominit: null branch fires the ~apt verdict ------------------
    apt_mean = np.mean([float(r['auc100k']) for r in hist
                        if r['mode'].startswith('ax1fq1')])
    rif_null = [dict(mode='rif', seed=str(i),
                     auc100k=f'{apt_mean + 2 * rng.standard_normal():.3f}')
                for i in range(1, 9)]
    p = os.path.join(tmp, 'rif.csv'); _write_csv(p, rif_null)
    out = os.path.join(tmp, 'ri.json')
    cmd_randominit(argparse.Namespace(auc=p, baseline_auc=[hist_md_csv],
                                      rif_mode='rif', output=out))
    v = json.load(open(out))
    assert 'indistinguishable' in v['verdict'], v['verdict']
    assert v['p_ri2_floor_band']['inside'] is True
    # cup rows did NOT pool: apt population mean must be ~81, not ~340
    assert abs(v['p_ri1']['mean_b'] - apt_mean) < 1.0, v['p_ri1']
    # below-apt branch
    rif_low = [dict(mode='rif', seed=str(i),
                    auc100k=f'{30 + 5 * rng.standard_normal():.3f}')
               for i in range(1, 9)]
    p2 = os.path.join(tmp, 'rif2.csv'); _write_csv(p2, rif_low)
    cmd_randominit(argparse.Namespace(auc=p2, baseline_auc=[hist_csv],
                                      rif_mode='rif', output=out))
    assert 'BELOW apt' in json.load(open(out))['verdict']
    # above-apt branch (m13)
    rif_hi = [dict(mode='rif', seed=str(i),
                   auc100k=f'{160 + 5 * rng.standard_normal():.3f}')
              for i in range(1, 9)]
    p2b = os.path.join(tmp, 'rif2b.csv'); _write_csv(p2b, rif_hi)
    cmd_randominit(argparse.Namespace(auc=p2b, baseline_auc=[hist_csv],
                                      rif_mode='rif', output=out))
    assert 'ABOVE apt' in json.load(open(out))['verdict']
    # underpowered branch (M5): big |diff| but ns via huge spread
    rif_up = [dict(mode='rif', seed=str(i),
                   auc100k=f'{81 + 30 * (1 if i % 2 else -1) + 20:.3f}')
              for i in range(1, 9)]
    p2c = os.path.join(tmp, 'rif2c.csv'); _write_csv(p2c, rif_up)
    cmd_randominit(argparse.Namespace(auc=p2c, baseline_auc=[hist_csv],
                                      rif_mode='rif', output=out))
    vv = json.load(open(out))['verdict']
    assert ('UNDERPOWERED' in vv) or ('BELOW' in vv) or ('ABOVE' in vv), vv

    # --- guards: sub-modal exclusion + duplicate fatal -------------------
    rif_bad = rif_null[:7] + [dict(mode='rif', seed='8', auc100k='500.0',
                                   n_ep_100k='40')]
    p3 = os.path.join(tmp, 'rif3.csv'); _write_csv(p3, rif_bad)
    try:
      cmd_randominit(argparse.Namespace(auc=p3, baseline_auc=[hist_csv],
                                        rif_mode='rif', output=out))
      raise AssertionError('sub-modal row consumed without tripping n=8 gate')
    except AssertionError as e:
      if 'sub-modal' in str(e):
        raise
      pass  # excluded -> only 7 runs -> n-assert trips: guard works
    dup = rif_null + [rif_null[0]]
    p4 = os.path.join(tmp, 'rif4.csv'); _write_csv(p4, dup)
    try:
      cmd_randominit(argparse.Namespace(auc=p4, baseline_auc=[hist_csv],
                                        rif_mode='rif', output=out))
      raise AssertionError('duplicate row survived')
    except AssertionError as e:
      assert 'duplicate' in str(e)
    # qc_pass=0 row is excluded+disclosed, not fatal (m11)
    rif_qc = rif_null[:8]
    rif_qc[0] = dict(rif_qc[0]); rif_qc[0]['qc_pass'] = '0'
    p4b = os.path.join(tmp, 'rif4b.csv'); _write_csv(p4b, rif_qc)
    try:
      cmd_randominit(argparse.Namespace(auc=p4b, baseline_auc=[hist_csv],
                                        rif_mode='rif', output=out))
      raise AssertionError('qc-fail row did not reduce n (guard silent)')
    except AssertionError as e:
      assert 'need 8 rif runs' in str(e), e

    # --- external: both branches ----------------------------------------
    ext = ([dict(mode='p2eof', seed=str(i),
                 auc100k=f'{82 + 8 * rng.standard_normal():.3f}')
            for i in range(1, 6)] +
           [dict(mode='p2eou', seed=str(i),
                 auc100k=f'{85 + 8 * rng.standard_normal():.3f}')
            for i in range(1, 6)])
    p5 = os.path.join(tmp, 'ext.csv'); _write_csv(p5, ext)
    out2 = os.path.join(tmp, 'ext.json')
    cmd_external(argparse.Namespace(auc=p5, baseline_auc=[hist_csv],
                                    frozen_mode='p2eof',
                                    unfrozen_mode='p2eou', output=out2))
    v = json.load(open(out2))['verdict']
    assert 'fairly represents' in v and 'extends to the canonical' in v, v
    # m13: opposite branches — frozen clearly above apt, unfrozen ~scratch
    ext2 = ([dict(mode='p2eof', seed=str(i),
                  auc100k=f'{170 + 6 * rng.standard_normal():.3f}')
             for i in range(1, 6)] +
            [dict(mode='p2eou', seed=str(i),
                  auc100k=f'{150 + 6 * rng.standard_normal():.3f}')
             for i in range(1, 6)])
    p5b = os.path.join(tmp, 'ext2.csv'); _write_csv(p5b, ext2)
    cmd_external(argparse.Namespace(auc=p5b, baseline_auc=[hist_csv],
                                    frozen_mode='p2eof',
                                    unfrozen_mode='p2eou', output=out2))
    v2 = json.load(open(out2))['verdict']
    assert 'differs from apt' in v2 and 'stays scoped' in v2, v2

    # --- orthpos: fire / floor / off-floor / negative branches -----------
    floor_rows = [dict(mode='rndte', seed=str(i),
                       auc100k=f'{120 + 10 * rng.standard_normal():.3f}')
                  for i in range(1, 6)]
    p_fl = os.path.join(tmp, 'floor.csv'); _write_csv(p_fl, floor_rows)
    for rgo_mu, sgb_mu, expect in (
        (260, 130, 'positive control passes'),
        (125, 122, 'floor-confound confirmed'),
        (300, 300, 'above the measured turn_easy floor'),
        (130, 260, 'SIGNIFICANTLY NEGATIVE')):
      cells = []
      for side in ('s0', 's1'):
        for i in range(1, 5):
          cells.append(dict(mode=f'ax1terrgoq1{side}', seed=str(i),
                            auc100k=f'{rgo_mu + 4 * rng.standard_normal():.3f}'))
          cells.append(dict(mode=f'ax1tersgbq1{side}', seed=str(i),
                            auc100k=f'{sgb_mu + 4 * rng.standard_normal():.3f}'))
      p6 = os.path.join(tmp, f'op{rgo_mu}_{sgb_mu}.csv'); _write_csv(p6, cells)
      out3 = os.path.join(tmp, 'op.json')
      cmd_orthpos(argparse.Namespace(auc=p6, floor_auc=p_fl,
                                     floor_mode='rndte', prefix='ax1ter',
                                     output=out3))
      assert expect in json.load(open(out3))['verdict'], (rgo_mu, sgb_mu,
                                                          expect)

    # --- tm2mpc: fire vs null branches -----------------------------------
    for gap, expect in ((90, 'without the planner'),
                        (0, 'boundary survives')):
      cells = []
      for i in range(1, 17):
        aw0 = 200 + 30 * rng.standard_normal()
        cells += [dict(mode='tm2mfawareq1s0', seed=str(i),
                       auc100k=f'{aw0:.3f}'),
                  dict(mode='tm2mfawareq1s1', seed=str(i),
                       auc100k=f'{aw0 + gap + 12 * rng.standard_normal():.3f}'),
                  dict(mode='tm2mffreeq1s0', seed=str(i),
                       auc100k=f'{90 + 10 * rng.standard_normal():.3f}'),
                  dict(mode='tm2mffreeq1s1', seed=str(i),
                       auc100k=f'{92 + 10 * rng.standard_normal():.3f}')]
      p7 = os.path.join(tmp, f'mf{gap}.csv'); _write_csv(p7, cells)
      out4 = os.path.join(tmp, 'mf.json')
      cmd_tm2mpc(argparse.Namespace(auc=p7, prefix='tm2mf', output=out4))
      assert expect in json.load(open(out4))['verdict'], (gap, expect)
  print('review_waves_read selfcheck PASS (branches: rif~apt/below/above/'
        'underpowered, multi-domain contamination, sub-modal + duplicate + '
        'qc trips, external all four wordings, orthpos fire/floor/off-floor/'
        'negative vs measured rndte band, tm2mpc fire+null w/ n=16 assert)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='cmd', required=True)
  r = sub.add_parser('randominit')
  r.add_argument('--auc', required=True)
  r.add_argument('--baseline_auc', required=True, nargs='+',
                 help='archived csvs jointly covering ax1fq1s* (e.g. the '
                      'volume_repl csv) + scratch (the U1 csv); max-n_ep '
                      'row wins across files')
  r.add_argument('--rif_mode', default='rif')
  r.add_argument('--output', required=True)
  r.set_defaults(fn=cmd_randominit)
  e = sub.add_parser('external')
  e.add_argument('--auc', required=True)
  e.add_argument('--baseline_auc', required=True, nargs='+')
  e.add_argument('--frozen_mode', default='p2eof')
  e.add_argument('--unfrozen_mode', default='p2eou')
  e.add_argument('--output', required=True)
  e.set_defaults(fn=cmd_external)
  o = sub.add_parser('orthpos')
  o.add_argument('--auc', required=True)
  o.add_argument('--floor_auc', required=True,
                 help='csv with the turn_easy random-policy floor runs')
  o.add_argument('--floor_mode', default='rndte')
  o.add_argument('--prefix', default='ax1ter')
  o.add_argument('--output', required=True)
  o.set_defaults(fn=cmd_orthpos)
  t = sub.add_parser('tm2mpc')
  t.add_argument('--auc', required=True)
  t.add_argument('--prefix', default='tm2mf')
  t.add_argument('--output', required=True)
  t.set_defaults(fn=cmd_tm2mpc)
  s = sub.add_parser('selfcheck')
  s.set_defaults(fn=cmd_selfcheck)
  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
