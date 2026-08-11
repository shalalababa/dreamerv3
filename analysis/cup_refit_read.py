"""Frozen reader for the finger/cup regeneration reads (2026-08-10).

Registrations: PREREG_finger_refit_20260808.md (#9 apt-trunk probe +
finger Delta_dom + replication rider) and PREREG_cup_refit_20260808.md
(P-C3 adjudication + cup rider), as amended by
PREREG_refit_reads_amend1_20260810.md (checkpoint-step witness substituted
for the overlaid fit counters; mixed finger substrate with fresh-only
sensitivity; regime-leg structural facts).

Two executions total, one per registration:
  python -m analysis.cup_refit_read --domain finger --bundle <regen bundle> \
      --out artifacts/finger_refit_read_20260810
  python -m analysis.cup_refit_read --domain cup --bundle <regen bundle> \
      --finger_json artifacts/finger_refit_read_20260810/read.json \
      --out artifacts/cup_refit_read_20260810
Selfcheck (synthetic fixtures, no bundle): python -m analysis.cup_refit_read selfcheck
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
HEADLINE_ALPHA = 'alpha_0.001'
ALPHA_KEYS = ('alpha_0.0001', 'alpha_0.001', 'alpha_0.01')
CKPT_STEP = 500_000
AUROC_BAR = 0.70          # registered point bar (#9)
PARITY_BAND = 0.10        # registered |Delta_dom| band (P-C3 cup side)
FRESH_PREFIXES = ('20260809', '20260810')
JULY_ORIGINALS = {'ax1wm_finger_fq1s0_seed6', 'ax1wm_finger_fq1s1_seed5'}
ARCHIVED_E4 = {               # replication-rider anchors (14-Jul panel)
    'finger': 'artifacts/e4_goodhart_20260714/e4_finger_v1.csv',
    'cup': 'artifacts/e4_goodhart_20260714/e4_cup_v1.csv',
}


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def _ndtri(p):
  return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def bca(vals, clusters, rng_seed=RNG_SEED, b=B_BOOT):
  """Cluster bootstrap + BCa (house implementation, as w1_read)."""
  vals = np.asarray(vals, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  idx = {c: np.where(cl == c)[0] for c in uniq}
  rng = np.random.default_rng(rng_seed)
  boots = np.empty(b)
  for i in range(b):
    pick = rng.choice(len(uniq), len(uniq), replace=True)
    sel = np.concatenate([idx[uniq[p]] for p in pick])
    boots[i] = np.nanmean(vals[sel])
  theta = float(np.nanmean(vals))
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
  z0 = _ndtri(prop)
  jack = np.asarray([float(np.nanmean(vals[cl != c])) for c in uniq])
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  a_acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha_pt):
    z = _ndtri(alpha_pt)
    adj = z0 + (z0 + z) / (1.0 - a_acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2.0)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  return theta, (q(0.025), q(0.975))


def cluster_signflip_p(diffs, clusters, rng_seed=RNG_SEED, b=B_BOOT):
  """Cluster-level sign-flip permutation for mean != 0 (two-sided).
  Flips whole clusters together (8 seed clusters -> 256 distinct patterns;
  resolution disclosed in output)."""
  diffs = np.asarray(diffs, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  rng = np.random.default_rng(rng_seed)
  obs = abs(np.nanmean(diffs))
  masks = np.stack([cl == c for c in uniq])  # (C, n)
  flips = rng.choice([-1.0, 1.0], size=(b, len(uniq)))
  signs = flips @ masks  # (b, n) each entry +-1
  null = np.abs(np.nanmean(signs * diffs[None], axis=1))
  return float((np.sum(null >= obs - 1e-15) + 1) / (b + 1))


# --------------------------------------------------------------------------
# loading (fail-closed gates per the preregs + amendment 1)
# --------------------------------------------------------------------------

def load_rows(bundle, domain):
  probeset = f'{domain}_v1'
  pat = os.path.join(bundle, 'runs', f'ax1wm_{domain}_*')
  rows = []
  for rd in sorted(glob.glob(pat)):
    run = os.path.basename(rd)
    rj = json.load(open(os.path.join(rd, f'ridge_probe_{probeset}.json')))
    ej = json.load(open(os.path.join(
        rd, f'e4_{probeset}refit', 'summary.json')))
    arm = 'apt' if f'ax1wm_{domain}_f' in run else 'task'
    ck = os.path.basename(rj['checkpoint'])
    ck_e4 = os.path.basename(ej['checkpoint'])
    assert rj['run_id'] == run and ej['run_id'] == run, run
    assert int(ck.split('-')[-1]) == CKPT_STEP, (run, ck)
    assert int(ck_e4.split('-')[-1]) == CKPT_STEP, (run, ck_e4)
    assert rj['witness_match'] is True, (run, 'witness')
    assert rj['probeset_id'] == probeset, run
    assert rj['reward_override'] is None and ej['reward_override'] is None, run
    if arm == 'task':
      assert rj['expl_mode'] == 'task' and rj['reward_aware'] is True, run
    else:
      assert rj['expl_mode'] == 'apt' and rj['reward_aware'] is False, run
    assert all(k in rj['probe'] for k in ALPHA_KEYS), run
    mid = rj['probe'][HEADLINE_ALPHA]
    h0 = ej['horizon_stats']['0']
    rh = h0.get('reward_head')
    fresh = ck.startswith(FRESH_PREFIXES) and ck_e4.startswith(FRESH_PREFIXES)
    if not fresh:
      assert run in JULY_ORIGINALS, (run, ck, 'unexpected non-fresh substrate')
    side = 1 if f'q1s1' in run else 0
    seed = int(run.rsplit('seed', 1)[1])
    rows.append(dict(
        run=run, arm=arm, side=side, seed=seed, fresh=fresh,
        ckpt=ck, sha=rj['probeset_sha256'],
        auroc=mid['auroc'], auroc_in=mid['auroc_in_regime'],
        auroc_out=mid['auroc_out_regime'], r2=mid['r2'],
        rew_nll_in=(rh['in_regime']['mean'] if rh else None),
        rew_nll_out=(rh['out_regime']['mean'] if rh else None),
        rew_nll_all=(rh['all']['mean'] if rh else None),
    ))
  assert len(rows) == 32, len(rows)
  for arm in ('task', 'apt'):
    assert sum(r['arm'] == arm for r in rows) == 16, arm
  assert len({r['sha'] for r in rows}) == 1, 'probeset sha not uniform'
  return rows


# --------------------------------------------------------------------------
# analysis (pure; fixture-testable)
# --------------------------------------------------------------------------

def arm_auroc_stats(rows, arm):
  sub = [r for r in rows if r['arm'] == arm]
  out = {'n': len(sub)}
  for leg in ('auroc', 'auroc_in'):
    vals = [r[leg] for r in sub]
    if any(v is None for v in vals):
      out[leg] = None
      continue
    mean, ci = bca(vals, [r['seed'] for r in sub])
    out[leg] = dict(mean=mean, ci=list(ci))
  out['auroc_out_defined'] = all(r['auroc_out'] is not None for r in sub)
  out['r2_mean'] = float(np.mean([r['r2'] for r in sub]))
  return out


def grade_nine(task_stats, apt_stats):
  """#9 apt-trunk grading on the DEFINED legs (amendment 1 section 3)."""
  def leg_strong(s):
    return s is not None and s['ci'][0] > 0.5 and s['mean'] >= AUROC_BAR
  def leg_straddles(s):
    return s is not None and s['ci'][0] <= 0.5 <= s['ci'][1]
  legs = [k for k in ('auroc', 'auroc_in')
          if task_stats[k] is not None and apt_stats[k] is not None]
  if not legs:
    return 'INSTRUMENT-UNINFORMATIVE', 'no defined AUROC leg'
  if not all(leg_strong(task_stats[k]) for k in legs):
    return ('INSTRUMENT-UNINFORMATIVE',
            'task calibration arm fails the probe (registered escape)')
  if all(leg_strong(apt_stats[k]) for k in legs):
    return ('PRESENT-BUT-UNUSABLE',
            'apt CI>0.5 and point>=0.70 on all defined legs')
  if any(leg_straddles(apt_stats[k]) for k in legs):
    return ('ABSENT-OR-NONLINEAR',
            'apt AUROC CI includes 0.5 on a defined leg')
  return ('UNCLASSIFIED-DISCLOSED',
          'pattern outside the registered branches (e.g. CI>0.5 with '
          'point<0.70, or CI entirely <0.5) - no wording licensed')


def delta_dom(rows):
  """Domain contrast task-apt on pooled AUROC, paired within (side, seed);
  seed-cluster BCa + cluster sign-flip permutation."""
  cells = {}
  for r in rows:
    cells.setdefault((r['side'], r['seed']), {})[r['arm']] = r['auroc']
  diffs, clusters = [], []
  for (side, seed), d in sorted(cells.items()):
    if 'task' in d and 'apt' in d:
      diffs.append(d['task'] - d['apt'])
      clusters.append(seed)
  mean, ci = bca(diffs, clusters)
  p = cluster_signflip_p(diffs, clusters)
  return dict(mean=mean, ci=list(ci), perm_p=p, n_pairs=len(diffs),
              perm_note='cluster-level sign flips; 2^8=256 distinct '
                        'patterns bound the p resolution')


def grade_pc3(cup_delta, finger_delta):
  cup_lo, cup_hi = cup_delta['ci']
  fin_lo, fin_hi = finger_delta['ci']
  cup_straddles = cup_lo <= 0.0 <= cup_hi
  if cup_straddles and abs(cup_delta['mean']) <= PARITY_BAND and fin_lo > 0.0:
    return 'P-C3 FIRES', ('cup parity (CI includes 0, |Delta|<=0.10) with '
                          'finger violation (CI entirely > 0)')
  if cup_lo > 0.0 and not (cup_hi < fin_lo or fin_hi < cup_lo):
    return 'P-C3 REFUTED', ('cup CI entirely > 0 at magnitude overlapping '
                            'the finger contrast')
  return 'AMBIGUOUS', ('registered catch-all: neither parity-with-violation '
                       'nor overlap-refutation pattern')


def replication_rider(rows, domain, archived_csv):
  """Descriptive: regen task-arm h0 reward-head NLL vs archived 14-Jul
  panel (per side). Never a verdict."""
  if not os.path.exists(archived_csv):
    return {'error': f'archived panel missing: {archived_csv}'}
  arch = {}
  with open(archived_csv) as f:
    for row in csv.DictReader(f):
      if int(float(row['horizon'])) != 0:
        continue
      rid = row['run_id']
      if f'ax1wm_{domain}_q1s' not in rid:
        continue
      side = 1 if 'q1s1' in rid else 0
      arch.setdefault(side, []).append(
          tuple(float(row[k]) for k in
                ('rew_nll_in', 'rew_nll_out', 'rew_nll_all')))
  out = {}
  for side in (0, 1):
    regen = [(r['rew_nll_in'], r['rew_nll_out'], r['rew_nll_all'])
             for r in rows if r['arm'] == 'task' and r['side'] == side
             and r['rew_nll_in'] is not None]
    a = np.asarray(arch.get(side, []), float)
    g = np.asarray(regen, float)
    out[f's{side}'] = dict(
        regen_mean=[float(x) for x in g.mean(0)] if len(g) else None,
        archived_mean=[float(x) for x in a.mean(0)] if len(a) else None,
        n_regen=len(g), n_archived=len(a),
        fields=['rew_nll_in', 'rew_nll_out', 'rew_nll_all'])
  return out


def analyse(rows, domain, finger_ref=None, archived_csv=None):
  res = {'domain': domain, 'n_rows': len(rows),
         'substrate': {r['run']: ('fresh' if r['fresh'] else 'JULY-ORIGINAL')
                       for r in rows if not r['fresh']} or 'all-fresh',
         'n_fresh': sum(r['fresh'] for r in rows)}
  task = arm_auroc_stats(rows, 'task')
  apt = arm_auroc_stats(rows, 'apt')
  res['task'] = task
  res['apt'] = apt
  res['delta_dom'] = delta_dom(rows)
  if domain == 'finger':
    v, why = grade_nine(task, apt)
    res['verdict_9'] = {'grade': v, 'why': why,
                        'out_regime_leg': 'structurally unevaluable '
                        '(0 rewarded out-of-regime frames; amendment 1)',
                        'in_regime_power_note': 'in-regime leg has 6 '
                        'negative frames (low-powered)'}
    fresh = [r for r in rows if r['fresh']]
    if len(fresh) < len(rows):
      t2, a2 = arm_auroc_stats(fresh, 'task'), arm_auroc_stats(fresh, 'apt')
      v2, why2 = grade_nine(t2, a2)
      res['sensitivity_fresh_only'] = {
          'n': len(fresh), 'grade': v2, 'why': why2,
          'task': t2, 'apt': a2, 'delta_dom': delta_dom(fresh),
          'agrees_with_primary': v2 == v}
  if domain == 'cup':
    assert finger_ref is not None, 'cup read requires the finger read json'
    fdelta = finger_ref['delta_dom']
    v, why = grade_pc3(res['delta_dom'], fdelta)
    res['verdict_pc3'] = {
        'grade': v, 'why': why,
        'cup_delta': res['delta_dom'],
        'finger_delta': {'mean': fdelta['mean'], 'ci': fdelta['ci']},
        'in_regime_note': 'cup in-regime AUROC undefined (0 rewarded '
                          'in-regime frames); pooled AUROC per prereg rule'}
  if archived_csv is not None:
    res['replication_rider'] = replication_rider(rows, domain, archived_csv)
  return res


# --------------------------------------------------------------------------
# selfcheck (synthetic fixtures; no bundle access)
# --------------------------------------------------------------------------

def _fake_rows(task_auroc, apt_auroc, task_in=None, apt_in=None, jitter=0.01,
               rng=None):
  rng = rng or np.random.default_rng(7)
  rows = []
  for arm, mu, mu_in in (('task', task_auroc, task_in),
                         ('apt', apt_auroc, apt_in)):
    for side in (0, 1):
      for seed in range(1, 9):
        a = float(np.clip(mu + rng.normal(0, jitter), 0.01, 0.999))
        ai = (float(np.clip((mu_in if mu_in is not None else mu)
                            + rng.normal(0, jitter), 0.01, 0.999)))
        rows.append(dict(
            run=f'fake_{arm}_s{side}_seed{seed}', arm=arm, side=side,
            seed=seed, fresh=True, ckpt='x', sha='s',
            auroc=a, auroc_in=ai, auroc_out=None, r2=0.1,
            rew_nll_in=1.0, rew_nll_out=0.2, rew_nll_all=0.3))
  return rows


def selfcheck():
  # 1. BCa sanity: CI brackets the sample mean tightly, far from 0.5.
  rng = np.random.default_rng(0)
  vals = rng.normal(0.8, 0.02, 16)
  m, ci = bca(vals, list(range(8)) * 2)
  assert ci[0] < m < ci[1] and ci[0] > 0.6 and (ci[1] - ci[0]) < 0.1, (m, ci)
  # 2. #9 branches.
  r = analyse(_fake_rows(0.95, 0.90), 'finger')
  assert r['verdict_9']['grade'] == 'PRESENT-BUT-UNUSABLE', r['verdict_9']
  r = analyse(_fake_rows(0.95, 0.50, jitter=0.04), 'finger')
  assert r['verdict_9']['grade'] == 'ABSENT-OR-NONLINEAR', r['verdict_9']
  r = analyse(_fake_rows(0.55, 0.90, jitter=0.02), 'finger')
  assert r['verdict_9']['grade'] == 'INSTRUMENT-UNINFORMATIVE', r['verdict_9']
  r = analyse(_fake_rows(0.95, 0.60, jitter=0.005), 'finger')
  assert r['verdict_9']['grade'] == 'UNCLASSIFIED-DISCLOSED', r['verdict_9']
  # 3. mixed-substrate sensitivity path.
  rows = _fake_rows(0.95, 0.90)
  for x in rows[:2]:
    x['fresh'] = False
  r = analyse(rows, 'finger')
  assert r['sensitivity_fresh_only']['agrees_with_primary'] is True
  # 4. P-C3 branches (finger_ref with CI > 0).
  fref = {'delta_dom': {'mean': 0.3, 'ci': [0.2, 0.4]}}
  # exact-parity fixture: apt = task -+ 0.02 alternating per cell (mean 0).
  par = _fake_rows(0.80, 0.80, jitter=0.0)
  tv = {(x['side'], x['seed']): x['auroc'] for x in par if x['arm'] == 'task'}
  for i, x in enumerate(p for p in par if p['arm'] == 'apt'):
    x['auroc'] = tv[(x['side'], x['seed'])] + (0.02 if i % 2 else -0.02)
  r = analyse(par, 'cup', finger_ref=fref)
  assert r['verdict_pc3']['grade'] == 'P-C3 FIRES', r['verdict_pc3']
  r = analyse(_fake_rows(0.95, 0.65, jitter=0.005), 'cup', finger_ref=fref)
  assert r['verdict_pc3']['grade'] == 'P-C3 REFUTED', r['verdict_pc3']
  fref0 = {'delta_dom': {'mean': 0.0, 'ci': [-0.1, 0.1]}}
  r = analyse(_fake_rows(0.80, 0.80), 'cup', finger_ref=fref0)
  assert r['verdict_pc3']['grade'] == 'AMBIGUOUS', r['verdict_pc3']
  # parity band: cup delta ~0.2 straddling would still not FIRE (>0.10).
  fr = _fake_rows(0.80, 0.60, jitter=0.30, rng=np.random.default_rng(3))
  rr = analyse(fr, 'cup', finger_ref=fref)
  if (rr['delta_dom']['ci'][0] <= 0 <= rr['delta_dom']['ci'][1]
      and abs(rr['delta_dom']['mean']) > PARITY_BAND):
    assert rr['verdict_pc3']['grade'] != 'P-C3 FIRES'
  # 5. permutation: null data -> large p; strong signal -> small p.
  d0 = np.random.default_rng(1).normal(0, 1, 16)
  assert cluster_signflip_p(d0, list(range(8)) * 2) > 0.05
  d1 = np.abs(np.random.default_rng(2).normal(5, 0.5, 16))
  assert cluster_signflip_p(d1, list(range(8)) * 2) < 0.02
  print('cup_refit_read selfcheck PASS')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  sub.add_parser('selfcheck')
  ap.add_argument('--domain', choices=('finger', 'cup'))
  ap.add_argument('--bundle')
  ap.add_argument('--finger_json')
  ap.add_argument('--out')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck()
    return
  assert args.domain and args.bundle and args.out
  rows = load_rows(args.bundle, args.domain)
  fref = json.load(open(args.finger_json)) if args.finger_json else None
  if args.domain == 'cup':
    assert fref is not None, '--finger_json required for the cup read'
  res = analyse(rows, args.domain, finger_ref=fref,
                archived_csv=ARCHIVED_E4[args.domain])
  res['bundle'] = args.bundle
  res['amendment'] = 'PREREG_refit_reads_amend1_20260810.md'
  os.makedirs(args.out, exist_ok=True)
  out = os.path.join(args.out, 'read.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(f'wrote {out}')
  key = 'verdict_9' if args.domain == 'finger' else 'verdict_pc3'
  print(json.dumps({key: res[key], 'delta_dom': res['delta_dom']}, indent=1))


if __name__ == '__main__':
  main()
