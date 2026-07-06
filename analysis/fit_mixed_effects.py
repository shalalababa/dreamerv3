"""Pre-registered Phase-5a mixed-effects analysis (PREREG_phase5a §5-6).

Fits, on the dose-response grid:
  M0  auc_z ~ 1 + milestone_c            (+ random intercept | pretrain_run)
  M1  auc_z ~ 1 + <driver>_z             one model per driver
  M2  auc_z ~ all drivers jointly        secondary; VIFs reported

Primary population = the decoupled domains (default cup+finger), pooled after
within-domain z-scoring; the control domain (walker) is fit separately and
never pooled. Inference: cluster bootstrap over pretrain_run (B=1000, seed 0),
percentile 95% CIs.

Inputs:
  --auc      auc.csv from analysis/adaptation_auc.py
  --drivers  optional CSV with columns mode,domain,seed,milestone,<driver...>
             (from probing/measure_drivers.py + probe/vsa collation);
             without it only M0 is fit.

Paired contrasts (Phase 6/6b) live in `paired_contrast()`, exposed via
  python -m analysis.fit_mixed_effects paired --csv deltas.csv \
      --value auc100k --cond_a A --cond_b B
where deltas.csv has columns seed, cond, <value>.

Self-check (synthetic data with known effects):
  python -m analysis.fit_mixed_effects --selfcheck
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

DRIVERS = ('cov', 'occ_phys', 'occ_rew', 'fwd', 'geom', 'vsa', 'retdec')
N_BOOT = 1000
N_BOOT_PAIRED = 10_000
SEED = 0


def zscore_within_domain(df, cols):
  out = df.copy()
  for col in cols:
    if col not in out.columns:
      continue
    z = out.groupby('domain')[col].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
    out[col + '_z'] = z
  return out


def fit_lmm(df, formula):
  import statsmodels.formula.api as smf
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model = smf.mixedlm(formula, df, groups=df['pretrain_run'])
    return model.fit(reml=True)


def cluster_bootstrap(df, formula, params, n_boot, rng):
  """Percentile CIs from resampling pretrain_run clusters with replacement."""
  groups = df['pretrain_run'].unique()
  draws = {p: [] for p in params}
  n_fail = 0
  for _ in range(n_boot):
    picked = rng.choice(groups, size=len(groups), replace=True)
    parts = []
    for j, g in enumerate(picked):
      part = df[df['pretrain_run'] == g].copy()
      part['pretrain_run'] = f'{g}__b{j}'
      parts.append(part)
    boot = pd.concat(parts, ignore_index=True)
    try:
      res = fit_lmm(boot, formula)
      for p in params:
        draws[p].append(res.params[p])
    except Exception:
      n_fail += 1
  cis = {}
  for p in params:
    arr = np.asarray(draws[p])
    cis[p] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) \
        if len(arr) else (float('nan'), float('nan'))
  return cis, n_fail


def vif_table(df, cols):
  X = df[cols].to_numpy(float)
  X = np.column_stack([np.ones(len(X)), X])
  vifs = {}
  for i, col in enumerate(cols, start=1):
    others = [j for j in range(X.shape[1]) if j != i]
    beta, *_ = np.linalg.lstsq(X[:, others], X[:, i], rcond=None)
    resid = X[:, i] - X[:, others] @ beta
    tss = ((X[:, i] - X[:, i].mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / tss if tss > 0 else 0.0
    vifs[col] = float(1.0 / max(1.0 - r2, 1e-12))
  return vifs


def summarize(res, term):
  return dict(beta=float(res.params[term]),
              se=float(res.bse[term]),
              wald_lo=float(res.params[term] - 1.96 * res.bse[term]),
              wald_hi=float(res.params[term] + 1.96 * res.bse[term]),
              n_obs=int(res.nobs))


def run_population(df, drivers, outcome, n_boot, rng, label):
  """Fit M0/M1/M2 on one population; return a results dict."""
  results = {'population': label, 'n_rows': int(len(df)),
             'n_runs': int(df['pretrain_run'].nunique()), 'models': {}}
  df = df.copy()
  df['milestone_c'] = (df['milestone'] - 300_000) / 100_000.0

  m0 = fit_lmm(df, f'{outcome}_z ~ milestone_c')
  ci, _ = cluster_bootstrap(df, f'{outcome}_z ~ milestone_c',
                            ['milestone_c'], n_boot, rng)
  results['models']['M0_dose'] = {
      'milestone_c': {**summarize(m0, 'milestone_c'),
                      'boot_lo': ci['milestone_c'][0],
                      'boot_hi': ci['milestone_c'][1]}}

  present = [d for d in drivers if f'{d}_z' in df.columns
             and df[f'{d}_z'].notna().all()]
  for d in drivers:
    if d in present:
      continue
    if f'{d}_z' in df.columns:
      print(f'  [{label}] driver {d}: missing values, rows dropped per-model')
  for d in [x for x in drivers if f'{x}_z' in df.columns]:
    sub = df[df[f'{d}_z'].notna()]
    if not len(sub):
      continue
    formula = f'{outcome}_z ~ {d}_z'
    res = fit_lmm(sub, formula)
    ci, nf = cluster_bootstrap(sub, formula, [f'{d}_z'], n_boot, rng)
    results['models'][f'M1_{d}'] = {
        f'{d}_z': {**summarize(res, f'{d}_z'),
                   'boot_lo': ci[f'{d}_z'][0], 'boot_hi': ci[f'{d}_z'][1],
                   'boot_failures': nf}}

  joint = [d for d in ('cov', 'occ_phys', 'fwd', 'geom', 'vsa')
           if f'{d}_z' in df.columns]
  if len(joint) >= 2:
    sub = df.dropna(subset=[f'{d}_z' for d in joint])
    if len(sub) >= 10:
      formula = f'{outcome}_z ~ ' + ' + '.join(f'{d}_z' for d in joint)
      res = fit_lmm(sub, formula)
      terms = [f'{d}_z' for d in joint]
      ci, nf = cluster_bootstrap(sub, formula, terms, n_boot, rng)
      m2 = {t: {**summarize(res, t), 'boot_lo': ci[t][0], 'boot_hi': ci[t][1]}
            for t in terms}
      m2['vif'] = vif_table(sub, terms)
      m2['boot_failures'] = nf
      results['models']['M2_joint'] = m2
  return results


def paired_contrast(df, value, cond_a, cond_b, n_boot=N_BOOT_PAIRED,
                    seed=SEED):
  """Within-seed deltas between two matched conditions (PREREG §6)."""
  a = df[df['cond'] == cond_a].set_index('seed')[value]
  b = df[df['cond'] == cond_b].set_index('seed')[value]
  seeds = sorted(set(a.index) & set(b.index))
  deltas = np.asarray([a[s] - b[s] for s in seeds], float)
  rng = np.random.default_rng(seed)
  boots = [rng.choice(deltas, size=len(deltas), replace=True).mean()
           for _ in range(n_boot)]
  nz = deltas[deltas != 0]
  k, n = int((nz > 0).sum()), len(nz)
  from scipy.stats import binomtest
  p_sign = float(binomtest(k, n, 0.5).pvalue) if n else float('nan')
  return dict(cond_a=cond_a, cond_b=cond_b, n_pairs=len(seeds),
              mean_delta=float(deltas.mean()) if len(deltas) else float('nan'),
              boot_lo=float(np.percentile(boots, 2.5)),
              boot_hi=float(np.percentile(boots, 97.5)),
              sign_test_p=p_sign, deltas=[float(d) for d in deltas])


def selfcheck():
  """Synthetic recovery test: known driver effect + run intercepts."""
  rng = np.random.default_rng(1)
  rows = []
  for domain in ('cup', 'finger'):
    for mode in ('p2e', 'apt', 'random'):
      for seed in range(1, 6):
        run = f'{mode}_{domain}_{seed}'
        u = rng.normal(0, 0.5)
        for ms in (100, 200, 300, 400, 500):
          cov = rng.normal(ms / 500.0, 0.3)
          occ = rng.normal(0, 1)
          auc = 0.8 * cov + 0.0 * occ + u + rng.normal(0, 0.3)
          rows.append(dict(mode=mode, domain=domain, seed=seed,
                           milestone=ms * 1000, pretrain_run=run,
                           auc100k=auc, cov=cov, occ_phys=occ))
  df = zscore_within_domain(pd.DataFrame(rows),
                            ['auc100k', 'cov', 'occ_phys'])
  res = run_population(df, ('cov', 'occ_phys'), 'auc100k', 100,
                       np.random.default_rng(SEED), 'selfcheck')
  b_cov = res['models']['M1_cov']['cov_z']
  b_occ = res['models']['M1_occ_phys']['occ_phys_z']
  assert b_cov['beta'] > 0.4, b_cov
  assert b_cov['boot_lo'] > 0.2, b_cov
  assert abs(b_occ['beta']) < 0.25, b_occ
  assert b_occ['boot_lo'] < 0 < b_occ['boot_hi'], b_occ
  dose = res['models']['M0_dose']['milestone_c']
  assert dose['beta'] > 0, dose

  pdf = pd.DataFrame(
      [dict(seed=s, cond='A', auc100k=1.0 + 0.5 * s % 3 + 0.4)
       for s in range(8)] +
      [dict(seed=s, cond='B', auc100k=1.0 + 0.5 * s % 3)
       for s in range(8)])
  pc = paired_contrast(pdf, 'auc100k', 'A', 'B', n_boot=2000)
  assert abs(pc['mean_delta'] - 0.4) < 1e-9, pc
  assert pc['sign_test_p'] < 0.01, pc
  print('selfcheck PASS')
  print(json.dumps({k: res['models'][k] for k in
                    ('M0_dose', 'M1_cov', 'M1_occ_phys')}, indent=2))


def main():
  if len(sys.argv) > 1 and sys.argv[1] == 'paired':
    p = argparse.ArgumentParser()
    p.add_argument('paired')
    p.add_argument('--csv', required=True)
    p.add_argument('--value', default='auc100k')
    p.add_argument('--cond_a', required=True)
    p.add_argument('--cond_b', required=True)
    args = p.parse_args()
    out = paired_contrast(pd.read_csv(args.csv), args.value,
                          args.cond_a, args.cond_b)
    print(json.dumps(out, indent=2))
    return

  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--auc', help='auc.csv from adaptation_auc.py')
  p.add_argument('--drivers', default='',
                 help='Optional driver CSV (mode,domain,seed,milestone,...)')
  p.add_argument('--output', default='',
                 help='Output directory for results.json')
  p.add_argument('--outcome', default='auc100k')
  p.add_argument('--primary_domains', nargs='+', default=['cup', 'finger'])
  p.add_argument('--control_domain', default='walker')
  p.add_argument('--n_boot', type=int, default=N_BOOT)
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()

  if args.selfcheck:
    selfcheck()
    return
  if not args.auc or not args.output:
    raise SystemExit('--auc and --output are required (or --selfcheck)')

  df = pd.read_csv(args.auc)
  df = df[df['qc_pass'] == 1].copy()
  df['pretrain_run'] = (df['mode'] + '_' + df['domain'] + '_'
                        + df['seed'].astype(str))
  drivers = ()
  if args.drivers:
    drv = pd.read_csv(args.drivers)
    df = df.merge(drv, on=['mode', 'domain', 'seed', 'milestone'], how='left')
    drivers = tuple(d for d in DRIVERS if d in df.columns)
  df = zscore_within_domain(df, [args.outcome, *drivers])

  rng = np.random.default_rng(args.seed)
  out = {'outcome': args.outcome, 'spec': 'analysis/PREREG_phase5a.md',
         'populations': []}
  primary = df[df['domain'].isin(args.primary_domains)]
  if len(primary):
    out['populations'].append(run_population(
        primary, drivers, args.outcome, args.n_boot, rng,
        '+'.join(args.primary_domains)))
  control = df[df['domain'] == args.control_domain]
  if len(control):
    out['populations'].append(run_population(
        control, drivers, args.outcome, args.n_boot, rng,
        args.control_domain))

  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, 'results.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'Wrote {path}')


if __name__ == '__main__':
  main()
