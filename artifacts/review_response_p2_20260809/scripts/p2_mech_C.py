"""Paper-2 mechanism mining — pass C (2026-08-09; labeled exploratory,
post-read). Digs into WHY the dissociation holds and where the signal
lives, from committed arrays only.

  C1  Flatness vs misranking: is the value ensemble's within-state
      candidate spread (sd over m of q_mean) small relative to (a) its
      own head disagreement and (b) the G spread? "Cannot rank" splits
      into "does not discriminate" (flat) vs "discriminates wrongly".
  C2  Probe-skill localization: per-state Spearman(real_scores, g_all)
      binned by G spread, q spread, domain, dose, maturity — where does
      the one-real-step signal live?
  C3  Probe-vs-ensemble agreement: do the two choosers agree with each
      other more than either agrees with G (shared wrongness)?
  C4  TM2 distinctness: does TM2's larger probe skill (+0.130) ride on
      states with DISTINCT candidate actions (its genuine action
      dependence), vs dv3?
  C5  Concentration curves: share of total opportunity carried by top
      q% of states (both families) — is value-of-information rare-and-
      concentrated?

Caveat registered up front: Spearman(real_scores, g_all) shares the
first-step env reward between score and target under a common RNG mark
(review §1.3 leak) — the leak-free decomposition (store r_real,
independent probe/eval marks) is W1's job; C-results are structure
maps, not final mechanism verdicts.
"""

import glob
import json
import os
import re

import numpy as np

ROOT = '/home/rickybao/projects/dreamerv3'
OUT = os.path.join(ROOT, 'artifacts/review_response_p2_20260809')
DV3 = os.path.join(ROOT, 'local_results/r3_competence_20260729_105146/labels')
TM2 = os.path.join(ROOT, 'local_results/tm2_r3_20260803_175551')
FILE_RE = re.compile(
    r'^r3_(cup|finger)_(e1|e4)_seed(3[1-8])_(early|late)\.npz$')


def spearman(a, b):
  ra = np.argsort(np.argsort(a)).astype(float)
  rb = np.argsort(np.argsort(b)).astype(float)
  sa, sb = ra.std(), rb.std()
  if sa < 1e-12 or sb < 1e-12:
    return np.nan
  return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def load(dirpath, pattern):
  cells = []
  for path in sorted(glob.glob(os.path.join(dirpath, '**', '*.npz'),
                               recursive=True)):
    name = os.path.basename(path)
    m = pattern.match(name) if hasattr(pattern, 'match') else None
    if pattern is FILE_RE and not m:
      continue
    if pattern is not FILE_RE and 'label' not in path:
      continue
    z = np.load(path, allow_pickle=True)
    if 'g_all' not in z.files or 'real_scores' not in z.files:
      continue
    cells.append(dict(
        name=name,
        g=z['g_all'].astype(float), q=z['qfull'].astype(float),
        rs=z['real_scores'].astype(float), c=z['cands'].astype(float),
        mn=z['m_now'].astype(int)))
  return cells


def analyze(cells, tag):
  out = {}
  rows = []
  for cell in cells:
    g, q, rs, c = cell['g'], cell['q'], cell['rs'], cell['c']
    qm = q.mean(1)                      # (S, M) candidate head-mean
    qs = q.std(1)                       # head disagreement per candidate
    for s in range(len(g)):
      gspread = float(g[s].std())
      if gspread < 1e-9:
        continue
      qspread = float(qm[s].std())
      hnoise = float(qs[s].mean())
      ndist = len(np.unique(np.round(c[s], 6), axis=0))
      rows.append(dict(
          cell=cell['name'], gspread=gspread, qspread=qspread,
          hnoise=hnoise, ndistinct=ndist,
          rho_q=spearman(qm[s], g[s]),
          rho_p=spearman(rs[s], g[s]),
          rho_qp=spearman(qm[s], rs[s])))
  R = {k: np.array([r[k] for r in rows], float)
       for k in ('gspread', 'qspread', 'hnoise', 'ndistinct',
                 'rho_q', 'rho_p', 'rho_qp')}
  fin = np.isfinite(R['rho_q']) & np.isfinite(R['rho_p'])

  # C1 flatness
  out['C1_flatness'] = dict(
      n_live_states=len(rows),
      median_q_candidate_spread=float(np.median(R['qspread'])),
      median_q_head_noise=float(np.median(R['hnoise'])),
      spread_over_noise_median=float(np.median(
          R['qspread'] / np.maximum(R['hnoise'], 1e-9))),
      median_g_spread=float(np.median(R['gspread'])),
      qspread_over_gspread_median=float(np.median(
          R['qspread'] / R['gspread'])))

  # C2 localization by G-spread quartile
  qt = np.quantile(R['gspread'], [0.25, 0.5, 0.75])
  loc = []
  for lo, hi in zip([-np.inf, *qt], [*qt, np.inf]):
    sel = fin & (R['gspread'] > lo) & (R['gspread'] <= hi)
    loc.append(dict(n=int(sel.sum()),
                    rho_p=float(np.nanmean(R['rho_p'][sel])),
                    rho_q=float(np.nanmean(R['rho_q'][sel]))))
  out['C2_by_gspread_quartile'] = loc

  # C3 shared wrongness
  out['C3_agreement'] = dict(
      mean_rho_probe_vs_ensemble=float(np.nanmean(R['rho_qp'])),
      mean_rho_probe_vs_G=float(np.nanmean(R['rho_p'])),
      mean_rho_ensemble_vs_G=float(np.nanmean(R['rho_q'])))

  # C4 distinctness
  dist = []
  for nd in (1, 2, 3):
    sel = fin & (R['ndistinct'] == nd) if nd < 3 else fin & (
        R['ndistinct'] >= 3)
    dist.append(dict(ndistinct=('%d' % nd if nd < 3 else '>=3'),
                     n=int(sel.sum()),
                     rho_p=float(np.nanmean(R['rho_p'][sel]))
                     if sel.any() else None))
  out['C4_by_distinct_candidates'] = dist
  return out


def concentration(cells):
  opps = []
  for cell in cells:
    g, mn = cell['g'], cell['mn']
    opps.append(g.max(1) - g[np.arange(len(g)), mn])
  o = np.sort(np.concatenate(opps))[::-1]
  tot = max(o.sum(), 1e-12)
  return {f'top_{q}pct_share': float(o[:max(1, int(len(o) * q / 100))].sum()
                                     / tot) for q in (1, 5, 10, 25)}


def main():
  out = {}
  dv3 = load(DV3, FILE_RE)
  out['dv3'] = analyze(dv3, 'dv3')
  out['dv3']['C5_concentration'] = concentration(dv3)
  tm2 = load(TM2, re.compile('.*'))
  if tm2:
    out['tm2'] = analyze(tm2, 'tm2')
    out['tm2']['C5_concentration'] = concentration(tm2)
    out['tm2']['n_cells'] = len(tm2)
  path = os.path.join(OUT, 'p2_mech_C.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)


if __name__ == '__main__':
  main()
