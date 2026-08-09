"""Paper-2 review resolution — verification pass A (2026-08-09).

LABELED POST-READ ANALYSIS (both reviews already delivered; every
registered read executed long before). Reproduces, from the committed
label arrays, the load-bearing claims of
reviews/Review_FullRecord_20260807.md §1 and GPT_Diagnosis_20260809.md:

  A1  registered primaries reproduce exactly (validates cohort + arithmetic)
  A2  opportunity vs max-mean; m_now/m_real vs mean; m_now oracle rank
  A3  permutation exchangeability null (candidate->slot shuffle per state)
  A4  state-level floor fractions / P(opp>0) / concentration
  A5  non-floor (opp>0) primaries: achieved + m_now-vs-random
  A6  within-state dissociation: Spearman(q-mean, g_all) vs
      Spearman(real_scores, g_all) on non-degenerate states
  A7  g_all_boot opportunity (noise-doubling diagnostic)
  A8  variogram: sigma(delta) by action distance

Estimator conventions copied from the frozen analysis/r3_read.py
(cell = (dom,dose,seed,mat) mean of per-state values; cluster = run
(dom,dose,seed); pooled point = mean over cell means; run-clustered
percentile bootstrap B=10K rng 0).
"""

import glob
import json
import os
import re
import sys

import numpy as np

ROOT = '/home/rickybao/projects/dreamerv3'
LABELS = os.path.join(ROOT, 'local_results/r3_competence_20260729_105146/labels')
OUT = os.path.join(ROOT, 'artifacts/review_response_p2_20260809')
FILE_RE = re.compile(
    r'^r3_(cup|finger)_(e1|e4)_seed(3[1-8])_(early|late)\.npz$')
B_BOOT = 10_000


def load_cells():
  cells = {}
  for path in sorted(glob.glob(os.path.join(LABELS, '*.npz'))):
    m = FILE_RE.match(os.path.basename(path))
    if not m:
      continue
    z = np.load(path, allow_pickle=True)
    assert 'd1fix_20260724' in str(z['meta'])
    key = (m.group(1), m.group(2), int(m.group(3)), m.group(4))
    cells[key] = {k: np.asarray(z[k]) for k in
                  ('g_all', 'g_now', 'm_now', 'm_real', 'qfull',
                   'real_scores', 'g_all_boot', 'cands')}
  assert len(cells) == 64, len(cells)
  return cells


def cluster_boot(cell_vals, clusters, rng_seed=0, b=B_BOOT):
  """Run-clustered percentile bootstrap over cell means (r3_read style)."""
  vals = np.asarray(cell_vals, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  idx = {c: np.where(cl == c)[0] for c in uniq}
  rng = np.random.default_rng(rng_seed)
  means = []
  for _ in range(b):
    pick = rng.choice(len(uniq), len(uniq), replace=True)
    sel = np.concatenate([idx[uniq[p]] for p in pick])
    means.append(vals[sel].mean())
  return float(vals.mean()), (float(np.percentile(means, 2.5)),
                              float(np.percentile(means, 97.5)))


def per_state(cell, which):
  g, gn, mr, mn = (cell['g_all'].astype(float), cell['g_now'].astype(float),
                   cell['m_real'].astype(int), cell['m_now'].astype(int))
  i = np.arange(len(gn))
  opp = g.max(1) - gn
  ach = g[i, mr] - gn
  if which == 'opp':
    return opp
  if which == 'ach':
    return ach
  if which == 'gap':
    return opp - ach
  if which == 'maxmean':
    return g.max(1) - g.mean(1)
  if which == 'now_vs_mean':
    return gn - g.mean(1)
  if which == 'real_vs_mean':
    return g[i, mr] - g.mean(1)
  raise KeyError(which)


def pooled(cells, which, transform=None):
  vals, cl = [], []
  for key, cell in cells.items():
    x = per_state(cell, which)
    vals.append(float(np.mean(x)))
    cl.append(key[:3])
  return cluster_boot(vals, ['%s_%s_%s' % c for c in cl])


def main():
  cells = load_cells()
  out = {}

  # ---- A1 + A2 -----------------------------------------------------------
  opp_pt, opp_ci = pooled(cells, 'opp')
  gap_pt, gap_ci = pooled(cells, 'gap')
  mm_pt, mm_ci = pooled(cells, 'maxmean')
  now_pt, now_ci = pooled(cells, 'now_vs_mean')
  real_pt, real_ci = pooled(cells, 'real_vs_mean')
  reg = json.load(open(os.path.join(
      ROOT, 'artifacts/r3_competence_20260729/r3.json')))
  out['A1_primary_reproduction'] = dict(
      opportunity=dict(mine=opp_pt, registered=reg['primaries']
                       ['p_r3a_opportunity']['point'],
                       match=abs(opp_pt - reg['primaries']
                                 ['p_r3a_opportunity']['point']) < 1e-9),
      gap=dict(mine=gap_pt, registered=reg['primaries']['p_r3b_gap']['point'],
               match=abs(gap_pt - reg['primaries']['p_r3b_gap']['point'])
               < 1e-9))
  # m_now oracle rank (rank of g_now among the state's g_all, average rank
  # for ties, 1=worst..M=best; chance = (M+1)/2 = 4.5)
  ranks = []
  for cell in cells.values():
    g = cell['g_all'].astype(float)
    gn = cell['g_now'].astype(float)
    r = (g < gn[:, None]).sum(1) + 0.5 * (g == gn[:, None]).sum(1) + 0.5
    ranks.append(float(np.mean(r)))
  out['A2_selection_columns'] = dict(
      opportunity=[opp_pt, list(opp_ci)],
      max_minus_mean=[mm_pt, list(mm_ci)],
      m_now_minus_mean=[now_pt, list(now_ci)],
      m_real_minus_mean=[real_pt, list(real_ci)],
      m_now_mean_oracle_rank=float(np.mean(ranks)),
      chance_rank=4.5)

  # ---- A3 permutation null ----------------------------------------------
  rng = np.random.default_rng(0)
  perm_cells = {}
  for key, cell in cells.items():
    g = cell['g_all'].astype(float)
    P = np.array([rng.permutation(g.shape[1]) for _ in range(len(g))])
    gp = np.take_along_axis(g, P, axis=1)
    # m_now/m_real indices unchanged; their slots now hold random candidates
    perm_cells[key] = dict(cell, g_all=gp,
                           g_now=gp[np.arange(len(g)),
                                    cell['m_now'].astype(int)])
  popp_pt, popp_ci = pooled(perm_cells, 'opp')
  pach_pt, pach_ci = pooled(perm_cells, 'ach')
  out['A3_permutation_null'] = dict(
      observed_opportunity=opp_pt,
      null_opportunity=[popp_pt, list(popp_ci)],
      null_achieved=[pach_pt, list(pach_ci)])

  # ---- A4 state-level floors --------------------------------------------
  n_states = deg = zero_opp = live = 0
  all_opps = []
  for cell in cells.values():
    g = cell['g_all'].astype(float)
    opp = per_state(cell, 'opp')
    n_states += len(g)
    deg += int(np.sum(np.all(g == g[:, :1], axis=1)))
    zero_opp += int(np.sum(opp == 0))
    live += int(np.sum(opp > 0))
    all_opps.append(opp)
  all_opps = np.concatenate(all_opps)
  srt = np.sort(all_opps)[::-1]
  top1 = srt[:max(1, len(srt) // 100)].sum() / max(srt.sum(), 1e-12)
  out['A4_state_floors'] = dict(
      n_states=n_states,
      frac_all_candidates_identical=deg / n_states,
      frac_opportunity_zero=zero_opp / n_states,
      p_opportunity_positive=live / n_states,
      top1pct_share_of_total_opportunity=float(top1))

  # ---- A5 non-floor primaries -------------------------------------------
  nf_ach, nf_now, nf_opp, nf_cl = [], [], [], []
  for key, cell in cells.items():
    opp = per_state(cell, 'opp')
    sel = opp > 0
    if not sel.any():
      continue
    g = cell['g_all'].astype(float)[sel]
    gn = cell['g_now'].astype(float)[sel]
    mr = cell['m_real'].astype(int)[sel]
    i = np.arange(sel.sum())
    nf_ach.append(float(np.mean(g[i, mr] - gn)))
    nf_now.append(float(np.mean(gn - g.mean(1))))
    nf_opp.append(float(np.mean(opp[sel])))
    nf_cl.append('%s_%s_%s' % key[:3])
  a_pt, a_ci = cluster_boot(nf_ach, nf_cl)
  n_pt, n_ci = cluster_boot(nf_now, nf_cl)
  out['A5_nonfloor'] = dict(
      n_cells=len(nf_ach), n_clusters=len(set(nf_cl)),
      opportunity=float(np.mean(nf_opp)),
      achieved=[a_pt, list(a_ci)],
      m_now_vs_random=[n_pt, list(n_ci)])

  # ---- A6 within-state dissociation -------------------------------------
  def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))

  q_rhos, p_rhos, d_cl = [], [], []
  for key, cell in cells.items():
    g = cell['g_all'].astype(float)
    qm = cell['qfull'].astype(float).mean(1)          # (S, M)
    ps = cell['real_scores'].astype(float)
    keep = ~np.all(g == g[:, :1], axis=1)
    if not keep.any():
      continue
    qr = [spearman(qm[s], g[s]) for s in np.where(keep)[0]]
    pr = [spearman(ps[s], g[s]) for s in np.where(keep)[0]]
    q_rhos.append(float(np.mean(qr)))
    p_rhos.append(float(np.mean(pr)))
    d_cl.append('%s_%s_%s' % key[:3])
  qq = cluster_boot(q_rhos, d_cl)
  pp = cluster_boot(p_rhos, d_cl)
  out['A6_dissociation'] = dict(
      n_cells=len(q_rhos), n_clusters=len(set(d_cl)),
      spearman_qmean_gall=[qq[0], list(qq[1])],
      spearman_probescore_gall=[pp[0], list(pp[1])])

  # ---- A7 g_all_boot -----------------------------------------------------
  boot_cells = {}
  for key, cell in cells.items():
    gb = cell['g_all_boot'].astype(float)
    boot_cells[key] = dict(cell, g_all=gb,
                           g_now=gb[np.arange(len(gb)),
                                    cell['m_now'].astype(int)])
  bopp = pooled(boot_cells, 'opp')
  bgap = pooled(boot_cells, 'gap')
  out['A7_g_all_boot'] = dict(opportunity=[bopp[0], list(bopp[1])],
                              gap=[bgap[0], list(bgap[1])])

  # ---- A8 variogram ------------------------------------------------------
  bins = [(0.0, 0.0), (0.0, 0.005), (0.005, 0.05), (0.05, 0.3),
          (0.3, 0.6), (0.6, 1.2), (1.2, 99.0)]
  acc = {b: [0.0, 0] for b in bins}
  for cell in cells.values():
    g = cell['g_all'].astype(float)
    c = cell['cands'].astype(float)
    M = g.shape[1]
    iu, ju = np.triu_indices(M, 1)
    for s in range(len(g)):
      dg2 = (g[s, iu] - g[s, ju]) ** 2
      da = np.linalg.norm(c[s, iu] - c[s, ju], axis=1)
      for lo, hi in bins:
        sel = (da == 0.0) if hi == 0.0 else (da > lo) & (da <= hi)
        if sel.any():
          acc[(lo, hi)][0] += float(dg2[sel].sum())
          acc[(lo, hi)][1] += int(sel.sum())
  out['A8_variogram_sigma_by_action_distance'] = {
      f'({lo},{hi}]': dict(n=n, sigma=float(np.sqrt(s2 / (2 * n)))
                           if n else None)
      for (lo, hi), (s2, n) in acc.items()}

  path = os.path.join(OUT, 'p2_verify_A.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)


if __name__ == '__main__':
  main()
