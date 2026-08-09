"""Paper-2 mechanism pass D — the shared-mark leak witness (2026-08-09,
labeled exploratory, post-read).

`op_real` probes each candidate under the SAME rng_mark as that
candidate's G rollout (review §1.3 noted this as a small upward leak on
`achieved`). Consequence for the DISSOCIATION result: real_scores and
g_all share the first-step realization, so Spearman(real_scores, g_all)
has a mechanical component that Spearman(qfull, g_all) does not.

TM2 provides a built-in witness: many TM2 states have DUPLICATE
candidate actions (median 1 distinct). On a set of exactly-identical
candidates the true expected returns are identical, so ANY within-
duplicate-set rank agreement between real_scores and g_all is pure
shared-realization leak. dv3 has no duplicates, so the dv3 witness is
indirect (reported for structure only).

Registered-cohort discipline: TM2 = tm2r3_{cup,finger}_{e1,e4}_seed
{51..58}_{early,late} (64 cells, smoke files excluded); dv3 = the 64
R3 cells. Cell-pooled means with run-clustered bootstrap (B=10K, rng 0)
— same conventions as the frozen readers.

Outputs:
  D1  TM2 duplicate-set leak: within-state Spearman(rs, g) computed
      ONLY inside duplicate-action groups (>=3 members) — must be 0 if
      no leak; each group's true signal is 0 by construction.
  D2  TM2 distinct-only dissociation: Spearman across DISTINCT-action
      representatives only (first index per unique action) — the
      leak-reduced version of the review's +0.130.
  D3  dv3/TM2 review-construction reproduction (non-floor states,
      cell-pooled) for side-by-side context.
"""

import glob
import json
import os
import re

import numpy as np

ROOT = '/home/rickybao/projects/dreamerv3'
OUT = os.path.join(ROOT, 'artifacts/review_response_p2_20260809')
DV3_DIR = os.path.join(ROOT,
                       'local_results/r3_competence_20260729_105146/labels')
TM2_DIR = os.path.join(ROOT, 'local_results/tm2_r3_20260803_175551/tm2r3/labels')
DV3_RE = re.compile(r'^r3_(cup|finger)_(e1|e4)_seed(3[1-8])_(early|late)\.npz$')
TM2_RE = re.compile(
    r'^tm2r3_(cup|finger)_(e1|e4)_seed(5[1-8])_(early|late)\.npz$')


def spearman(a, b):
  ra = np.argsort(np.argsort(a)).astype(float)
  rb = np.argsort(np.argsort(b)).astype(float)
  sa, sb = ra.std(), rb.std()
  if sa < 1e-12 or sb < 1e-12:
    return np.nan
  return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def cluster_boot(cell_vals, clusters, b=10_000, seed=0):
  vals = np.asarray(cell_vals, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  idx = {c: np.where(cl == c)[0] for c in uniq}
  rng = np.random.default_rng(seed)
  means = []
  for _ in range(b):
    pick = rng.choice(len(uniq), len(uniq), replace=True)
    sel = np.concatenate([idx[uniq[p]] for p in pick])
    means.append(np.nanmean(vals[sel]))
  return float(np.nanmean(vals)), (float(np.percentile(means, 2.5)),
                                   float(np.percentile(means, 97.5)))


def load(dirpath, rex):
  cells = []
  for path in sorted(glob.glob(os.path.join(dirpath, '*.npz'))):
    m = rex.match(os.path.basename(path))
    if not m:
      continue
    z = np.load(path, allow_pickle=True)
    cells.append(dict(
        cluster=f'{m.group(1)}_{m.group(2)}_{m.group(3)}',
        g=z['g_all'].astype(float), q=z['qfull'].astype(float),
        rs=z['real_scores'].astype(float), c=z['cands'].astype(float)))
  return cells


def dup_leak(cells):
  """Within duplicate-action groups: rank agreement rs<->g (true = 0)."""
  vals, cl = [], []
  for cell in cells:
    per_state = []
    for s in range(len(cell['g'])):
      acts = np.round(cell['c'][s], 6)
      _, inv = np.unique(acts, axis=0, return_inverse=True)
      for grp in range(inv.max() + 1):
        idx = np.where(inv == grp)[0]
        if len(idx) < 3:
          continue
        g = cell['g'][s, idx]
        if g.std() < 1e-12:
          continue
        rho = spearman(cell['rs'][s, idx], g)
        if np.isfinite(rho):
          per_state.append(rho)
    if per_state:
      vals.append(float(np.mean(per_state)))
      cl.append(cell['cluster'])
  return cluster_boot(vals, cl), len(vals)


def distinct_only(cells):
  """Dissociation on distinct-action representatives only."""
  p_vals, q_vals, cl = [], [], []
  for cell in cells:
    qm = cell['q'].mean(1)
    pr, qr = [], []
    for s in range(len(cell['g'])):
      acts = np.round(cell['c'][s], 6)
      _, first = np.unique(acts, axis=0, return_index=True)
      idx = np.sort(first)
      if len(idx) < 3:
        continue
      g = cell['g'][s, idx]
      if g.std() < 1e-12:
        continue
      a = spearman(cell['rs'][s, idx], g)
      b = spearman(qm[s, idx], g)
      if np.isfinite(a):
        pr.append(a)
      if np.isfinite(b):
        qr.append(b)
    if pr:
      p_vals.append(float(np.mean(pr)))
      q_vals.append(float(np.mean(qr)) if qr else np.nan)
      cl.append(cell['cluster'])
  return cluster_boot(p_vals, cl), cluster_boot(q_vals, cl), len(p_vals)


def review_construction(cells):
  """Non-degenerate states, all M slots (the §6 construction)."""
  p_vals, q_vals, cl = [], [], []
  for cell in cells:
    qm = cell['q'].mean(1)
    pr, qr = [], []
    for s in range(len(cell['g'])):
      g = cell['g'][s]
      if g.std() < 1e-12:
        continue
      a = spearman(cell['rs'][s], g)
      b = spearman(qm[s], g)
      if np.isfinite(a):
        pr.append(a)
      if np.isfinite(b):
        qr.append(b)
    if pr:
      p_vals.append(float(np.mean(pr)))
      q_vals.append(float(np.mean(qr)) if qr else np.nan)
      cl.append(cell['cluster'])
  return cluster_boot(p_vals, cl), cluster_boot(q_vals, cl), len(p_vals)


def main():
  out = {}
  tm2 = load(TM2_DIR, TM2_RE)
  assert len(tm2) == 64, len(tm2)
  dv3 = load(DV3_DIR, DV3_RE)
  assert len(dv3) == 64, len(dv3)

  (leak, leak_ci), n = dup_leak(tm2)
  out['D1_tm2_duplicate_leak'] = dict(
      n_cells=n, rho=[leak, list(leak_ci)],
      note='true signal is 0 by construction inside duplicate groups; '
           'nonzero = shared first-step realization leak')

  (dp, dq, n2) = distinct_only(tm2)
  out['D2_tm2_distinct_only'] = dict(
      n_cells=n2,
      probe=[dp[0][0] if isinstance(dp[0], tuple) else dp[0], list(dp[1])],
      ensemble=[dq[0][0] if isinstance(dq[0], tuple) else dq[0],
                list(dq[1])])

  for tag, cells in (('tm2', tm2), ('dv3', dv3)):
    (rp, rq, n3) = review_construction(cells)
    out[f'D3_{tag}_review_construction'] = dict(
        n_cells=n3, probe=[rp[0], list(rp[1])],
        ensemble=[rq[0], list(rq[1])])

  path = os.path.join(OUT, 'p2_mech_D_leak.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)


if __name__ == '__main__':
  main()
