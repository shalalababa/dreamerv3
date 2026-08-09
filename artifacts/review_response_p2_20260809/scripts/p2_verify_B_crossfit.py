"""Paper-2 review resolution — verification pass B: the cross-fit
(2026-08-09; labeled post-read analysis).

Reproduces review §1.2c: select the oracle argmax on the committed R3
late-cell pass (A), evaluate that slot on the repair wave's independent
re-pass (B) of the same 32 cells at the same (episode, step) index rule.

  within-pass opportunity  = mean_s [max_m G_A[m] - G_A[m_now_A]]
  cross-fit  opportunity   = mean_s [G_B[argmax_A] - G_B[m_now_B]]
                             (variant b: baseline G_B[m_now_A])

If the max-mean selection-bias account is right, argmax_A carries no
information about pass-B returns and the cross-fit collapses to ~0.
Alignment caveat (review D3b): the env is unseeded, so pass B's states
drift from pass A's — reported alongside a candidate-drift gating.

Run-clustered percentile bootstrap over the 32 late cells (cluster =
run), B=10K, rng 0 — same conventions as the frozen reader.
"""

import glob
import json
import os
import re

import numpy as np

ROOT = '/home/rickybao/projects/dreamerv3'
A_DIR = os.path.join(ROOT, 'local_results/r3_competence_20260729_105146/labels')
B_DIR = os.path.join(ROOT,
                     'local_results/r3_repair_20260802_143054/r3_local/rep_labels')
OUT = os.path.join(ROOT, 'artifacts/review_response_p2_20260809')
CELL_RE = re.compile(r'^rep_(cup|finger)_(e1|e4)_seed(3[1-8])_late\.npz$')


def boot(vals, b=10_000, seed=0):
  vals = np.asarray(vals, float)
  rng = np.random.default_rng(seed)
  n = len(vals)
  means = vals[rng.integers(0, n, (b, n))].mean(1)
  return float(vals.mean()), (float(np.percentile(means, 2.5)),
                              float(np.percentile(means, 97.5)))


def main():
  rows = []
  for path in sorted(glob.glob(os.path.join(B_DIR, 'rep_*.npz'))):
    m = CELL_RE.match(os.path.basename(path))
    if not m:
      continue
    dom, dose, seed = m.group(1), m.group(2), m.group(3)
    a = np.load(os.path.join(
        A_DIR, f'r3_{dom}_{dose}_seed{seed}_late.npz'), allow_pickle=True)
    b = np.load(path, allow_pickle=True)
    assert np.array_equal(a['episode'], b['episode'])
    assert np.array_equal(a['step'], b['step'])
    gA = a['g_all'].astype(float)
    gB = b['g_all'].astype(float)
    i = np.arange(len(gA))
    mnA = a['m_now'].astype(int)
    mnB = b['m_now'].astype(int)
    argA = gA.argmax(1)
    cand_drift = float(np.mean(np.linalg.norm(
        a['cands'].astype(float) - b['cands'].astype(float), axis=2)))
    rows.append(dict(
        cell=f'{dom}_{dose}_{seed}',
        within=float(np.mean(gA.max(1) - gA[i, mnA])),
        cross_bnow=float(np.mean(gB[i, argA] - gB[i, mnB])),
        cross_anow=float(np.mean(gB[i, argA] - gB[i, mnA])),
        within_ach=float(np.mean(
            gA[i, a['m_real'].astype(int)] - gA[i, mnA])),
        cand_drift=cand_drift))
  assert len(rows) == 32, len(rows)

  out = {}
  w = boot([r['within'] for r in rows])
  cb = boot([r['cross_bnow'] for r in rows])
  ca = boot([r['cross_anow'] for r in rows])
  ach = np.mean([r['within_ach'] for r in rows])
  gap_cross = boot([r['cross_bnow'] - r['within_ach'] for r in rows])
  out['all_32_cells'] = dict(
      within_pass_opportunity=[w[0], list(w[1])],
      crossfit_opportunity_baseline_mnowB=[cb[0], list(cb[1])],
      crossfit_opportunity_baseline_mnowA=[ca[0], list(ca[1])],
      crossfit_gap=[gap_cross[0], list(gap_cross[1])],
      registered_opp8_late=1.3014062499999999)

  drifts = np.array([r['cand_drift'] for r in rows])
  gate = drifts < np.median(drifts)
  wg = boot([r['within'] for r, g in zip(rows, gate) if g])
  cg = boot([r['cross_bnow'] for r, g in zip(rows, gate) if g])
  out['low_drift_half'] = dict(
      n_cells=int(gate.sum()),
      median_cand_drift=float(np.median(drifts)),
      drift_range=[float(drifts.min()), float(drifts.max())],
      within=[wg[0], list(wg[1])],
      crossfit=[cg[0], list(cg[1])])

  path = os.path.join(OUT, 'p2_verify_B_crossfit.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)


if __name__ == '__main__':
  main()
