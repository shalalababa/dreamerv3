"""Tier-0 EXPLORATORY kill: legibility-gated compute allocation, retrospective.

Algorithm candidate (Algorithm_Ideation_20260819 §3.6): an ASHA-like sweep
scheduler whose rung-0 is the fit-time in-regime reward-NLL probe (minutes)
— in-band candidates fast-tracked, out-of-band ones still evaluated
(promote-never-eliminate). This retrospective replays the program's own
(probe, realized-transfer) pairs as a simulated scheduler. Zero compute.

POST-READ EXPLORATORY. Consumes no registered estimand. Arm-level means at a
fixed buffer are the licensed unit (FullRecord §10) — the simulation's
candidates are arm×side cells, never runs.

Pair assembly:
- probe  = mean rew_nll_in over horizon==0 rows of the finger-proprio E4
  csvs (per arm×side run-group; pixel/cup csvs excluded — different probeset
  scale).
- transfer = mean auc100k over qc_pass rows of the consolidated AUC table
  (vgo_extended_20260726/auc.csv) for the matching mode, plus the dose read's
  per-level (membership mean, cell mean) pairs.
- E4 run-group 'ax1wm_finger_<tag>_seedN' maps to AUC mode 'ax1<tag>'.
- Arms present in more than one E4 csv keep the largest-n csv's value.

Success criteria (v2 — DISCLOSED REVISION: v1's SC2 used the E4
`reward_aware` column, which turned out constant across all matched arms
(it marks the probeset labeling, not the arm's objective) and was therefore
vacuous; v2 replaces it with the competitive-subpool test, which is the
honest form of the same caveat. Revised before any verdict was adopted):
- SC1: gated ordering (ascending probe) reaches >=95% of the pool's best
  transfer with less adapt-spend than the median random ordering.
- SC2: SC1 still holds on the COMPETITIVE subpool (arms with transfer above
  the 147.68 scratch anchor — the pool a practitioner would actually sweep;
  the ideation doc's caveat is that resolution may be near zero there).
- SC3 (sensitivity): SC1 survives dropping the single best-probe arm.
Verdict: PROCEED requires all three; SC1-only or SC1+SC3 = RESCOPE
(coarse triage tool); SC1 fails = DROP.

Usage:
  python -m analysis.tier0_alloc_retro --artifacts artifacts \
      --output artifacts/algo_tier0_20260820/alloc_retro
"""

import argparse
import csv as csvlib
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

E4_CSVS = [  # finger-proprio probesets only
    'p3_wave_20260717/e4_factorial_finger_v1.csv',
    'stamping_20260718/e4_stamping_true_label_finger_v1.csv',
    'e4_amend1_optionc_20260723/e4_finger_v1_p3amend1_rgo.csv',
    'e4_amend1_optionc_20260723/e4_finger_v1_p3amend1_sgb.csv',
    'e4_amend1_optionc_20260723/e4_finger_v1_optionc_volume.csv',
    'e4_amend1_optionc_20260723/e4_finger_v1_sgbq1_full_20260724.csv',
    'e4_goodhart_20260714/e4_finger_v1.csv',
    'scaling_optionb_20260726/e4_finger_v1_scaling_b.csv',
]
AUC_CSV = 'vgo_extended_20260726/auc.csv'
DOSE_JSON = 'dose_task_read_20260811/dose_task.json'
RUN_RE = re.compile(r'^ax1wm_finger_(?P<tag>.+)_seed\d+$')
N_PERM = 2000


def read_e4(path):
  """arm-tag -> (mean rew_nll_in @ horizon 0, n_runs, reward_aware)."""
  by_run = defaultdict(list)
  aware = {}
  with open(path) as f:
    for r in csvlib.DictReader(f):
      if int(float(r['horizon'])) != 0 or not r.get('rew_nll_in'):
        continue
      by_run[r['run_id']].append(float(r['rew_nll_in']))
      aware[r['run_id']] = int(float(r.get('reward_aware', 1)))
  arms = defaultdict(list)
  arm_aware = {}
  for run, vals in by_run.items():
    m = RUN_RE.match(run)
    if not m:
      continue
    arms[m.group('tag')].append(np.mean(vals))
    arm_aware[m.group('tag')] = aware[run]
  return {t: (float(np.mean(v)), len(v), arm_aware[t])
          for t, v in arms.items()}


def read_auc(path):
  by_mode = defaultdict(list)
  with open(path) as f:
    for r in csvlib.DictReader(f):
      if r.get('qc_pass') not in (None, '', '1'):
        continue
      if r.get('auc100k'):
        by_mode[r['mode']].append(float(r['auc100k']))
  return {m: (float(np.mean(v)), len(v)) for m, v in by_mode.items()}


def simulate(probe, transfer):
  """Spend (1 unit/adapt) to reach >=95% of best transfer: gated vs random."""
  n = len(probe)
  target = 0.95 * max(transfer)
  order_g = np.argsort(probe)  # ascending NLL = most legible first
  spend_g = next(i + 1 for i in range(n)
                 if max(transfer[j] for j in order_g[:i + 1]) >= target)
  rng = np.random.default_rng(0)
  spends = []
  for _ in range(N_PERM):
    p = rng.permutation(n)
    spends.append(next(i + 1 for i in range(n)
                       if max(transfer[j] for j in p[:i + 1]) >= target))
  return spend_g, float(np.median(spends)), [int(s) for s in
                                             np.percentile(spends, [25, 75])]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--artifacts', default='artifacts')
  ap.add_argument('--output', required=True)
  args = ap.parse_args()
  A = args.artifacts

  e4 = {}
  e4_src = {}
  for rel in E4_CSVS:
    path = os.path.join(A, rel)
    if not os.path.exists(path):
      print(f'skip (missing): {rel}')
      continue
    for tag, (nll, n, aware) in read_e4(path).items():
      if tag not in e4 or n > e4[tag][1]:
        e4[tag] = (nll, n, aware)
        e4_src[tag] = rel
  auc = read_auc(os.path.join(A, AUC_CSV))

  pairs = []
  unmatched = []
  for tag, (nll, n, aware) in sorted(e4.items()):
    mode = 'ax1' + tag
    if mode in auc:
      pairs.append(dict(arm=mode, probe=nll, transfer=auc[mode][0],
                        n_probe=n, n_auc=auc[mode][1],
                        reward_aware=aware, src=e4_src[tag]))
    else:
      unmatched.append(mode)
  d = json.load(open(os.path.join(A, DOSE_JSON)))
  for lvl, cm in d['cell_means'].items():
    pl = d['p_dr3_membership']['per_level'][lvl]
    pairs.append(dict(arm=f'dose_level_{lvl}', probe=float(pl['mean']),
                      transfer=float(cm), n_probe=int(pl['n']), n_auc=8,
                      reward_aware=1, src=DOSE_JSON))

  probe = np.array([p['probe'] for p in pairs])
  trans = np.array([p['transfer'] for p in pairs])
  SCRATCH = 147.6823
  comp = trans > SCRATCH

  def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

  g_all, r_all, iqr_all = simulate(probe, trans)
  g_c, r_c, iqr_c = simulate(probe[comp], trans[comp])
  drop = np.arange(len(pairs)) != int(np.argmin(probe))
  g_d, r_d, iqr_d = simulate(probe[drop], trans[drop])
  sc1 = g_all < r_all
  sc2 = g_c < r_c
  sc3 = g_d < r_d
  verdict = ('PROCEED' if sc1 and sc2 and sc3 else
             'RESCOPE-TRIAGE' if sc1 else 'DROP')

  out = dict(
      label='EXPLORATORY (post-read; no registered estimand consumed)',
      n_pairs=len(pairs), n_competitive=int(comp.sum()),
      scratch_anchor=SCRATCH,
      unmatched_modes=unmatched,
      spearman_all=spearman(probe, -trans),
      spearman_competitive=spearman(probe[comp], -trans[comp]),
      spearman_note='rank corr of probe NLL with NEGATIVE transfer: '
                    'positive = legible arms transfer better',
      sim_all=dict(gated_spend=g_all, random_median=r_all,
                   random_iqr=iqr_all),
      sim_competitive=dict(gated_spend=g_c, random_median=r_c,
                           random_iqr=iqr_c),
      sim_drop_best_probe=dict(gated_spend=g_d, random_median=r_d,
                               random_iqr=iqr_d),
      criteria=dict(SC1=bool(sc1), SC2_competitive=bool(sc2),
                    SC3_drop_best=bool(sc3)),
      verdict=verdict,
      pairs=pairs)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'alloc_retro.json'), 'w') as f:
    json.dump(out, f, indent=1)
  print(f'{len(pairs)} pairs ({int(comp.sum())} competitive); '
        f'unmatched: {len(unmatched)}')
  print(f'spearman(probe, -transfer): all {out["spearman_all"]:+.3f}  '
        f'competitive {out["spearman_competitive"]:+.3f}')
  print(f'spend-to-95%: ALL gated {g_all} vs random {r_all} {iqr_all}; '
        f'COMP gated {g_c} vs random {r_c} {iqr_c}; '
        f'DROP-BEST gated {g_d} vs random {r_d} {iqr_d}')
  print(f'SC1={sc1} SC2={sc2} SC3={sc3} -> {verdict}')
  print(f'-> {args.output}/alloc_retro.json')


if __name__ == '__main__':
  main()
