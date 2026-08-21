"""Substrate screen — per-state decision-value ceiling of the W1 bundle.

NON-REGISTERED ARCHIVED COMPUTE (user decision 21 Aug: ship). The
estimand was first computed pre-registration during the 21 Aug
ThreeAxes ideation; this script is the repo-tracked reproduction so the
ceiling figures cited as context in the committed preregs are
auditable. It carries NO registered decision rule and adjudicates
nothing.

REPRODUCTION STATUS (report at every citation):
  - the naive single-draw PEDESTAL (within-repeat max - mean, averaged
    over repeats) reproduces the ideation-doc values BIT-EXACTLY:
    1.5399 (dv3) / 0.4591 (tm2);
  - the ideation doc's CEILING figures +0.2026 (dv3) / +0.1184 (tm2)
    do NOT reproduce under any natural definition tried (per-state
    mean decision-sd, sqrt of pooled clipped S2, per-cell wdc-style,
    ddof 0/1 variants) — same status as the already-withdrawn "12-29x"
    ratio. The reproducible ceilings are LOWER (see below), i.e. the
    substrate is at least as flat as claimed;
  - consequently ALL inflation ratios (incl. the interim 7.6x/3.9x,
    which used the unreproduced denominators) are DROPPED: quote the
    pedestal and the ceiling-by-definition table, never a ratio.

Estimator (ThreeAxes §2.2 corrected form, ddof=1 — validated on
planted truth in the selfcheck):

    S2(s) = Var_m[mean_r g] - (1/R) * mean_m Var_r[g]

Two pinned ceiling definitions are reported:
  ceiling_state = mean over states of sqrt(clip(S2, 0))   [G units]
  ceiling_cell  = mean over cells of sqrt(clip(mean_s S2, 0))
                  (the frozen wdc_read._ceiling form, so the WDC wave's
                  numbers will be directly comparable)

Duplicate arms (planted truth-zero): the between-candidate variance is
EXACTLY zero bit-wise (identical columns), so both ceilings are exactly
zero — the screen's built-in validation.

Usage:
  python -m analysis.substrate_screen --labels <dir> --output <dir>
  python -m analysis.substrate_screen --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

FILE_RES = dict(
    dv3=re.compile(r'^w1_(cup|finger)_(e1|e4)_seed(3[1-8])_late\.npz$'),
    tm2=re.compile(r'^w1tm2_(cup|finger)_(e1|e4)_seed(5[1-8])_late\.npz$'))
DUP_RES = dict(
    dv3=re.compile(r'^w1dup_(cup|finger)_(e1|e4)_seed(3[1-8])_late\.npz$'),
    tm2=re.compile(r'^w1tm2dup_(cup|finger)_(e1|e4)_seed(5[1-8])_'
                   r'late\.npz$'))
UNREPRODUCED = dict(
    note=('ideation-doc ceilings NOT reproduced under any tried '
          'definition; reproducible values are lower'),
    doc_ceiling=dict(dv3=0.2026, tm2=0.1184),
    doc_ratio='12-29x WITHDRAWN; interim 7.6x/3.9x also dropped '
              '(unreproduced denominators)')


def per_state_s2(g):
  """g (S, R, M) -> ddof=1 variance-components S2 per state."""
  between = g.mean(1).var(1, ddof=1)
  within = g.var(1, ddof=1).mean(1)
  return between - within / g.shape[1]


def per_state_pedestal(g):
  """Naive single-draw opportunity: within-repeat max - mean, averaged
  over repeats (reproduces the ideation doc bit-exactly)."""
  return (g.max(2) - g.mean(2)).mean(1)


def screen_family(labels_dir, family):
  s2_all, ped_all, zero_all, cell_ceils = [], [], [], []
  dup_between_max, n_main, n_dup = 0.0, 0, 0
  dup_ceils = []
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    is_main = bool(FILE_RES[family].match(name))
    is_dup = bool(DUP_RES[family].match(name))
    if not (is_main or is_dup):
      continue
    z = np.load(path, allow_pickle=True)
    g = np.asarray(z['g_all_rep'], float)
    s2 = per_state_s2(g)
    if is_dup:
      n_dup += 1
      dup_between_max = max(dup_between_max, float(
          np.max(np.abs(g.mean(1).var(1, ddof=1)))))
      dup_ceils.append(float(np.sqrt(max(float(s2.mean()), 0.0))))
      continue
    n_main += 1
    s2_all.append(s2)
    ped_all.append(per_state_pedestal(g))
    zero_all.append(np.all(g == 0.0, axis=(1, 2)))
    cell_ceils.append(float(np.sqrt(max(float(s2.mean()), 0.0))))
  s2 = np.concatenate(s2_all)
  sd = np.sqrt(np.clip(s2, 0.0, None))
  return dict(
      n_files=n_main, n_states=int(len(s2)),
      ceiling_state=float(sd.mean()),
      ceiling_state_p99=float(np.percentile(sd, 99)),
      ceiling_cell=float(np.mean(cell_ceils)),
      cell_ceilings=[round(c, 4) for c in cell_ceils],
      frac_s2_nonpositive=float(np.mean(s2 <= 0)),
      frac_all_g_zero=float(np.mean(np.concatenate(zero_all))),
      pedestal_single_draw=float(np.concatenate(ped_all).mean()),
      dup_gates=dict(
          n_files=n_dup,
          between_var_max_abs=dup_between_max,
          between_exactly_zero=bool(dup_between_max == 0.0),
          ceilings=dup_ceils))


def run(labels_dir, output):
  os.makedirs(output, exist_ok=True)
  out = {fam: screen_family(labels_dir, fam) for fam in ('dv3', 'tm2')}
  out['status'] = ('NON-REGISTERED ARCHIVED COMPUTE; adjudicates '
                   'nothing; estimand first computed pre-registration '
                   '(21 Aug ideation)')
  out['reproduction'] = UNREPRODUCED
  path = os.path.join(output, 'substrate_screen.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


def selfcheck():
  rng = np.random.default_rng(0)
  # estimator unbiased at planted truth (S2 in variance units)
  for truth in (0.0, 0.25, 4.0):
    vals = []
    for _ in range(30):
      mu = (rng.normal(0, np.sqrt(truth), (400, 8)) if truth
            else np.zeros((400, 8)))
      g = mu[:, None, :] + rng.normal(0, 1.0, (400, 8, 8))
      vals.append(float(per_state_s2(g).mean()))
    got = float(np.mean(vals))
    assert abs(got - truth) < 0.05 + 0.05 * truth, (truth, got)
  # duplicate arms: identical columns => between-variance EXACTLY zero,
  # clipped ceiling exactly zero (raw S2 is negative: -within/R)
  g = np.repeat(rng.normal(0, 1.0, (50, 8, 1)), 8, axis=2)
  assert float(np.max(np.abs(g.mean(1).var(1, ddof=1)))) == 0.0
  s2d = per_state_s2(g)
  assert np.all(s2d <= 0.0)
  assert float(np.sqrt(max(float(s2d.mean()), 0.0))) == 0.0
  # pedestal positive on pure noise (the bias the screen exposes)
  gn = rng.normal(0, 1.0, (400, 8, 8))
  assert float(per_state_pedestal(gn).mean()) > 1.0
  print('substrate_screen selfcheck PASS (ddof=1 unbiased at planted '
        '0/.25/4; duplicate between-variance bitwise zero + clipped '
        'ceiling zero; single-draw pedestal positive on noise)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.labels and args.output):
    p.error('--labels and --output required (or --selfcheck)')
  run(args.labels, args.output)


if __name__ == '__main__':
  main()
