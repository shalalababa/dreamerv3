"""Synth re-diagnosis read (PREREG_synth_rediag_20260718.md).

Frozen BEFORE any ax1rd* outcome exists (2026-07-18). One-way
random-effects variance decomposition of AUC100k over the fits(4) x
adapt-seeds(3) grid, per side, task arm:

  MS_w (within-fit / adapt-seed), MS_b (between-fit);
  sigma2_w = MS_w; sigma2_b = max(0, (MS_b - MS_w)/3);
  WITHIN SHARE = sigma2_w / (sigma2_w + sigma2_b), pooled = mean of sides.

Directional prediction (adapt-lottery): pooled within share >= 0.5
=> Phase-B'' (adapt-averaged re-test) authorized for registration.

Usage:
  python -m analysis.synth_rediag_read --auc <csv> --output <dir>
  python -m analysis.synth_rediag_read --selfcheck
"""

import argparse
import csv
import json
import os
import re

import numpy as np

FITS = (1, 2, 3, 4)
ADAPT_SEEDS = (101, 102, 103)
MODE_RE = re.compile(r"ax1rd(\d)q1s(\d)$")


def load(path):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = MODE_RE.match(r["mode"])
      if not m:
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      fit, side, aseed = int(m.group(1)), int(m.group(2)), int(r["seed"])
      grid[(side, fit, aseed)] = float(r["auc100k"])
  return grid


def decompose(grid, side):
  data = np.array([[grid[(side, f, a)] for a in ADAPT_SEEDS] for f in FITS])
  k = data.shape[1]
  fit_means = data.mean(1)
  grand = data.mean()
  ms_b = k * ((fit_means - grand) ** 2).sum() / (len(FITS) - 1)
  ms_w = ((data - fit_means[:, None]) ** 2).sum() / (len(FITS) * (k - 1))
  s2_w = ms_w
  s2_b = max(0.0, (ms_b - ms_w) / k)
  share = s2_w / (s2_w + s2_b) if (s2_w + s2_b) > 0 else float("nan")
  return dict(cells=[[float(v) for v in row] for row in data],
              fit_means=[float(v) for v in fit_means],
              ms_between=float(ms_b), ms_within=float(ms_w),
              sigma2_between=float(s2_b), sigma2_within=float(s2_w),
              within_share=float(share))


def analyze(grid):
  for side in (0, 1):
    for f in FITS:
      for a in ADAPT_SEEDS:
        assert (side, f, a) in grid, f"missing s{side} fit{f} adapt{a}"
  res = {f"side{s}": decompose(grid, s) for s in (0, 1)}
  pooled = float(np.mean([res[f"side{s}"]["within_share"] for s in (0, 1)]))
  res["pooled_within_share"] = pooled
  res["lottery_confirmed"] = bool(pooled >= 0.5)
  res["gate"] = ("Phase-B'' (adapt-averaged re-test) AUTHORIZED for "
                 "registration" if res["lottery_confirmed"] else
                 "gate CLOSED: fit-level heterogeneity dominates - "
                 "trunk-quality instrument next (own registration)")
  return res


def selfcheck():
  rng = np.random.default_rng(0)
  # Lottery branch: tiny fit effects, huge adapt noise.
  g = {(s, f, a): 100 + 2 * f + rng.normal(0, 80)
       for s in (0, 1) for f in FITS for a in ADAPT_SEEDS}
  r = analyze(g)
  assert r["lottery_confirmed"] and r["pooled_within_share"] > 0.5, \
      r["pooled_within_share"]
  # Fit-dominated branch: big fit effects, tiny adapt noise.
  g = {(s, f, a): 100 + 80 * f + rng.normal(0, 5)
       for s in (0, 1) for f in FITS for a in ADAPT_SEEDS}
  r = analyze(g)
  assert not r["lottery_confirmed"] and r["pooled_within_share"] < 0.2
  print("selfcheck PASS: lottery branch and fit-dominated branch recovered")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--output", default="analysis_out/synth_rediag")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  res = analyze(load(args.auc))
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "synth_rediag.json"), "w") as f:
    json.dump(res, f, indent=1)
  for s in (0, 1):
    d = res[f"side{s}"]
    print(f"side{s}: within-share {d['within_share']:.3f} "
          f"(s2_w {d['sigma2_within']:.1f}, s2_b {d['sigma2_between']:.1f})")
  print(f"pooled within-share {res['pooled_within_share']:.3f} -> "
        f"{res['gate']}")


if __name__ == "__main__":
  main()
