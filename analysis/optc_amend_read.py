"""Option C Amendment-1 read (PREREG_scaling_optc_amend1_20260718.md).

Frozen BEFORE any v400s0 outcome exists (2026-07-18). Existing
v200s1/v400s1 rows come from the archived Option C record
(artifacts/scaling_optc_20260718/auc.csv); any overlapping rows in the
new csv must bit-match it. Machinery: frozen family conventions
(AUC100k, B=10,000 cluster bootstrap default_rng(0), seed-paired,
decision CI alone, sensitivity robustness-only).

Registered quantities (task decisional, apt descriptive):
  C1  task [v400s0 - v400s1]   fixed-volume occupancy contrast
  C2  task [v400s0 - v200s1]   ~fraction-matched volume contrast (P-B4)

Usage:
  python -m analysis.optc_amend_read --auc <new csv> --output <dir>
  python -m analysis.optc_amend_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import math
import os

import numpy as np
from scipy import stats

B, RNG_SEED = 10_000, 0
SEEDS = list(range(1, 7))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_AUC = os.path.join(REPO, "artifacts/scaling_optc_20260718/auc.csv")
NEW_MODES = ("ax1v4q1v400s0", "ax1fv4q1v400s0")


def load_auc(path):
  out = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      if not r["mode"].startswith(("ax1v", "ax1fv")):
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      out[(r["mode"], int(r["seed"]))] = r["auc100k"]
  return out


def boot_ci(x, b=B, seed=RNG_SEED):
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(x), size=(b, len(x)))
  m = x[idx].mean(axis=1)
  return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sensitivity(x):
  n = len(x)
  m, sd = x.mean(), x.std(ddof=1)
  perms = np.array(list(itertools.product([1, -1], repeat=n)))
  p_perm = float((np.abs((perms * x).mean(1)) >= abs(m) - 1e-12).mean())
  loo = [float(np.delete(x, i).mean()) for i in range(n)]
  return dict(t=float(m / (sd / math.sqrt(n))), perm_p=p_perm,
              d_z=float(m / sd), loo_range=[min(loo), max(loo)],
              wilcoxon_p=float(stats.wilcoxon(x).pvalue))


def block(x):
  return dict(deltas=[float(v) for v in x], mean=float(x.mean()),
              ci=boot_ci(x), pos=int((x > 0).sum()), n=len(x),
              sensitivity=sensitivity(x))


def analyze(old, new):
  for key in set(old) & set(new):
    assert old[key] == new[key], f"bit-check FAIL {key}"
  merged = dict(new)
  merged.update(old)
  for m in NEW_MODES + ("ax1v4q1v400s1", "ax1v2q1v200s1",
                        "ax1fv4q1v400s1", "ax1fv2q1v200s1"):
    for k in SEEDS:
      assert (m, k) in merged, f"missing {m} seed {k}"
  v = lambda m: np.array([float(merged[(m, k)]) for k in SEEDS])
  res = {}
  res["c1_task_occ_at_v400"] = block(v("ax1v4q1v400s0") - v("ax1v4q1v400s1"))
  res["c2_task_volume_matched_frac"] = block(
      v("ax1v4q1v400s0") - v("ax1v2q1v200s1"))
  res["c1_apt_desc"] = block(v("ax1fv4q1v400s0") - v("ax1fv4q1v400s1"))
  res["c2_apt_desc"] = block(v("ax1fv4q1v400s0") - v("ax1fv2q1v200s1"))
  res["cell_means"] = {m: dict(mean=float(v(m).mean()),
                               sd=float(v(m).std(ddof=1)))
                       for m in ("ax1v4q1v400s0", "ax1v4q1v400s1",
                                 "ax1v2q1v200s1", "ax1fv4q1v400s0")}
  c1, c2 = res["c1_task_occ_at_v400"], res["c2_task_volume_matched_frac"]
  res["c1_verdict"] = ("occupancy still binds at 400 eps" if c1["ci"][0] > 0
                       else "anomaly persists: lo>hi -> composition-driven; audit"
                       if c1["ci"][1] < 0 else
                       "occupancy saturates below .094 at this volume")
  res["c2_verdict"] = ("volume matters at ~fixed fraction -> P-B4 strong "
                       "form REFUTED" if c2["ci"][0] > 0 else
                       "unexpected negative; audit" if c2["ci"][1] < 0 else
                       "no volume effect at matched fraction -> P-B4 stands; "
                       "18-Jul anomaly = composition/occupancy mixture")
  return res


def selfcheck():
  rng = np.random.default_rng(3)

  def synth(v400s0):
    d = {}
    for m, base in (("ax1v2q1v200s1", 200.0), ("ax1v4q1v400s1", 430.0),
                    ("ax1fv2q1v200s1", 65.0), ("ax1fv4q1v400s1", 85.0),
                    ("ax1fv4q1v400s0", 80.0), ("ax1v4q1v400s0", v400s0)):
      for k in SEEDS:
        d[(m, k)] = f"{base + rng.normal(0, 15):.4f}"
    return d

  a = synth(600.0)  # s0 wins big: C1>0 and C2>0 -> P-B4 refuted branch
  old = {k: v for k, v in a.items() if k[0] != "ax1v4q1v400s0"}
  r = analyze(old, a)
  assert r["c1_task_occ_at_v400"]["ci"][0] > 0
  assert "REFUTED" in r["c2_verdict"], r["c2_verdict"]
  b = synth(205.0)  # s0 ~ v200s1: C1<0 (anomaly persists), C2 ~ 0
  r = analyze({k: v for k, v in b.items() if k[0] != "ax1v4q1v400s0"}, b)
  assert r["c1_task_occ_at_v400"]["ci"][1] < 0
  assert "P-B4 stands" in r["c2_verdict"], r["c2_verdict"]
  bad = dict(a)
  bad[("ax1v2q1v200s1", 1)] = "1.0"
  try:
    analyze(old, bad)
    raise SystemExit("selfcheck FAIL: bit-check did not trip")
  except AssertionError:
    pass
  print("selfcheck PASS: refuted branch, stands branch, bit-check trip")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--output", default="analysis_out/optc_amend1")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  res = analyze(load_auc(FROZEN_AUC), load_auc(args.auc))
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "optc_amend_read.json"), "w") as f:
    json.dump(res, f, indent=1)
  for tag in ("c1_task_occ_at_v400", "c2_task_volume_matched_frac"):
    r = res[tag]
    print(f"{tag}: {r['mean']:+.1f} CI=[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}] "
          f"pos={r['pos']}/{r['n']}")
  print("C1:", res["c1_verdict"])
  print("C2:", res["c2_verdict"])


if __name__ == "__main__":
  main()
