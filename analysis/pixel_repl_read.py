"""Pixel X1 read (PREREG_pixel_repl_20260719.md).

Frozen BEFORE any ax1(f)px*pxq1m adapt outcome exists (2026-07-19; the
only pixel adapt outcomes to date are the excluded seed-99 X0 smoke runs,
modes ax1pxpxq1imgs*, which this reader cannot match).

PRIMARY (sole inferential endpoint, AUC100k only): the occupancy x
reward-supervision interaction on the collector-matched pixel pair,

  I = mean_k [ (task: s1-s0)_k - (apt: s1-s0)_k ],   k = seeds 1..8,

cluster bootstrap over seeds B=10,000 percentile CI with
np.random.default_rng(0); FIRES iff the CI excludes 0 from above.
Decision on the CI alone.

P-C2 (registered directional secondary, never a gate): if the primary
fires, the pixel point estimate is compared with the proprio anchor band
[+84.0 (E3v2 within-collector), +98.7 (W0 pooled)]; "meets" iff
I >= 84.0. Same task, same reward scale, same AUC100k window, so the
comparison is unit-consistent; it is reported directionally, never
pooled.

Arm simples (task and apt s1-s0 with their own CIs) and the sensitivity
suite (exact sign-flip permutation, d_z, leave-one-seed-out range) are
robustness-only descriptives.

Usage:
  python -m analysis.pixel_repl_read --auc <csv> --output <dir> \
      [--audit_runroot <runroot>]
  python -m analysis.pixel_repl_read --selfcheck
"""

import argparse
import csv
import glob
import itertools
import json
import os
import re

import numpy as np

SEEDS = tuple(range(1, 9))
MODE_RE = re.compile(r"^ax1(f?)pxpxq1ms([01])$")
DOMAIN = "finger"
B = 10_000
PROPRIO_BAND = (84.0, 98.7)


def load(path):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = MODE_RE.match(r["mode"])
      if not m or r["domain"] != DOMAIN:
        continue
      seed = int(r["seed"])
      if seed not in SEEDS:
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      arm = "apt" if m.group(1) == "f" else "task"
      key = (arm, int(m.group(2)), seed)
      assert key not in grid, f"duplicate row: {r['run_id']}"
      grid[key] = float(r["auc100k"])
  return grid


def stats(diffs):
  d = np.asarray(diffs, np.float64)
  n = len(d)
  rng = np.random.default_rng(0)
  boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
  lo, hi = np.percentile(boots, [2.5, 97.5])
  obs = d.mean()
  flips = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
  perm = (flips * d).mean(1)
  p_perm = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
  sd = d.std(ddof=1)
  loo = [np.delete(d, i).mean() for i in range(n)]
  return dict(mean=float(obs), ci=[float(lo), float(hi)], n=n,
              perm_p=p_perm, d_z=float(obs / sd) if sd > 0 else None,
              loo_range=[float(min(loo)), float(max(loo))])


def analyze(grid):
  for arm in ("task", "apt"):
    for side in (0, 1):
      for k in SEEDS:
        assert (arm, side, k) in grid, f"missing {arm} s{side} seed{k}"
  b = {arm: np.array([grid[(arm, 1, k)] - grid[(arm, 0, k)] for k in SEEDS])
       for arm in ("task", "apt")}
  inter = b["task"] - b["apt"]
  res = dict(primary_interaction=stats(inter),
             task_simple=stats(b["task"]), apt_simple=stats(b["apt"]))
  ci = res["primary_interaction"]["ci"]
  res["fires"] = bool(ci[0] > 0)
  point = res["primary_interaction"]["mean"]
  res["p_c2"] = dict(
      proprio_band=list(PROPRIO_BAND),
      status=("not-evaluated (primary did not fire)" if not res["fires"]
              else "meets" if point >= PROPRIO_BAND[0] else "below"))
  res["verdict"] = (
      "interaction REPLICATES in pixels"
      + (f"; P-C2 {res['p_c2']['status']}" if res["fires"] else "")
      if res["fires"] else
      "interaction does NOT fire in pixels -> G-X3: amplification NO-GO; "
      "run the registered reconstruction-swamping diagnostic instead")
  return res


def audit_runroot(runroot):
  """Non-inferential config audit: every X1 wm run must train image-only
  (model_obs: image) with the arm's registered expl mode."""
  report, bad = {}, []
  for arm, infix, want_mode in (("task", "px", "task"), ("apt", "fpx", "apt")):
    for side in (0, 1):
      for k in SEEDS:
        run = f"ax1wm_{DOMAIN}_{infix}pxq1ms{side}_seed{k}"
        paths = glob.glob(os.path.join(runroot, run, "config.yaml"))
        if not paths:
          bad.append(f"{run}: no config.yaml")
          continue
        text = open(paths[0]).read()
        ok_obs = re.search(r"model_obs:\s*image\b", text) is not None
        ok_mode = re.search(r"mode:\s*" + want_mode + r"\b", text) is not None
        report[run] = dict(model_obs_image=ok_obs, expl_mode_ok=ok_mode)
        if not (ok_obs and ok_mode):
          bad.append(f"{run}: model_obs_image={ok_obs} expl_mode={ok_mode}")
  return dict(runs=report, violations=bad, clean=not bad)


def _fake_grid(rng, task_eff, apt_eff, noise):
  g = {}
  for arm, eff in (("task", task_eff), ("apt", apt_eff)):
    for k in SEEDS:
      base = 200 + rng.normal(0, noise)
      g[(arm, 0, k)] = base
      g[(arm, 1, k)] = base + eff + rng.normal(0, noise)
  return g


def selfcheck():
  rng = np.random.default_rng(0)
  r = analyze(_fake_grid(rng, task_eff=150.0, apt_eff=10.0, noise=15.0))
  assert r["fires"] and r["p_c2"]["status"] == "meets", r
  rng = np.random.default_rng(1)
  r = analyze(_fake_grid(rng, task_eff=20.0, apt_eff=15.0, noise=60.0))
  assert not r["fires"] and "not-evaluated" in r["p_c2"]["status"], r
  try:
    g = _fake_grid(np.random.default_rng(2), 150.0, 10.0, 15.0)
    del g[("apt", 1, 5)]
    analyze(g)
    raise SystemExit("missing-cell assert did not trip")
  except AssertionError:
    pass
  smoke = MODE_RE.match("ax1pxpxq1imgs0")
  assert smoke is None, "X0 smoke mode must not parse"
  print("selfcheck PASS: fires/meets, null, missing-cell trip, "
        "smoke-mode exclusion")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--output", default="analysis_out/pixel_repl")
  ap.add_argument("--audit_runroot", default=None)
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  res = analyze(load(args.auc))
  if args.audit_runroot:
    res["config_audit"] = audit_runroot(args.audit_runroot)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "pixel_repl.json"), "w") as f:
    json.dump(res, f, indent=1)
  p = res["primary_interaction"]
  print(f"PRIMARY interaction {p['mean']:+.1f} "
        f"[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] "
        f"(perm p={p['perm_p']:.4f}) -> "
        f"{'FIRES' if res['fires'] else 'does not fire'}")
  for arm in ("task", "apt"):
    s = res[f"{arm}_simple"]
    print(f"  {arm} s1-s0 {s['mean']:+.1f} "
          f"[{s['ci'][0]:+.1f},{s['ci'][1]:+.1f}]")
  print(f"P-C2: {res['p_c2']['status']}  |  {res['verdict']}")
  if args.audit_runroot:
    a = res["config_audit"]
    print(f"config audit: {'CLEAN' if a['clean'] else a['violations']}")


if __name__ == "__main__":
  main()
