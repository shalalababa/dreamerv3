"""Synth Phase-B'' read (PREREG_synth_phasebpp_20260720.md).

Frozen BEFORE any ax1rb* outcome exists (2026-07-20). Adapt-averaged
re-test of the Phase-B interaction, authorized by the re-diagnosis gate
(artifacts/synth_rediag_20260719/: pooled within-share 0.865).

Grid: {task, apt} x {s0, s1} x fit seeds 1-8 x adapt seeds {104,105,106}
= 96 adapt-only rows. Fit-level AUC100k = MEAN over the 3 adapt seeds;
B_arm(fit) = fit-level(s1) - fit-level(s0);

  PRIMARY: interaction I = mean_fit [B_task(fit) - B_apt(fit)],
  cluster bootstrap over fit seeds B=10,000 percentile CI,
  np.random.default_rng(0); FIRES iff CI > 0. Decision on the CI alone.

Secondary (robustness/descriptive): arm simples with own CIs; exact
sign-flip permutation, d_z, leave-one-fit-out range; per-cell
within-fit adapt-seed SD (variance re-check against the re-diagnosis
decomposition).

Usage:
  python -m analysis.synth_phasebpp_read --auc <csv> --output <dir>
  python -m analysis.synth_phasebpp_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import os
import re

import numpy as np

FITS = tuple(range(1, 9))
ADAPT_SEEDS = (104, 105, 106)
MODE_RE = re.compile(r"^ax1rb(f?)([1-8])q1s([01])$")
B = 10_000


def load(path):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = MODE_RE.match(r["mode"])
      if not m:
        continue
      aseed = int(r["seed"])
      if aseed not in ADAPT_SEEDS:
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      arm = "apt" if m.group(1) == "f" else "task"
      key = (arm, int(m.group(3)), int(m.group(2)), aseed)
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
      for f in FITS:
        for a in ADAPT_SEEDS:
          assert (arm, side, f, a) in grid, \
              f"missing {arm} s{side} fit{f} adapt{a}"
  fit_level = {}
  within_sd = {}
  for arm in ("task", "apt"):
    for side in (0, 1):
      vals = np.array([[grid[(arm, side, f, a)] for a in ADAPT_SEEDS]
                       for f in FITS])
      for i, f in enumerate(FITS):
        fit_level[(arm, side, f)] = float(vals[i].mean())
      within_sd[f"{arm}_s{side}"] = float(vals.std(axis=1, ddof=1).mean())
  b = {arm: np.array([fit_level[(arm, 1, f)] - fit_level[(arm, 0, f)]
                      for f in FITS]) for arm in ("task", "apt")}
  inter = b["task"] - b["apt"]
  res = dict(primary_interaction=stats(inter),
             task_simple=stats(b["task"]), apt_simple=stats(b["apt"]),
             cell_fit_level_means={
                 f"{arm}_s{side}": float(np.mean(
                     [fit_level[(arm, side, f)] for f in FITS]))
                 for arm in ("task", "apt") for side in (0, 1)},
             mean_within_fit_adapt_sd=within_sd)
  ci = res["primary_interaction"]["ci"]
  res["fires"] = bool(ci[0] > 0)
  res["verdict"] = (
      "interaction DETECTED with adapt averaging -> the Phase-B null was "
      "lottery-masked; synth rejoins as a positive interaction domain"
      if res["fires"] else
      "interaction not detected at B'' power -> synth interaction leg "
      "CLOSED (registered consequence: no Phase-C; synth remains a "
      "shuffle-collapse-only exhibit)")
  return res


def _fake_grid(rng, task_eff, apt_eff, base_noise, adapt_noise):
  g = {}
  for arm, eff in (("task", task_eff), ("apt", apt_eff)):
    for f in FITS:
      base = {0: 100 + rng.normal(0, base_noise)}
      base[1] = base[0] + eff + rng.normal(0, base_noise)
      for side in (0, 1):
        for a in ADAPT_SEEDS:
          g[(arm, side, f, a)] = base[side] + rng.normal(0, adapt_noise)
  return g


def selfcheck():
  rng = np.random.default_rng(0)
  r = analyze(_fake_grid(rng, 120.0, 5.0, base_noise=10.0, adapt_noise=60.0))
  assert r["fires"], r["primary_interaction"]
  rng = np.random.default_rng(1)
  r = analyze(_fake_grid(rng, 10.0, 8.0, base_noise=10.0, adapt_noise=120.0))
  assert not r["fires"], r["primary_interaction"]
  try:
    g = _fake_grid(np.random.default_rng(2), 120.0, 5.0, 10.0, 60.0)
    del g[("apt", 1, 5, 105)]
    analyze(g)
    raise SystemExit("missing-cell assert did not trip")
  except AssertionError:
    pass
  assert MODE_RE.match("ax1rd1q1s0") is None, "rediag modes must not parse"
  print("selfcheck PASS: fires, lottery-null, missing-cell trip, "
        "rediag-mode exclusion")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--output", default="analysis_out/synth_phasebpp")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  res = analyze(load(args.auc))
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "synth_phasebpp.json"), "w") as f:
    json.dump(res, f, indent=1)
  p = res["primary_interaction"]
  print(f"PRIMARY interaction {p['mean']:+.1f} "
        f"[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] (perm p={p['perm_p']:.4f}) "
        f"-> {'FIRES' if res['fires'] else 'does not fire'}")
  for arm in ("task", "apt"):
    s = res[f"{arm}_simple"]
    print(f"  {arm} s1-s0 {s['mean']:+.1f} "
          f"[{s['ci'][0]:+.1f},{s['ci'][1]:+.1f}]")
  print(f"within-fit adapt SD by cell: "
        f"{ {k: round(v, 1) for k, v in res['mean_within_fit_adapt_sd'].items()} }")
  print(res["verdict"])


if __name__ == "__main__":
  main()
