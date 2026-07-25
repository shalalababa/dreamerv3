"""Orthogonal-objective generalization read (PREREG_orthogonal_obj_20260723.md).

Frozen BEFORE any ax1og* outcome exists (2026-07-23). The band-ledger
"one orthogonal generalization test" leg (editorial 17 Jul): the
confirmed rgo-vs-sgb contrast, frozen-readout, adapted under the
finger:spin objective evaluated on the SAME turn_hard env (orthreward
wrapper: obs space/dynamics unchanged, only the reward channel).

Grid (registered conditional, fit existence is not an outcome): {rgo,
sgb} x {s0, s1} x fit seeds 1-16 if the sgb 9-16 fits exist at
submission, else the registered fallback seeds 1-8. The read asserts
the realized grid is EXACTLY one of the two and records which.

  VALIDITY GATE (registered): pooled mean AUC100k over all orthogonal
  rows < FLOOR (20.0) => "FLOOR - uninformative" (the sparse spin
  objective was not reachable at this adaptation budget; the
  prediction is NOT adjudicated). Guards against reading a floor as an
  objective-specificity null.

  PRIMARY: seed-level d(seed) = mean over sides of [rgo - sgb] on the
  orthogonal objective; cluster bootstrap over fit seeds, B=10,000
  percentile CI, default_rng(0); FIRES iff CI > 0.

Registered interpretation map (both outcomes land the band leg):
  FIRES => the retained features are objective-general within the
  domain: legibility governs inclusion at fit time, but the included
  support serves other objectives (headline scope widens).
  Does not fire (above floor) => objective-specific support,
  consistent with the strict headline reading; realized power
  disclosed from the grid.

Descriptors (never decisional): same-seed original-task effects
(committed auc_pooled_1_16.csv) - d_z comparison and ratio, with the
registered scale caveat (turn and spin AUC scales differ); side
simples; permutation/d_z/loo.

Usage:
  python -m analysis.orthogonal_obj_read --auc <csv> --output <dir>
  python -m analysis.orthogonal_obj_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import os
import re

import numpy as np

ARMS = ("rgo", "sgb")
SIDES = (0, 1)
GRID_FULL = tuple(range(1, 17))
GRID_FALLBACK = tuple(range(1, 9))
MODE_RE = re.compile(r"^ax1og(rgo|sgb)q1s([01])$")
BASE_RE = re.compile(r"^ax1(rgo|sgb)q1s([01])$")
BASELINE = "artifacts/p3_amend1_20260717/auc_pooled_1_16.csv"
FLOOR = 20.0
B = 10_000


def _load(path, mode_re, seeds=None):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = mode_re.match(r["mode"])
      if not m:
        continue
      seed = int(r["seed"])
      if seeds is not None and seed not in seeds:
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      key = (m.group(1), int(m.group(2)), seed)
      assert key not in grid, f"duplicate row: {r['run_id']}"
      grid[key] = float(r["auc100k"])
  return grid


def load(auc_path, baseline_path=BASELINE):
  new = _load(auc_path, MODE_RE)
  found = sorted({k[2] for k in new})
  if found == list(GRID_FULL):
    seeds = GRID_FULL
  elif found == list(GRID_FALLBACK):
    seeds = GRID_FALLBACK
  else:
    raise AssertionError(
        f"realized seed set {found} is neither the registered full grid "
        f"{list(GRID_FULL)} nor the fallback {list(GRID_FALLBACK)}")
  for arm in ARMS:
    for side in SIDES:
      for s in seeds:
        assert (arm, side, s) in new, f"missing orth {arm} s{side} seed{s}"
  base = _load(baseline_path, BASE_RE, seeds)
  for arm in ARMS:
    for side in SIDES:
      for s in seeds:
        assert (arm, side, s) in base, f"missing baseline {arm} s{side} seed{s}"
  return new, base, seeds


def stats(diffs):
  d = np.asarray(diffs, np.float64)
  n = len(d)
  rng = np.random.default_rng(0)
  boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
  lo, hi = np.percentile(boots, [2.5, 97.5])
  obs = d.mean()
  if n <= 12:
    flips = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    p_perm = float((np.abs((flips * d).mean(1)) >= abs(obs) - 1e-12).mean())
  else:
    rng2 = np.random.default_rng(1)
    signs = rng2.choice((1.0, -1.0), size=(20_000, n))
    p_perm = float((np.abs((signs * d).mean(1)) >= abs(obs) - 1e-12).mean())
  sd = d.std(ddof=1)
  loo = [np.delete(d, i).mean() for i in range(n)]
  return dict(mean=float(obs), ci=[float(lo), float(hi)], n=n,
              perm_p=p_perm, d_z=float(obs / sd) if sd > 0 else None,
              loo_range=[float(min(loo)), float(max(loo))])


def analyze(new, base, seeds):
  pooled_level = float(np.mean([new[k] for k in new]))
  res = dict(grid=list(seeds), pooled_orth_level=pooled_level,
             floor=FLOOR)
  if pooled_level < FLOOR:
    res["fires"] = False
    res["floor_uninformative"] = True
    res["verdict"] = (
        f"FLOOR - uninformative: pooled orthogonal AUC100k "
        f"{pooled_level:.1f} < {FLOOR}; the sparse spin objective was "
        "not reachable at this adaptation budget; the objective-axis "
        "prediction is NOT adjudicated (registered validity gate).")
    return res
  res["floor_uninformative"] = False
  d_orth = [np.mean([new[("rgo", side, s)] - new[("sgb", side, s)]
                     for side in SIDES]) for s in seeds]
  d_orig = [np.mean([base[("rgo", side, s)] - base[("sgb", side, s)]
                     for side in SIDES]) for s in seeds]
  res["primary_orth_effect"] = stats(d_orth)
  res["orig_same_seed_effect"] = stats(d_orig)
  res["side_simple"] = {
      f"s{side}": stats([new[("rgo", side, s)] - new[("sgb", side, s)]
                         for s in seeds]) for side in SIDES}
  mo = float(np.mean(d_orig))
  res["ratio_orth_over_orig"] = (
      float(np.mean(d_orth) / mo) if abs(mo) >= 1.0 else None)
  ci = res["primary_orth_effect"]["ci"]
  res["fires"] = bool(ci[0] > 0)
  res["verdict"] = (
      "orthogonal-objective transfer CONFIRMED -> the retained features "
      "are objective-general within the domain (legibility governs "
      "inclusion at fit time; the included support serves other "
      "objectives - headline scope widens)" if res["fires"] else
      "orthogonal-objective transfer not detected (above floor) -> "
      "objective-specific support, consistent with the strict headline "
      "reading; power bounded by the realized grid")
  return res


def _fake(rng, orth_eff, level, seeds=GRID_FULL, noise=20.0):
  new, base = {}, {}
  for s in seeds:
    for side in SIDES:
      nb = level + rng.normal(0, noise)
      new[("sgb", side, s)] = nb
      new[("rgo", side, s)] = nb + orth_eff + rng.normal(0, noise)
      bb = 250 + rng.normal(0, noise)
      base[("sgb", side, s)] = bb
      base[("rgo", side, s)] = bb + 50 + rng.normal(0, noise)
  return new, base


def selfcheck():
  rng = np.random.default_rng(0)
  new, base = _fake(rng, 60.0, 200.0)
  r = analyze(new, base, GRID_FULL)
  assert r["fires"] and not r["floor_uninformative"], r
  new, base = _fake(np.random.default_rng(1), 0.0, 200.0)
  r = analyze(new, base, GRID_FULL)
  assert not r["fires"] and not r["floor_uninformative"], r
  new, base = _fake(np.random.default_rng(2), 5.0, 3.0, noise=1.0)
  r = analyze(new, base, GRID_FULL)
  assert r["floor_uninformative"] and not r["fires"], r
  new, base = _fake(np.random.default_rng(3), 60.0, 200.0, seeds=GRID_FALLBACK)
  r = analyze(new, base, GRID_FALLBACK)
  assert r["fires"] and r["grid"] == list(GRID_FALLBACK)
  assert MODE_RE.match("ax1rgoq1s0") is None
  assert MODE_RE.match("ax1ogrgoq1s1") is not None
  print("selfcheck PASS: planted effect fires (full + fallback grids), "
        "null does not, floor gate trips, original modes excluded")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--baseline", default=BASELINE)
  ap.add_argument("--output", default="analysis_out/orthogonal_obj")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  new, base, seeds = load(args.auc, args.baseline)
  res = analyze(new, base, seeds)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "orthogonal_obj.json"), "w") as f:
    json.dump(res, f, indent=1)
  print(f"grid seeds {res['grid'][0]}-{res['grid'][-1]}; pooled orth "
        f"level {res['pooled_orth_level']:.1f} (floor {FLOOR})")
  if not res["floor_uninformative"]:
    p = res["primary_orth_effect"]
    print(f"PRIMARY orth rgo-sgb {p['mean']:+.1f} "
          f"[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] (perm p={p['perm_p']:.4f}) "
          f"-> {'FIRES' if res['fires'] else 'does not fire'}")
    print(f"orig same-seed effect {res['orig_same_seed_effect']['mean']:+.1f}; "
          f"ratio {res['ratio_orth_over_orig']}")
  print(res["verdict"])


if __name__ == "__main__":
  main()
