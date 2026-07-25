"""Unfrozen-adaptation calibration read (PREREG_unfrozen_calib_20260723.md).

Frozen BEFORE any ax1ufz* outcome exists (2026-07-23). The band-ledger
"full/unfrozen adaptation calibration" leg (Research Plan v4 item 4;
triage 17 Jul): the confirmed rgo-vs-sgb contrast re-run with the WM
UNFROZEN during adaptation (config unfrozen_readout: same enc/dyn/dec
init, nothing frozen).

Grid: {rgo, sgb} x {s0, s1} x fit seeds 1-8, TASK finger_turn_hard,
adapt seed = fit seed, from the EXISTING 500K fits. Seed-level effect
d(seed) = mean over sides of [rgo - sgb] AUC100k.

  PRIMARY: mean_seed d(seed); cluster bootstrap over the 8 fit seeds,
  B=10,000 percentile CI, default_rng(0); FIRES iff CI > 0.
  Decision on the CI alone.

Registered descriptors: side simples; paired per-arm level change
(unfrozen - frozen, same cells; frozen baseline = the committed
artifacts/p3_amend1_20260717/auc_pooled_1_16.csv, seeds 1-8 - KNOWN at
freeze, disclosed); attenuation ratio vs the frozen same-seed effect;
sign-flip permutation, d_z, leave-one-seed-out range.

Usage:
  python -m analysis.unfrozen_calib_read --auc <csv> --output <dir>
  python -m analysis.unfrozen_calib_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import os
import re

import numpy as np

SEEDS = tuple(range(1, 9))
ARMS = ("rgo", "sgb")
SIDES = (0, 1)
MODE_RE = re.compile(r"^ax1ufz(rgo|sgb)q1s([01])$")
BASE_RE = re.compile(r"^ax1(rgo|sgb)q1s([01])$")
BASELINE = "artifacts/p3_amend1_20260717/auc_pooled_1_16.csv"
B = 10_000


def _load(path, mode_re, seeds):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = mode_re.match(r["mode"])
      if not m:
        continue
      seed = int(r["seed"])
      if seed not in seeds:
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      key = (m.group(1), int(m.group(2)), seed)
      assert key not in grid, f"duplicate row: {r['run_id']}"
      grid[key] = float(r["auc100k"])
  return grid


def load(auc_path, baseline_path=BASELINE):
  new = _load(auc_path, MODE_RE, SEEDS)
  base = _load(baseline_path, BASE_RE, SEEDS)
  for arm in ARMS:
    for side in SIDES:
      for s in SEEDS:
        assert (arm, side, s) in new, f"missing unfrozen {arm} s{side} seed{s}"
        assert (arm, side, s) in base, f"missing frozen baseline {arm} s{side} seed{s}"
  return new, base


def stats(diffs):
  d = np.asarray(diffs, np.float64)
  n = len(d)
  rng = np.random.default_rng(0)
  boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
  lo, hi = np.percentile(boots, [2.5, 97.5])
  obs = d.mean()
  flips = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
  p_perm = float((np.abs((flips * d).mean(1)) >= abs(obs) - 1e-12).mean())
  sd = d.std(ddof=1)
  loo = [np.delete(d, i).mean() for i in range(n)]
  return dict(mean=float(obs), ci=[float(lo), float(hi)], n=n,
              perm_p=p_perm, d_z=float(obs / sd) if sd > 0 else None,
              loo_range=[float(min(loo)), float(max(loo))])


def _seed_effect(grid):
  return [np.mean([grid[("rgo", side, s)] - grid[("sgb", side, s)]
                   for side in SIDES]) for s in SEEDS]


def analyze(new, base):
  d_new = _seed_effect(new)
  d_base = _seed_effect(base)
  res = dict(primary_unfrozen_effect=stats(d_new))
  res["side_simple"] = {
      f"s{side}": stats([new[("rgo", side, s)] - new[("sgb", side, s)]
                         for s in SEEDS]) for side in SIDES}
  res["level_change"] = {
      arm: stats([np.mean([new[(arm, side, s)] - base[(arm, side, s)]
                           for side in SIDES]) for s in SEEDS])
      for arm in ARMS}
  mb = float(np.mean(d_base))
  res["frozen_same_seed_effect"] = stats(d_base)
  res["attenuation_ratio"] = (
      float(np.mean(d_new) / mb) if abs(mb) >= 1.0 else None)
  ci = res["primary_unfrozen_effect"]["ci"]
  res["fires"] = bool(ci[0] > 0)
  res["verdict"] = (
      "unfrozen rgo-sgb CONFIRMED -> the transfer effect survives full "
      "adaptation; frozen-readout is not an artifact of freezing "
      "(band-ledger leg lands; attenuation ratio is the calibration "
      "number)" if res["fires"] else
      "unfrozen rgo-sgb not detected -> registered scope limitation: "
      "the confirmed effect is established for frozen-readout transfer; "
      "full-adaptation claims are not licensed at this n")
  return res


def _fake(rng, eff_new, eff_base, noise):
  new, base = {}, {}
  for s in SEEDS:
    for side in SIDES:
      b = 200 + rng.normal(0, noise)
      base[("sgb", side, s)] = b
      base[("rgo", side, s)] = b + eff_base + rng.normal(0, noise)
      nb = 250 + rng.normal(0, noise)
      new[("sgb", side, s)] = nb
      new[("rgo", side, s)] = nb + eff_new + rng.normal(0, noise)
  return new, base


def selfcheck():
  r = analyze(*_fake(np.random.default_rng(0), 60.0, 50.0, 15.0))
  assert r["fires"], r["primary_unfrozen_effect"]
  assert r["attenuation_ratio"] is not None
  r = analyze(*_fake(np.random.default_rng(1), 0.0, 50.0, 30.0))
  assert not r["fires"], r["primary_unfrozen_effect"]
  new, base = _fake(np.random.default_rng(2), 60.0, 50.0, 15.0)
  del new[("rgo", 1, 5)]
  tripped = False
  try:
    for arm in ARMS:
      for side in SIDES:
        for s in SEEDS:
          assert (arm, side, s) in new
  except AssertionError:
    tripped = True
  assert tripped, "missing-cell grid must trip"
  assert MODE_RE.match("ax1rgoq1s0") is None, "frozen modes must not parse"
  assert MODE_RE.match("ax1ufzrgoq1s0") is not None
  print("selfcheck PASS: planted effect fires, null does not, missing "
        "cell trips, frozen modes excluded")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--baseline", default=BASELINE)
  ap.add_argument("--output", default="analysis_out/unfrozen_calib")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  new, base = load(args.auc, args.baseline)
  res = analyze(new, base)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "unfrozen_calib.json"), "w") as f:
    json.dump(res, f, indent=1)
  p = res["primary_unfrozen_effect"]
  print(f"PRIMARY unfrozen rgo-sgb {p['mean']:+.1f} "
        f"[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] (perm p={p['perm_p']:.4f}) "
        f"-> {'FIRES' if res['fires'] else 'does not fire'}")
  print(f"frozen same-seed effect {res['frozen_same_seed_effect']['mean']:+.1f}; "
        f"attenuation ratio {res['attenuation_ratio']}")
  for arm in ARMS:
    lc = res["level_change"][arm]
    print(f"  {arm} level change (ufz-frozen) {lc['mean']:+.1f} "
          f"[{lc['ci'][0]:+.1f},{lc['ci'][1]:+.1f}]")
  print(res["verdict"])


if __name__ == "__main__":
  main()
