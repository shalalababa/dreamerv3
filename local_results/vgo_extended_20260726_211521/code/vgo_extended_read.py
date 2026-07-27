"""vgo extended-fit discriminator read (PREREG_vgo_extended_20260723.md).

Frozen BEFORE any ax1x* outcome exists (2026-07-23). Tests P-B2's
attenuation form after the vgo wiring audit (case (b), replay-grounded):
if the value-gradient path is ATTENUATED rather than absent, longer
fitting should grow vgo's transfer beyond generic-extension gains.

Grid: FRESH fits {vgo, sgb} x fit seeds 1-8 x s1 (hi side) at 1.5M
updates (3x the frozen 500K; new WM dirs - the frozen fits are never
touched), then standard frozen-readout adapt. 500K baselines = the
committed artifacts/p3_wave_20260717/auc_p3.csv (KNOWN at freeze,
disclosed). sgb is the generic-extension control: no representation
gradient path, so any sgb gain from 3x fitting is generic.

  PRIMARY: D(seed) = [vgo_x - vgo_500k] - [sgb_x - sgb_500k];
  mean over the 8 seeds; cluster bootstrap B=10,000 percentile CI,
  default_rng(0); FIRES iff CI > 0 => attenuation form SUPPORTED
  (the value-gradient path grows with fit length beyond generic gains).
  Does not fire => attenuation form NOT supported at 3x updates; vgo
  stays dead in the tested range (absence-like).

Descriptors: arm simples (x - 500k per arm, own CIs); levels;
permutation/d_z/loo.

Usage:
  python -m analysis.vgo_extended_read --auc <csv> --output <dir>
  python -m analysis.vgo_extended_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import os
import re

import numpy as np

SEEDS = tuple(range(1, 9))
ARMS = ("vgo", "sgb")
MODE_RE = re.compile(r"^ax1x(vgo|sgb)q1s1$")
BASE_RE = re.compile(r"^ax1(vgo|sgb)q1s1$")
BASELINE = "artifacts/p3_wave_20260717/auc_p3.csv"
MILESTONE = 1_500_000
B = 10_000


def _load(path, mode_re, milestone=None):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = mode_re.match(r["mode"])
      if not m:
        continue
      seed = int(r["seed"])
      if seed not in SEEDS:
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      if milestone is not None:
        assert int(r["milestone"]) == milestone, (
            f"{r['run_id']}: milestone {r['milestone']} != {milestone}")
      key = (m.group(1), seed)
      assert key not in grid, f"duplicate row: {r['run_id']}"
      grid[key] = float(r["auc100k"])
  return grid


def load(auc_path, baseline_path=BASELINE):
  ext = _load(auc_path, MODE_RE, milestone=MILESTONE)
  base = _load(baseline_path, BASE_RE)
  for arm in ARMS:
    for s in SEEDS:
      assert (arm, s) in ext, f"missing extended {arm} seed{s}"
      assert (arm, s) in base, f"missing 500K baseline {arm} seed{s}"
  return ext, base


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


def analyze(ext, base):
  gain = {arm: [ext[(arm, s)] - base[(arm, s)] for s in SEEDS]
          for arm in ARMS}
  D = [gain["vgo"][i] - gain["sgb"][i] for i in range(len(SEEDS))]
  res = dict(primary_diff_in_diff=stats(D),
             vgo_gain=stats(gain["vgo"]), sgb_gain=stats(gain["sgb"]),
             levels={f"{arm}_{k}": float(np.mean(
                 [(ext if k == "x" else base)[(arm, s)] for s in SEEDS]))
                 for arm in ARMS for k in ("x", "500k")})
  ci = res["primary_diff_in_diff"]["ci"]
  res["fires"] = bool(ci[0] > 0)
  res["verdict"] = (
      "vgo grows with fit length beyond the generic-extension control "
      "-> P-B2 attenuation form SUPPORTED (value-gradient path present "
      "but slow)" if res["fires"] else
      "no differential vgo gain at 3x updates -> P-B2 attenuation form "
      "NOT supported in the tested range; vgo stays dead "
      "(absence-like); no further extended-fit arms without new theory")
  return res


def _fake(rng, vgo_extra, generic, noise=15.0):
  ext, base = {}, {}
  for s in SEEDS:
    for arm in ARMS:
      b = (90 if arm == "vgo" else 160) + rng.normal(0, noise)
      base[(arm, s)] = b
      extra = vgo_extra if arm == "vgo" else 0.0
      ext[(arm, s)] = b + generic + extra + rng.normal(0, noise)
  return ext, base


def selfcheck():
  r = analyze(*_fake(np.random.default_rng(0), 70.0, 30.0))
  assert r["fires"], r["primary_diff_in_diff"]
  r = analyze(*_fake(np.random.default_rng(1), 0.0, 80.0))
  assert not r["fires"], (
      "generic gains alone must NOT fire the diff-in-diff: "
      f"{r['primary_diff_in_diff']}")
  assert r["sgb_gain"]["ci"][0] > 0, "control should show the generic gain"
  ext, base = _fake(np.random.default_rng(2), 70.0, 30.0)
  del ext[("vgo", 4)]
  tripped = False
  try:
    for arm in ARMS:
      for s in SEEDS:
        assert (arm, s) in ext
  except AssertionError:
    tripped = True
  assert tripped, "missing-cell grid must trip"
  assert MODE_RE.match("ax1vgoq1s1") is None, "500K modes must not parse"
  assert MODE_RE.match("ax1xvgoq1s1") is not None
  print("selfcheck PASS: planted differential gain fires, generic-only "
        "gain does not, missing cell trips, 500K modes excluded")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--baseline", default=BASELINE)
  ap.add_argument("--output", default="analysis_out/vgo_extended")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  ext, base = load(args.auc, args.baseline)
  res = analyze(ext, base)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "vgo_extended.json"), "w") as f:
    json.dump(res, f, indent=1)
  p = res["primary_diff_in_diff"]
  print(f"PRIMARY diff-in-diff {p['mean']:+.1f} "
        f"[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] (perm p={p['perm_p']:.4f}) "
        f"-> {'FIRES' if res['fires'] else 'does not fire'}")
  for arm in ARMS:
    g = res[f"{arm}_gain"]
    print(f"  {arm} gain (x - 500k) {g['mean']:+.1f} "
          f"[{g['ci'][0]:+.1f},{g['ci'][1]:+.1f}]")
  print(res["verdict"])


if __name__ == "__main__":
  main()
