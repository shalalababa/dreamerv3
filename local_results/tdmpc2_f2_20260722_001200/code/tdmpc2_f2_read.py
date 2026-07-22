"""TD-MPC2 F2 mechanism read (PREREG_tdmpc2_f2_20260720.md).

Frozen BEFORE any tm2 stratified-error outcome exists (2026-07-20).
Input: the collate csv from probing.tdmpc2_stratified_error over all 64
fit checkpoints ({aware, free} x {s0, s1} x seeds 1-16).

PRIMARY **P-F2a (reward-decodability separation)**: per (side, seed)
pair, D = ln(free probe_mse_in_regime) - ln(aware probe_mse_in_regime);
mean over the 32 pairs; cluster bootstrap over SEEDS (16 clusters, both
sides resampled together) B=10,000 percentile CI, default_rng(0);
FIRES iff CI > 0. Decision on the CI alone.

Registered descriptors (never decisional):
- **P-F2b consistency invariance**: mean ln-ratio of cons_err_in
  (free/aware) with CI; tagged invariant-like iff |mean| < 0.5
  (mirrors the Dreamer d_errin arm-invariance qualitatively).
- **Lottery discriminator**: lottery_fraction = fraction of free runs
  whose ln probe_mse_in falls at or below the aware runs' 90th
  percentile; dispersion ratio = sd(free ln probe_mse_in) /
  sd(aware ln probe_mse_in).
- Interpretive map (frozen): fires AND lottery_fraction >= 0.15 =>
  "lottery-with-partial-inclusion"; fires AND < 0.15 =>
  "never-included-like"; not fires => "no representational separation
  (consistency alone includes the reward direction at this scale)".
- rew_head_mse reported for the aware arm only (free head untrained).

Usage:
  python -m analysis.tdmpc2_f2_read --csv <collate csv> --output <dir>
  python -m analysis.tdmpc2_f2_read --selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

SEEDS = tuple(range(1, 17))
SIDES = (0, 1)
B = 10_000
LOTTERY_FRAC_BAR = 0.15
INVARIANCE_BAND = 0.5


def load(path):
  grid = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      assert r["smoke"] == "0", f"smoke row in read input: {r['run_id']}"
      seed = int(r["seed"])
      if seed not in SEEDS:
        continue
      key = (r["arm"], int(r["side"]), seed)
      assert key not in grid, f"duplicate row: {r['run_id']}"
      grid[key] = dict(
          probe_in=float(r["probe_mse_in_regime"]),
          cons_in=float(r["cons_err_in_regime"]),
          rewhead_in=float(r["rew_head_mse_in_regime"]),
          head_trained=int(r["head_trained"]),
          sha=r["probeset_sha"])
  shas = {v["sha"] for v in grid.values()}
  assert len(shas) == 1, f"mixed probe sets: {shas}"
  return grid


def boot_ci(values_by_seed):
  """values_by_seed: {seed: [values]} -> mean + cluster bootstrap CI."""
  seeds = sorted(values_by_seed)
  allv = np.concatenate([values_by_seed[s] for s in seeds])
  obs = float(allv.mean())
  rng = np.random.default_rng(0)
  boots = np.empty(B)
  for b in range(B):
    pick = rng.integers(0, len(seeds), len(seeds))
    boots[b] = np.concatenate(
        [values_by_seed[seeds[i]] for i in pick]).mean()
  lo, hi = np.percentile(boots, [2.5, 97.5])
  return dict(mean=obs, ci=[float(lo), float(hi)], n=int(len(allv)))


def analyze(grid):
  for arm in ("aware", "free"):
    for side in SIDES:
      for k in SEEDS:
        assert (arm, side, k) in grid, f"missing {arm} s{side} seed{k}"
  assert all(grid[("aware", s, k)]["head_trained"] == 1
             for s in SIDES for k in SEEDS), "aware head must be trained"
  assert all(grid[("free", s, k)]["head_trained"] == 0
             for s in SIDES for k in SEEDS), "free head must be untrained"

  d_probe, d_cons = {}, {}
  for k in SEEDS:
    d_probe[k] = np.array([
        np.log(grid[("free", s, k)]["probe_in"])
        - np.log(grid[("aware", s, k)]["probe_in"]) for s in SIDES])
    d_cons[k] = np.array([
        np.log(grid[("free", s, k)]["cons_in"])
        - np.log(grid[("aware", s, k)]["cons_in"]) for s in SIDES])
  primary = boot_ci(d_probe)
  cons = boot_ci(d_cons)
  fires = bool(primary["ci"][0] > 0)

  aware_ln = np.array([np.log(grid[("aware", s, k)]["probe_in"])
                       for s in SIDES for k in SEEDS])
  free_ln = np.array([np.log(grid[("free", s, k)]["probe_in"])
                      for s in SIDES for k in SEEDS])
  p90 = float(np.percentile(aware_ln, 90))
  lottery_fraction = float((free_ln <= p90).mean())
  disp = float(free_ln.std(ddof=1) / aware_ln.std(ddof=1)) \
      if aware_ln.std(ddof=1) > 0 else None

  if fires:
    label = ("lottery-with-partial-inclusion"
             if lottery_fraction >= LOTTERY_FRAC_BAR else
             "never-included-like")
  else:
    label = ("no representational separation (consistency alone includes "
             "the reward direction at this scale)")
  res = dict(
      primary_pf2a=primary, fires=fires,
      consistency_pf2b=dict(
          **cons, invariant_like=bool(abs(cons["mean"]) < INVARIANCE_BAND)),
      lottery=dict(fraction=lottery_fraction, aware_p90_ln=p90,
                   dispersion_ratio=disp, bar=LOTTERY_FRAC_BAR),
      rew_head_mse_in_aware=float(np.mean(
          [grid[("aware", s, k)]["rewhead_in"] for s in SIDES for k in SEEDS])),
      interpretation=label)
  return res


def _fake(rng, aware_mu, free_mu, free_sd, cons_gap=0.0):
  grid = {}
  for s in SIDES:
    for k in SEEDS:
      grid[("aware", s, k)] = dict(
          probe_in=float(np.exp(rng.normal(aware_mu, 0.15))),
          cons_in=float(np.exp(rng.normal(-2.0, 0.1))),
          rewhead_in=0.02, head_trained=1, sha="x")
      grid[("free", s, k)] = dict(
          probe_in=float(np.exp(rng.normal(free_mu, free_sd))),
          cons_in=float(np.exp(rng.normal(-2.0 + cons_gap, 0.1))),
          rewhead_in=9.0, head_trained=0, sha="x")
  return grid


def selfcheck():
  # Lottery: free mean higher but heterogeneous - many free runs dip
  # into the aware band.
  r = analyze(_fake(np.random.default_rng(0), -4.0, -3.0, 1.2))
  assert r["fires"] and r["interpretation"].startswith("lottery"), r
  assert r["lottery"]["dispersion_ratio"] > 2, r["lottery"]
  # Never-included: free uniformly high, tight.
  r = analyze(_fake(np.random.default_rng(1), -4.0, -1.5, 0.15))
  assert r["fires"] and r["interpretation"] == "never-included-like", r
  # Null: arms indistinguishable; consistency invariant.
  r = analyze(_fake(np.random.default_rng(2), -4.0, -4.0, 0.15))
  assert not r["fires"] and "no representational" in r["interpretation"]
  assert r["consistency_pf2b"]["invariant_like"]
  # Missing cell trips.
  g = _fake(np.random.default_rng(3), -4.0, -3.0, 1.2)
  del g[("free", 1, 7)]
  try:
    analyze(g)
    raise SystemExit("missing-cell assert did not trip")
  except AssertionError:
    pass
  print("selfcheck PASS: lottery, never-included, null, missing-cell trip")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv")
  ap.add_argument("--output", default="analysis_out/tdmpc2_f2")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  res = analyze(load(args.csv))
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "tdmpc2_f2.json"), "w") as f:
    json.dump(res, f, indent=1)
  p = res["primary_pf2a"]
  print(f"P-F2a ln-ratio free/aware probe_mse_in {p['mean']:+.3f} "
        f"[{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] -> "
        f"{'FIRES' if res['fires'] else 'does not fire'}")
  c = res["consistency_pf2b"]
  print(f"P-F2b cons ln-ratio {c['mean']:+.3f} "
        f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
        f"invariant_like={c['invariant_like']}")
  lo = res["lottery"]
  print(f"lottery fraction {lo['fraction']:.3f} (bar {lo['bar']}), "
        f"dispersion ratio {lo['dispersion_ratio']:.2f}")
  print(f"=> {res['interpretation']}")


if __name__ == "__main__":
  main()
