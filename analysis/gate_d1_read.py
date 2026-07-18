"""Gate D1 read: Stage-1B oracle labels vs amortized EVSI-hat.

Implements PREREG_gate_d1_stage1a_20260714.md SB on the Stage-1B label
npz files produced by ``d0/oracle_labels.py`` (D1 pilot runs, E1 dose).
The three decision criteria, their thresholds (0.3 / 2x / net>0), the
run-clustered bootstrap, and cross-fitting by run were frozen 2026-07-14
before any label existed. The following are MECHANICAL choices made at
read time (2026-07-18, after labels existed but before any calibration
number was seen) and are disclosed, not registered:

  - EVSI-hat = OLS on the five standardized d0 signal features
    (uq, aflip, evpi_plugin, evpi_split, gap from ``d0.signals.compute``
    on the stored qfull), cross-fitted leave-one-run-out WITHIN domain
    (train on the two sibling runs). Sensitivity: rank of the raw
    evpi_split feature alone (no fitting at all).
  - Criterion-3 cost price is NOT pinned by the prereg; the CI-bearing
    statement is evaluated at price 0 (gross return), with a price sweep
    over real env steps reported as sensitivity. A dated amendment must
    pin the price before criterion 3 is treated as adjudicated.
  - Bootstraps resample RUNS (clusters) with replacement over the FIXED
    cross-fitted per-state predictions (no refit inside the bootstrap).

Binding identification condition (addendum s4): features come from the
stored qfull (computed at the decision state by the run's own d0 path,
imagination RNG); oracle deltas come from separate CRN env rollouts --
labels share no RNG or rollouts with feature computation.

Integrity check unique to this design: exact CRN pairing implies
delta_o == 0 exactly whenever m_o == m_now; any violation means
restore_env determinism was broken and the read aborts.

Usage:
  python -m analysis.gate_d1_read --labels <dir with *.npz> --output <dir>
  python -m analysis.gate_d1_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats

from d0 import signals as d0_signals

B, RNG_SEED = 10_000, 0
FEATURES = ("uq", "aflip", "evpi_plugin", "evpi_split", "gap")
OPS = ("real", "imag")
PRICE_SWEEP = (0.0005, 0.005, 0.05)  # return units per real env step


def load_labels(labels_dir):
  runs = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, "*.npz"))):
    if path.endswith("_smoke.npz"):
      continue
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    run = os.path.basename(path)[:-4]
    m = re.match(r".*_(cup|finger)_e\d+_seed(\d+)", run)
    assert m, f"unparseable run id {run}"
    runs[run] = dict(
        domain=m.group(1), seed=int(m.group(2)), meta=meta,
        qfull=np.transpose(np.asarray(d["qfull"], np.float64), (0, 1, 2)),
        **{k: np.asarray(d[k]) for k in (
            "delta_real", "delta_imag", "m_now", "m_real", "m_imag",
            "g_now", "episode", "step",
            "cost_real_env", "cost_real_calls",
            "cost_imag_env", "cost_imag_calls")})
  assert runs, f"no label npz found in {labels_dir}"
  return runs


def integrity(runs):
  report = {}
  for run, r in runs.items():
    row = dict(n=len(r["step"]), meta_dials={
        k: r["meta"][k] for k in
        ("states", "horizon", "label_every", "actions", "rollouts")})
    for op in OPS:
      same = r[f"m_{op}"] == r["m_now"]
      viol = int(np.sum(same & (r[f"delta_{op}"] != 0.0)))
      assert viol == 0, f"CRN violation: {run} op={op} ({viol} states)"
      row[f"change_rate_{op}"] = float((~same).mean())
      row[f"mean_delta_{op}"] = float(r[f"delta_{op}"].mean())
    report[run] = row
  return report


def crossfit_evsi(runs, domain, op):
  """Leave-one-run-out OLS within domain. Returns per-run predictions."""
  dom_runs = {k: v for k, v in runs.items() if v["domain"] == domain}
  feats = {k: d0_signals.compute(v["qfull"]) for k, v in dom_runs.items()}
  X = {k: np.stack([f[c] for c in FEATURES], 1) for k, f in feats.items()}
  y = {k: np.asarray(v[f"delta_{op}"], np.float64)
       for k, v in dom_runs.items()}
  preds = {}
  for held in dom_runs:
    tr = [k for k in dom_runs if k != held]
    Xtr = np.concatenate([X[k] for k in tr], 0)
    ytr = np.concatenate([y[k] for k in tr], 0)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Ah = np.column_stack(
        [np.ones(len(X[held])), (X[held] - mu) / sd])
    preds[held] = Ah @ w
  raw = {k: feats[k]["evpi_split"] for k in dom_runs}
  return preds, y, raw


def _cluster_ci(stat_fn, by_run, b=B, seed=RNG_SEED):
  """Percentile CI over run-resampled (cluster) bootstrap."""
  keys = sorted(by_run)
  rng = np.random.default_rng(seed)
  vals = []
  for _ in range(b):
    take = rng.choice(len(keys), len(keys), replace=True)
    vals.append(stat_fn([by_run[keys[i]] for i in take]))
  vals = np.asarray(vals, np.float64)
  return [float(np.nanpercentile(vals, 2.5)),
          float(np.nanpercentile(vals, 97.5))]


def criteria(preds, y):
  """The three registered Gate-D1 criteria for one domain x op cell."""
  by_run = {k: (np.asarray(preds[k]), np.asarray(y[k])) for k in preds}

  def _sp(chunks):
    p = np.concatenate([c[0] for c in chunks])
    d = np.concatenate([c[1] for c in chunks])
    return stats.spearmanr(p, d).statistic

  def _lift(chunks):
    p = np.concatenate([c[0] for c in chunks])
    d = np.concatenate([c[1] for c in chunks])
    pop = d.mean()
    if pop <= 0:
      return np.nan
    top = d[p >= np.quantile(p, 0.8)]
    return top.mean() / pop

  def _netgain(chunks, price=0.0, cost_env=0):
    p = np.concatenate([c[0] for c in chunks])
    d = np.concatenate([c[1] for c in chunks])
    net = d - price * cost_env
    gated = float(np.where(p - price * cost_env > 0, net, 0.0).mean())
    better = max(float(net.mean()), 0.0)
    return gated - better

  chunks_all = [by_run[k] for k in sorted(by_run)]
  rho = float(_sp(chunks_all))
  rho_ci = _cluster_ci(_sp, by_run)
  c1 = bool(rho >= 0.3 and rho_ci[0] > 0)

  pop_mean = float(np.concatenate([c[1] for c in chunks_all]).mean())
  lift = float(_lift(chunks_all))
  lift_ci = _cluster_ci(_lift, by_run)
  c2 = bool(pop_mean > 0 and np.isfinite(lift) and lift >= 2.0
            and np.isfinite(lift_ci[0]) and lift_ci[0] > 1)

  gain = float(_netgain(chunks_all))
  gain_ci = _cluster_ci(_netgain, by_run)
  c3 = bool(gain > 0 and gain_ci[0] > 0)
  return dict(
      spearman=rho, spearman_ci=rho_ci, c1_pass=c1,
      pop_mean_delta=pop_mean, lift=lift, lift_ci=lift_ci, c2_pass=c2,
      net_gain_price0=gain, net_gain_ci=gain_ci, c3_pass=c3,
      gate_pass=bool(c1 and c2 and c3))


def price_sweep(preds, y, cost_env):
  out = {}
  for pr in PRICE_SWEEP:
    p = np.concatenate([np.asarray(preds[k]) for k in sorted(preds)])
    d = np.concatenate([np.asarray(y[k]) for k in sorted(y)])
    net = d - pr * cost_env
    gated = float(np.where(p - pr * cost_env > 0, net, 0.0).mean())
    out[str(pr)] = dict(gated=gated, always=float(net.mean()), never=0.0)
  return out


def read(labels_dir, output):
  runs = load_labels(labels_dir)
  integ = integrity(runs)
  domains = sorted({r["domain"] for r in runs.values()})
  result = dict(labels_dir=labels_dir, integrity=integ, cells={})
  for domain in domains:
    for op in OPS:
      preds, y, raw = crossfit_evsi(runs, domain, op)
      cell = criteria(preds, y)
      cell["raw_evpi_split_spearman"] = float(stats.spearmanr(
          np.concatenate([raw[k] for k in sorted(raw)]),
          np.concatenate([y[k] for k in sorted(y)])).statistic)
      cost_env = 0 if op == "imag" else int(
          next(iter(runs.values()))["cost_real_env"][0])
      cell["price_sweep_gross"] = price_sweep(preds, y, cost_env)
      per_feat = {}
      dom_runs = {k: v for k, v in runs.items() if v["domain"] == domain}
      fdict = {k: d0_signals.compute(v["qfull"])
               for k, v in dom_runs.items()}
      for c in FEATURES:
        per_feat[c] = float(stats.spearmanr(
            np.concatenate([fdict[k][c] for k in sorted(fdict)]),
            np.concatenate([y[k] for k in sorted(y)])).statistic)
      cell["per_feature_spearman"] = per_feat
      result["cells"][f"{domain}_{op}"] = cell
  os.makedirs(output, exist_ok=True)
  with open(os.path.join(output, "gate_d1_read.json"), "w") as f:
    json.dump(result, f, indent=1, default=float)
  print("INTEGRITY: CRN pairing exact in all runs; dials:",
        next(iter(integ.values()))["meta_dials"])
  for run, row in sorted(integ.items()):
    print(f"  {run}: change real {row['change_rate_real']:.2f} / imag "
          f"{row['change_rate_imag']:.2f}; mean d_real "
          f"{row['mean_delta_real']:+.3f} d_imag {row['mean_delta_imag']:+.3f}")
  print()
  n_pass = 0
  for cell, r in result["cells"].items():
    n_pass += r["gate_pass"]
    print(f"{cell:14s} rho={r['spearman']:+.3f} CI={r['spearman_ci']} "
          f"C1={'PASS' if r['c1_pass'] else 'fail'} | "
          f"lift={r['lift']:.2f} CI={r['lift_ci']} "
          f"C2={'PASS' if r['c2_pass'] else 'fail'} | "
          f"net={r['net_gain_price0']:+.4f} CI={r['net_gain_ci']} "
          f"C3={'PASS' if r['c3_pass'] else 'fail'} "
          f"=> {'GATE PASS' if r['gate_pass'] else 'GATE FAIL'}")
  print(f"\nGATE D1: {n_pass}/{len(result['cells'])} domain-x-op cells "
        "pass all three registered criteria.")
  return result


def selfcheck():
  rng = np.random.default_rng(0)
  runs = {}
  for dom, signal in (("cup", True), ("finger", False)):
    for seed in (1, 2, 3):
      n, K, M = 300, 5, 8
      q = rng.normal(0, 1, (n, K, M))
      feats = d0_signals.compute(q)
      if signal:  # planted: delta follows evpi_split, positive pop mean
        d = 2.0 * feats["evpi_split"] + rng.normal(0, .3, n)
      else:       # null: delta independent of everything
        d = rng.normal(0.0, 1.0, n)
      runs[f"pilot_{dom}_e1_seed{seed}"] = dict(
          domain=dom, seed=seed, qfull=q,
          meta=dict(states=n, horizon=1, label_every=1,
                    actions=M, rollouts=1),
          delta_real=d, delta_imag=d * 0.5,
          m_now=np.zeros(n, int), m_real=np.ones(n, int),
          m_imag=np.ones(n, int),
          g_now=np.zeros(n), episode=np.arange(n) // 25,
          step=np.arange(n),
          cost_real_env=np.full(n, M), cost_real_calls=np.full(n, M),
          cost_imag_env=np.zeros(n, int), cost_imag_calls=np.full(n, M))
  integrity(runs)
  preds, y, _ = crossfit_evsi(runs, "cup", "real")
  c = criteria(preds, y)
  assert c["c1_pass"] and c["gate_pass"] is (c["c1_pass"] and
                                             c["c2_pass"] and c["c3_pass"])
  assert c["spearman"] > 0.6, c
  preds, y, _ = crossfit_evsi(runs, "finger", "real")
  c0 = criteria(preds, y)
  assert not c0["c1_pass"] and not c0["gate_pass"], c0
  print("selfcheck PASS: planted-signal cell passes C1, null cell fails")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--labels")
  ap.add_argument("--output", default="analysis_out/gate_d1")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  read(args.labels, args.output)


if __name__ == "__main__":
  main()
