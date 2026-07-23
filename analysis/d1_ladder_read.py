"""D1 actionability-ladder read (PREREG_d1_ladder_20260723.md).

Registered-descriptive addendum on the EXISTING R2 labels (real op
only; imag excluded as defective per Amendment 1). Ladder per cell
({early,late} x {e1,e4}, domains pooled, 6 runs):

  L0  constants: always (mean delta_real) vs never (0)
  L1  LORO OLS on F0 scalars  (known from the R2 read; re-presented)
  L2  LORO OLS on F1 scalars  (known from the R2 read; re-presented)
  L3  LORO ridge on [deter (512) + F1 scalars], lambda in
      {1e1,1e2,1e3,1e4} chosen per outer fold by inner
      leave-one-training-run-out mean Spearman (nested).

Metrics: the frozen Gate-D1 criteria() outputs per level. Registered
NON-decisional heuristic: "compression signature" in a cell iff L3
Spearman >= 0.2 with cluster CI > 0 while L2 Spearman < 0.1; its only
consequence is resource-side (support-diverse-ensemble arm design).

Usage:
  python -m analysis.d1_ladder_read --labels <dir> --output <dir>
  python -m analysis.d1_ladder_read --selfcheck
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

import analysis.gate_d1_read as gd1
import analysis.gate_d1_r2_read as r2

OP = "real"
LAMBDAS = (1e1, 1e2, 1e3, 1e4)
SIG_L3_MIN = 0.2
SIG_L2_MAX = 0.1


def l3_features(run):
  X_f1, _ = r2.features(run, "F1")
  return np.column_stack([np.asarray(run["deter"], np.float64), X_f1])


def _ridge_fit(Xtr, ytr, lam):
  mu, sd = Xtr.mean(0), Xtr.std(0)
  sd[sd == 0] = 1.0
  Z = (Xtr - mu) / sd
  ym = ytr.mean()
  p = Z.shape[1]
  w = np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ (ytr - ym))
  return mu, sd, ym, w


def _ridge_pred(X, fit):
  mu, sd, ym, w = fit
  return ((X - mu) / sd) @ w + ym


def crossfit_l3(cell_runs):
  """Nested LORO ridge: lambda chosen per outer fold by inner
  leave-one-training-run-out mean Spearman."""
  X = {k: l3_features(v) for k, v in cell_runs.items()}
  y = {k: np.asarray(v[f"delta_{OP}"], np.float64)
       for k, v in cell_runs.items()}
  preds, chosen = {}, {}
  for held in sorted(cell_runs):
    tr = [k for k in sorted(cell_runs) if k != held]
    best_lam, best_score = None, -np.inf
    for lam in LAMBDAS:
      scores = []
      for inner in tr:
        itr = [k for k in tr if k != inner]
        fit = _ridge_fit(np.concatenate([X[k] for k in itr], 0),
                         np.concatenate([y[k] for k in itr], 0), lam)
        rho = stats.spearmanr(_ridge_pred(X[inner], fit),
                              y[inner]).statistic
        scores.append(0.0 if np.isnan(rho) else rho)
      score = float(np.mean(scores))
      if score > best_score:
        best_lam, best_score = lam, score
    fit = _ridge_fit(np.concatenate([X[k] for k in tr], 0),
                     np.concatenate([y[k] for k in tr], 0), best_lam)
    preds[held] = _ridge_pred(X[held], fit)
    chosen[held] = best_lam
  return preds, y, chosen


def l0_constants(cell_runs):
  by_run = {k: np.asarray(v[f"delta_{OP}"], np.float64)
            for k, v in cell_runs.items()}

  def _mean(chunks):
    return float(np.concatenate(chunks).mean())

  always = _mean(list(by_run.values()))
  ci = gd1._cluster_ci(_mean, by_run)
  return dict(always_mean=always, always_ci=ci, never=0.0,
              best_constant=max(always, 0.0))


def analyze(runs):
  result = dict(cells={}, compression_signature=[], op=OP)
  for ckpt in r2.CKPTS:
    for dose in r2.DOSES:
      cell_runs = {k: v for k, v in runs.items()
                   if v["ckpt"] == ckpt and v["dose"] == dose}
      assert len(cell_runs) == len(r2.DOMS) * len(r2.SEEDS)
      name = f"{ckpt}_{dose}"
      cell = dict(L0=l0_constants(cell_runs))
      for level, fset in (("L1", "F0"), ("L2", "F1")):
        preds, y = r2.crossfit(cell_runs, OP, fset)
        cell[level] = gd1.criteria(preds, y)
      preds, y, chosen = crossfit_l3(cell_runs)
      cell["L3"] = gd1.criteria(preds, y)
      cell["L3"]["lambda_per_fold"] = {
          "_".join(map(str, k)): v for k, v in chosen.items()}
      sig = (cell["L3"]["spearman"] >= SIG_L3_MIN
             and cell["L3"]["spearman_ci"][0] > 0
             and cell["L2"]["spearman"] < SIG_L2_MAX)
      cell["compression_signature"] = bool(sig)
      if sig:
        result["compression_signature"].append(name)
      result["cells"][name] = cell
  if result["compression_signature"]:
    result["verdict"] = (
        "COMPRESSION SIGNATURE (non-decisional heuristic) in "
        f"{result['compression_signature']}: the belief state predicts "
        "delta_real where scalar summaries do not => authorizes "
        "designing the support-diverse-ensemble arm (resource only).")
  else:
    result["verdict"] = (
        "No compression signature: at this information level the "
        "calibration null is not a scalar-summary artifact => "
        "no-exploitable-heterogeneity reading stands; boundary framing "
        "unchanged. (Resource consequence: ensemble arm stays gated.)")
  return result


def read(labels_dir, output):
  runs = r2.load_labels(labels_dir)
  result = analyze(runs)
  result["labels_dir"] = labels_dir
  os.makedirs(output, exist_ok=True)
  with open(os.path.join(output, "d1_ladder.json"), "w") as f:
    json.dump(result, f, indent=1, default=float)
  for name, cell in result["cells"].items():
    row = "  ".join(
        f"{lv} rho={cell[lv]['spearman']:+.3f}"
        f" CI=[{cell[lv]['spearman_ci'][0]:+.3f},"
        f"{cell[lv]['spearman_ci'][1]:+.3f}]"
        for lv in ("L1", "L2", "L3"))
    print(f"{name:10s} alw={cell['L0']['always_mean']:+.3f}  {row}"
          f"  sig={cell['compression_signature']}")
  print()
  print(result["verdict"])
  return result


# --------------------------------------------------------------------------
# Selfcheck: synthetic label grid, no MuJoCo/jax
# --------------------------------------------------------------------------

def _synth_run(rng, dom, dose, seed, ckpt, n=120, refn=200, mode="null"):
  deter = rng.normal(size=(n, 32))
  qfull = rng.normal(size=(n, 3, 4))
  udyn = rng.normal(size=n)
  noise = 0.05 * rng.normal(size=n)
  if mode == "deter":
    delta = deter[:, :4].sum(1) + noise         # belief-only signal
  elif mode == "scalar":
    from d0 import signals as d0_signals
    uq = d0_signals.u_q(np.asarray(qfull, np.float64))
    delta = 3.0 * (uq - uq.mean()) + noise
  else:
    delta = noise
  return dict(
      domain=dom, dose=dose, seed=seed, ckpt=ckpt, ckpt_step=None,
      meta={}, qfull=qfull, udyn=udyn, deter=deter,
      ref_deter=rng.normal(size=(refn, 32)),
      ref_episode=np.zeros(refn, np.int64),
      ref_step=np.arange(refn) * 5,
      episode=np.ones(n, np.int64),
      step=np.arange(n) * 25 + 25,
      delta_real=delta, delta_imag=np.zeros(n),
      delta_real_boot=delta, delta_imag_boot=np.zeros(n),
      m_now=np.zeros(n, np.int64), m_real=np.ones(n, np.int64),
      m_imag=np.zeros(n, np.int64), cost_real_env=np.full(n, 4))


def _synth_grid(mode_by_cell):
  rng = np.random.default_rng(0)
  runs = {}
  for dom in r2.DOMS:
    for dose in r2.DOSES:
      for seed in r2.SEEDS:
        for ckpt in r2.CKPTS:
          mode = mode_by_cell.get((ckpt, dose), "null")
          runs[(dom, dose, seed, ckpt)] = _synth_run(
              rng, dom, dose, seed, ckpt, mode=mode)
  return runs


def selfcheck():
  res = analyze(_synth_grid({("early", "e1"): "deter",
                             ("late", "e4"): "scalar"}))
  cell = res["cells"]["early_e1"]
  assert cell["compression_signature"], (
      "deter-planted cell must show the compression signature; got "
      f"L2 {cell['L2']['spearman']:+.3f} L3 {cell['L3']['spearman']:+.3f}")
  assert cell["L2"]["spearman"] < SIG_L2_MAX
  sc = res["cells"]["late_e4"]
  assert sc["L1"]["spearman"] >= 0.2 and sc["L2"]["spearman"] >= 0.2, (
      "scalar-planted cell must be picked up by scalar levels; got "
      f"L1 {sc['L1']['spearman']:+.3f} L2 {sc['L2']['spearman']:+.3f}")
  assert not sc["compression_signature"], (
      "scalar-planted cell must NOT count as compression")
  null = res["cells"]["late_e1"]
  assert not null["compression_signature"]
  assert abs(null["L3"]["spearman"]) < SIG_L3_MIN, (
      f"null cell L3 rho {null['L3']['spearman']:+.3f} suspiciously high")
  res2 = analyze(_synth_grid({}))
  assert not res2["compression_signature"], "all-null grid must be clean"
  print("selfcheck PASS: deter-planted cell fires the signature, "
        "scalar-planted cell is caught by L1/L2 without the signature, "
        "null cells and the all-null grid stay clean")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--labels")
  ap.add_argument("--output", default="analysis_out/d1_ladder")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.labels, "--labels required"
  read(args.labels, args.output)


if __name__ == "__main__":
  main()
