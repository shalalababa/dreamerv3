"""Route-A R2 probe read (PREREG_gate_d1_r2probe_20260722.md).

Frozen BEFORE any R2-probe pilot or label exists (2026-07-22). Input: the
extended oracle-label npz files (d0/oracle_labels.py, labeler_version
r2_ext_20260722) for the 12 local pilots x {early, late} checkpoints:

    d1r2_{cup|finger}_{e1|e4}_seed{11|12|13}_{early|late}.npz

Cells = {early, late} x {e1 dose-zero, e4 = d0_dose3} x {real, imag},
domains POOLED within cell (6 runs/cell = 2 domains x 3 seeds; the
run-clustered bootstrap and LORO cross-fit treat runs as units exactly as
in the frozen Gate-D1 read, whose criteria() this file imports).

Feature sets, both read per cell:
  F0 = the five Gate-D1 signal features (relocation test alone);
  F1 = F0 + udyn + logdens + udyn_resid (the Stage-1A density-residual
       asset as an EVSI feature; R1 riding along). logdens = -ln(mean
       distance of k=10 nearest reference latents), standardized deter
       space, excluding same-episode reference points within +-10 steps;
       udyn_resid = within-run OLS residual of udyn on [1, logdens].

REGISTERED DECISION RULE (probe = resource gate, deliberately liberal;
multiplicity disclosed, any full Route-A wave carries its own
registration): R2 FIRES iff registered criterion C1 (Spearman >= 0.3
with run-clustered bootstrap CI > 0) passes in >= 1 of the three
PREDICTED cells {early x e1, early x e4, late x e4} for either op under
either feature set. The {late x e1} cell is the registered
boundary-replication check: predicted FAIL (this is the cell where the
cluster Gate-D1 read failed 0/4); a C1 pass there is flagged as a
port-integrity anomaly, not evidence.

C2/C3 and per-feature Spearmans (incl. against delta_*_boot) are
reported per cell as registered descriptors, never decisional.

Usage:
  python -m analysis.gate_d1_r2_read --labels <dir> --output <dir>
  python -m analysis.gate_d1_r2_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats

import analysis.gate_d1_read as gd1
from d0 import signals as d0_signals

FEATURES_F0 = gd1.FEATURES  # (uq, aflip, evpi_plugin, evpi_split, gap)
FEATURES_F1 = FEATURES_F0 + ("udyn", "logdens", "udyn_resid")
OPS = ("real", "imag")
DOMS = ("cup", "finger")
DOSES = ("e1", "e4")
CKPTS = ("early", "late")
SEEDS = (11, 12, 13)
KNN_K = 10
KNN_EXCLUDE = 10
EARLY_STEP_BOUNDS = (10_000, 40_000)
LATE_STEP_MIN = 90_000
PINNED_DIALS = dict(states=200, horizon=100, label_every=25,
                    actions=8, rollouts=16, ref_stride=5)
PREDICTED_CELLS = (("early", "e1"), ("early", "e4"), ("late", "e4"))
BOUNDARY_CELL = ("late", "e1")
FILE_RE = re.compile(
    r"^d1r2_(cup|finger)_(e1|e4)_seed(\d+)_(early|late)\.npz$")


def load_labels(labels_dir):
  runs = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, "*.npz"))):
    name = os.path.basename(path)
    if name.endswith("_smoke.npz"):
      continue
    m = FILE_RE.match(name)
    assert m, f"unparseable label file {name}"
    dom, dose, seed, ckpt = (m.group(1), m.group(2),
                             int(m.group(3)), m.group(4))
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    for k, v in PINNED_DIALS.items():
      assert meta[k] == v, f"{name}: dial {k}={meta[k]} != pinned {v}"
    assert meta.get("labeler_version") == "r2_ext_20260722", name
    mdose = meta["dose"]
    if dose == "e1":
      assert int(mdose["dim"]) == 0, f"{name}: e1 but dose {mdose}"
    else:
      assert int(mdose["dim"]) == 32 and float(mdose["scale"]) == 3.0, (
          f"{name}: e4 but dose {mdose}")
    step_m = re.search(r"step(\d+)", str(meta["checkpoint"]))
    if ckpt == "early":
      assert step_m, f"{name}: early labels must come from a step snapshot"
      step = int(step_m.group(1))
      assert EARLY_STEP_BOUNDS[0] <= step <= EARLY_STEP_BOUNDS[1], (
          f"{name}: early step {step} outside {EARLY_STEP_BOUNDS}")
    else:
      step = int(step_m.group(1)) if step_m else None
      assert step is None or step >= LATE_STEP_MIN, (
          f"{name}: late step {step} < {LATE_STEP_MIN}")
    key = (dom, dose, seed, ckpt)
    assert key not in runs, f"duplicate cell member {key}"
    runs[key] = dict(
        domain=dom, dose=dose, seed=seed, ckpt=ckpt, meta=meta,
        ckpt_step=step,
        qfull=np.asarray(d["qfull"], np.float64),
        udyn=np.asarray(d["udyn"], np.float64),
        deter=np.asarray(d["deter"], np.float64),
        ref_deter=np.asarray(d["ref_deter"], np.float64),
        ref_episode=np.asarray(d["ref_episode"]),
        ref_step=np.asarray(d["ref_step"]),
        **{k: np.asarray(d[k]) for k in (
            "delta_real", "delta_imag", "delta_real_boot",
            "delta_imag_boot", "m_now", "m_real", "m_imag",
            "episode", "step", "cost_real_env")})
  for dom in DOMS:
    for dose in DOSES:
      for seed in SEEDS:
        for ckpt in CKPTS:
          assert (dom, dose, seed, ckpt) in runs, (
              f"missing labels: {dom} {dose} seed{seed} {ckpt}")
  return runs


def integrity(runs):
  report = {}
  for key, r in sorted(runs.items()):
    for op in OPS:
      same = r[f"m_{op}"] == r["m_now"]
      viol = int(np.sum(same & (r[f"delta_{op}"] != 0.0)))
      assert viol == 0, f"CRN violation in {key} op={op}"
    report["_".join(map(str, key))] = dict(
        n=len(r["step"]), ckpt_step=r["ckpt_step"],
        change_rate_real=float((r["m_real"] != r["m_now"]).mean()),
        change_rate_imag=float((r["m_imag"] != r["m_now"]).mean()),
        mean_delta_real=float(r["delta_real"].mean()),
        mean_delta_imag=float(r["delta_imag"].mean()))
  return report


def knn_logdens(run, k=KNN_K, exclude=KNN_EXCLUDE):
  """-ln(mean distance to k nearest base-trajectory latents), per labeled
  state; reference standardized per run; same-episode points within
  +-exclude steps removed (the labeled state's own base twin)."""
  ref = run["ref_deter"]
  mu, sd = ref.mean(0), ref.std(0)
  sd[sd == 0] = 1.0
  refz = (ref - mu) / sd
  lab = (run["deter"] - mu) / sd
  out = np.empty(len(lab))
  for i in range(len(lab)):
    d = np.linalg.norm(refz - lab[i], axis=1)
    mask = ((run["ref_episode"] == run["episode"][i])
            & (np.abs(run["ref_step"] - run["step"][i]) <= exclude))
    d = d[~mask]
    kk = min(k, len(d))
    out[i] = -np.log(np.sort(d)[:kk].mean() + 1e-8)
  return out


def features(run, fset):
  f = d0_signals.compute(run["qfull"])
  cols = [f[c] for c in FEATURES_F0]
  extra = {}
  if fset == "F1":
    ld = knn_logdens(run)
    A = np.column_stack([np.ones(len(ld)), ld])
    w, *_ = np.linalg.lstsq(A, run["udyn"], rcond=None)
    resid = run["udyn"] - A @ w
    cols += [run["udyn"], ld, resid]
    extra = dict(logdens=ld, udyn_resid=resid)
  return np.stack(cols, 1), extra


def crossfit(cell_runs, op, fset):
  """LORO OLS over the (pooled-domain) runs of one cell."""
  X = {k: features(v, fset)[0] for k, v in cell_runs.items()}
  y = {k: np.asarray(v[f"delta_{op}"], np.float64)
       for k, v in cell_runs.items()}
  preds = {}
  for held in cell_runs:
    tr = [k for k in cell_runs if k != held]
    Xtr = np.concatenate([X[k] for k in tr], 0)
    ytr = np.concatenate([y[k] for k in tr], 0)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Ah = np.column_stack([np.ones(len(X[held])), (X[held] - mu) / sd])
    preds[held] = Ah @ w
  return preds, y


def per_feature_spearman(cell_runs, op):
  out = {}
  feats = {}
  for k, v in cell_runs.items():
    f = d0_signals.compute(v["qfull"], udyn=v["udyn"])
    _, extra = features(v, "F1")
    f.update(extra)
    feats[k] = f
  for c in FEATURES_F1:
    xs = np.concatenate([feats[k][c] for k in sorted(feats)])
    for lab, col in (("", f"delta_{op}"), ("_boot", f"delta_{op}_boot")):
      ys = np.concatenate(
          [np.asarray(cell_runs[k][col], np.float64)
           for k in sorted(feats)])
      out[f"{c}{lab}"] = float(stats.spearmanr(xs, ys).statistic)
  return out


def analyze(runs):
  integ = integrity(runs)
  result = dict(integrity=integ, cells={}, fires=False,
                fired_cells=[], boundary_anomaly=[])
  for ckpt in CKPTS:
    for dose in DOSES:
      cell_runs = {k: v for k, v in runs.items()
                   if v["ckpt"] == ckpt and v["dose"] == dose}
      assert len(cell_runs) == len(DOMS) * len(SEEDS)
      for op in OPS:
        for fset in ("F0", "F1"):
          preds, y = crossfit(cell_runs, op, fset)
          cell = gd1.criteria(preds, y)
          name = f"{ckpt}_{dose}_{op}_{fset}"
          result["cells"][name] = cell
          if cell["c1_pass"]:
            if (ckpt, dose) in PREDICTED_CELLS:
              result["fires"] = True
              result["fired_cells"].append(name)
            else:
              result["boundary_anomaly"].append(name)
        result["cells"][f"{ckpt}_{dose}_{op}_per_feature"] = (
            per_feature_spearman(cell_runs, op))
  if result["fires"]:
    result["verdict"] = (
        "R2 FIRES: C1 calibration appears in predicted cell(s) "
        f"{result['fired_cells']} => Route A revived as the boundary-law "
        "paper (full Route-A wave needs its own registration; 24+24 stay "
        "frozen until that wave passes its own gate).")
  else:
    result["verdict"] = (
        "R2 does not fire: no predicted cell reaches C1 => Route A "
        "CLOSED; Route B absorbs the package (incl. this probe as a "
        "registered negative); the 24+24 stay frozen permanently.")
  if result["boundary_anomaly"]:
    result["verdict"] += (
        " WARNING boundary-replication anomaly: C1 passed in late x e1 "
        f"({result['boundary_anomaly']}) where the cluster Gate-D1 read "
        "failed — treat the port with caution; not evidence for firing.")
  return result


def read(labels_dir, output):
  runs = load_labels(labels_dir)
  result = analyze(runs)
  result["labels_dir"] = labels_dir
  os.makedirs(output, exist_ok=True)
  with open(os.path.join(output, "gate_d1_r2.json"), "w") as f:
    json.dump(result, f, indent=1, default=float)
  for name, cell in result["cells"].items():
    if name.endswith("per_feature"):
      continue
    print(f"{name:22s} rho={cell['spearman']:+.3f} "
          f"CI={[round(c, 3) for c in cell['spearman_ci']]} "
          f"C1={'PASS' if cell['c1_pass'] else 'fail'}")
  print(f"\n{result['verdict']}")
  return result


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _synth_run(rng, dom, dose, seed, ckpt, mode, n=150):
  q = rng.normal(0, 1, (n, 5, 8))
  feats = d0_signals.compute(q)
  # latents: 3 gaussian clusters of different spread => density varies
  centers = np.array([[0, 0], [6, 6], [12, 0]], np.float64)
  ref_c = rng.integers(0, 3, 400)
  ref = centers[ref_c] + rng.normal(0, [1, 1, 3][0], (400, 2))
  lab_c = rng.integers(0, 3, n)
  deter = centers[lab_c] + rng.normal(0, 1, (n, 2))
  run = dict(
      domain=dom, dose=dose, seed=seed, ckpt=ckpt, ckpt_step=None,
      meta={}, qfull=q, deter=deter,
      ref_deter=ref, ref_episode=np.zeros(400, int),
      ref_step=np.arange(400) * 1000,  # never inside the exclusion window
      episode=np.ones(n, int) * 7, step=np.arange(n),
      m_now=np.zeros(n, int), m_real=np.ones(n, int),
      m_imag=np.ones(n, int), cost_real_env=np.full(n, 8))
  hidden = rng.normal(0, 1, n)
  ld = knn_logdens(run)
  run["udyn"] = 1.0 * ld + hidden
  if mode == "signal_f0":       # relocation: the plain signals work
    d = 2.0 * feats["evpi_split"] + rng.normal(0, .3, n)
  elif mode == "signal_f1":     # only the density-residualized udyn works
    d = 2.0 * hidden + rng.normal(0, .3, n)
  else:                         # null (the boundary cell)
    d = rng.normal(0, 1.0, n)
  for op in OPS:
    run[f"delta_{op}"] = d if op == "real" else d * 0.5
    run[f"delta_{op}_boot"] = run[f"delta_{op}"] + rng.normal(0, .05, n)
  return run


def selfcheck():
  # runs at the full B=10K cluster bootstrap (~2.5 min): the same code
  # path the real read uses, nothing patched
  # density estimator sanity: a point inside the dense cluster beats an
  # outlier
  rng = np.random.default_rng(0)
  r = _synth_run(rng, "cup", "e1", 11, "late", "null")
  r["deter"] = np.array([[0.0, 0.0], [30.0, 30.0]])
  r["episode"] = np.array([7, 7])
  r["step"] = np.array([0, 1])
  ld = knn_logdens(r)
  assert ld[0] > ld[1], ld

  rng = np.random.default_rng(1)
  runs = {}
  for dom in DOMS:
    for seed in SEEDS:
      for ckpt, dose in [(c, d) for c in CKPTS for d in DOSES]:
        if (ckpt, dose) == BOUNDARY_CELL:
          mode = "null"
        elif (ckpt, dose) == ("late", "e4"):
          mode = "signal_f1"
        else:
          mode = "signal_f0"
        runs[(dom, dose, seed, ckpt)] = _synth_run(
            rng, dom, dose, seed, ckpt, mode)
  res = analyze(runs)
  assert res["fires"], res["fired_cells"]
  assert any(n.startswith("early_e1_real_F0") for n in res["fired_cells"])
  assert any(n.startswith("late_e4_real_F1") for n in res["fired_cells"])
  assert not any(n.startswith("late_e4_real_F0") for n in res["fired_cells"]), (
      "F0 must NOT catch the density-residual-only cell")
  assert not res["boundary_anomaly"], res["boundary_anomaly"]

  # all-null: does not fire
  rng = np.random.default_rng(2)
  runs = {(dom, dose, seed, ckpt): _synth_run(rng, dom, dose, seed, ckpt,
                                              "null")
          for dom in DOMS for dose in DOSES
          for seed in SEEDS for ckpt in CKPTS}
  res0 = analyze(runs)
  assert not res0["fires"], res0["fired_cells"]

  # missing cell member trips
  del runs[("cup", "e1", 11, "early")]
  try:
    analyze(runs)
    raise SystemExit("missing-cell assert did not trip")
  except AssertionError:
    pass
  print("selfcheck PASS: density ordering, F0-cell fires, F1-only cell "
        "fires under F1 not F0, boundary stays clean, all-null does not "
        "fire, missing-cell trips")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--labels")
  ap.add_argument("--output", default="analysis_out/gate_d1_r2")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  read(args.labels, args.output)


if __name__ == "__main__":
  main()
