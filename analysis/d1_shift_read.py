"""Route-B shift-consequence probe read (PREREG_d1_shift_20260723.md).

Frozen BEFORE any shift-probe pilot or label exists. Input: 18 label
files from the shift-extended labeler (labeler_version
shift_ext_20260723), 6 fresh e1 pilots x 3 arms:

    d1s_{cup|finger}_seed{21|22|23}_{base|xpol|phys}.npz

  base = own-policy, unshifted (fresh baseline on the new pilots)
  xpol = behavior policy DRIVES the trajectory (sibling final ckpt,
         seed rotation 21->22->23->21 within domain); evaluation
         agent unchanged
  phys = all dm_control body masses scaled x1.3 at label time;
         own-policy trajectory

All labels from FINAL checkpoints (late only), dose zero only.

REGISTERED PRIMARY (one per arm, 2 looks total, run-clustered):
for arm in {xpol, phys}: D_r = mean(delta_real | arm) -
mean(delta_real | base) per run r (paired by domain x seed, 6 runs,
domains pooled); the arm FIRES iff the cluster-bootstrap 95% CI of
mean_r(D_r) is > 0. This is the Route-B consequence leg: "the real
purchase becomes more valuable under task-relevant shift."

REGISTERED SECONDARY (descriptive, never decisional, NOT a Route-A
revival): C1-style calibration (frozen Gate-D1 criteria via the R2
machinery, F0/F1) inside each arm's pooled 6-run cell; per-domain
splits; change rates.

AMENDMENT 1 (PREREG_d1_relabel_amend1_20260724.md, frozen before any
d1pilot corrected label exists): a second cohort of 18 passes on the
six Gate-D1 pilots (d1pilot_{cup,finger}_e1_seed{1,2,3}, same 1e5
maturity, e1, rotation 1->2->3->1). The d1s primary above is
UNCHANGED and still decides `fired_arms`. Added outputs when the
d1pilot cohort is present (all-or-nothing): `replication` = the same
2-look rule on the d1pilot cohort alone; `pooled` = the 12-cluster
estimate (headline magnitude). Cross-cohort consequence map lives in
the amendment, not in this script's verdict logic.

Usage:
  python -m analysis.d1_shift_read --labels <dir> --output <dir>
  python -m analysis.d1_shift_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import analysis.gate_d1_read as gd1
import analysis.gate_d1_r2_read as r2

ARMS = ("base", "xpol", "phys")
SHIFT_ARMS = ("xpol", "phys")
DOMS = ("cup", "finger")
MASS_SCALE = 1.3
LATE_STEP_MIN = 90_000
# d1fix_20260724: corrected-instrument campaign (candidate-conditioned
# branch carries + policy-RNG CRN; PREREG_d1_relabel_20260724). The
# 23-Jul defective-labeler wave (shift_ext_20260723) was read at commit
# cf0eeead; this read no longer accepts those labels.
LABELER_VERSION = "d1fix_20260724"
COHORTS = {
    "d1s": dict(
        seeds=(21, 22, 23), rotation={21: 22, 22: 23, 23: 21},
        run_fmt="d1s_{dom}_seed{seed}"),
    "d1pilot": dict(
        seeds=(1, 2, 3), rotation={1: 2, 2: 3, 3: 1},
        run_fmt="d1pilot_{dom}_e1_seed{seed}"),
}
SEEDS = COHORTS["d1s"]["seeds"]          # frozen-primary cohort
ROTATION = COHORTS["d1s"]["rotation"]
FILE_RES = {
    "d1s": re.compile(r"^d1s_(cup|finger)_seed(\d+)_(base|xpol|phys)\.npz$"),
    "d1pilot": re.compile(
        r"^d1pilot_(cup|finger)_e1_seed(\d+)_(base|xpol|phys)\.npz$"),
}


def load_labels(labels_dir):
  runs = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, "*.npz"))):
    name = os.path.basename(path)
    if name.endswith("_smoke.npz"):
      continue
    cohort = m = None
    for co, rx in FILE_RES.items():
      m = rx.match(name)
      if m:
        cohort = co
        break
    assert m, f"unparseable label file {name}"
    dom, seed, arm = m.group(1), int(m.group(2)), m.group(3)
    spec = COHORTS[cohort]
    assert seed in spec["seeds"], f"{name}: seed outside cohort {cohort}"
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    for k, v in r2.PINNED_DIALS.items():
      assert meta[k] == v, f"{name}: dial {k}={meta[k]} != pinned {v}"
    assert meta.get("labeler_version") == LABELER_VERSION, name
    assert int(meta["dose"]["dim"]) == 0, f"{name}: shift probe is e1-only"
    step_m = re.search(r"step(\d+)", str(meta["checkpoint"]))
    assert step_m is None or int(step_m.group(1)) >= LATE_STEP_MIN, (
        f"{name}: shift probe labels final checkpoints only")
    beh = str(meta.get("behavior_checkpoint", ""))
    mass = float(meta.get("mass_scale", 1.0))
    if arm == "base":
      assert beh == "" and mass == 1.0, f"{name}: base arm shifted"
    elif arm == "xpol":
      assert mass == 1.0, f"{name}: xpol arm must not mass-shift"
      sib = spec["run_fmt"].format(dom=dom, seed=spec["rotation"][seed])
      assert sib in beh, (
          f"{name}: behavior ckpt {beh!r} is not sibling {sib}")
      own = spec["run_fmt"].format(dom=dom, seed=seed)
      assert own not in beh, f"{name}: behavior ckpt is the eval run"
    else:
      assert beh == "" and mass == MASS_SCALE, (
          f"{name}: phys arm needs mass_scale={MASS_SCALE}, got "
          f"beh={beh!r} mass={mass}")
    key = (cohort, dom, seed, arm)
    assert key not in runs, f"duplicate {key}"
    runs[key] = dict(
        cohort=cohort, domain=dom, dose="e1", seed=seed, ckpt="late",
        arm=arm, meta=meta, ckpt_step=None,
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
  # d1s (frozen primary) must be complete; d1pilot is all-or-nothing.
  present = {co: any(k[0] == co for k in runs) for co in COHORTS}
  assert present["d1s"], "no d1s cohort labels found"
  for cohort, spec in COHORTS.items():
    if not present[cohort]:
      continue
    for dom in DOMS:
      for seed in spec["seeds"]:
        for arm in ARMS:
          assert (cohort, dom, seed, arm) in runs, (
              f"missing {cohort} {dom} seed{seed} {arm}")
  return runs


def integrity(runs):
  report = {}
  for key, r in sorted(runs.items()):
    same = r["m_real"] == r["m_now"]
    viol = int(np.sum(same & (r["delta_real"] != 0.0)))
    assert viol == 0, f"CRN violation in {key}"
    report["_".join(map(str, key))] = dict(
        n=len(r["step"]),
        change_rate_real=float((r["m_real"] != r["m_now"]).mean()),
        mean_delta_real=float(r["delta_real"].mean()))
  return report


def primary(runs, arm, cohorts=("d1s",)):
  diffs = {}
  for cohort in cohorts:
    for dom in DOMS:
      for seed in COHORTS[cohort]["seeds"]:
        diffs[(cohort, dom, seed)] = (
            float(runs[(cohort, dom, seed, arm)]["delta_real"].mean())
            - float(runs[(cohort, dom, seed, "base")]["delta_real"].mean()))

  def _mean(chunks):
    return float(np.mean(chunks))

  ci = gd1._cluster_ci(_mean, diffs)
  point = _mean(list(diffs.values()))
  return dict(
      point=point, ci=ci, fires=bool(ci[0] > 0),
      n_clusters=len(diffs),
      per_run={"_".join(map(str, k)): v for k, v in diffs.items()},
      per_domain={dom: float(np.mean(
          [v for k, v in diffs.items() if k[1] == dom]))
          for dom in DOMS})


def secondary_calibration(runs, arm, cohort="d1s"):
  cell_runs = {k: v for k, v in runs.items()
               if k[0] == cohort and k[3] == arm}
  out = {}
  for fset in ("F0", "F1"):
    preds, y = r2.crossfit(cell_runs, "real", fset)
    out[fset] = gd1.criteria(preds, y)
  out["per_feature"] = r2.per_feature_spearman(cell_runs, "real")
  return out


def analyze(runs):
  have_pilot = any(k[0] == "d1pilot" for k in runs)
  result = dict(integrity=integrity(runs), primary={}, secondary={},
                fired_arms=[])
  for arm in SHIFT_ARMS:
    result["primary"][arm] = primary(runs, arm, cohorts=("d1s",))
    if result["primary"][arm]["fires"]:
      result["fired_arms"].append(arm)
  for arm in ARMS:
    result["secondary"][arm] = secondary_calibration(runs, arm, "d1s")
  if have_pilot:
    # Amendment 1: replication family (d1pilot alone) + pooled headline.
    result["replication"] = {}
    result["pooled"] = {}
    result["fired_arms_replication"] = []
    for arm in SHIFT_ARMS:
      result["replication"][arm] = primary(runs, arm, cohorts=("d1pilot",))
      if result["replication"][arm]["fires"]:
        result["fired_arms_replication"].append(arm)
      result["pooled"][arm] = primary(runs, arm, cohorts=("d1s", "d1pilot"))
    result["secondary_d1pilot"] = {
        arm: secondary_calibration(runs, arm, "d1pilot") for arm in ARMS}
  if result["fired_arms"]:
    result["verdict"] = (
        f"SHIFT CONSEQUENCE FIRES ({result['fired_arms']}): the real "
        "purchase gains value under task-relevant shift — Route B's "
        "downstream-consequence leg lands. (Calibration numbers are "
        "descriptive; a C1 pass would be mechanism-consistent but does "
        "NOT revive Route A.)")
  else:
    result["verdict"] = (
        "Shift consequence does not fire: purchase value does not "
        "detectably rise under either shift arm — the consequence leg "
        "fails; Route B rests on mechanism + boundary alone.")
  if have_pilot:
    result["verdict"] += (
        f" [Amendment 1: replication cohort fires "
        f"{result['fired_arms_replication'] or 'nothing'}; cross-cohort "
        "consequence map per PREREG_d1_relabel_amend1_20260724.md.]")
  return result


def read(labels_dir, output):
  runs = load_labels(labels_dir)
  result = analyze(runs)
  result["labels_dir"] = labels_dir
  os.makedirs(output, exist_ok=True)
  with open(os.path.join(output, "d1_shift.json"), "w") as f:
    json.dump(result, f, indent=1, default=float)
  for label, block in (("primary", result["primary"]),
                       ("replication", result.get("replication", {})),
                       ("pooled", result.get("pooled", {}))):
    for arm, p in block.items():
      print(f"{label}/{arm}: D={p['point']:+.3f} CI=[{p['ci'][0]:+.3f},"
            f"{p['ci'][1]:+.3f}] fires={p['fires']} n={p['n_clusters']} "
            f"per-domain={ {k: round(v, 3) for k, v in p['per_domain'].items()} }")
  print()
  print(result["verdict"])
  return result


# --------------------------------------------------------------------------
# Selfcheck: synthetic label grid, no MuJoCo/jax
# --------------------------------------------------------------------------

def _synth_run(rng, dom, seed, arm, delta_mean, n=120, refn=200,
               cohort="d1s"):
  delta = delta_mean + 0.3 * rng.normal(size=n)
  m_now = np.zeros(n, np.int64)
  m_real = np.ones(n, np.int64)
  return dict(
      cohort=cohort, domain=dom, dose="e1", seed=seed, ckpt="late",
      arm=arm, meta={}, ckpt_step=None,
      qfull=rng.normal(size=(n, 3, 4)), udyn=rng.normal(size=n),
      deter=rng.normal(size=(n, 16)),
      ref_deter=rng.normal(size=(refn, 16)),
      ref_episode=np.zeros(refn, np.int64),
      ref_step=np.arange(refn) * 5,
      episode=np.ones(n, np.int64), step=np.arange(n) * 25 + 25,
      delta_real=delta, delta_imag=np.zeros(n),
      delta_real_boot=delta, delta_imag_boot=np.zeros(n),
      m_now=m_now, m_real=m_real, m_imag=m_now,
      cost_real_env=np.full(n, 4))


def _synth_grid(xpol_lift=0.0, phys_lift=0.0, cohorts=("d1s",),
                pilot_xpol_lift=None):
  rng = np.random.default_rng(0)
  runs = {}
  for cohort in cohorts:
    xl = xpol_lift if (cohort == "d1s" or pilot_xpol_lift is None) \
        else pilot_xpol_lift
    for dom in DOMS:
      for seed in COHORTS[cohort]["seeds"]:
        base_mean = float(rng.normal(0.1, 0.05))
        lifts = dict(base=0.0, xpol=xl, phys=phys_lift)
        for arm in ARMS:
          runs[(cohort, dom, seed, arm)] = _synth_run(
              rng, dom, seed, arm, base_mean + lifts[arm], cohort=cohort)
  return runs


def selfcheck():
  res = analyze(_synth_grid(xpol_lift=0.5))
  assert res["primary"]["xpol"]["fires"], res["primary"]["xpol"]
  assert not res["primary"]["phys"]["fires"], res["primary"]["phys"]
  assert res["fired_arms"] == ["xpol"]
  assert "replication" not in res, "single-cohort grid must not replicate"
  res2 = analyze(_synth_grid())
  assert res2["fired_arms"] == [], res2["fired_arms"]
  # Amendment 1: two-cohort grids.
  both = analyze(_synth_grid(xpol_lift=0.5, cohorts=("d1s", "d1pilot")))
  assert both["fired_arms"] == ["xpol"]
  assert both["fired_arms_replication"] == ["xpol"], both["replication"]
  assert both["pooled"]["xpol"]["fires"], both["pooled"]["xpol"]
  assert both["pooled"]["xpol"]["n_clusters"] == 12
  assert not both["pooled"]["phys"]["fires"]
  split = analyze(_synth_grid(
      xpol_lift=0.5, cohorts=("d1s", "d1pilot"), pilot_xpol_lift=0.0))
  assert split["fired_arms"] == ["xpol"]
  assert split["fired_arms_replication"] == [], split["replication"]
  runs = _synth_grid()
  del runs[("d1s", "cup", 22, "phys")]
  tripped = False
  try:
    analyze(runs)
  except (AssertionError, KeyError):
    tripped = True
  assert tripped, "missing-cell grid must trip"
  runs = _synth_grid(cohorts=("d1s", "d1pilot"))
  del runs[("d1pilot", "finger", 2, "xpol")]
  tripped = False
  try:
    analyze(runs)
  except (AssertionError, KeyError):
    tripped = True
  assert tripped, "missing d1pilot cell must trip when cohort present"
  print("selfcheck PASS: planted xpol lift fires xpol only (both "
        "cohorts + pooled n=12), cohort-split grid separates primary "
        "from replication, null grid fires nothing, missing cells trip "
        "in both cohorts")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--labels")
  ap.add_argument("--output", default="analysis_out/d1_shift")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.labels, "--labels required"
  read(args.labels, args.output)


if __name__ == "__main__":
  main()
