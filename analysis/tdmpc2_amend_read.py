"""TD-MPC2 Amendment-1 pooled read (PREREG_tdmpc2_amend1_20260718.md).

Frozen BEFORE any seed-9-16 TD-MPC2 job exists (2026-07-18). Mirrors
the P3 Amendment-1 pattern: seeds 1-8 come from the frozen record
(artifacts/tdmpc2_goodhart_d1pilot_20260717/auc.csv, the 17-Jul read's
input); any seed-1-8 rows in the new csv must bit-match it. Machinery
identical to the frozen family: B_arm(k) = AUC100k(s1) - AUC100k(s0),
cluster-bootstrap percentile 95% CI (B=10,000, default_rng(0)),
decisions on CI alone, sensitivity suite robustness-only.

Registered quantities:
  PRIMARY   pooled seeds 1-16 [B_aware - B_free]      (decision CI)
  FRESH     seeds 9-16 alone                          (replication panel)
  SEC       pooled B_aware, pooled B_free             (CI-bearing for P-C1)
  P-C1a     share (primary mean / B_aware mean) < 0.6 (directional)
  P-C1b     B_free CI > 0 lands / spans & point>=+10 unresolved /
            spans & point<+10 fails (refutes decoder-free limit)
  P-C1c     B_aware CI > 0
  Verdict   LANDS iff C1a & C1b-lands & C1c; REFUTED iff C1b-fails;
            else PARTIAL.
  G-F2      coherence note (resource gate only, never inferential).

Audit (registered rule, --runroot): per seed-9-16 run, wm config.yaml
arm coefficients (free: reward_coef=value_coef=0.0; aware: 0.1/0.1;
consistency_coef 20 both) + TM2_FIT_DONE + ADAPT_DONE.

Usage:
  python -m analysis.tdmpc2_amend_read --auc <new csv (seeds 9-16 or 1-16)> \
      [--runroot <snapshot runroot>] --output <dir>
  python -m analysis.tdmpc2_amend_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import math
import os

import numpy as np
from scipy import stats

B, RNG_SEED = 10_000, 0
SEEDS_OLD = list(range(1, 9))
SEEDS_NEW = list(range(9, 17))
SEEDS_ALL = SEEDS_OLD + SEEDS_NEW
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_AUC = os.path.join(
    REPO, "artifacts/tdmpc2_goodhart_d1pilot_20260717/auc.csv")
ARMS = ("aware", "free")
COEFS = {"aware": (0.1, 0.1), "free": (0.0, 0.0)}
CONSISTENCY = 20


def load_auc(path):
  out = {}
  with open(path) as f:
    for r in csv.DictReader(f):
      if not r["mode"].startswith("tm2"):
        continue
      assert r["qc_pass"] == "1", f"QC fail: {r['run_id']}"
      out[(r["mode"], int(r["seed"]))] = r["auc100k"]
  return out


def b_arm(auc, arm, seeds):
  return np.array([float(auc[(f"tm2{arm}q1s1", k)]) -
                   float(auc[(f"tm2{arm}q1s0", k)]) for k in seeds])


def boot_ci(x, b=B, seed=RNG_SEED):
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(x), size=(b, len(x)))
  means = x[idx].mean(axis=1)
  return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sensitivity(x):
  n = len(x)
  m, sd = x.mean(), x.std(ddof=1)
  t = m / (sd / math.sqrt(n))
  if n <= 16:
    perms = np.array(list(itertools.product([1, -1], repeat=n)))
    pm = (perms * x).mean(axis=1)
    p_perm = float((np.abs(pm) >= abs(m) - 1e-12).mean())
  else:
    p_perm = float("nan")
  loo = [float(np.delete(x, i).mean()) for i in range(n)]
  return dict(t=float(t), t_p=float(2 * stats.t.sf(abs(t), n - 1)),
              perm_p=p_perm, d_z=float(m / sd),
              loo_range=[min(loo), max(loo)],
              wilcoxon_p=float(stats.wilcoxon(x).pvalue))


def stat_block(x):
  return dict(deltas=[float(v) for v in x], mean=float(x.mean()),
              ci=boot_ci(x), pos=int((x > 0).sum()), n=len(x),
              sensitivity=sensitivity(x))


def audit_runroot(runroot):
  problems, n_ok = [], 0
  for arm in ARMS:
    for side in (0, 1):
      for k in SEEDS_NEW:
        wm = os.path.join(runroot, f"tm2wm_finger_{arm}q1s{side}_seed{k}")
        ad = os.path.join(
            runroot, f"adapt_tm2{arm}q1s{side}_finger_seed{k}_ckpt500000")
        try:
          assert os.path.exists(os.path.join(wm, "TM2_FIT_DONE")), \
              f"{wm}: no TM2_FIT_DONE"
          assert os.path.exists(os.path.join(ad, "ADAPT_DONE")), \
              f"{ad}: no ADAPT_DONE"
          with open(os.path.join(wm, "config.yaml")) as f:
            cfg = json.load(f)
          rc, vc = COEFS[arm]
          assert cfg["arm"] == arm and cfg["seed"] == k, cfg
          assert float(cfg["reward_coef"]) == rc, \
              f"{wm}: reward_coef {cfg['reward_coef']} != {rc}"
          assert float(cfg["value_coef"]) == vc, \
              f"{wm}: value_coef {cfg['value_coef']} != {vc}"
          assert float(cfg["consistency_coef"]) == CONSISTENCY, wm
          n_ok += 1
        except (AssertionError, FileNotFoundError, KeyError) as e:
          problems.append(str(e))
  return dict(n_ok=n_ok, n_expected=32, problems=problems)


def analyze(auc_old, auc_new):
  overlap = sorted(set(auc_old) & set(auc_new))
  for key in overlap:
    assert auc_old[key] == auc_new[key], \
        f"bit-check FAIL {key}: frozen {auc_old[key]} != new {auc_new[key]}"
  merged = dict(auc_new)
  merged.update(auc_old)  # frozen record authoritative for seeds 1-8
  for arm in ARMS:
    for side in (0, 1):
      for k in SEEDS_ALL:
        assert (f"tm2{arm}q1s{side}", k) in merged, \
            f"missing cell tm2{arm}q1s{side} seed {k}"

  res = dict(n_overlap_bitchecked=len(overlap))
  for tag, seeds in (("pooled", SEEDS_ALL), ("old", SEEDS_OLD),
                     ("fresh", SEEDS_NEW)):
    baw, bfr = b_arm(merged, "aware", seeds), b_arm(merged, "free", seeds)
    res[f"primary_{tag}"] = stat_block(baw - bfr)
    res[f"aware_{tag}"] = stat_block(baw)
    res[f"free_{tag}"] = stat_block(bfr)

  prim, aw, fr = res["primary_pooled"], res["aware_pooled"], res["free_pooled"]
  res["primary_fires"] = bool(prim["ci"][0] > 0)

  share = prim["mean"] / aw["mean"] if aw["mean"] > 0 else float("nan")
  c1a = bool(np.isfinite(share) and share < 0.6)
  if fr["ci"][0] > 0:
    c1b = "lands"
  elif fr["mean"] >= 10:
    c1b = "unresolved"
  else:
    c1b = "fails"
  c1c = bool(aw["ci"][0] > 0)
  verdict = ("REFUTED" if c1b == "fails" else
             "LANDS" if (c1a and c1b == "lands" and c1c) else "PARTIAL")
  res["p_c1"] = dict(share=float(share), c1a_pass=c1a, c1b=c1b,
                     c1c_pass=c1c, verdict=verdict)

  fresh, old = res["primary_fresh"], res["primary_old"]
  incoherent = bool(
      np.sign(fresh["mean"]) != np.sign(old["mean"])
      and aw["ci"][0] <= 0 and aw["mean"] < 10)
  res["gate_f2"] = dict(incoherent=incoherent,
                        note="resource gate only; F2 blocked iff incoherent")

  cells = {}
  for arm in ARMS:
    for side in (0, 1):
      v = [float(merged[(f"tm2{arm}q1s{side}", k)]) for k in SEEDS_ALL]
      cells[f"tm2{arm}_s{side}"] = dict(
          mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
  res["cell_means_pooled"] = cells
  return res


def report(res):
  def line(tag, r):
    s = r["sensitivity"]
    print(f"{tag:28s} {r['mean']:+7.1f} CI=[{r['ci'][0]:+8.1f},"
          f"{r['ci'][1]:+8.1f}] pos={r['pos']}/{r['n']} "
          f"perm_p={s['perm_p']:.4f} d_z={s['d_z']:+.2f}")
  print(f"bit-checked overlap rows: {res['n_overlap_bitchecked']}")
  line("PRIMARY pooled 1-16", res["primary_pooled"])
  line("  fresh batch 9-16", res["primary_fresh"])
  line("  seeds 1-8 (frozen record)", res["primary_old"])
  line("SEC aware pooled", res["aware_pooled"])
  line("SEC free  pooled", res["free_pooled"])
  print()
  print("PRIMARY:", "FIRES (CI>0) -> attenuated-but-present cross-family; "
        "family-scope limitation revised" if res["primary_fires"] else
        "does NOT fire -> family-scope limitation stands at doubled power")
  pc = res["p_c1"]
  print(f"P-C1: share={pc['share']:.3f} C1a(<0.6)="
        f"{'pass' if pc['c1a_pass'] else 'fail'}  C1b={pc['c1b']}  "
        f"C1c={'pass' if pc['c1c_pass'] else 'fail'}  "
        f"=> P-C1 {pc['verdict']}")
  print(f"Gate G-F2: {'INCOHERENT - F2 blocked' if res['gate_f2']['incoherent'] else 'coherent - F2 proceeds (default GO)'}")
  print()
  for cell, v in res["cell_means_pooled"].items():
    print(f"cell {cell}: mean AUC100k {v['mean']:7.1f} (sd {v['sd']:5.1f})")


def selfcheck():
  rng = np.random.default_rng(1)

  def synth(aware_eff, free_eff, sd):
    auc = {}
    for arm, eff in (("aware", aware_eff), ("free", free_eff)):
      for k in SEEDS_ALL:
        base = 100 + rng.normal(0, 5)
        auc[(f"tm2{arm}q1s0", k)] = f"{base:.4f}"
        auc[(f"tm2{arm}q1s1", k)] = f"{base + eff + rng.normal(0, sd):.4f}"
    return auc

  # Branch A: aware strong, free modest-positive, tight -> fires + LANDS
  auc = synth(60.0, 35.0, 12.0)
  old = {k: v for k, v in auc.items() if k[1] <= 8}
  res = analyze(old, auc)
  assert res["n_overlap_bitchecked"] == 32
  assert res["primary_fires"], res["primary_pooled"]
  assert res["p_c1"]["verdict"] == "LANDS", res["p_c1"]
  # Branch B: free dead -> C1b fails -> REFUTED
  auc = synth(50.0, 0.0, 15.0)
  res = analyze({k: v for k, v in auc.items() if k[1] <= 8}, auc)
  assert res["p_c1"]["c1b"] == "fails" and \
      res["p_c1"]["verdict"] == "REFUTED", res["p_c1"]
  # bit-check must trip on a corrupted overlap row
  bad = dict(auc)
  bad[("tm2awareq1s0", 1)] = "999.0"
  try:
    analyze({k: v for k, v in auc.items() if k[1] <= 8}, bad)
    raise SystemExit("selfcheck FAIL: bit-check did not trip")
  except AssertionError:
    pass
  # missing-cell guard
  incomplete = {k: v for k, v in auc.items() if k[1] != 16}
  try:
    analyze({k: v for k, v in auc.items() if k[1] <= 8}, incomplete)
    raise SystemExit("selfcheck FAIL: missing-cell guard did not trip")
  except AssertionError:
    pass
  print("selfcheck PASS: fires+LANDS branch, REFUTED branch, "
        "bit-check trip, missing-cell trip")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--runroot", default=None)
  ap.add_argument("--output", default="analysis_out/tdmpc2_amend1")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  auc_old = load_auc(FROZEN_AUC)
  assert len(auc_old) == 32, f"frozen record has {len(auc_old)} rows, not 32"
  auc_new = load_auc(args.auc)
  res = analyze(auc_old, auc_new)
  if args.runroot:
    res["audit"] = audit_runroot(args.runroot)
    print(f"AUDIT: {res['audit']['n_ok']}/32 seed-9-16 runs pass; "
          f"problems: {len(res['audit']['problems'])}")
    for p in res["audit"]["problems"]:
      print("  ", p)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "tdmpc2_amend_read.json"), "w") as f:
    json.dump(res, f, indent=1)
  report(res)


if __name__ == "__main__":
  main()
