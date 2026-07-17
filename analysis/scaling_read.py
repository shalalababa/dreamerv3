"""Scaling-pilot read (PREREG_scaling_pilot_20260717.md; Option B).

Frozen BEFORE any size12m outcome exists (2026-07-17; no 12m fit or
adapt has been run — the deleted timing smoke is excluded by rule).
Machinery identical to the frozen P3/stamping reads: AUC100k rows from
the untouched ``analysis/adaptation_auc.py`` pipeline, cluster-bootstrap
percentile 95% CI (B=10,000, numpy default_rng seed 0), decisions on the
CI alone, sensitivity suite robustness-only.

Cells (finger q1 pair, frozen buffers, seeds 1-8 paired across ALL four
cells):

  task@12m  mode ax1s12q1s{0,1}   (AXIS1_SIZE=size12m AXIS1_ID_PREFIX=ax1s12)
  apt@12m   mode ax1fs12q1s{0,1}  (AXIS1_ID_PREFIX=ax1fs12)
  task@1m   mode ax1q1s{0,1}      historical (10-Jul read, auc_axis1.csv)
  apt@1m    mode ax1fq1s{0,1}     historical (P0 corrective, auc_p0.csv)

Registered quantities (B(k) = endpoint(s1) - endpoint(s0), seed k):

  PRIMARY  mean over seeds 1-8 of
           [B_task@12m - B_apt@12m] - [B_task@1m - B_apt@1m]
           on AUC100k (decision CI).  CI < 0 => supervision effect
           shrinks with capacity (substitution begins; membership
           account); CI ~ 0 => capacity does not substitute at 12x
           (compression account unchallenged; interaction is
           scale-robust for Paper 1); CI > 0 => supervision effect
           grows (unexpected; theory revision).
  S1       same three-way on the final10 endpoint (P-B5 dissociation;
           registered DIRECTIONAL, no decision): shrinkage larger on
           the near-asymptotic endpoint than on AUC100k.
  S2       B_apt@12m simple effect (P-B6-lite; prediction: CI includes
           0 - reward-free transfer stays null at 12m).
  descriptive  all 8 cell means on both endpoints; level shifts.

Disclosed-known caveat (registered): the 1m task stratum seeds 1-8 is
the optimistic draw (+159; honest pooled 1-16 = +98.7, W0). The
three-way therefore overstates 1m-side supervision benefit; a
robustness variant replacing the 1m strata with pooled-1-16 records is
reported alongside, never decision-bearing.

Theory hooks: P-B5, P-B6 (PREREG_theory_predictions_20260717.md).

Usage:
  python -m analysis.scaling_read --auc <auc.csv from the 12m snapshot> \
      [--runroot <snapshot runroot for config/size audit>] --output <dir>
  python -m analysis.scaling_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import os

import numpy as np
from scipy import stats

B, RNG_SEED = 10_000, 0
SEEDS = list(range(1, 9))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK1M_CSV = os.path.join(REPO, "artifacts/phase6_axis1_20260710/auc_axis1.csv")
APT1M_CSV = os.path.join(REPO, "artifacts/p0_axis1_corrective_20260713/auc_p0.csv")
FULL_JSON = os.path.join(
    REPO, "artifacts/phase6_axis1_20260710/paired/"
    "paired_finger_q1_auc100k_s1_minus_s0.json")
APT_JSON = os.path.join(
    REPO, "artifacts/p0_axis1_corrective_20260713/paired/"
    "paired_finger_q1_auc100k_s1_minus_s0.json")
MODES = dict(task12="ax1s12q1s", apt12="ax1fs12q1s",
             task1="ax1q1s", apt1="ax1fq1s")
DETER_12M = 2048


def boot_ci(x, b=B, seed=RNG_SEED):
  rng = np.random.default_rng(seed)
  boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(b)]
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sensitivity(deltas):
  deltas = np.asarray(deltas, float)
  n = len(deltas)
  t = stats.ttest_1samp(deltas, 0.0)
  ci = t.confidence_interval()
  obs = deltas.mean()
  flips = [np.mean(deltas * np.array(s))
           for s in itertools.product((1, -1), repeat=n)]
  p_perm = float(np.mean([abs(f) >= abs(obs) - 1e-12 for f in flips]))
  w = stats.wilcoxon(deltas)
  d_z = float(deltas.mean() / deltas.std(ddof=1))
  loo = [np.delete(deltas, i).mean() for i in range(n)]
  return dict(t_ci=[float(ci.low), float(ci.high)], t_p=float(t.pvalue),
              perm_p=p_perm, wilcoxon_p=float(w.pvalue), d_z=d_z,
              loo_range=[float(min(loo)), float(max(loo))])


def contrast(x, decision=False):
  x = np.asarray(x, float)
  lo, hi = boot_ci(x)
  out = dict(deltas=x.tolist(), mean=float(x.mean()), ci=[lo, hi],
             pos=int((x > 0).sum()), n=len(x), sensitivity=sensitivity(x))
  if decision:
    out["decision"] = ("CI excludes 0" if (lo > 0) == (hi > 0)
                       else "CI includes 0")
  return out


def load_b(auc_csv, mode_prefix, endpoint, seeds=SEEDS, domain="finger"):
  """Per-seed B = endpoint(s1) - endpoint(s0) for one cell."""
  s0, s1 = {}, {}
  for r in csv.DictReader(open(auc_csv)):
    if r["domain"] != domain or int(r["qc_pass"]) != 1:
      continue
    if r["mode"] == f"{mode_prefix}0":
      s0[int(r["seed"])] = float(r[endpoint])
    elif r["mode"] == f"{mode_prefix}1":
      s1[int(r["seed"])] = float(r[endpoint])
  missing = [s for s in seeds if s not in s0 or s not in s1]
  if missing:
    raise SystemExit(f"cell {mode_prefix}*: missing seeds {missing} "
                     f"in {auc_csv} (have s0={sorted(s0)}, s1={sorted(s1)})")
  b = np.array([s1[s] - s0[s] for s in seeds])
  means = dict(s0=float(np.mean([s0[s] for s in seeds])),
               s1=float(np.mean([s1[s] for s in seeds])))
  return b, means


def audit(runroot):
  """12m cells: saved WM config has rssm deter == 2048 (size audit) and
  the correct expl.mode per arm; ADAPT_DONE present. The 1m strata were
  audited in their own frozen reads (bit-checked artifacts)."""
  import yaml
  problems, n_ok = [], 0
  for cell, mode_prefix, exp_mode in (
      ("task12", "s12q1s", "task"), ("apt12", "fs12q1s", "apt")):
    for side in (0, 1):
      for seed in SEEDS:
        wm = os.path.join(
            runroot, f"ax1wm_finger_{mode_prefix}{side}_seed{seed}")
        run = os.path.join(
            runroot,
            f"adapt_ax1{mode_prefix}{side}_finger_seed{seed}_ckpt500000")
        tag = os.path.basename(run)
        if not os.path.isdir(wm):
          problems.append(f"{tag}: missing WM dir"); continue
        cfg = yaml.safe_load(open(os.path.join(wm, "config.yaml")))["agent"]
        deter = int(cfg["dyn"]["rssm"]["deter"])
        if deter != DETER_12M:
          problems.append(f"{tag}: deter {deter} != {DETER_12M}"); continue
        if cfg["expl"]["mode"] != exp_mode:
          problems.append(
              f"{tag}: expl.mode {cfg['expl']['mode']} != {exp_mode}")
          continue
        if not os.path.isfile(os.path.join(run, "ADAPT_DONE")):
          problems.append(f"{tag}: no ADAPT_DONE"); continue
        n_ok += 1
  return n_ok, problems


def run_read(auc_csv, runroot, outdir,
             task1m_csv=TASK1M_CSV, apt1m_csv=APT1M_CSV, bitcheck=True):
  os.makedirs(outdir, exist_ok=True)
  out = {"auc_csv": auc_csv, "seeds": SEEDS}

  if runroot:
    n_ok, problems = audit(runroot)
    out["audit"] = dict(n_ok=n_ok, problems=problems)
    print(f"AUDIT: {n_ok}/32 12m runs pass; problems: {len(problems)}")
    for p in problems[:20]:
      print("  ", p)

  srcs = dict(task12=(auc_csv, MODES["task12"]),
              apt12=(auc_csv, MODES["apt12"]),
              task1=(task1m_csv, MODES["task1"]),
              apt1=(apt1m_csv, MODES["apt1"]))
  b, cells = {}, {}
  for ep in ("auc100k", "final10"):
    for cell, (path, prefix) in srcs.items():
      b[(cell, ep)], cells[(cell, ep)] = load_b(path, prefix, ep)

  if bitcheck:
    # Internal consistency: the 1m AUC100k deltas must reproduce the
    # bit-checked paired JSONs of the original frozen reads.
    full = np.asarray(json.load(open(FULL_JSON))["deltas"], float)
    apt = np.asarray(json.load(open(APT_JSON))["deltas"], float)
    assert np.allclose(b[("task1", "auc100k")], full, atol=1e-4), \
        "task@1m deltas do not reproduce the 10-Jul paired JSON"
    assert np.allclose(b[("apt1", "auc100k")], apt, atol=1e-4), \
        "apt@1m deltas do not reproduce the P0 paired JSON"

  def threeway(ep):
    return (b[("task12", ep)] - b[("apt12", ep)]) \
        - (b[("task1", ep)] - b[("apt1", ep)])

  out["b_cells"] = {f"{c}_{ep}": v.tolist() for (c, ep), v in b.items()}
  out["cell_means"] = {f"{c}_{ep}": m for (c, ep), m in cells.items()}

  out["primary_threeway_auc100k"] = contrast(threeway("auc100k"),
                                             decision=True)
  out["s1_threeway_final10"] = contrast(threeway("final10"))
  out["s1_dissociation_final10_minus_auc"] = contrast(
      threeway("final10") - threeway("auc100k"))
  out["s2_b_apt12_auc100k"] = contrast(b[("apt12", "auc100k")])
  out["interaction_at_12m"] = contrast(
      b[("task12", "auc100k")] - b[("apt12", "auc100k")])
  out["interaction_at_1m"] = contrast(
      b[("task1", "auc100k")] - b[("apt1", "auc100k")])

  p = out["primary_threeway_auc100k"]
  if p["decision"] == "CI excludes 0" and p["mean"] < 0:
    branch = ("PRIMARY fires NEGATIVE: supervision effect shrinks with "
              "capacity - substitution begins (membership account); "
              "parameterizes the 9b ladder around the crossing.")
  elif p["decision"] == "CI excludes 0":
    branch = ("PRIMARY fires POSITIVE: supervision effect GROWS with "
              "capacity - outside both registered accounts; theory "
              "revision required before any follow-up design.")
  else:
    branch = ("PRIMARY includes 0: capacity does not substitute at 12x "
              "- compression account unchallenged; the interaction is "
              "scale-robust (Paper-1 strengthens); 9b needs a wider "
              "size ladder to find any crossing.")
  out["registered_branch"] = branch

  json.dump(out, open(os.path.join(outdir, "scaling_read.json"), "w"),
            indent=1)

  def line(name, c):
    extra = f" -> {c['decision']}" if "decision" in c else ""
    print(f"{name:<34} {c['mean']:+9.2f} "
          f"CI=[{c['ci'][0]:+8.2f},{c['ci'][1]:+8.2f}]"
          f" pos={c['pos']}/{c['n']}"
          f" perm_p={c['sensitivity']['perm_p']:.4f}{extra}")

  print("\ncell means (auc100k):",
        {c: (round(m["s0"], 1), round(m["s1"], 1))
         for (c, ep), m in cells.items() if ep == "auc100k"})
  line("PRIMARY  three-way (AUC100k)", out["primary_threeway_auc100k"])
  line("S1       three-way (final10)", out["s1_threeway_final10"])
  line("S1d      final10 - auc100k  ", out["s1_dissociation_final10_minus_auc"])
  line("S2       B_apt@12m          ", out["s2_b_apt12_auc100k"])
  line("desc     interaction @12m   ", out["interaction_at_12m"])
  line("desc     interaction @1m    ", out["interaction_at_1m"])
  print("\nREGISTERED BRANCH:", branch)
  return out


def selfcheck():
  """Synthetic four-cell tables with known structure; both branches."""
  import tempfile
  rng = np.random.default_rng(7)

  def make_csv(path, cells):
    with open(path, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["run_id", "mode", "domain", "seed", "milestone", "auc50k",
                  "n_ep_50k", "auc100k", "n_ep_100k", "auc125k", "n_ep_125k",
                  "final10", "qc_pass"])
      for prefix, eff in cells.items():
        for seed in SEEDS:
          base = 100 + rng.normal(0, 5)
          for side, val in ((0, base), (1, base + eff + rng.normal(0, 8))):
            rid = f"adapt_{prefix}{side}_finger_seed{seed}_ckpt500000"
            w.writerow([rid, f"{prefix}{side}", "finger", seed, 500000,
                        0, 48, f"{val:.4f}", 96, 0, 112, f"{val:.4f}", 1])

  tmp = tempfile.mkdtemp()
  hist = os.path.join(tmp, "hist_task.csv")
  make_csv(hist, {MODES["task1"]: 150.0})
  hist_apt = os.path.join(tmp, "hist_apt.csv")
  make_csv(hist_apt, {MODES["apt1"]: 0.0})
  # Scenario A: 12m halves the supervision effect -> three-way ~ -75.
  new_a = os.path.join(tmp, "new_a.csv")
  make_csv(new_a, {MODES["task12"]: 75.0, MODES["apt12"]: 0.0})
  out = run_read(new_a, None, os.path.join(tmp, "a"),
                 task1m_csv=hist, apt1m_csv=hist_apt, bitcheck=False)
  p = out["primary_threeway_auc100k"]
  assert p["decision"] == "CI excludes 0" and -100 < p["mean"] < -50, p
  assert "substitution" in out["registered_branch"]
  # Scenario B: effect preserved at 12m -> three-way ~ 0.
  new_b = os.path.join(tmp, "new_b.csv")
  make_csv(new_b, {MODES["task12"]: 150.0, MODES["apt12"]: 0.0})
  out = run_read(new_b, None, os.path.join(tmp, "b"),
                 task1m_csv=hist, apt1m_csv=hist_apt, bitcheck=False)
  assert out["primary_threeway_auc100k"]["decision"] == "CI includes 0"
  assert "scale-robust" in out["registered_branch"]
  # Bit-check path runs against the real artifacts (loads only).
  full = np.asarray(json.load(open(FULL_JSON))["deltas"], float)
  b1, _ = load_b(TASK1M_CSV, MODES["task1"], "auc100k")
  assert np.allclose(b1, full, atol=1e-4)
  print("\nSELFCHECK PASS")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--runroot", default=None)
  ap.add_argument("--output", default="analysis_out/scaling")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not args.auc:
    ap.error("--auc required (or --selfcheck)")
  run_read(args.auc, args.runroot, args.output)


if __name__ == "__main__":
  main()
