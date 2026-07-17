"""Synth Phase-B read (PREREG_synth_phaseb_20260717.md).

Frozen BEFORE any synth fit or adapt outcome exists (2026-07-17; only
the deleted Phase-A debug smoke has ever touched the domain). Machinery
identical to the frozen P3/stamping/scaling reads: AUC100k rows from
the untouched ``analysis/adaptation_auc.py``, cluster-bootstrap
percentile 95% CI (B=10,000, numpy default_rng seed 0), decisions on
the CI alone, sensitivity suite robustness-only.

Cells (synth q1 pair from probing/synth_buffers.py, seeds 1-8 paired,
domain 'synth'):

  task  mode ax1q1s{0,1}    full task objective  (axis1-bundles, task)
  apt   mode ax1fq1s{0,1}   reward-free          (axis1-bundles, apt)
  sh    mode ax1shq1s{0,1}  shuffled labels      (factorial-bundles, sh)

Registered quantities (B_arm(k) = AUC100k(s1) - AUC100k(s0)):

  PRIMARY-1  mean of [B_task - B_apt]  (the interaction; CI > 0 fires)
  PRIMARY-2  mean of [B_sh - B_task]   (shuffle collapse; CI < 0 fires)
  S1         B_apt simple effect       (prediction: null - 4th
                                        reward-free null, engineered)
  Both primaries must fire for "third domain" status; the registered
  branch map covers all four outcomes.

Usage:
  python -m analysis.synth_phaseb_read --auc <auc.csv> \
      [--runroot <snapshot runroot for config audit>] --output <dir>
  python -m analysis.synth_phaseb_read --selfcheck
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
ARMS = ("", "f", "sh")  # id infixes: task, apt, shuffle
ARM_FLAGS = {  # expl.mode, reward_grad, repval_loss, repval_grad
    "": ("task", True, True, True),
    "f": ("apt", None, None, None),  # reward-free: only mode audited
    "sh": ("task", True, True, True),
}


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


def load_b_arm(auc_csv, infix, seeds=SEEDS, domain="synth"):
  s0, s1 = {}, {}
  for r in csv.DictReader(open(auc_csv)):
    if r["domain"] != domain or int(r["qc_pass"]) != 1:
      continue
    if r["mode"] == f"ax1{infix}q1s0":
      s0[int(r["seed"])] = float(r["auc100k"])
    elif r["mode"] == f"ax1{infix}q1s1":
      s1[int(r["seed"])] = float(r["auc100k"])
  missing = [s for s in seeds if s not in s0 or s not in s1]
  if missing:
    raise SystemExit(f"arm ax1{infix}: missing seeds {missing} "
                     f"(have s0={sorted(s0)}, s1={sorted(s1)})")
  return np.array([s1[s] - s0[s] for s in seeds])


def audit(runroot):
  """Saved WM config flags per arm; ADAPT_DONE present."""
  import yaml
  problems, n_ok = [], 0
  for infix in ARMS:
    exp_mode, exp_rg, exp_rl, exp_rvg = ARM_FLAGS[infix]
    for side in (0, 1):
      for seed in SEEDS:
        wm = os.path.join(runroot, f"ax1wm_synth_{infix}q1s{side}_seed{seed}")
        run = os.path.join(
            runroot, f"adapt_ax1{infix}q1s{side}_synth_seed{seed}_ckpt500000")
        tag = os.path.basename(run)
        if not os.path.isdir(wm):
          problems.append(f"{tag}: missing WM dir"); continue
        cfg = yaml.safe_load(open(os.path.join(wm, "config.yaml")))["agent"]
        if cfg["expl"]["mode"] != exp_mode:
          problems.append(
              f"{tag}: expl.mode {cfg['expl']['mode']} != {exp_mode}")
          continue
        if exp_rg is not None:
          got = (bool(cfg["reward_grad"]), bool(cfg["repval_loss"]),
                 bool(cfg["repval_grad"]))
          if got != (exp_rg, exp_rl, exp_rvg):
            problems.append(f"{tag}: flags {got}"); continue
        if not os.path.isfile(os.path.join(run, "ADAPT_DONE")):
          problems.append(f"{tag}: no ADAPT_DONE"); continue
        n_ok += 1
  return n_ok, problems


def run_read(auc_csv, runroot, outdir):
  os.makedirs(outdir, exist_ok=True)
  out = {"auc_csv": auc_csv, "seeds": SEEDS}

  if runroot:
    n_ok, problems = audit(runroot)
    out["audit"] = dict(n_ok=n_ok, problems=problems)
    print(f"AUDIT: {n_ok}/48 runs pass; problems: {len(problems)}")
    for p in problems[:20]:
      print("  ", p)

  b = {infix: load_b_arm(auc_csv, infix) for infix in ARMS}
  out["b_arm"] = {("task" if k == "" else "apt" if k == "f" else "sh"):
                  v.tolist() for k, v in b.items()}

  out["primary1_interaction"] = contrast(b[""] - b["f"], decision=True)
  out["primary2_shuffle_collapse"] = contrast(b["sh"] - b[""], decision=True)
  out["s1_apt_simple"] = contrast(b["f"])
  out["task_simple"] = contrast(b[""])
  out["arm_benefit_means"] = dict(
      task=float(b[""].mean()), apt=float(b["f"].mean()),
      sh=float(b["sh"].mean()))

  p1 = out["primary1_interaction"]
  p2 = out["primary2_shuffle_collapse"]
  p1_fires = p1["decision"] == "CI excludes 0" and p1["mean"] > 0
  p2_fires = p2["decision"] == "CI excludes 0" and p2["mean"] < 0
  if p1_fires and p2_fires:
    branch = ("BOTH primaries fire: interaction + binding collapse "
              "replicate in the engineered domain - synth becomes Paper "
              "1's third domain (ground-truth version of the law); "
              "Phase C authorized for scoping.")
  elif p1_fires:
    branch = ("Interaction fires, shuffle does NOT collapse: in the "
              "engineered domain density suffices - boundary condition "
              "against the DMC binding result; report as-is, Phase C "
              "redesigns around the binding factor.")
  elif p2_fires:
    branch = ("Interaction null but shuffle collapses: no occupancy x "
              "supervision interaction here (occ may not gate transfer "
              "in synth) - boundary condition; Phase C only after "
              "diagnosing which DMC ingredient synth lacks.")
  else:
    branch = ("Neither fires: the law does not reproduce under "
              "engineered ground truth - major boundary condition, "
              "informative and cheap; no Phase C without redesign.")
  out["registered_branch"] = branch

  json.dump(out, open(os.path.join(outdir, "synth_phaseb_read.json"), "w"),
            indent=1)

  def line(name, c):
    extra = f" -> {c['decision']}" if "decision" in c else ""
    print(f"{name:<30} {c['mean']:+9.2f} "
          f"CI=[{c['ci'][0]:+8.2f},{c['ci'][1]:+8.2f}]"
          f" pos={c['pos']}/{c['n']}"
          f" perm_p={c['sensitivity']['perm_p']:.4f}{extra}")

  print("\narm benefit means:",
        {k: round(v, 1) for k, v in out["arm_benefit_means"].items()})
  line("PRIMARY-1  B_task - B_apt", out["primary1_interaction"])
  line("PRIMARY-2  B_sh - B_task ", out["primary2_shuffle_collapse"])
  line("S1         B_apt simple  ", out["s1_apt_simple"])
  line("desc       B_task simple ", out["task_simple"])
  print("\nREGISTERED BRANCH:", branch)
  return out


def selfcheck():
  """Synthetic tables; verifies the both-fire and neither-fire branches."""
  import tempfile

  def make_csv(path, eff, rng):
    with open(path, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["run_id", "mode", "domain", "seed", "milestone", "auc50k",
                  "n_ep_50k", "auc100k", "n_ep_100k", "auc125k", "n_ep_125k",
                  "final10", "qc_pass"])
      for infix, e in eff.items():
        for seed in SEEDS:
          base = 100 + rng.normal(0, 5)
          for side, val in ((0, base), (1, base + e + rng.normal(0, 8))):
            rid = f"adapt_ax1{infix}q1s{side}_synth_seed{seed}_ckpt500000"
            w.writerow([rid, f"ax1{infix}q1s{side}", "synth", seed, 500000,
                        0, 48, f"{val:.4f}", 96, 0, 112, 0, 1])

  tmp = tempfile.mkdtemp()
  rng = np.random.default_rng(7)
  path_a = os.path.join(tmp, "auc_fire.csv")
  make_csv(path_a, {"": 120.0, "f": 0.0, "sh": 10.0}, rng)
  out = run_read(path_a, None, os.path.join(tmp, "a"))
  assert "BOTH primaries fire" in out["registered_branch"], \
      out["registered_branch"]
  path_b = os.path.join(tmp, "auc_null.csv")
  make_csv(path_b, {"": 0.0, "f": 0.0, "sh": 0.0}, rng)
  out = run_read(path_b, None, os.path.join(tmp, "b"))
  assert "Neither fires" in out["registered_branch"]
  print("\nSELFCHECK PASS")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--runroot", default=None)
  ap.add_argument("--output", default="analysis_out/synth_phaseb")
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
