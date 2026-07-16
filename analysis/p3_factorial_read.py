"""P3 gradient-path factorial read (PREREG_p3_gradient_path_20260714.md).

Frozen BEFORE any sgb/rgo/vgo/sh/rl outcome exists (2026-07-16; the runs
were submitted 14 Jul and no factorial snapshot has been pulled or read).
Machinery identical to the W0/W123 reads: AUC100k rows from the frozen
``analysis/adaptation_auc.py`` pipeline, cluster-bootstrap percentile 95%
CI (B=10,000, numpy default_rng seed 0), decisions on the CI alone,
sensitivity suite (paired t, exact sign-flip permutation, Wilcoxon, d_z,
LOO) robustness-only.

Registered quantities (B_arm(k) = AUC100k(s1) - AUC100k(s0), seed k):
  PRIMARY      mean over seeds 1-8 of [B_rgo - B_sgb]     (decision CI)
  S1           mean of [B_vgo - B_sgb]
  S2           mean of [B_sh  - B_full(true)]             (binding necessity)
  S3           mean of [B_rl  - B_full(true)]
  S4 (placebo) mean of [B_sgb - B_apt]                    (estimation only)
  descriptive  additivity: B_full vs B_rgo + B_vgo - B_sgb

B_full(true) and B_apt seeds 1-8 come from the frozen paired JSONs of the
10-Jul and P0 reads (bit-checked records), as disclosed in the prereg.

AUDIT CLARIFICATION (pre-outcome, 2026-07-16): the prereg's audit sentence
expects "loss lines containing both rew and repval in every task arm".
``probing/offline_fit.py`` computes repval in every task-mode fit
(dreamerv3/agent.py:370, gated only by ``repval_loss`` and
``reward_free``) but deliberately EXCLUDES it from printed update lines
and the end-of-fit summary (NON_WM_LOSS filter). The repval component of
the audit is therefore verified from the saved config flags
(``repval_loss`` recorded per run), and stdout verifies ``rew`` presence
plus the absence of AC keys only. Flag expectations per arm:

  arm   expl.mode  reward_grad  repval_loss  repval_grad
  sgb   task       False        True         False
  rgo   task       True         True         False
  vgo   task       False        True         True
  sh    task       True         True         True   (full objective)
  rl    task       True         True         True   (full objective)

Usage:
  python -m analysis.p3_factorial_read --auc <auc.csv> \
      [--runroot <snapshot runroot for config/log audit>] --output <dir>
  python -m analysis.p3_factorial_read --selfcheck
"""

import argparse
import csv
import itertools
import json
import os
import re

import numpy as np
from scipy import stats

B, RNG_SEED = 10_000, 0
SEEDS = list(range(1, 9))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_JSON = os.path.join(
    REPO, "artifacts/phase6_axis1_20260710/paired/"
    "paired_finger_q1_auc100k_s1_minus_s0.json")
APT_JSON = os.path.join(
    REPO, "artifacts/p0_axis1_corrective_20260713/paired/"
    "paired_finger_q1_auc100k_s1_minus_s0.json")
ARMS = ("sgb", "rgo", "vgo", "sh", "rl")
ARM_FLAGS = {  # expl.mode, reward_grad, repval_loss, repval_grad
    "sgb": ("task", False, True, False),
    "rgo": ("task", True, True, False),
    "vgo": ("task", False, True, True),
    "sh": ("task", True, True, True),
    "rl": ("task", True, True, True),
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


def load_b_arm(auc_csv, code, seeds=SEEDS, domain="finger"):
  """Per-seed B_arm = AUC100k(s1) - AUC100k(s0) for one factorial arm."""
  s0, s1 = {}, {}
  for r in csv.DictReader(open(auc_csv)):
    if r["domain"] != domain or int(r["qc_pass"]) != 1:
      continue
    if r["mode"] == f"ax1{code}q1s0":
      s0[int(r["seed"])] = float(r["auc100k"])
    elif r["mode"] == f"ax1{code}q1s1":
      s1[int(r["seed"])] = float(r["auc100k"])
  missing = [s for s in seeds if s not in s0 or s not in s1]
  if missing:
    raise SystemExit(f"arm {code}: missing seeds {missing} "
                     f"(have s0={sorted(s0)}, s1={sorted(s1)})")
  return np.array([s1[s] - s0[s] for s in seeds])


def audit(runroot):
  """Per-run audit: saved WM config flags per arm table; fit-stage stdout
  has rew and no AC keys (repval verified via flags; see module docstring)."""
  import yaml
  problems, n_ok = [], 0
  for code in ARMS:
    exp_mode, exp_rg, exp_rl, exp_rvg = ARM_FLAGS[code]
    for side in (0, 1):
      for seed in SEEDS:
        wm = os.path.join(runroot, f"ax1wm_finger_{code}q1s{side}_seed{seed}")
        run = os.path.join(
            runroot, f"adapt_ax1{code}q1s{side}_finger_seed{seed}_ckpt500000")
        tag = os.path.basename(run)
        if not os.path.isdir(wm):
          problems.append(f"{tag}: missing WM dir"); continue
        cfg = yaml.safe_load(open(os.path.join(wm, "config.yaml")))["agent"]
        got = (cfg["expl"]["mode"], bool(cfg["reward_grad"]),
               bool(cfg["repval_loss"]), bool(cfg["repval_grad"]))
        if got != (exp_mode, exp_rg, exp_rl, exp_rvg):
          problems.append(f"{tag}: flags {got} != {(exp_mode, exp_rg, exp_rl, exp_rvg)}")
          continue
        if not os.path.isfile(os.path.join(run, "ADAPT_DONE")):
          problems.append(f"{tag}: no ADAPT_DONE"); continue
        log = os.path.join(runroot, "_cloud_logs", f"{tag}.out")
        if os.path.isfile(log):
          txt = open(log, errors="replace").read()
          am = re.search(r"\] adapt ", txt)
          fit_txt = txt[:am.start()] if am else txt
          keys = set()
          for line in re.findall(r"update\s+\d+/500000:([^\n]*)", fit_txt):
            keys |= set(re.findall(r"([a-z_2]+)=", line))
          if keys and "rew" not in keys:
            problems.append(f"{tag}: fit stdout lacks rew"); continue
          if keys & {"policy", "value", "repval", "actor", "critic"}:
            problems.append(f"{tag}: AC keys in fit stdout"); continue
        n_ok += 1
  return n_ok, problems


def run_read(auc_csv, runroot, outdir):
  os.makedirs(outdir, exist_ok=True)
  out = {"auc_csv": auc_csv, "seeds": SEEDS}

  if runroot:
    n_ok, problems = audit(runroot)
    out["audit"] = dict(n_ok=n_ok, problems=problems)
    print(f"AUDIT: {n_ok}/80 runs pass; problems: {len(problems)}")
    for p in problems[:20]:
      print("  ", p)

  b = {code: load_b_arm(auc_csv, code) for code in ARMS}
  full = np.asarray(json.load(open(FULL_JSON))["deltas"], float)
  apt = np.asarray(json.load(open(APT_JSON))["deltas"], float)
  assert len(full) == 8 and len(apt) == 8
  out["b_arm"] = {k: v.tolist() for k, v in b.items()}
  out["b_full_true"], out["b_apt"] = full.tolist(), apt.tolist()

  out["primary_rgo_minus_sgb"] = contrast(b["rgo"] - b["sgb"], decision=True)
  out["s1_vgo_minus_sgb"] = contrast(b["vgo"] - b["sgb"])
  out["s2_shuffle_minus_true"] = contrast(b["sh"] - full)
  out["s3_relocate_minus_true"] = contrast(b["rl"] - full)
  out["s4_placebo_sgb_minus_apt"] = contrast(b["sgb"] - apt)
  combo = b["rgo"] + b["vgo"] - b["sgb"]
  out["descriptive_additivity"] = dict(
      b_full_mean=float(full.mean()), combo_mean=float(combo.mean()),
      full_minus_combo=contrast(full - combo))
  out["arm_benefit_means"] = {k: float(v.mean()) for k, v in b.items()}
  out["arm_benefit_means"]["full_true"] = float(full.mean())
  out["arm_benefit_means"]["apt"] = float(apt.mean())

  # registered interpretation map
  p_fires = out["primary_rgo_minus_sgb"]["decision"] == "CI excludes 0" and \
      out["primary_rgo_minus_sgb"]["mean"] > 0
  s1lo, s1hi = out["s1_vgo_minus_sgb"]["ci"]
  s1_fires = (s1lo > 0) == (s1hi > 0) and out["s1_vgo_minus_sgb"]["mean"] > 0
  s2lo, s2hi = out["s2_shuffle_minus_true"]["ci"]
  s2_collapse = s2hi < 0
  if p_fires and s2_collapse:
    branch = ("PRIMARY fires + shuffle collapses: mechanism = reward-gradient "
              "hidden curriculum on reward-dense data (P3 core claim)")
  elif not p_fires and s1_fires:
    branch = "PRIMARY null, S1 fires: carrier is the value path"
  elif not p_fires and not s1_fires:
    branch = ("PRIMARY and S1 null: interaction needs both paths (or "
              "interplay) - reported as-is (check full-arm reproduction)")
  else:
    branch = "PRIMARY fires, shuffle does NOT collapse: density-not-binding"
  out["registered_branch"] = branch

  json.dump(out, open(os.path.join(outdir, "p3_read.json"), "w"), indent=1)

  def line(name, c):
    extra = f" -> {c['decision']}" if "decision" in c else ""
    print(f"{name:<28} {c['mean']:+9.2f} CI=[{c['ci'][0]:+8.2f},{c['ci'][1]:+8.2f}]"
          f" pos={c['pos']}/{c['n']} perm_p={c['sensitivity']['perm_p']:.4f}{extra}")

  print("\narm benefit means:", {k: round(v, 1) for k, v in out["arm_benefit_means"].items()})
  line("PRIMARY  B_rgo - B_sgb", out["primary_rgo_minus_sgb"])
  line("S1       B_vgo - B_sgb", out["s1_vgo_minus_sgb"])
  line("S2       B_sh  - B_full", out["s2_shuffle_minus_true"])
  line("S3       B_rl  - B_full", out["s3_relocate_minus_true"])
  line("S4       B_sgb - B_apt ", out["s4_placebo_sgb_minus_apt"])
  a = out["descriptive_additivity"]
  print(f"additivity: B_full {a['b_full_mean']:+.1f} vs combo {a['combo_mean']:+.1f}"
        f" (full-combo CI [{a['full_minus_combo']['ci'][0]:+.1f},"
        f"{a['full_minus_combo']['ci'][1]:+.1f}])")
  print("\nREGISTERED BRANCH:", branch)
  return out


def selfcheck():
  """Synthetic AUC table with known structure; verifies recovery."""
  import tempfile
  rng = np.random.default_rng(7)
  tmp = tempfile.mkdtemp()
  path = os.path.join(tmp, "auc.csv")
  # construct: rgo benefit ~ +120, sgb ~ 0, vgo ~ +10, sh ~ +20, rl ~ +5
  eff = dict(sgb=0.0, rgo=120.0, vgo=10.0, sh=20.0, rl=5.0)
  with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "mode", "domain", "seed", "milestone", "auc50k",
                "n_ep_50k", "auc100k", "n_ep_100k", "auc125k", "n_ep_125k",
                "final10", "qc_pass"])
    for code, e in eff.items():
      for seed in SEEDS:
        base = 100 + rng.normal(0, 5)
        for side, val in ((0, base), (1, base + e + rng.normal(0, 8))):
          rid = f"adapt_ax1{code}q1s{side}_finger_seed{seed}_ckpt500000"
          w.writerow([rid, f"ax1{code}q1s{side}", "finger", seed, 500000,
                      0, 48, f"{val:.4f}", 96, 0, 112, 0, 1])
  out = run_read(path, None, tmp)
  p = out["primary_rgo_minus_sgb"]
  assert p["decision"] == "CI excludes 0" and 100 < p["mean"] < 140, p
  assert abs(out["s4_placebo_sgb_minus_apt"]["mean"]
             - (0 - np.mean(out["b_apt"]))) < 25
  s2 = out["s2_shuffle_minus_true"]  # 20 - full(+157) => strongly negative
  assert s2["mean"] < -100 and s2["ci"][1] < 0, s2
  print("\nSELFCHECK PASS")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--runroot", default=None)
  ap.add_argument("--output", default="analysis_out/p3")
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
