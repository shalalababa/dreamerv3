"""Synthetic-reward-stamping read (PREREG_stamping_20260717.md).

Frozen BEFORE any srd0/srd1/sid outcome exists (2026-07-17; the stamped
buffers may exist but no fit or adapt has been run on them). Machinery
identical to the frozen P3 read (`analysis/p3_factorial_read.py`):
AUC100k rows from the untouched ``analysis/adaptation_auc.py`` pipeline,
cluster-bootstrap percentile 95% CI (B=10,000, numpy default_rng seed 0),
decisions on the CI alone, sensitivity suite (paired t, exact sign-flip
permutation, Wilcoxon, d_z, LOO) robustness-only.

Arms (finger q1 pair, seeds 1-8, full task objective on relabeled
buffers via AXIS1_TRANSFORM):

  srd0  q1_srd0  frozen random MLP g0(obs), per-side marginal-matched
  srd1  q1_srd1  second function draw g1(obs)
  sid   q1_sid   state-independent exact-count placement (same marginals)

Registered quantities (B_arm(k) = AUC100k(s1) - AUC100k(s0), seed k;
B_srd(k) = mean of B_srd0(k) and B_srd1(k)):

  PRIMARY      mean over seeds 1-8 of [B_srd - B_sid]      (decision CI)
  S1           mean of [B_srd  - B_apt(hist)]   learnable-unaligned vs nothing
  S2           mean of [B_srd  - B_sh(hist)]    vs unlearnable-density-preserving
  S3           mean of [B_srd0 - B_srd1]        function-draw heterogeneity
  descriptive  arm benefit means incl. historical apt/sh/full

Historical records (disclosed-known comparators, never decision-bearing):
B_apt seeds 1-8 from the P0 corrective paired JSON; B_sh seeds 1-8 from
the P3 wave read JSON (both bit-checked artifacts).

Theory hook: P-B1 (PREREG_theory_predictions_20260717.md). The E4-style
mechanism panel (stamp-NLL vs true-NLL; inclusion-without-transfer
signature) is measured separately by ``probing/stratified_error measure
--reward_override`` and is descriptive-only -- never part of this read's
decision.

Usage:
  python -m analysis.stamping_read --auc <auc.csv> \
      [--runroot <snapshot runroot for config audit>] --output <dir>
  python -m analysis.stamping_read --selfcheck
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
APT_JSON = os.path.join(
    REPO, "artifacts/p0_axis1_corrective_20260713/paired/"
    "paired_finger_q1_auc100k_s1_minus_s0.json")
P3_JSON = os.path.join(REPO, "artifacts/p3_wave_20260717/p3_read.json")
ARMS = ("srd0", "srd1", "sid")
# All stamp arms run the FULL task objective on the relabeled buffers
# (expl.mode, reward_grad, repval_loss, repval_grad) -- same audit row as
# the P3 sh/rl transform arms.
FULL_FLAGS = ("task", True, True, True)


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
  """Per-seed B_arm = AUC100k(s1) - AUC100k(s0) for one stamp arm."""
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
  """Saved WM config flags == full-objective row for every stamp run
  (the registered per-run rule, as in the P3 read); ADAPT_DONE present;
  fit stdout (where retained) shows the registered transformed replay
  path (offline_fit prints 'Static replay: ...'; the saved config does
  NOT record it), has rew, and no AC keys."""
  import yaml
  problems, n_ok = [], 0
  for code in ARMS:
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
        if got != FULL_FLAGS:
          problems.append(f"{tag}: flags {got} != {FULL_FLAGS}"); continue
        if not os.path.isfile(os.path.join(run, "ADAPT_DONE")):
          problems.append(f"{tag}: no ADAPT_DONE"); continue
        log = os.path.join(runroot, "_cloud_logs", f"{tag}.out")
        if os.path.isfile(log):
          txt = open(log, errors="replace").read()
          am = re.search(r"\] adapt ", txt)
          fit_txt = txt[:am.start()] if am else txt
          sr = re.search(r"Static replay: (\S+)", fit_txt)
          want = f"q1_{code}/side{side}"
          if sr and want not in sr.group(1):
            problems.append(
                f"{tag}: static replay {sr.group(1)!r} lacks {want}")
            continue
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
    print(f"AUDIT: {n_ok}/48 runs pass; problems: {len(problems)}")
    for p in problems[:20]:
      print("  ", p)

  b = {code: load_b_arm(auc_csv, code) for code in ARMS}
  apt = np.asarray(json.load(open(APT_JSON))["deltas"], float)
  sh = np.asarray(json.load(open(P3_JSON))["b_arm"]["sh"], float)
  assert len(apt) == 8 and len(sh) == 8
  b_srd = (b["srd0"] + b["srd1"]) / 2.0
  out["b_arm"] = {k: v.tolist() for k, v in b.items()}
  out["b_srd_pooled"] = b_srd.tolist()
  out["b_apt_hist"], out["b_sh_hist"] = apt.tolist(), sh.tolist()

  out["primary_srd_minus_sid"] = contrast(b_srd - b["sid"], decision=True)
  out["s1_srd_minus_apt"] = contrast(b_srd - apt)
  out["s2_srd_minus_sh"] = contrast(b_srd - sh)
  out["s3_srd0_minus_srd1"] = contrast(b["srd0"] - b["srd1"])
  out["arm_benefit_means"] = {k: float(v.mean()) for k, v in b.items()}
  out["arm_benefit_means"]["srd_pooled"] = float(b_srd.mean())
  out["arm_benefit_means"]["apt_hist"] = float(apt.mean())
  out["arm_benefit_means"]["sh_hist"] = float(sh.mean())

  # Registered interpretation map (P-B1,
  # PREREG_theory_predictions_20260717.md).
  p = out["primary_srd_minus_sid"]
  fires = p["decision"] == "CI excludes 0"
  if fires and p["mean"] > 0:
    branch = ("PRIMARY fires positive: P-B1 STRONG FORM REFUTED - "
              "learnable-but-unaligned scalars transfer; subspace-"
              "expansion amendment to the theory; stamping becomes a "
              "constructive-method candidate (alignment-graded follow-up "
              "wave).")
  elif fires:
    branch = ("PRIMARY fires NEGATIVE: stamps actively hurt relative to "
              "the unlearnable control - outside both registered "
              "accounts; report as-is and audit before interpreting.")
  else:
    branch = ("PRIMARY includes 0: P-B1 strong form STANDS - "
              "learnable-but-unaligned scalar supervision does not "
              "transfer; Paper-1 discussion gains 'not any scalar - "
              "aligned scalars'. Check S1 (vs apt) descriptively and the "
              "E4 stamp-NLL panel for the inclusion-without-transfer "
              "signature.")
  out["registered_branch"] = branch

  json.dump(out, open(os.path.join(outdir, "stamping_read.json"), "w"),
            indent=1)

  def line(name, c):
    extra = f" -> {c['decision']}" if "decision" in c else ""
    print(f"{name:<28} {c['mean']:+9.2f} CI=[{c['ci'][0]:+8.2f},{c['ci'][1]:+8.2f}]"
          f" pos={c['pos']}/{c['n']} perm_p={c['sensitivity']['perm_p']:.4f}{extra}")

  print("\narm benefit means:",
        {k: round(v, 1) for k, v in out["arm_benefit_means"].items()})
  line("PRIMARY  B_srd - B_sid", out["primary_srd_minus_sid"])
  line("S1       B_srd - B_apt ", out["s1_srd_minus_apt"])
  line("S2       B_srd - B_sh  ", out["s2_srd_minus_sh"])
  line("S3       B_srd0 - B_srd1", out["s3_srd0_minus_srd1"])
  print("\nREGISTERED BRANCH:", branch)
  return out


def selfcheck():
  """Synthetic AUC tables with known structure; verifies both branches."""
  import tempfile

  def make_csv(path, eff, rng):
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

  tmp = tempfile.mkdtemp()
  rng = np.random.default_rng(7)
  # Scenario A: all stamp arms null -> strong form stands.
  path_a = os.path.join(tmp, "auc_null.csv")
  make_csv(path_a, dict(srd0=0.0, srd1=0.0, sid=0.0), rng)
  out = run_read(path_a, None, os.path.join(tmp, "a"))
  assert out["primary_srd_minus_sid"]["decision"] == "CI includes 0"
  assert "STANDS" in out["registered_branch"]
  # Scenario B: srd transfers (+80), sid null -> strong form refuted.
  path_b = os.path.join(tmp, "auc_pos.csv")
  make_csv(path_b, dict(srd0=80.0, srd1=80.0, sid=0.0), rng)
  out = run_read(path_b, None, os.path.join(tmp, "b"))
  p = out["primary_srd_minus_sid"]
  assert p["decision"] == "CI excludes 0" and 60 < p["mean"] < 100, p
  assert "REFUTED" in out["registered_branch"]
  print("\nSELFCHECK PASS")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--auc")
  ap.add_argument("--runroot", default=None)
  ap.add_argument("--output", default="analysis_out/stamping")
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
