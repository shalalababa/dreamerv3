"""A2 REWARD-INVISIBILITY CERTIFICATE (engineering-mode, descriptive).

Certifies that the A2 scarecrow fence (a planted position-modulated
misprice) does not change the reward stream the agent experiences,
while the penalty comparator arms — whose -lambda term IS in the
reward — are detected by the same instrument.

Statistics and tests are PRE-STATED in PRESTATE.md (written before
this script ran). Do not add statistics here.

Inputs : local_results/a2_scarecrow_20260824_184000_light/runroot/
         sc_*_s*/scores.jsonl   (20 runs x 496 episodes)
Outputs: a2_reward_invisibility.json
CPU only, saved data only.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np
from scipy import stats as scipy_stats

BUNDLE = ("/home/rickybao/projects/dreamerv3/local_results/"
          "a2_scarecrow_20260824_184000_light")
RUNROOT = os.path.join(BUNDLE, "runroot")
OUT = os.path.dirname(os.path.abspath(__file__))

ARMS = {
    "ctrl": [76, 77, 78, 79],
    "scare": [80, 81, 82, 83],
    "scare2": [84, 85, 86, 87],
    "pen1": [88, 89, 90, 91],
    "pen2": [92, 93, 94, 95],
}
FINAL_WINDOW = 100          # T1: last 100 episodes
N_EPISODES = 496            # asserted
BOOT = 10000
BOOT_SEED = 20260830


# ---------------------------------------------------------------- load

def run_dir(arm, seed):
  return os.path.join(RUNROOT, f"sc_{arm}_s{seed}")


def load_run(arm, seed):
  d = run_dir(arm, seed)
  steps, scores = [], []
  with open(os.path.join(d, "scores.jsonl")) as f:
    for line in f:
      rec = json.loads(line)
      steps.append(rec["step"])
      scores.append(rec["episode/score"])
  steps = np.asarray(steps, np.float64)
  scores = np.asarray(scores, np.float64)
  assert len(scores) == N_EPISODES, (d, len(scores))
  assert np.all(np.diff(steps) > 0), d
  cfg = open(os.path.join(d, "config.yaml")).read()
  return {"dir": d, "arm": arm, "seed": seed, "steps": steps,
          "scores": scores, "config": cfg}


def arm_pin_check(runs):
  """Assert the reward-channel wiring each arm is supposed to have."""
  pins = {}
  for r in runs:
    cfg, arm = r["config"], r["arm"]
    # regionpenalty is constructed iff penalty.scale != 0 (main.py:275)
    pen_on = ("penalty: {gate_index: 0, gate_key: '', gate_threshold: 0.0, "
              "scale: 0.0}") not in cfg
    pmix = "penalty_mix: true" in cfg
    scare_mod = "mod_key: position" in cfg
    if arm in ("ctrl", "scare", "scare2"):
      assert not pen_on, f"{r['dir']}: penalty wrapper ACTIVE on {arm}"
      assert not pmix, f"{r['dir']}: penalty_mix true on {arm}"
    else:
      assert pen_on, f"{r['dir']}: penalty wrapper INACTIVE on {arm}"
      assert pmix, f"{r['dir']}: penalty_mix false on {arm}"
    if arm == "ctrl":
      assert not scare_mod, f"{r['dir']}: ctrl carries the misprice mod"
    if arm in ("scare", "scare2"):
      assert scare_mod, f"{r['dir']}: {arm} lacks the misprice mod"
    pins[os.path.basename(r["dir"])] = {
        "regionpenalty_wrapper": pen_on, "expl.penalty_mix": pmix,
        "distractor.mod_key_position": scare_mod}
  return pins


# ------------------------------------------------------- test statistics

def t1_final_mean(r):
  return float(r["scores"][-FINAL_WINDOW:].mean())


def t2_auc(r):
  s, y = r["steps"], r["scores"]
  return float(np.trapz(y, s) / (s[-1] - s[0]))


def t3_pool(r):
  return r["scores"][-FINAL_WINDOW:]


def ks_stat(a, b):
  a = np.sort(np.asarray(a)); b = np.sort(np.asarray(b))
  allv = np.concatenate([a, b])
  fa = np.searchsorted(a, allv, "right") / len(a)
  fb = np.searchsorted(b, allv, "right") / len(b)
  return float(np.max(np.abs(fa - fb)))


# ------------------------------------------------------- permutation

_MASK_CACHE = {}


def _masks(n, k):
  key = (n, k)
  if key not in _MASK_CACHE:
    m = np.zeros((math.comb(n, k), n), bool)
    for i, comb in enumerate(itertools.combinations(range(n), k)):
      m[i, list(comb)] = True
    _MASK_CACHE[key] = m
  return _MASK_CACHE[key]


def perm_mean_diff(x_treat, x_ctrl):
  """Exact permutation on difference of means. x_treat first.

  Returns dict with observed diff, two-sided p, one-sided p (in the
  observed direction), n assignments, and the achievable floors.
  """
  x_treat = np.asarray(x_treat, np.float64)
  x_ctrl = np.asarray(x_ctrl, np.float64)
  pool = np.concatenate([x_treat, x_ctrl])
  n_t, n_c = len(x_treat), len(x_ctrl)
  obs = float(x_treat.mean() - x_ctrl.mean())
  m = _masks(len(pool), n_t)
  sums = m @ pool
  total = pool.sum()
  diffs = sums / n_t - (total - sums) / n_c
  n_perm = len(diffs)
  p_two = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
  if obs >= 0:
    p_one = float(np.mean(diffs >= obs - 1e-12))
  else:
    p_one = float(np.mean(diffs <= obs + 1e-12))
  return {"observed_diff": obs, "p_two_sided": p_two,
          "p_one_sided_observed_direction": p_one,
          "n_assignments": n_perm,
          "floor_two_sided": 2.0 / n_perm,
          "floor_one_sided": 1.0 / n_perm,
          "at_floor_two_sided": bool(abs(p_two - 2.0 / n_perm) < 1e-12),
          "perm_null_sd": float(diffs.std(ddof=0))}


def perm_ks(pool_treat_runs, pool_ctrl_runs):
  """Run-level exact permutation on the pooled-episode KS statistic."""
  runs = list(pool_treat_runs) + list(pool_ctrl_runs)
  n_t = len(pool_treat_runs)
  obs = ks_stat(np.concatenate(list(pool_treat_runs)),
                np.concatenate(list(pool_ctrl_runs)))
  stats = []
  for comb in itertools.combinations(range(len(runs)), n_t):
    m = set(comb)
    a = np.concatenate([runs[i] for i in range(len(runs)) if i in m])
    b = np.concatenate([runs[i] for i in range(len(runs)) if i not in m])
    stats.append(ks_stat(a, b))
  stats = np.asarray(stats)
  return {"observed_KS_D": obs,
          "p_permutation": float(np.mean(stats >= obs - 1e-12)),
          "n_assignments": int(len(stats)),
          "floor": 1.0 / len(stats)}


def perm_invert_ci(x_treat, x_ctrl, alpha=0.05, grid=4001):
  """95% CI by inverting the two-sided exact permutation test.

  The CI is {delta : two-sided p for (x_treat - delta) vs x_ctrl > alpha}.
  With C(8,4)=70 assignments the achievable two-sided levels are
  k/70; alpha=0.05 rejects only at p = 2/70 = .0286, so this interval
  is CONSERVATIVE (true coverage >= 95%).
  """
  x_treat = np.asarray(x_treat, np.float64)
  x_ctrl = np.asarray(x_ctrl, np.float64)
  obs = float(x_treat.mean() - x_ctrl.mean())
  span = max(np.ptp(np.concatenate([x_treat, x_ctrl])), 1e-9)
  lo_g, hi_g = obs - 6 * span, obs + 6 * span
  deltas = np.linspace(lo_g, hi_g, grid)
  keep = []
  for d in deltas:
    r = perm_mean_diff(x_treat - d, x_ctrl)
    if r["p_two_sided"] > alpha:
      keep.append(d)
  if not keep:
    return {"lo": None, "hi": None, "alpha": alpha,
            "note": "empty at this alpha"}
  return {"lo": float(min(keep)), "hi": float(max(keep)), "alpha": alpha,
          "achievable_levels": "k/C(n,k); conservative"}


def welch_ci(x_treat, x_ctrl, alpha=0.05):
  a = np.asarray(x_treat, np.float64); b = np.asarray(x_ctrl, np.float64)
  na, nb = len(a), len(b)
  va, vb = a.var(ddof=1), b.var(ddof=1)
  se = math.sqrt(va / na + vb / nb)
  diff = float(a.mean() - b.mean())
  if se == 0:
    return {"diff": diff, "lo": diff, "hi": diff, "df": None, "se": 0.0}
  df = (va / na + vb / nb) ** 2 / (
      (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
  tcrit = float(scipy_stats.t.ppf(1 - alpha / 2, df))
  return {"diff": diff, "lo": diff - tcrit * se, "hi": diff + tcrit * se,
          "df": float(df), "se": float(se), "t": float(diff / se),
          "t_crit_0975": float(tcrit)}


def boot_ci(x_treat, x_ctrl, alpha=0.05, n=BOOT, seed=BOOT_SEED):
  rng = np.random.default_rng(seed)
  a = np.asarray(x_treat, np.float64); b = np.asarray(x_ctrl, np.float64)
  d = np.empty(n)
  for i in range(n):
    d[i] = (rng.choice(a, len(a), True).mean()
            - rng.choice(b, len(b), True).mean())
  return {"lo": float(np.quantile(d, alpha / 2)),
          "hi": float(np.quantile(d, 1 - alpha / 2)),
          "n_draws": n, "seed": seed}


# ------------------------------------------------------------------ main

def main():
  runs = {}
  for arm, seeds in ARMS.items():
    runs[arm] = [load_run(arm, s) for s in seeds]
  allruns = [r for a in ARMS for r in runs[a]]
  pins = arm_pin_check(allruns)

  # (a) per-run distributions
  per_run = {}
  for arm in ARMS:
    for r in runs[arm]:
      s = r["scores"]; fw = s[-FINAL_WINDOW:]
      per_run[os.path.basename(r["dir"])] = {
          "arm": arm, "seed": r["seed"], "n_episodes": int(len(s)),
          "step_first": float(r["steps"][0]),
          "step_last": float(r["steps"][-1]),
          "T1_final100_mean": t1_final_mean(r),
          "T2_auc_norm": t2_auc(r),
          "all_mean": float(s.mean()), "all_sd": float(s.std(ddof=1)),
          "all_median": float(np.median(s)),
          "all_min": float(s.min()), "all_max": float(s.max()),
          "final100_sd": float(fw.std(ddof=1)),
          "final100_median": float(np.median(fw)),
          "final100_q10": float(np.quantile(fw, 0.10)),
          "final100_q90": float(np.quantile(fw, 0.90)),
      }

  arm_stats = {}
  for arm in ARMS:
    t1 = [t1_final_mean(r) for r in runs[arm]]
    t2 = [t2_auc(r) for r in runs[arm]]
    arm_stats[arm] = {
        "T1_per_run": t1, "T1_mean": float(np.mean(t1)),
        "T1_sd": float(np.std(t1, ddof=1)),
        "T2_per_run": t2, "T2_mean": float(np.mean(t2)),
        "T2_sd": float(np.std(t2, ddof=1)),
    }

  # (b)+(c) contrasts
  def T1(arm): return [t1_final_mean(r) for r in runs[arm]]
  def T2(arm): return [t2_auc(r) for r in runs[arm]]
  def P3(arm): return [t3_pool(r) for r in runs[arm]]

  contrasts = {}
  pairs = [("scare_vs_ctrl", "scare", "ctrl", "INDISTINGUISHABLE"),
           ("scare2_vs_ctrl", "scare2", "ctrl", "INDISTINGUISHABLE"),
           ("pen1_vs_ctrl", "pen1", "ctrl", "DISTINGUISHABLE"),
           ("pen2_vs_ctrl", "pen2", "ctrl", "DISTINGUISHABLE"),
           ("scare2_vs_scare", "scare2", "scare", "descriptive")]
  for name, a, b, expect in pairs:
    contrasts[name] = {
        "prestated_expectation": expect,
        "T1_final100_mean": perm_mean_diff(T1(a), T1(b)),
        "T2_auc": perm_mean_diff(T2(a), T2(b)),
        "T3_pooled_KS": perm_ks(P3(a), P3(b)),
    }

  # supplementary pooled 8v4
  pooled_t1 = T1("scare") + T1("scare2")
  pooled_t2 = T2("scare") + T2("scare2")
  pooled_p3 = P3("scare") + P3("scare2")
  contrasts["scarepooled_vs_ctrl"] = {
      "prestated_expectation": "INDISTINGUISHABLE (pooled 8v4)",
      "T1_final100_mean": perm_mean_diff(pooled_t1, T1("ctrl")),
      "T2_auc": perm_mean_diff(pooled_t2, T2("ctrl")),
      "T3_pooled_KS": perm_ks(pooled_p3, P3("ctrl")),
  }

  # (d) effect-size bounds
  bounds = {}
  for name, a in [("scare_vs_ctrl", "scare"), ("scare2_vs_ctrl", "scare2")]:
    bounds[name] = {}
    for lab, fn in [("T1_final100_mean", T1), ("T2_auc", T2)]:
      bounds[name][lab] = {
          "diff": float(np.mean(fn(a)) - np.mean(fn("ctrl"))),
          "perm_inverted_CI95": perm_invert_ci(fn(a), fn("ctrl")),
          "welch_CI95": welch_ci(fn(a), fn("ctrl")),
          "bootstrap_CI95": boot_ci(fn(a), fn("ctrl")),
      }
  bounds["scarepooled_vs_ctrl"] = {}
  for lab, xs in [("T1_final100_mean", pooled_t1), ("T2_auc", pooled_t2)]:
    ctrl_x = T1("ctrl") if lab.startswith("T1") else T2("ctrl")
    bounds["scarepooled_vs_ctrl"][lab] = {
        "diff": float(np.mean(xs) - np.mean(ctrl_x)),
        "perm_inverted_CI95": perm_invert_ci(xs, ctrl_x),
        "welch_CI95": welch_ci(xs, ctrl_x),
        "bootstrap_CI95": boot_ci(xs, ctrl_x),
    }

  # footprint ratio vs the penalty arms
  footprint = {}
  for lab, fn in [("T1_final100_mean", T1), ("T2_auc", T2)]:
    d_pen1 = abs(np.mean(fn("pen1")) - np.mean(fn("ctrl")))
    d_pen2 = abs(np.mean(fn("pen2")) - np.mean(fn("ctrl")))
    ent = {"abs_diff_pen1_vs_ctrl": float(d_pen1),
           "abs_diff_pen2_vs_ctrl": float(d_pen2)}
    for name, xs in [("scare", fn("scare")), ("scare2", fn("scare2")),
                     ("scare_pooled",
                      (pooled_t1 if lab.startswith("T1") else pooled_t2))]:
      ctrl_x = fn("ctrl")
      ci = perm_invert_ci(xs, ctrl_x)
      wc = welch_ci(xs, ctrl_x)
      widest = max(abs(ci["lo"]), abs(ci["hi"]))
      ent[name] = {
          "abs_diff": float(abs(np.mean(xs) - np.mean(ctrl_x))),
          "perm_CI_abs_max": float(widest),
          "welch_CI_abs_max": float(max(abs(wc["lo"]), abs(wc["hi"]))),
          "pct_of_pen1_footprint_point": float(
              100 * abs(np.mean(xs) - np.mean(ctrl_x)) / d_pen1),
          "pct_of_pen2_footprint_point": float(
              100 * abs(np.mean(xs) - np.mean(ctrl_x)) / d_pen2),
          "pct_of_pen1_footprint_permCIbound": float(100 * widest / d_pen1),
          "pct_of_pen2_footprint_permCIbound": float(100 * widest / d_pen2),
      }
    footprint[lab] = ent

  # Holm over the 9 invisibility tests (3 contrasts x 3 pre-stated
  # statistics). Derived from the pre-stated tests; no new statistic.
  inv = []
  for c in ("scare_vs_ctrl", "scare2_vs_ctrl", "scarepooled_vs_ctrl"):
    inv.append((f"{c}:T1", contrasts[c]["T1_final100_mean"]["p_two_sided"]))
    inv.append((f"{c}:T2", contrasts[c]["T2_auc"]["p_two_sided"]))
    inv.append((f"{c}:T3_KS", contrasts[c]["T3_pooled_KS"]["p_permutation"]))
  inv.sort(key=lambda kv: kv[1])
  m, run_max, holm = len(inv), 0.0, {}
  for i, (k, p) in enumerate(inv):
    run_max = max(run_max, min(1.0, p * (m - i)))
    holm[k] = {"p_raw": p, "p_holm": run_max}
  multiplicity = {
      "family": "9 invisibility tests (scare, scare2, pooled) x (T1,T2,T3)",
      "method": "Holm step-down on two-sided permutation p",
      "holm": holm,
      "min_p_holm": min(v["p_holm"] for v in holm.values()),
      "any_significant_at_0.05": bool(
          min(v["p_holm"] for v in holm.values()) < 0.05),
  }

  # occupancy -> return conversion (occupancy Deltas from the registered
  # read artifacts/a2_scarecrow_read_20260824/RESULTS.md, quoted)
  OCC = {"scare": -0.0632, "scare2": -0.0736, "pen1": -0.479, "pen2": -0.495}
  conversion = {}
  for arm in ("scare", "scare2"):
    d1 = np.mean(T1(arm)) - np.mean(T1("ctrl"))
    d2 = np.mean(T2(arm)) - np.mean(T2("ctrl"))
    conversion[arm] = {
        "occupancy_delta_quoted": OCC[arm],
        "T1_delta": float(d1), "T2_delta": float(d2),
        "T1_delta_pct_of_ctrl": float(100 * d1 / np.mean(T1("ctrl"))),
        "T2_delta_pct_of_ctrl": float(100 * d2 / np.mean(T2("ctrl"))),
        "T1_return_pts_per_pp_occupancy": float(d1 / (100 * OCC[arm])),
    }

  # ---- EXPLORATORY, NOT IN THE PRE-STATED PLAN, BEARS ON NO VERDICT ----
  # Where in training does the scare-vs-ctrl return gap live? Reported
  # as a diagnostic only; the certificate verdict uses T1/T2/T3 above.
  quarts = {}
  for arm in ARMS:
    qs = []
    for r in runs[arm]:
      s = r["scores"]
      qs.append([float(s[i * 124:(i + 1) * 124].mean()) for i in range(4)])
    qs = np.asarray(qs)
    quarts[arm] = {"per_run_quartile_means": qs.tolist(),
                   "arm_quartile_means": qs.mean(0).tolist()}
  gap = {}
  for arm in ("scare", "scare2"):
    d = (np.asarray(quarts[arm]["arm_quartile_means"])
         - np.asarray(quarts["ctrl"]["arm_quartile_means"]))
    perq = []
    for q in range(4):
      a = [quarts[arm]["per_run_quartile_means"][i][q] for i in range(4)]
      c = [quarts["ctrl"]["per_run_quartile_means"][i][q] for i in range(4)]
      perq.append(perm_mean_diff(a, c))
    gap[f"{arm}_minus_ctrl_by_quartile"] = {
        "diffs": d.tolist(),
        "p_two_sided": [x["p_two_sided"] for x in perq]}
  exploratory = {
      "WARNING": ("post-hoc diagnostic, NOT in PRESTATE.md, does not "
                  "enter the certificate verdict"),
      "quartile_means_124_episodes_each": quarts,
      "gap_over_training": gap,
  }

  out = {
      "artifact": "a2_reward_invisibility_20260830",
      "mode": "engineering / descriptive (NOT a registered primary)",
      "prestate": "PRESTATE.md (written 2026-08-30 13:18 CST before compute)",
      "bundle": BUNDLE,
      "manifest": ("manifests/a2_scarecrow_20260824_184000_light.sha256 "
                   "— 40/40 scores.jsonl+config.yaml verified, 0 mismatch"),
      "n_runs": len(allruns),
      "final_window_episodes": FINAL_WINDOW,
      "arm_pins": pins,
      "per_run": per_run,
      "arm_stats": arm_stats,
      "contrasts": contrasts,
      "effect_size_bounds": bounds,
      "footprint_ratio": footprint,
      "multiplicity": multiplicity,
      "occupancy_to_return_conversion": conversion,
      "exploratory_not_in_prestate": exploratory,
      "python": sys.version.split()[0],
      "numpy": np.__version__,
  }
  path = os.path.join(OUT, "a2_reward_invisibility.json")
  with open(path, "w") as f:
    json.dump(out, f, indent=2, sort_keys=False)
  print("wrote", path)

  # console summary
  for arm in ARMS:
    a = arm_stats[arm]
    print(f"{arm:8s} T1 {a['T1_mean']:9.3f} +-{a['T1_sd']:7.3f}   "
          f"T2 {a['T2_mean']:9.3f} +-{a['T2_sd']:7.3f}")
  for k, v in contrasts.items():
    print(f"{k:24s} T1 d={v['T1_final100_mean']['observed_diff']:+9.3f} "
          f"p2={v['T1_final100_mean']['p_two_sided']:.4f} | "
          f"T2 d={v['T2_auc']['observed_diff']:+9.3f} "
          f"p2={v['T2_auc']['p_two_sided']:.4f} | "
          f"KS D={v['T3_pooled_KS']['observed_KS_D']:.3f} "
          f"p={v['T3_pooled_KS']['p_permutation']:.4f}")


if __name__ == "__main__":
  main()
