"""W0 registered read: fully-paired n=16 interaction (Amendment 2 SA).

PREREG_axis1_corrective_amendment2_20260713.md SA:
- Primary confirmatory: finger Q1 occupancy x reward-supervision interaction,
  per-seed delta_task - delta_apt on AUC100k, seeds 1-16, cluster-bootstrap
  percentile 95% CI (B=10,000, seed 0). Decision on this CI alone.
- Labeled subsidiary: fresh-batch (seeds 9-16 only) interaction, same machinery.
- Sensitivity (robustness only): paired t CI, exact sign-flip permutation,
  Wilcoxon, d_z, leave-one-seed-out range.
- Secondary: reward-free simple effect pooled 1-16 (precision update of P0 null).
"""
import csv
import itertools
import json

import numpy as np
from scipy import stats

SP = "/tmp/claude-1000/-home-rickybao-projects-dreamerv3/1fd435d6-7962-41e3-b5ca-5ffe131ca911/scratchpad/w0_analysis"
ART = "/home/rickybao/projects/dreamerv3/artifacts/p0_axis1_corrective_20260713"
B, SEED = 10_000, 0
SEEDS = list(range(1, 17))


def load_auc(path, mode_s0, mode_s1, domain):
  s0, s1 = {}, {}
  for r in csv.DictReader(open(path)):
    if r["domain"] != domain or int(r["qc_pass"]) != 1:
      continue
    seed = int(r["seed"])
    if r["mode"] == mode_s0:
      s0[seed] = float(r["auc100k"])
    elif r["mode"] == mode_s1:
      s1[seed] = float(r["auc100k"])
  return s0, s1


def deltas_by_seed(s0, s1, seeds):
  assert all(s in s0 and s in s1 for s in seeds), (sorted(s0), sorted(s1))
  return np.array([s1[s] - s0[s] for s in seeds])


def boot_ci(x, b=B, seed=SEED):
  rng = np.random.default_rng(seed)
  boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(b)]
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sensitivity(deltas):
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
  return dict(
      t_ci=[float(ci.low), float(ci.high)], t_p=float(t.pvalue),
      perm_p=p_perm, wilcoxon_p=float(w.pvalue), d_z=d_z,
      loo_range=[float(min(loo)), float(max(loo))])


# --- assemble apt (reward-free) deltas: seeds 1-8 from P0 CSV, 9-16 from W0 ---
apt_s0_18, apt_s1_18 = load_auc(f"{ART}/auc_p0.csv", "ax1fq1s0", "ax1fq1s1", "finger")
apt_s0_916, apt_s1_916 = load_auc(f"{SP}/auc.csv", "ax1fq1s0", "ax1fq1s1", "finger")
apt_s0 = {**apt_s0_18, **apt_s0_916}
apt_s1 = {**apt_s1_18, **apt_s1_916}
apt = deltas_by_seed(apt_s0, apt_s1, SEEDS)

# --- aware (task) deltas seeds 1-16 from pooled CSV ---
aw_s0, aw_s1 = load_auc(f"{ART}/auc_aware_1_16.csv", "ax1q1s0", "ax1q1s1", "finger")
aware = deltas_by_seed(aw_s0, aw_s1, SEEDS)

# --- cross-check seeds 1-8 against the frozen paired JSONs (bit-consistency) ---
p0_json = json.load(open(f"{ART}/paired/paired_finger_q1_auc100k_s1_minus_s0.json"))
assert np.allclose(apt[:8], np.asarray(p0_json["deltas"], float)), "apt 1-8 mismatch"
aw_json = json.load(open(
    "/home/rickybao/projects/dreamerv3/artifacts/phase6_axis1_20260710/paired/"
    "paired_finger_q1_auc100k_s1_minus_s0.json"))
assert np.allclose(aware[:8], np.asarray(aw_json["deltas"], float)), "aware 1-8 mismatch"

inter = aware - apt

out = {"seeds": SEEDS,
       "apt_deltas": apt.tolist(), "aware_deltas": aware.tolist(),
       "interaction_deltas": inter.tolist()}

# PRIMARY: n=16 interaction, decision on bootstrap CI alone
lo, hi = boot_ci(inter)
out["primary_n16_interaction"] = dict(
    mean=float(inter.mean()), ci=[lo, hi], pos=int((inter > 0).sum()),
    decision="CI excludes 0" if (lo > 0) == (hi > 0) else "CI includes 0",
    sensitivity=sensitivity(inter))

# SUBSIDIARY (labeled): fresh-batch seeds 9-16 interaction
fresh = inter[8:]
flo, fhi = boot_ci(fresh)
out["subsidiary_fresh_9_16_interaction"] = dict(
    mean=float(fresh.mean()), ci=[flo, fhi], pos=int((fresh > 0).sum()),
    sensitivity=sensitivity(fresh))

# SECONDARY: reward-free simple effect pooled 1-16 (precision update of P0 null)
alo, ahi = boot_ci(apt)
out["secondary_apt_simple_1_16"] = dict(
    mean=float(apt.mean()), ci=[alo, ahi], pos=int((apt > 0).sum()),
    sensitivity=sensitivity(apt))

# descriptive: aware simple effect n=16, fresh-batch apt simple effect, cell means
slo, shi = boot_ci(aware)
out["descriptive_aware_simple_1_16"] = dict(
    mean=float(aware.mean()), ci=[slo, shi], pos=int((aware > 0).sum()))
a9 = apt[8:]
a9lo, a9hi = boot_ci(a9)
out["descriptive_apt_simple_9_16"] = dict(
    mean=float(a9.mean()), ci=[a9lo, a9hi], pos=int((a9 > 0).sum()))
out["cell_means"] = dict(
    apt_s0_1_8=float(np.mean([apt_s0[s] for s in SEEDS[:8]])),
    apt_s1_1_8=float(np.mean([apt_s1[s] for s in SEEDS[:8]])),
    apt_s0_9_16=float(np.mean([apt_s0[s] for s in SEEDS[8:]])),
    apt_s1_9_16=float(np.mean([apt_s1[s] for s in SEEDS[8:]])),
    aware_s0_1_8=float(np.mean([aw_s0[s] for s in SEEDS[:8]])),
    aware_s1_1_8=float(np.mean([aw_s1[s] for s in SEEDS[:8]])),
    aware_s0_9_16=float(np.mean([aw_s0[s] for s in SEEDS[8:]])),
    aware_s1_9_16=float(np.mean([aw_s1[s] for s in SEEDS[8:]])),
)

json.dump(out, open(f"{SP}/n16_interaction_read.json", "w"), indent=1)

p = out["primary_n16_interaction"]
print(f"PRIMARY  n=16 interaction: {p['mean']:+8.2f} CI=[{p['ci'][0]:+8.2f},{p['ci'][1]:+8.2f}] "
      f"pos={p['pos']}/16  -> {p['decision']}")
s = p["sensitivity"]
print(f"  sens: tCI=[{s['t_ci'][0]:+.1f},{s['t_ci'][1]:+.1f}] perm_p={s['perm_p']:.5f} "
      f"wilcoxon_p={s['wilcoxon_p']:.5f} d_z={s['d_z']:+.2f} "
      f"LOO=[{s['loo_range'][0]:+.1f},{s['loo_range'][1]:+.1f}]")
f = out["subsidiary_fresh_9_16_interaction"]
print(f"SUBSID   9-16 interaction: {f['mean']:+8.2f} CI=[{f['ci'][0]:+8.2f},{f['ci'][1]:+8.2f}] "
      f"pos={f['pos']}/8  perm_p={f['sensitivity']['perm_p']:.4f}")
a = out["secondary_apt_simple_1_16"]
print(f"SECOND   apt simple 1-16 : {a['mean']:+8.2f} CI=[{a['ci'][0]:+8.2f},{a['ci'][1]:+8.2f}] "
      f"pos={a['pos']}/16")
d = out["descriptive_aware_simple_1_16"]
print(f"DESC     aware simple1-16: {d['mean']:+8.2f} CI=[{d['ci'][0]:+8.2f},{d['ci'][1]:+8.2f}] "
      f"pos={d['pos']}/16")
d = out["descriptive_apt_simple_9_16"]
print(f"DESC     apt simple 9-16 : {d['mean']:+8.2f} CI=[{d['ci'][0]:+8.2f},{d['ci'][1]:+8.2f}] "
      f"pos={d['pos']}/8")
print("cells:", {k: round(v, 1) for k, v in out["cell_means"].items()})
print("per-seed interaction:", [round(x, 1) for x in inter])
