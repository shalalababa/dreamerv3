"""W1/W2/W3 registered reads (PREREG_axis1_corrective_amendment2_20260713.md SB/SC/SD).

SB (W1, E3v2 within-collector, BOTH arms):
  - Delta_hat_c = mean over nested seeds {1,2} of [(s1-s0)_task - (s1-s0)_apt]
    on AUC100k, per collector run c.
  - PRIMARY confirmatory (finger): exact two-sided sign-flip permutation
    across collector runs on {Delta_hat_c}. SECONDARY: same in cup.
  - Labeled non-decisional: per-arm simple effects (same permutation
    machinery); hierarchical collector-level bootstrap CI (B=10K, seed 0).

SC (W2): cup Q1 apt pooled seeds 1-16 within-seed paired contrast on AUC100k,
  cluster-bootstrap percentile 95% CI (B=10K, seed 0). CI excludes 0 =>
  named secondary finding; otherwise directional-only.

SD (W3): finger Q2 aware completion seeds 1-16 - descriptive only.
"""
import csv
import itertools
import json

import numpy as np
from scipy import stats

SP = "/tmp/claude-1000/-home-rickybao-projects-dreamerv3/1fd435d6-7962-41e3-b5ca-5ffe131ca911/scratchpad"
AUC = f"{SP}/w123_auc/auc.csv"
ART_P0 = "/home/rickybao/projects/dreamerv3/artifacts/p0_axis1_corrective_20260713"
SNAP = "/home/rickybao/projects/dreamerv3/local_results/w1_w2_w3_20260716_163331/runroot_light"
B, SEED = 10_000, 0

rows = [r for r in csv.DictReader(open(AUC)) if int(r["qc_pass"]) == 1]
auc = {r["run_id"]: float(r["auc100k"]) for r in rows}


def boot_ci(x, b=B, seed=SEED):
  rng = np.random.default_rng(seed)
  boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(b)]
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sign_flip_perm(x):
  """Exact two-sided sign-flip permutation p on mean(x)."""
  n = len(x)
  obs = x.mean()
  flips = [np.mean(x * np.array(s)) for s in itertools.product((1, -1), repeat=n)]
  return float(np.mean([abs(f) >= abs(obs) - 1e-12 for f in flips]))


def sensitivity(deltas):
  n = len(deltas)
  t = stats.ttest_1samp(deltas, 0.0)
  ci = t.confidence_interval()
  w = stats.wilcoxon(deltas)
  d_z = float(deltas.mean() / deltas.std(ddof=1))
  loo = [np.delete(deltas, i).mean() for i in range(n)]
  return dict(t_ci=[float(ci.low), float(ci.high)], t_p=float(t.pvalue),
              perm_p=sign_flip_perm(deltas), wilcoxon_p=float(w.pvalue), d_z=d_z,
              loo_range=[float(min(loo)), float(max(loo))])


def hier_boot_ci(mat, b=B, seed=SEED):
  """Hierarchical bootstrap: resample collectors, then nested seed deltas.

  mat: list of per-collector arrays of seed-level interaction deltas."""
  rng = np.random.default_rng(seed)
  nc = len(mat)
  boots = []
  for _ in range(b):
    cs = rng.integers(0, nc, size=nc)
    means = [rng.choice(mat[c], size=len(mat[c]), replace=True).mean() for c in cs]
    boots.append(np.mean(means))
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


summary = {d: json.load(open(f"{SNAP}/axis1_{d}/within/within_summary.json"))
           for d in ("finger", "cup")}
order = summary["finger"]["collector_order"]

out = {}

# ---------------------------------------------------------------- W1 (SB)
for dom, role in (("finger", "PRIMARY"), ("cup", "SECONDARY")):
  built = sorted(int(e["collector_index"]) for e in summary[dom]["sources"].values()
                 if e.get("decision") == "OK" and not str(e.get("collector_index")) == "None")
  per_c = []
  for ci in built:
    label = order[ci]
    ent = summary[dom]["sources"][label]
    seed_deltas, arm_effects = [], {"task": [], "apt": []}
    for k in (1, 2):
      d_task = auc[f"adapt_ax1w{ci}s1_{dom}_seed{k}_ckpt500000"] - \
               auc[f"adapt_ax1w{ci}s0_{dom}_seed{k}_ckpt500000"]
      d_apt = auc[f"adapt_ax1fw{ci}s1_{dom}_seed{k}_ckpt500000"] - \
              auc[f"adapt_ax1fw{ci}s0_{dom}_seed{k}_ckpt500000"]
      seed_deltas.append(d_task - d_apt)
      arm_effects["task"].append(d_task)
      arm_effects["apt"].append(d_apt)
    cells = {f"{arm}_s{s}": np.mean([auc[f"adapt_ax1{p}{ci}s{s}_{dom}_seed{k}_ckpt500000"]
                                     for k in (1, 2)])
             for arm, p in (("task", "w"), ("apt", "fw")) for s in (0, 1)}
    per_c.append(dict(
        collector=label, index=ci, docc=ent["docc"], dcov=ent["dcov"],
        lo_occ=ent["lo_occ"], hi_occ=ent["hi_occ"],
        seed_interaction_deltas=seed_deltas,
        interaction=float(np.mean(seed_deltas)),
        task_simple=float(np.mean(arm_effects["task"])),
        apt_simple=float(np.mean(arm_effects["apt"])),
        cells={k: float(v) for k, v in cells.items()}))

  D = np.array([c["interaction"] for c in per_c])
  task_s = np.array([c["task_simple"] for c in per_c])
  apt_s = np.array([c["apt_simple"] for c in per_c])
  mat = [np.array(c["seed_interaction_deltas"]) for c in per_c]
  hlo, hhi = hier_boot_ci(mat)
  docc = np.array([c["docc"] for c in per_c])
  fam = [c["collector"].rstrip("0123456789") for c in per_c]

  res = dict(
      role=role, n_collectors=len(per_c), per_collector=per_c,
      interaction_mean=float(D.mean()),
      interaction_pos=int((D > 0).sum()),
      perm_p_primary=sign_flip_perm(D),
      sensitivity=sensitivity(D),
      hier_boot_ci=[hlo, hhi],
      task_simple=dict(mean=float(task_s.mean()), pos=int((task_s > 0).sum()),
                       perm_p=sign_flip_perm(task_s), ci=list(boot_ci(task_s))),
      apt_simple=dict(mean=float(apt_s.mean()), pos=int((apt_s > 0).sum()),
                      perm_p=sign_flip_perm(apt_s), ci=list(boot_ci(apt_s))),
      descriptive_dose=dict(
          spearman_interaction_vs_docc=[float(v) for v in
                                        stats.spearmanr(docc, D)],
          family_means={f: float(D[[i for i, g in enumerate(fam) if g == f]].mean())
                        for f in sorted(set(fam))}),
  )
  out[f"w1_{dom}"] = res

# ---------------------------------------------------------------- W2 (SC)
seeds16 = list(range(1, 17))
w2 = np.array([auc[f"adapt_ax1fq1s1_cup_seed{s}_ckpt500000"] -
               auc[f"adapt_ax1fq1s0_cup_seed{s}_ckpt500000"] for s in seeds16])
# bit-consistency vs the frozen P0 paired JSON (seeds 1-8)
p0 = json.load(open(f"{ART_P0}/paired/paired_cup_q1_auc100k_s1_minus_s0.json"))
assert np.allclose(w2[:8], np.asarray(p0["deltas"], float)), "cup q1 apt 1-8 mismatch"
lo, hi = boot_ci(w2)
out["w2_cup_q1_apt_pooled_1_16"] = dict(
    deltas=w2.tolist(), mean=float(w2.mean()), ci=[lo, hi],
    pos=int((w2 > 0).sum()),
    decision="CI excludes 0" if (lo > 0) == (hi > 0) else "CI includes 0",
    sensitivity=sensitivity(w2),
    fresh_9_16=dict(mean=float(w2[8:].mean()), ci=list(boot_ci(w2[8:])),
                    pos=int((w2[8:] > 0).sum())),
    cell_means=dict(
        s0_1_16=float(np.mean([auc[f"adapt_ax1fq1s0_cup_seed{s}_ckpt500000"] for s in seeds16])),
        s1_1_16=float(np.mean([auc[f"adapt_ax1fq1s1_cup_seed{s}_ckpt500000"] for s in seeds16])),
        s0_9_16=float(np.mean([auc[f"adapt_ax1fq1s0_cup_seed{s}_ckpt500000"] for s in seeds16[8:]])),
        s1_9_16=float(np.mean([auc[f"adapt_ax1fq1s1_cup_seed{s}_ckpt500000"] for s in seeds16[8:]])),
    ))

# ---------------------------------------------------------------- W3 (SD)
w3 = np.array([auc[f"adapt_ax1q2s1_finger_seed{s}_ckpt500000"] -
               auc[f"adapt_ax1q2s0_finger_seed{s}_ckpt500000"] for s in seeds16])
out["w3_finger_q2_aware_1_16_descriptive"] = dict(
    deltas=w3.tolist(), mean=float(w3.mean()), ci=list(boot_ci(w3)),
    pos=int((w3 > 0).sum()),
    fresh_9_16=dict(mean=float(w3[8:].mean()), ci=list(boot_ci(w3[8:]))),
    cell_means=dict(
        s0=float(np.mean([auc[f"adapt_ax1q2s0_finger_seed{s}_ckpt500000"] for s in seeds16])),
        s1=float(np.mean([auc[f"adapt_ax1q2s1_finger_seed{s}_ckpt500000"] for s in seeds16]))))

json.dump(out, open(f"{SP}/w123_read.json", "w"), indent=1)

# ------------------------------------------------------------------ print
for dom in ("finger", "cup"):
  r = out[f"w1_{dom}"]
  print(f"\n=== W1 {dom} ({r['role']}), {r['n_collectors']} collectors ===")
  print(f"interaction mean {r['interaction_mean']:+8.2f}  pos {r['interaction_pos']}/{r['n_collectors']}"
        f"  PERM p={r['perm_p_primary']:.5f}  hierCI=[{r['hier_boot_ci'][0]:+.1f},{r['hier_boot_ci'][1]:+.1f}]")
  s = r["sensitivity"]
  print(f"  sens: tCI=[{s['t_ci'][0]:+.1f},{s['t_ci'][1]:+.1f}] d_z={s['d_z']:+.2f}"
        f" wilcoxon={s['wilcoxon_p']:.4f} LOO=[{s['loo_range'][0]:+.1f},{s['loo_range'][1]:+.1f}]")
  for arm in ("task", "apt"):
    a = r[f"{arm}_simple"]
    print(f"  {arm:>4} simple: {a['mean']:+8.2f} pos {a['pos']}/{r['n_collectors']}"
          f" perm_p={a['perm_p']:.4f} CI=[{a['ci'][0]:+.1f},{a['ci'][1]:+.1f}]")
  d = r["descriptive_dose"]
  print(f"  desc: spearman(D,docc)={d['spearman_interaction_vs_docc'][0]:+.3f}"
        f" (p={d['spearman_interaction_vs_docc'][1]:.3f}) family means:",
        {k: round(v, 1) for k, v in d["family_means"].items()})
  print("  per-collector (label docc D task apt):")
  for c in sorted(r["per_collector"], key=lambda c: c["index"]):
    print(f"    c{c['index']:>2} {c['collector']:<8} docc={c['docc']:.3f} "
          f"D={c['interaction']:+8.1f} task={c['task_simple']:+8.1f} apt={c['apt_simple']:+8.1f}"
          f"  seeds={[round(x,1) for x in c['seed_interaction_deltas']]}")

r = out["w2_cup_q1_apt_pooled_1_16"]
print(f"\n=== W2 cup Q1 apt pooled 1-16 ===\nmean {r['mean']:+8.2f} CI=[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]"
      f" pos {r['pos']}/16 -> {r['decision']}  perm_p={r['sensitivity']['perm_p']:.4f}")
print(f"  fresh 9-16: {r['fresh_9_16']['mean']:+.2f} CI=[{r['fresh_9_16']['ci'][0]:+.1f},"
      f"{r['fresh_9_16']['ci'][1]:+.1f}] pos {r['fresh_9_16']['pos']}/8   cells:",
      {k: round(v, 1) for k, v in r["cell_means"].items()})

r = out["w3_finger_q2_aware_1_16_descriptive"]
print(f"\n=== W3 finger Q2 aware 1-16 (descriptive) ===\nmean {r['mean']:+8.2f}"
      f" CI=[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] pos {r['pos']}/16"
      f"  fresh 9-16 {r['fresh_9_16']['mean']:+.2f}  cells:",
      {k: round(v, 1) for k, v in r["cell_means"].items()})
