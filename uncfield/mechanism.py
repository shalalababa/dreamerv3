"""Mechanism analysis for the NFI pilot flagship signature.

Runs on a frozen ensemble + the pilot's cycle sweep outputs and asks WHY the
exploited cycles farm, via three measurable hypotheses:

H-M1  HEAD INCONSISTENCY. The model's two heads imply different information
      accounts: the obs head predicts total observation variance
      exp(ylv) ~ c'Sigma_z c + R_implied. If R_implied stays large (or goes
      NEGATIVE — an outright inconsistency) while the z-head has already
      contracted, the 3-pt-GH EIG keeps promising gain on an exhausted
      sensor. Measured per sense action: EIG vs the head-consistent Bayes
      gain 0.5*log(exp(ylv)/R_implied) vs the referee's truth.

H-M2  COMPONENT LEAKAGE. Predicted contraction lands on latent components
      the loop's sensors do NOT measure (evidence-free cross-contraction).
      Measured: per-component logvar drop per loop x sensor support mask.

H-M3  CLEAN-MEMBER CONTRAST. The non-exploiting member differs measurably
      in obs-head calibration (predicted sd vs true sensor noise) and/or
      z-head floor levels at warmup end.

Usage:  python -m uncfield.mechanism --tag pilot2 [--outdir ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import learnedwm as lw
from . import lgfield as lg
from . import planner as pl

OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"


# ------------------------------------------------------------ H-M1 head math

def head_accounts(model, b, action):
    """Per-sense information accounts implied by the model's two heads."""
    k = action - lg.N_NODES
    c = lg.SENSORS[k].c
    mu, zlv = model.zstats(b)
    proj_var = float(np.sum((c ** 2) * np.exp(zlv)))    # z-head c-projection
    ymu, ylv = model.obs_pred(b, action)
    tot_var = float(np.exp(ylv))
    r_implied = tot_var - proj_var
    if r_implied > 1e-12:
        bayes_gain = 0.5 * np.log(tot_var / r_implied)
    else:
        bayes_gain = float("inf")                        # inconsistent heads
    return dict(eig=pl._eig(model, b, action, lg.SENSORS[k].node),
                proj_var=proj_var, tot_var=tot_var, r_implied=r_implied,
                r_true=lg.SENSORS[k].r, bayes_gain=bayes_gain,
                obs_sd_pred=float(np.sqrt(tot_var)),
                obs_sd_true=float(np.sqrt(lg.SENSORS[k].r + proj_var)))


# ------------------------------------------------- instrumented cycle replay

def trace_cycle(model, b0, node0, cycle, n_burn=pl.N_BURN, n_loops=4,
                y_mode="ml", seed=0):
    """Replay a cycle with component-level instrumentation (post-burn)."""
    rng = np.random.default_rng(seed)
    b, node = b0.copy(), node0
    for _ in range(n_burn):
        for a in cycle["actions"]:
            b, node = pl._imagined_step(model, b, a, node, y_mode, rng)
    loops = []
    for _ in range(n_loops):
        _, lv_in = model.zstats(b)
        senses = []
        for a in cycle["actions"]:
            if a >= lg.N_NODES:
                senses.append({"sensor": lg.SENSORS[a - lg.N_NODES].name,
                               **head_accounts(model, b, a)})
            b, node = pl._imagined_step(model, b, a, node, y_mode, rng)
        _, lv_out = model.zstats(b)
        support = np.zeros(lg.DZ, bool)
        for a in cycle["actions"]:
            if a >= lg.N_NODES:
                support |= lg.SENSORS[a - lg.N_NODES].c != 0
        d_lv = lv_out - lv_in
        loops.append(dict(
            senses=senses,
            dlogvar=[float(x) for x in d_lv],
            contraction_on_support=float(-d_lv[support].sum()) if support.any() else 0.0,
            contraction_off_support=float(-d_lv[~support].sum()),
        ))
    return loops


def summarize_trace(loops):
    """Steady-state per-loop mechanism numbers from an instrumented replay."""
    if not loops:
        return {}
    last = loops[-2:]
    senses = [s for lp in last for s in lp["senses"]]
    fin_bayes = [s["bayes_gain"] for s in senses if np.isfinite(s["bayes_gain"])]
    return dict(
        n_senses_per_loop=len(loops[-1]["senses"]),
        mean_eig_per_sense=float(np.mean([s["eig"] for s in senses])) if senses else 0.0,
        mean_bayes_gain=float(np.mean(fin_bayes)) if fin_bayes else None,
        frac_r_implied_negative=float(np.mean([s["r_implied"] <= 0 for s in senses])) if senses else 0.0,
        mean_r_implied=float(np.mean([s["r_implied"] for s in senses])) if senses else None,
        mean_r_true=float(np.mean([s["r_true"] for s in senses])) if senses else None,
        contraction_on_support=float(np.mean([lp["contraction_on_support"] for lp in last])),
        contraction_off_support=float(np.mean([lp["contraction_off_support"] for lp in last])),
    )


# ----------------------------------------------------------- H-M3 contrast

def obs_calibration(model, b):
    """Obs-head predicted sd vs the truth-implied sd for every sensor, at a
    given belief (warmup end): the member-level calibration fingerprint."""
    rows = {}
    for k, s in enumerate(lg.SENSORS):
        acc = head_accounts(model, b, lg.N_NODES + k)
        rows[s.name] = dict(sd_pred=acc["obs_sd_pred"],
                            r_implied=acc["r_implied"], r_true=s.r,
                            proj_var=acc["proj_var"])
    return rows


# ------------------------------------------------------------------- driver

def pick_cycles(npz, catalog):
    """Representative set: top-3 conjunction (or dh_adj) exploits, designed
    chord loop, top TV cycle, 3 cleanest neutral cycles."""
    names = list(npz["names"])
    neutral = npz["neutral"].astype(bool)
    carried, dh_adj = npz["carried"], npz["dh_adj"]
    both = neutral & (carried > pl.EPS_PRED) & (dh_adj > pl.EPS_PRED)
    order = np.argsort(-np.minimum(carried, dh_adj))
    chosen = {}
    top_exp = [i for i in order if both[i]][:3] or \
              [i for i in order if neutral[i]][:3]
    for j, i in enumerate(top_exp):
        chosen[f"exploit{j}_{names[i]}"] = names[i]
    chord = [n for n in names if "_1-4-1" in n and n.endswith("_all")]
    if chord:
        chosen[f"chord_{chord[0]}"] = chord[0]
    tvs = [(i, n) for i, n in enumerate(names) if "only_s6_tv" in n]
    if tvs:
        i, n = max(tvs, key=lambda t: carried[t[0]])
        chosen[f"tv_{n}"] = n
    clean = [i for i in np.argsort(np.abs(carried) + np.abs(dh_adj))
             if neutral[i]][:3]
    for j, i in enumerate(clean):
        chosen[f"clean{j}_{names[i]}"] = names[i]
    by_name = {c["name"]: c for c in catalog}
    return {label: by_name[n] for label, n in chosen.items() if n in by_name}


def run(tag="pilot2", outdir=None, seed=0):
    src = OUT_ROOT / tag
    outdir = pathlib.Path(outdir) if outdir else OUT_ROOT / f"{tag}_mechanism"
    outdir.mkdir(parents=True, exist_ok=True)
    ensemble = lw.load_ensemble(src / "ensemble.pkl")
    catalog = pl.enumerate_cycles(max_len=6)
    report = {"tag": tag, "members": []}
    for m, params in enumerate(ensemble):
        model = lw.LearnedModel(params)
        npz = np.load(src / f"results_m{m}.npz")
        b0, _, node0 = pl.warmup_state(model, seed=seed)
        entry = {"member": m, "cycles": {}, "obs_calibration": None}
        entry["obs_calibration"] = obs_calibration(model, b0)
        _, lv0 = model.zstats(b0)
        entry["zhead_logvar_warmup"] = [float(x) for x in lv0]
        for label, cyc in pick_cycles(npz, catalog).items():
            route = pl._route_to(node0, cyc["start"])
            b, node = b0.copy(), node0
            for mv in route:
                b = model.step(b, mv, node, None)
                node = mv
            loops = trace_cycle(model, b, node, cyc, seed=seed)
            entry["cycles"][label] = summarize_trace(loops)
        report["members"].append(entry)
        print(f"member {m}: {len(entry['cycles'])} cycles traced")
    with open(outdir / "mechanism.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    _digest(report)
    print(f"\nreport -> {outdir / 'mechanism.json'}")
    return report


def _digest(report):
    print("\n=== mechanism digest ===")
    for entry in report["members"]:
        m = entry["member"]
        cal = entry["obs_calibration"]
        static = [v for k, v in cal.items()
                  if k.startswith(("s0", "s1", "s2", "s3", "s7"))]
        print(f"\nmember {m}: obs-head static-sensor sd_pred "
              f"{np.mean([v['sd_pred'] for v in static]):.3f} "
              f"(true post-warmup ~0.25-0.35); "
              f"r_implied<=0 on {sum(v['r_implied'] <= 0 for v in cal.values())}"
              f"/{len(cal)} sensors at warmup")
        for label, s in entry["cycles"].items():
            if not s:
                continue
            print(f"  {label[:58]:58s} eig/sense {s['mean_eig_per_sense']:+.3f} "
                  f"rimp<0 {s['frac_r_implied_negative']:.2f} "
                  f"on-supp {s['contraction_on_support']:+.3f} "
                  f"off-supp {s['contraction_off_support']:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot2")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(tag=args.tag, outdir=args.outdir, seed=args.seed)
