"""Step-1 attribution-landscape diagnostic (EXPLORATORY, 22 Aug 2026).

Decides whether the planner wave's UNSTEERABLE cell is a property of
the FIELD (no reachable high-attribution region exists => terminal,
and the claim strengthens) or of the INSTRUMENT (a peak exists that
greedy CEM failed to reach => the trained-probe rescue wave is live).

Data: the executed planner wave's own npz (b3_planner overlay) — per
run, 5 arms x 2000 realized steps of per-step distractor attribution
computed at the ONLINE posterior states with shared per-run
normalizers. No new compute; no frozen reader touched; output is a
DIAGNOSTIC artifact feeding a GO decision, never a registered claim.

DECISION THRESHOLDS (declared before computation, in this header):
  headroom ratio H = mean(top-10% of pooled attr) / mean(policy-arm
  attr), per run.
  - H >= 1.5 in >= 6/8 runs  -> PEAKED: rescue live (spec the
    trained-probe wave).
  - H <= 1.2 in >= 6/8 runs  -> FLAT: terminal; UNSTEERABLE is a
    field property.
  - otherwise                -> AMBIGUOUS: report, decide by the
    secondary structure metrics.
Secondary (structure, no thresholds): policy-mean quantile in the
pooled distribution; exceedance fraction; in/out-region attr per
arm; lag-1 autocorrelation of attr within the policy arm (are
high-attr states dwellable?); best single EPISODE mean vs the policy
episodes (E=4 per arm).

Run: python -m analysis.planner_landscape --bundle <b3 runroot> \
        --output artifacts/planner_landscape_20260822
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ARMS = ("random", "policy", "cem_disag", "cem_real", "cem_distractor")
EP_LEN = 500
N_EP = 4
TOP_FRAC = 0.10
H_PEAK, H_FLAT, NEED = 1.5, 1.2, 6


def analyze_run(run_dir):
    npz = np.load(os.path.join(run_dir, "se_planner",
                               "se_planner.npz"))
    attr = {a: np.asarray(npz[f"{a}_attr_distractor"], np.float64)
            for a in ARMS}
    occ = {a: np.asarray(npz[f"{a}_occ"], np.float64) for a in ARMS}
    pooled = np.concatenate([attr[a] for a in ARMS])
    pol = attr["policy"]
    pol_mean = float(pol.mean())
    k = max(1, int(TOP_FRAC * pooled.size))
    top = np.sort(pooled)[-k:]
    headroom = float(top.mean() / pol_mean)
    # policy mean's quantile within the pooled landscape
    pol_q = float((pooled < pol_mean).mean())
    exceed = float((pooled > pol_mean).mean())
    # where do the top-decile states live? (arm composition)
    thresh = np.sort(pooled)[-k]
    comp = {a: float((attr[a] >= thresh).mean()) for a in ARMS}
    # in/out-region attr per arm
    region = {a: dict(
        inr=float(attr[a][occ[a] > 0.5].mean())
        if (occ[a] > 0.5).any() else None,
        out=float(attr[a][occ[a] <= 0.5].mean())
        if (occ[a] <= 0.5).any() else None) for a in ARMS}
    # dwellability: lag-1 autocorr of attr within each policy episode
    acs = []
    for e in range(N_EP):
        x = pol[e * EP_LEN:(e + 1) * EP_LEN]
        x = x - x.mean()
        d = float(np.sqrt((x[:-1] ** 2).sum() * (x[1:] ** 2).sum()))
        if d > 0:
            acs.append(float((x[:-1] * x[1:]).sum() / d))
    # best single episode mean across ALL arms vs policy episodes
    ep_means = {}
    for a in ARMS:
        ep_means[a] = [float(attr[a][e * EP_LEN:(e + 1) * EP_LEN]
                             .mean()) for e in range(N_EP)]
    best_ep_arm, best_ep = max(
        ((a, m) for a in ARMS for m in ep_means[a]),
        key=lambda t: t[1])
    return dict(run_dir=os.path.abspath(run_dir),
                pol_mean=pol_mean, headroom=headroom,
                pol_quantile_in_pooled=pol_q,
                exceed_frac=exceed,
                top_decile_arm_composition=comp,
                region_attr=region,
                policy_lag1_autocorr=float(np.mean(acs)),
                episode_means=ep_means,
                best_episode=dict(arm=best_ep_arm, mean=best_ep,
                                  vs_policy_best=best_ep
                                  / max(ep_means["policy"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(args.bundle,
                                         "se_cheetah_seed*")))
    assert len(dirs) == 8, dirs
    recs = [analyze_run(d) for d in dirs]
    hs = [r["headroom"] for r in recs]
    n_peak = sum(1 for h in hs if h >= H_PEAK)
    n_flat = sum(1 for h in hs if h <= H_FLAT)
    verdict = ("PEAKED-RESCUE-LIVE" if n_peak >= NEED else
               "FLAT-TERMINAL" if n_flat >= NEED else "AMBIGUOUS")
    out = dict(thresholds=dict(top_frac=TOP_FRAC, peak=H_PEAK,
                               flat=H_FLAT, need=NEED),
               headroom_per_run=hs, n_peak=n_peak, n_flat=n_flat,
               verdict=verdict, per_run=recs,
               status="EXPLORATORY DIAGNOSTIC — feeds a GO decision; "
                      "licenses no registered claim")
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output,
                           "planner_landscape.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"LANDSCAPE: headroom {[round(h, 3) for h in hs]}")
    print(f"  n_peak(>= {H_PEAK}) = {n_peak}, n_flat(<= {H_FLAT}) = "
          f"{n_flat} -> {verdict}")
    print(f"  policy quantile in pooled: "
          f"{[round(r['pol_quantile_in_pooled'], 3) for r in recs]}")
    print(f"  policy lag-1 autocorr: "
          f"{[round(r['policy_lag1_autocorr'], 2) for r in recs]}")


if __name__ == "__main__":
    main()
