"""NFI Level-1 pilot orchestrator.

Phases (composable flags; --all runs collect+train+search):

  --selfcheck   referee integrity + synergy referee + planted-NULL (exact
                Kalman behind the learned-model API must yield NO-EXPLOIT)
                + planted-POSITIVE (attractor-corrupted model must yield
                EXPLOIT-SURVIVES-CIG+PBIM) + holonomy-instrument sanity.
  --collect     random-policy episodes from the LG env (+ exact-filter refs).
  --train       train + freeze the GRU belief-WM ensemble on collected data.
  --search      cycle sweep on every ensemble member: three accounting modes,
                exact true gains, exploit classification, noisy-TV check,
                holonomy report, H* residue, ensemble disagreement.

Outputs land in local_results/uncfield/<tag>/ (gitignored):
  episodes.pkl, ensemble.pkl, results_<member>.npz, summary.json

Pilot verdict tree (registered in the design note, thresholds in planner.py):
  NO-EXPLOIT / EXPLOIT-NAIVE-ONLY        -> NFI kill branch
  EXPLOIT-SURVIVES-CIG-ONLY              -> PBIM repairs; weakened NFI
  EXPLOIT-SURVIVES-CIG+PBIM              -> flagship signature
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np

from . import learnedwm as lw
from . import lgfield as lg
from . import planner as pl
from . import residues as rs

OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"


# ------------------------------------------------------------------ selfcheck

def selfcheck(verbose=True):
    t0 = time.time()
    report = {"lgfield": lg.selfcheck(), "synergy": rs.synergy_selfcheck()}

    # Planted NULL: the exact filter is a coherent information accountant;
    # the detector must find no exploit of any kind on it.
    exact = lw.ExactFilterAdapter()
    results, _ = pl.search_exploits(exact, seed=0, max_len=4,
                                    n_loops=6, n_burn=20)
    verdict, counts = pl.classify(results)
    report["planted_null"] = {"verdict": verdict, **counts}
    assert verdict == "NO-EXPLOIT", f"planted-null failed: {verdict}"

    # Coherence: on NEUTRAL (static, evidence-saturating) cycles the exact
    # model's realized-dH rate must track the referee's true rate. (Dynamic
    # cycles legitimately differ: dH accounting nets out the Q entropy
    # inflow that update-gain accounting counts as extraction.)
    gaps = [abs(r["carried_dh_rate"] - r["true_rate"])
            for r in results if r["neutral"]]
    report["planted_null"]["max_dh_vs_true_gap_neutral"] = float(np.max(gaps))
    assert np.max(gaps) < 1e-3, "exact adapter accounting is not coherent"

    # Planted POSITIVE: attractor-style false contraction must be flagged as
    # the full CIG+PBIM-surviving exploit.
    corrupted = lw.CorruptedModel(lw.ExactFilterAdapter(), delta=0.05)
    results_c, _ = pl.search_exploits(corrupted, seed=0, max_len=4,
                                      n_loops=6, n_burn=20)
    verdict_c, counts_c = pl.classify(results_c)
    report["planted_positive"] = {"verdict": verdict_c, **counts_c}
    assert verdict_c == "EXPLOIT-SURVIVES-CIG+PBIM", \
        f"planted-positive failed: {verdict_c}"

    # Holonomy instrument sanity. Zero reference = MOVE-ONLY cycles on the
    # exact filter (no evidence, converged dynamics -> periodic beliefs ->
    # zero drift; sensing neutral cycles retain the genuine 1/n drain, which
    # the coherence check above already ties to the referee). Planted drift
    # on the corrupted model's exploited loops must register.
    exploited = [r for r in results_c if r["exploit_carried_dh"]]
    move_only = [r for r in results
                 if r["cycle"]["name"].endswith("move_only")]
    hol_exact = max(np.abs(rs.loop_holonomy(exact, r["snaps"])[1]).max()
                    for r in move_only)
    dphi_corr = max(rs.loop_holonomy(corrupted, r["snaps"])[1].min()
                    for r in exploited)
    report["holonomy"] = {"max_abs_dphi_exact_moveonly": float(hol_exact),
                          "min_dphi_corrupted_exploited": float(dphi_corr)}
    assert abs(hol_exact) < 1e-6, "holonomy instrument false positive"
    assert dphi_corr > 1e-3, "holonomy instrument missed planted drift"

    report["elapsed_s"] = round(time.time() - t0, 1)
    if verbose:
        print(json.dumps(report, indent=1, default=float))
        print("pilot selfcheck PASS")
    return report


# ------------------------------------------------------------------- phases

def collect(outdir, n_episodes, t_steps, seed):
    rng = np.random.default_rng(seed)
    episodes = []
    for e in range(n_episodes):
        env = lg.LGFieldEnv(seed=seed * 100003 + e)
        ref = lg.KalmanReferee()
        episodes.append(lg.rollout_random(env, ref, t_steps, rng))
    with open(outdir / "episodes.pkl", "wb") as f:
        pickle.dump(episodes, f)
    print(f"collected {n_episodes} episodes x {t_steps} steps")
    return episodes


def train(outdir, episodes, n_members, steps, seed):
    ensemble = []
    for m in range(n_members):
        params, loss = lw.train_model(episodes, seed=seed + m, steps=steps,
                                      verbose=True)
        print(f"  member {m}: final loss {loss:.4f}")
        ensemble.append(params)
    lw.save_ensemble(outdir / "ensemble.pkl", ensemble)
    return ensemble


def search(outdir, ensemble, max_len, n_loops, n_burn, seed):
    summary = {"members": [], "thresholds": dict(
        eps_true=pl.EPS_TRUE, eps_pred=pl.EPS_PRED,
        n_loops=n_loops, n_burn=n_burn, max_len=max_len)}
    models = [lw.LearnedModel(p) for p in ensemble]
    for m, model in enumerate(models):
        t0 = time.time()
        results, (b0, ref0, node0) = pl.search_exploits(
            model, seed=seed, max_len=max_len, n_loops=n_loops, n_burn=n_burn)
        verdict, counts = pl.classify(results)
        hol = rs.holonomy_report(model, results)

        # Noisy-TV check: cycles sensing only the TV channel.
        tv = [r for r in results if "only_s6_tv" in r["cycle"]["name"]]
        tv_max = max((r["carried_eig_rate"] for r in tv), default=0.0)

        # H* residue on the top carried_dh cycle.
        top = results[0]
        adapt = rs.adaptivity_residue(model, b0, node0, top["cycle"], seed=seed) \
            if top["cycle"]["start"] == node0 else None

        member = dict(member=m, verdict=verdict, **counts, holonomy=hol,
                      tv_max_pred_rate=float(tv_max),
                      top_cycle=top["cycle"]["name"],
                      top_true_rate=top["true_rate"],
                      top_dh_rate=top["carried_dh_rate"],
                      top_eig_rate=top["carried_eig_rate"],
                      adaptivity_residue=adapt,
                      elapsed_s=round(time.time() - t0, 1))
        summary["members"].append(member)
        np.savez(outdir / f"results_m{m}.npz",
                 names=np.array([r["cycle"]["name"] for r in results]),
                 true_rate=np.array([r["true_rate"] for r in results]),
                 naive=np.array([r["naive_eig_rate"] for r in results]),
                 carried=np.array([r["carried_eig_rate"] for r in results]),
                 dh=np.array([r["carried_dh_rate"] for r in results]),
                 holonomy=np.array([r.get("holonomy_rate", np.nan)
                                    for r in results]),
                 neutral=np.array([r["neutral"] for r in results]))
        print(f"member {m}: {verdict} "
              f"(neutral {counts['n_neutral']}/{counts['n_cycles']}, "
              f"exploits naive/cig/pbim = {counts['n_exploit_naive']}/"
              f"{counts['n_exploit_cig']}/{counts['n_exploit_pbim']}, "
              f"tv_max {tv_max:.3f}) [{member['elapsed_s']}s]")

    # Ensemble-level verdict: the exploit claim is ensemble-robust only if a
    # majority of members exhibit it.
    verdicts = [m["verdict"] for m in summary["members"]]
    for v in ("EXPLOIT-SURVIVES-CIG+PBIM", "EXPLOIT-SURVIVES-CIG-ONLY",
              "EXPLOIT-NAIVE-ONLY", "NO-EXPLOIT"):
        if sum(x == v for x in verdicts) * 2 > len(verdicts):
            summary["ensemble_verdict"] = v
            break
    else:
        summary["ensemble_verdict"] = "MIXED:" + "|".join(sorted(set(verdicts)))
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nENSEMBLE VERDICT: {summary['ensemble_verdict']}")
    print(f"summary -> {outdir / 'summary.json'}")
    return summary


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced sizes for an end-to-end shakeout")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    cfg = dict(n_episodes=1500, t_steps=40, n_members=4, train_steps=3000,
               max_len=6, n_loops=pl.N_LOOPS, n_burn=pl.N_BURN)
    if args.smoke:
        cfg = dict(n_episodes=120, t_steps=30, n_members=2, train_steps=400,
                   max_len=4, n_loops=6, n_burn=10)

    outdir = OUT_ROOT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps({**cfg, "seed": args.seed,
                                                    "smoke": args.smoke}, indent=1))

    episodes = ensemble = None
    if args.all or args.collect:
        episodes = collect(outdir, cfg["n_episodes"], cfg["t_steps"], args.seed)
    if args.all or args.train:
        if episodes is None:
            with open(outdir / "episodes.pkl", "rb") as f:
                episodes = pickle.load(f)
        ensemble = train(outdir, episodes, cfg["n_members"],
                         cfg["train_steps"], args.seed)
    if args.all or args.search:
        if ensemble is None:
            ensemble = lw.load_ensemble(outdir / "ensemble.pkl")
        search(outdir, ensemble, cfg["max_len"], cfg["n_loops"],
               cfg["n_burn"], args.seed)


if __name__ == "__main__":
    main()
