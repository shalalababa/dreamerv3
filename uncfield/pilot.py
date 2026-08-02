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
    """All planted runs execute at DEPLOY scale (max_len=6, burn 30, score 8)
    so the planted guarantees are not extrapolated across scan scale."""
    t0 = time.time()
    scale = dict(max_len=6, n_loops=pl.N_LOOPS, n_burn=pl.N_BURN)
    report = {"lgfield": lg.selfcheck(), "synergy": rs.synergy_selfcheck()}

    # Planted NULL #1: the exact filter (full-logdet entropy) is a coherent
    # information accountant; the detector must find no exploit of any kind.
    exact = lw.ExactFilterAdapter()
    results, _ = pl.search_exploits(exact, seed=0, **scale)
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

    # Planted NULL #2: the DIAGONAL-entropy variant — the surrogate the
    # learned members are actually scored with — must also be clean.
    results_d, _ = pl.search_exploits(lw.DiagExactFilterAdapter(), seed=0,
                                      **scale)
    verdict_d, counts_d = pl.classify(results_d)
    report["planted_null_diag"] = {"verdict": verdict_d, **counts_d}
    assert verdict_d == "NO-EXPLOIT", f"planted-null-diag failed: {verdict_d}"

    # Planted POSITIVE #1: constant-rate attractor-style false contraction
    # must be flagged as the full CIG+PBIM-surviving exploit (fires
    # carried_eig AND carried_dh on the same cycles).
    corrupted = lw.CorruptedModel(lw.ExactFilterAdapter(), delta=0.05)
    results_c, _ = pl.search_exploits(corrupted, seed=0, **scale)
    verdict_c, counts_c = pl.classify(results_c)
    report["planted_positive"] = {"verdict": verdict_c, **counts_c}
    assert verdict_c == "EXPLOIT-SURVIVES-CIG+PBIM", \
        f"planted-positive failed: {verdict_c}"

    # Planted POSITIVE #2: SATURATING farming (delta large, clipped at
    # LOGVAR_MIN — completes inside the burn window). The steady tiers are
    # blind to it by construction; the transient tier must catch it.
    saturating = lw.CorruptedModel(lw.ExactFilterAdapter(), delta=0.4,
                                   floor=True)
    results_s, _ = pl.search_exploits(saturating, seed=0, **scale)
    verdict_s, counts_s = pl.classify(results_s)
    report["planted_saturating"] = {"verdict": verdict_s, **counts_s}
    assert verdict_s == "TRANSIENT-EXPLOIT-ONLY", \
        f"planted-saturating failed: {verdict_s}"

    # Holonomy instrument sanity. Zero reference = MOVE-ONLY cycles on the
    # exact filter (no evidence, converged dynamics -> periodic beliefs ->
    # zero drift; sensing neutral cycles retain the genuine 1/n drain, which
    # the coherence check above already ties to the referee). BOTH channels
    # (sym-KL holonomy hs and signed potential drift dPhi) are asserted at
    # their zero reference; planted drift must register.
    exploited = [r for r in results_c if r["exploit_carried_dh"]]
    move_only = [r for r in results
                 if r["cycle"]["name"].endswith("move_only")]
    hol_pairs = [rs.loop_holonomy(exact, r["snaps"]) for r in move_only]
    hs_exact = max(hs.max() for hs, _ in hol_pairs)
    dphi_exact = max(np.abs(dphi).max() for _, dphi in hol_pairs)
    dphi_corr = max(rs.loop_holonomy(corrupted, r["snaps"])[1].min()
                    for r in exploited)
    report["holonomy"] = {"max_hs_exact_moveonly": float(hs_exact),
                          "max_abs_dphi_exact_moveonly": float(dphi_exact),
                          "min_dphi_corrupted_exploited": float(dphi_corr)}
    assert hs_exact < 1e-6, "holonomy sym-KL channel false positive"
    assert dphi_exact < 1e-6, "holonomy dPhi channel false positive"
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


def train(outdir, episodes, n_members, steps, batch, seed):
    ensemble = []
    for m in range(n_members):
        params, loss = lw.train_model(episodes, seed=seed + m, steps=steps,
                                      batch=batch, verbose=True)
        print(f"  member {m}: final loss {loss:.4f}")
        ensemble.append(params)
    lw.save_ensemble(outdir / "ensemble.pkl", ensemble)
    return ensemble


def _sanitize(obj):
    """NaN/inf -> None so summary.json stays strict JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (int, np.integer, bool)):
        return int(obj) if not isinstance(obj, bool) else obj
    return obj


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

        # Noisy-TV check: cycles sensing only the TV channel — max predicted
        # farming across ALL modes (drift-adjusted for dh).
        tv = [r for r in results if "only_s6_tv" in r["cycle"]["name"]]
        tv_max = max((max(r["naive_eig_rate"], r["carried_eig_rate"],
                          r["carried_dh_adj"]) for r in tv), default=0.0)

        # H* residue on the top carried_dh cycle.
        top = results[0]
        adapt = rs.adaptivity_residue(model, b0, node0, top["cycle"], seed=seed) \
            if top["cycle"]["start"] == node0 else None

        member = dict(member=m, verdict=verdict, **counts, holonomy=hol,
                      tv_max_pred_rate=float(tv_max),
                      top_cycle=top["cycle"]["name"],
                      top_true_rate=top["true_rate"],
                      top_dh_rate=top["carried_dh_rate"],
                      top_dh_adj=top["carried_dh_adj"],
                      top_eig_rate=top["carried_eig_rate"],
                      top_transient=top["transient_rate"],
                      adaptivity_residue=adapt,
                      elapsed_s=round(time.time() - t0, 1))
        summary["members"].append(member)
        np.savez(outdir / f"results_m{m}.npz",
                 names=np.array([r["cycle"]["name"] for r in results]),
                 true_rate=np.array([r["true_rate"] for r in results]),
                 naive=np.array([r["naive_eig_rate"] for r in results]),
                 carried=np.array([r["carried_eig_rate"] for r in results]),
                 dh=np.array([r["carried_dh_rate"] for r in results]),
                 dh_adj=np.array([r["carried_dh_adj"] for r in results]),
                 transient=np.array([r["transient_rate"] for r in results]),
                 holonomy=np.array([r.get("holonomy_rate", np.nan)
                                    for r in results]),
                 neutral=np.array([r["neutral"] for r in results]))
        print(f"member {m}: {verdict} "
              f"(neutral {counts['n_neutral']}/{counts['n_cycles']}, "
              f"naive/cig/pbim/both/trans = {counts['n_exploit_naive']}/"
              f"{counts['n_exploit_cig']}/{counts['n_exploit_pbim']}/"
              f"{counts['n_exploit_both']}/{counts['n_exploit_transient']}, "
              f"drift {counts['drift_per_action']:.4f}, "
              f"tv_max {tv_max:.3f}) [{member['elapsed_s']}s]")

    # Ensemble verdict: cumulative severity majority — the highest level L
    # such that at least half the members sit at level >= L (pre-declared
    # tie rule: exactly half counts).
    levels = [pl.VERDICTS.index(m["verdict"]) for m in summary["members"]]
    n = len(levels)
    ens_level = max((lvl for lvl in range(len(pl.VERDICTS))
                     if sum(x >= lvl for x in levels) * 2 >= n), default=0)
    summary["ensemble_verdict"] = pl.VERDICTS[ens_level]
    summary["member_verdicts"] = [m["verdict"] for m in summary["members"]]
    with open(outdir / "summary.json", "w") as f:
        json.dump(_sanitize(summary), f, indent=1)
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

    # t_steps=600 keeps the full imagined depth (warmup 60 + route + 30 burn
    # + 8 scored loops of <=12 actions ~ 520 steps) INSIDE the training
    # temporal support (review finding: no horizon extrapolation).
    cfg = dict(n_episodes=300, t_steps=600, n_members=4, train_steps=3000,
               batch=32, max_len=6, n_loops=pl.N_LOOPS, n_burn=pl.N_BURN)
    if args.smoke:
        cfg = dict(n_episodes=40, t_steps=120, n_members=2, train_steps=300,
                   batch=16, max_len=4, n_loops=6, n_burn=10)

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
                         cfg["train_steps"], cfg["batch"], args.seed)
    if args.all or args.search:
        if ensemble is None:
            ensemble = lw.load_ensemble(outdir / "ensemble.pkl")
        search(outdir, ensemble, cfg["max_len"], cfg["n_loops"],
               cfg["n_burn"], args.seed)


if __name__ == "__main__":
    main()
