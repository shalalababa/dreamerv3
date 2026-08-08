"""NFI family-generality factorial — runner (Paper 5, 7 Aug 2026).

The registered factorial is {world} x {architecture} with the flagship
family-1 cell (linear-Gaussian world x GRU, = pilot2 + dataseed3/4) already
read. This runner produces the three NEW cells at the anchor configuration
(300 ep x 600 steps, 4 members, 3000 train steps, batch 32, HID 64, ML
imagination, data seed 0, planner seed 0):

  lg_lstm   family-1 world (lgfield), LSTM belief WM, trained on the
            EXISTING anchor episodes (pilot2/episodes.pkl) — isolates the
            architecture axis on matched data
  dc_gru    family-2 world (dcfield), GRU belief WM — isolates the world
            axis at matched architecture
  dc_lstm   family-2 world, LSTM — the double-jump cell

Both dc cells train on ONE shared dc episode set (data seed 0), mirroring
the lg cells' shared anchor: architectures are data-matched within world.

Thresholds, windows, cycle enumeration, accounting modes and the verdict
tree are the family-1 planner's, inherited verbatim (planner.py defaults;
no retuning in either family). Verdict validity in the NEW world is
established pre-read by the planted battery (--selfcheck): exact-filter
NULL (joint + marginal-surrogate scoring) must classify NO-EXPLOIT and the
corrupted positives must fire the conjunction / transient tiers at the SAME
inherited thresholds.

Run:  python -m uncfield.family2 --selfcheck
      python -m uncfield.family2 --smoke          (disclosed tiny e2e)
      python -m uncfield.family2 [--cells lg_lstm,dc_gru,dc_lstm]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np

from . import dcfield as dc
from . import dcwm
from . import learnedwm as lw
from . import lgfield as lg
from . import pilot as pt
from . import planner as pl

OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
ANCHOR = OUT / "pilot2"
CELLS = ("lg_lstm", "dc_gru", "dc_lstm")
TV_TAG = {"lg": "only_s6_tv", "dc": "only_d6_tv"}


def _code_stamp(n_burn=None, n_loops=None):
    """Stamps the ACTUAL run window (review m1: module constants would
    self-contradict a smoke config) + a dirty-tree flag."""
    import subprocess
    cwd = pathlib.Path(__file__).parent
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=cwd).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, timeout=10, cwd=cwd).stdout.strip())
    except Exception:
        git, dirty = "unknown", True
    return dict(git=git or "unknown", dirty=dirty, eps_true=pl.EPS_TRUE,
                eps_pred=pl.EPS_PRED,
                n_burn=pl.N_BURN if n_burn is None else n_burn,
                n_loops=pl.N_LOOPS if n_loops is None else n_loops,
                steady=pl.STEADY, y_mode="ml")


def collect_dc(path, n_episodes, t_steps, seed):
    """Random-policy dc episodes (mirror of pilot.collect). Written
    atomically (tmp + rename) — both dc cells consume this ONE file, and
    the registered protocol runs them serially in one job; the atomic
    write is belt-and-suspenders against concurrent invocation."""
    rng = np.random.default_rng(seed)
    episodes = []
    for e in range(n_episodes):
        env = dc.ChainFieldEnv(seed=seed * 100003 + e)
        ref = dc.ChainReferee()
        episodes.append(dc.rollout_random(env, ref, t_steps, rng))
    import os
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(episodes, f)
    tmp.replace(path)
    print(f"collected {n_episodes} dc episodes x {t_steps} steps", flush=True)
    return episodes


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _search_cell(outdir, models, world, label, seed=0, max_len=6,
                 n_loops=pl.N_LOOPS, n_burn=pl.N_BURN, extra=None):
    """Search every member; write pilot-style summary + per-member npz.
    Idempotent: a completed cell is skipped on resume."""
    if (outdir / "summary.json").exists():
        with open(outdir / "summary.json") as f:
            summary = json.load(f)
        print(f"  {label}: SKIP (summary exists, "
              f"ensemble {summary['ensemble_verdict']})", flush=True)
        return summary
    wtag = "dc" if world is dc else "lg"
    summary = {"label": label, "world": wtag, "seed": seed, "members": []}
    if extra:
        summary.update(extra)
    for m, model in enumerate(models):
        t0 = time.time()
        results, _ = pl.search_exploits(model, seed=seed, max_len=max_len,
                                        n_loops=n_loops, n_burn=n_burn,
                                        world=world)
        verdict, counts = pl.classify(results)
        top = results[0]
        tv = [r for r in results if TV_TAG[wtag] in r["cycle"]["name"]]
        tv_max = max((max(r["naive_eig_rate"], r["carried_eig_rate"],
                          r["carried_dh_adj"]) for r in tv), default=0.0)
        # pinned secondary statistic (review M1): SAME-CYCLE eig+transient
        # conjunction — the eig conjunct self-excludes move-only drift
        # artifacts (move-only cycles have carried_eig ≡ 0)
        n_tconj = sum(r["exploit_carried_eig"] and r["exploit_transient"]
                      for r in results)
        member = dict(member=m, verdict=verdict, **counts,
                      n_exploit_transient_conj=int(n_tconj),
                      tv_max_pred_rate=float(tv_max),
                      top_cycle=top["cycle"]["name"],
                      top_true=top["true_rate"],
                      top_eig=top["carried_eig_rate"],
                      top_dh_adj=top["carried_dh_adj"],
                      top_transient=top["transient_rate"],
                      # bounded-entropy exhibit (dc): claimed contraction on
                      # the top cycle over burn+scored vs the world budget
                      top_claimed_cum=float(top["burn_pred_cum"]
                                            + np.sum(top["carried_dh_per_loop"])),
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
                 neutral=np.array([r["neutral"] for r in results]))
        print(f"    member {m}: {verdict} (both={counts['n_exploit_both']}, "
              f"cig={counts['n_exploit_cig']}, tv_max={tv_max:.3f}) "
              f"[{member['elapsed_s']}s]", flush=True)
    if not summary["members"]:
        # empty member list would make the majority rule vacuously true at
        # every level and fabricate the flagship verdict (sweeps review N1)
        raise RuntimeError(f"{label}: empty ensemble — refusing to aggregate")
    levels = [pl.VERDICTS.index(x["verdict"]) for x in summary["members"]]
    n = len(levels)
    lvl = max((L for L in range(len(pl.VERDICTS))
               if sum(x >= L for x in levels) * 2 >= n), default=0)
    summary["ensemble_verdict"] = pl.VERDICTS[lvl]
    summary["world_entropy_budget"] = (float(dc.N_CHAINS * np.log(2.0))
                                       if wtag == "dc" else None)
    summary["stamp"] = _code_stamp(n_burn=n_burn, n_loops=n_loops)
    import os
    tmp = outdir / f"summary.json.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(pt._sanitize(summary), f, indent=1)
    tmp.replace(outdir / "summary.json")
    print(f"  {label}: ENSEMBLE {summary['ensemble_verdict']}", flush=True)
    return summary


def run_cell(cell, root=None, cfg=None):
    if cell not in CELLS:                      # review n3: validate before
        raise ValueError(cell)                 # any directory side effects
    root = root or (OUT / "family2")
    cfg = cfg or dict(n_episodes=300, t_steps=600, n_members=4,
                      train_steps=3000, batch=32, hid=64, max_len=6,
                      n_loops=pl.N_LOOPS, n_burn=pl.N_BURN, seed=0)
    t0 = time.time()
    print(f"\n=== cell {cell} ===", flush=True)
    outdir = root / cell
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists():
        return _search_cell(outdir, [], None if cell.startswith("dc") else lg,
                            label=cell)   # skip via guard (world unused)
    (outdir / "config.json").write_text(
        json.dumps({**cfg, "steady": pl.STEADY, "y_mode": "ml"}, indent=1))

    if cell == "lg_lstm":
        with open(ANCHOR / "episodes.pkl", "rb") as f:
            episodes = pickle.load(f)
        episodes = episodes[: cfg["n_episodes"]]
        extra = dict(episodes_sha=_sha256(ANCHOR / "episodes.pkl"),
                     n_episodes_used=len(episodes))
        ensemble = []
        for m in range(cfg["n_members"]):
            params, loss = lw.train_model(
                episodes, seed=cfg["seed"] + m, steps=cfg["train_steps"],
                batch=cfg["batch"], hid=cfg["hid"], cell="lstm")
            print(f"    trained lg_lstm member {m} loss {loss:.3f}", flush=True)
            ensemble.append(params)
        lw.save_ensemble(outdir / "ensemble.pkl", ensemble)
        models = [lw.LearnedModel(p) for p in ensemble]
        return _search_cell(outdir, models, lg, label=cell, seed=cfg["seed"],
                            max_len=cfg["max_len"], n_loops=cfg["n_loops"],
                            n_burn=cfg["n_burn"], extra=extra)

    if cell in ("dc_gru", "dc_lstm"):
        epi_path = root / "dc_episodes.pkl"
        if epi_path.exists():
            with open(epi_path, "rb") as f:
                episodes = pickle.load(f)
        else:
            episodes = collect_dc(epi_path, cfg["n_episodes"],
                                  cfg["t_steps"], cfg["seed"])
        extra = dict(episodes_sha=_sha256(epi_path),
                     n_episodes_used=len(episodes))
        ensemble = []
        for m in range(cfg["n_members"]):
            params, loss = dcwm.train_model(
                episodes, seed=cfg["seed"] + m, steps=cfg["train_steps"],
                batch=cfg["batch"], hid=cfg["hid"],
                cell=("lstm" if cell == "dc_lstm" else "gru"))
            print(f"    trained {cell} member {m} loss {loss:.3f}", flush=True)
            ensemble.append(params)
        dcwm.save_ensemble(outdir / "ensemble.pkl", ensemble)
        models = [dcwm.DCLearnedModel(p) for p in ensemble]
        return _search_cell(outdir, models, dc, label=cell, seed=cfg["seed"],
                            max_len=cfg["max_len"], n_loops=cfg["n_loops"],
                            n_burn=cfg["n_burn"], extra=extra)

    raise ValueError(cell)


# ------------------------------------------------------------------ selfcheck

def selfcheck(verbose=True):
    """Pre-read validity gates, all at DEPLOY scale (max_len=6, burn 30,
    score 8), inherited thresholds:

      (a) dcfield referee integrity (incl. the observation-independence
          certificate + sampled-replay identity)
      (b) dc planted NULL (joint scoring) -> NO-EXPLOIT
      (c) dc planted NULL (marginal-sum surrogate scoring) -> NO-EXPLOIT
      (d) dc planted POSITIVE (constant-rate) -> EXPLOIT-SURVIVES-CIG+PBIM
      (e) dc planted POSITIVE (saturating) -> TRANSIENT-EXPLOIT-ONLY
      (f) dc null coherence: carried_dh tracks true rate on neutral cycles
      (g) family-1 regression: lg planted NULL through the world-threaded
          planner still classifies NO-EXPLOIT (default-path preservation)
      (h) lg LSTM API sanity: untrained params forward-pass finite
    """
    t0 = time.time()
    scale = dict(max_len=6, n_loops=pl.N_LOOPS, n_burn=pl.N_BURN)
    report = {"dcfield": dc.selfcheck()}

    exact = dcwm.ExactChainAdapter()
    results, _ = pl.search_exploits(exact, seed=0, world=dc, **scale)
    verdict, counts = pl.classify(results)
    report["dc_planted_null"] = {"verdict": verdict, **counts}
    assert verdict == "NO-EXPLOIT", f"dc planted-null failed: {verdict}"

    gaps = [abs(r["carried_dh_rate"] - r["true_rate"])
            for r in results if r["neutral"]]
    report["dc_planted_null"]["max_dh_vs_true_gap_neutral"] = float(np.max(gaps))
    assert np.max(gaps) < 1e-9, "dc exact adapter accounting is not coherent"

    marg = dcwm.MarginalChainAdapter()
    results_m, _ = pl.search_exploits(marg, seed=0, world=dc, **scale)
    verdict_m, counts_m = pl.classify(results_m)
    report["dc_planted_null_marginal"] = {"verdict": verdict_m, **counts_m}
    assert verdict_m == "NO-EXPLOIT", f"dc marginal-null failed: {verdict_m}"

    corrupted = dcwm.CorruptedDCModel(dcwm.ExactChainAdapter(), delta=0.05)
    results_c, _ = pl.search_exploits(corrupted, seed=0, world=dc, **scale)
    verdict_c, counts_c = pl.classify(results_c)
    report["dc_planted_positive"] = {"verdict": verdict_c, **counts_c}
    assert verdict_c == "EXPLOIT-SURVIVES-CIG+PBIM", \
        f"dc planted-positive failed: {verdict_c}"

    saturating = dcwm.CorruptedDCModel(dcwm.ExactChainAdapter(), delta=0.4,
                                       floor=True)
    results_s, _ = pl.search_exploits(saturating, seed=0, world=dc, **scale)
    verdict_s, counts_s = pl.classify(results_s)
    report["dc_planted_saturating"] = {"verdict": verdict_s, **counts_s}
    assert verdict_s == "TRANSIENT-EXPLOIT-ONLY", \
        f"dc planted-saturating failed: {verdict_s}"

    lg_null, _ = pl.search_exploits(lw.ExactFilterAdapter(), seed=0, **scale)
    v_lg, c_lg = pl.classify(lg_null)
    report["lg_regression_null"] = {"verdict": v_lg, **c_lg}
    assert v_lg == "NO-EXPLOIT", f"family-1 regression failed: {v_lg}"

    p_lstm = lw.init_params_lstm(0, hid=16)
    mdl = lw.LearnedModel(p_lstm)
    b = mdl.init_belief()
    b = mdl.step(b, 1, 0, None)
    b = mdl.step(b, lg.N_NODES + 4, 0, 0.3)
    ent = mdl.entropy(b)
    ymu, ylv = mdl.obs_pred(b, lg.N_NODES + 4)
    assert np.isfinite(ent) and np.isfinite(ymu) and np.isfinite(ylv)
    assert b.shape[0] == 32, "lstm belief must be [h, c]"
    report["lg_lstm_api"] = dict(entropy=float(ent), width=int(b.shape[0]))

    report["elapsed_s"] = round(time.time() - t0, 1)
    if verbose:
        print(json.dumps(report, indent=1, default=float))
        print("family2 selfcheck PASS")
    return report


def smoke():
    """Disclosed tiny end-to-end shakeout (quarantined outputs, smoke config
    != registered config; verdicts recorded in the prereg as pre-freeze
    disclosures, not used for tuning)."""
    cfg = dict(n_episodes=40, t_steps=120, n_members=2, train_steps=200,
               batch=16, hid=32, max_len=4, n_loops=6, n_burn=10, seed=77)
    root = OUT / "family2_smoke"
    for cell in CELLS:
        run_cell(cell, root=root, cfg=dict(cfg))
    print("family2 smoke complete (quarantined)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cells", default=",".join(CELLS))
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.smoke:
        smoke()
        return
    for cell in args.cells.split(","):
        run_cell(cell.strip())


if __name__ == "__main__":
    main()
