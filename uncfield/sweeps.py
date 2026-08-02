"""Sensitivity sweeps for the NFI flagship signature (pilot-grade).

The central objection to kill or confirm: "the exploit is an artifact of
undertrained / undersized models / ML-observation imagination / one data
draw." One axis varies per job, anchored at the pilot2 configuration
(300 ep x 600 steps, HID 64, 3000 train steps, ML-observation, data seed 0).

Queue (value order):
  ymode      pilot2 ensemble re-searched with SAMPLED-observation
             imagination, planner seeds {0,1,2}      (no training; cheap)
  train10k   4 members trained 10k steps on the pilot2 episodes
  train30k   4 members trained 30k steps  (the objection-killer long end)
  train1k    4 members trained 1k steps   (the decay-curve short end)
  dataseed1  full pilot, fresh data seed 1
  dataseed2  full pilot, fresh data seed 2
  hid32      4 members HID=32 on pilot2 episodes
  hid128     4 members HID=128 on pilot2 episodes

Each job writes local_results/uncfield/sweeps/<job>/summary.json with the
same member/ensemble verdict structure as the pilot, plus the
self-consistency ratio of the top exploited cycle (mechanism follow-up:
does EIG/own-heads-Bayes-gain predict exploitability across the sweep?).

Run:  python -m uncfield.sweeps [--jobs ymode,train10k,...]
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
from . import mechanism as mech
from . import pilot as pt
from . import planner as pl

OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
ANCHOR = OUT / "pilot2"

JOBS = ("ymode", "train10k", "train30k", "train1k",
        "dataseed1", "dataseed2", "hid32", "hid128")


def _anchor_episodes():
    with open(ANCHOR / "episodes.pkl", "rb") as f:
        return pickle.load(f)


def _search_ensemble(outdir, ensemble, y_mode="ml", seed=0, label=""):
    """pilot.search() equivalent with y_mode control + top-cycle
    self-consistency ratio. Idempotent: a completed job is skipped on
    resume (crash-restart safe)."""
    if (outdir / "summary.json").exists():
        with open(outdir / "summary.json") as f:
            summary = json.load(f)
        print(f"  {label}: SKIP (summary exists, "
              f"ensemble {summary['ensemble_verdict']})", flush=True)
        return summary
    summary = {"label": label, "y_mode": y_mode, "seed": seed, "members": []}
    for m, params in enumerate(ensemble):
        model = lw.LearnedModel(params)
        t0 = time.time()
        results, (b0, _, node0) = pl.search_exploits(
            model, seed=seed, max_len=6, y_mode=y_mode)
        verdict, counts = pl.classify(results)
        top = results[0]
        ratio = _self_consistency(model, b0, node0, top)
        summary["members"].append(dict(
            member=m, verdict=verdict, **counts,
            top_cycle=top["cycle"]["name"], top_true=top["true_rate"],
            top_eig=top["carried_eig_rate"], top_dh_adj=top["carried_dh_adj"],
            top_self_consistency=ratio,
            elapsed_s=round(time.time() - t0, 1)))
        print(f"    member {m}: {verdict} (both={counts['n_exploit_both']}, "
              f"selfcons={ratio if ratio is None else round(ratio, 2)}) "
              f"[{summary['members'][-1]['elapsed_s']}s]", flush=True)
    levels = [pl.VERDICTS.index(x["verdict"]) for x in summary["members"]]
    n = len(levels)
    lvl = max((L for L in range(len(pl.VERDICTS))
               if sum(x >= L for x in levels) * 2 >= n), default=0)
    summary["ensemble_verdict"] = pl.VERDICTS[lvl]
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "summary.json", "w") as f:
        json.dump(pt._sanitize(summary), f, indent=1)
    print(f"  {label}: ENSEMBLE {summary['ensemble_verdict']}", flush=True)
    return summary


def _self_consistency(model, b0, node0, res):
    """Mean EIG / own-heads Bayes gain over the top cycle's senses at the
    post-burn state (None when the cycle has no senses or heads license
    ~zero gain — there the numerator itself is the reading)."""
    cyc = res["cycle"]
    route = pl._route_to(node0, cyc["start"])
    b, node = b0.copy(), node0
    for mv in route:
        b = model.step(b, mv, node, None)
        node = mv
    loops = mech.trace_cycle(model, b, node, cyc, n_loops=2)
    s = mech.summarize_trace(loops)
    bg, eig = s.get("mean_bayes_gain"), s.get("mean_eig_per_sense")
    if not s.get("n_senses_per_loop") or bg is None or bg < 1e-4:
        return None
    return float(eig / bg)


def _train_and_search(job, steps=3000, hid=64, episodes=None, seed=0):
    outdir = OUT / "sweeps" / job
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists():
        return _search_ensemble(outdir, [], label=job)   # skip via guard
    episodes = episodes or _anchor_episodes()
    ensemble = []
    for m in range(4):
        params, loss = lw.train_model(episodes, seed=seed + m, steps=steps,
                                      batch=32, hid=hid, verbose=False)
        print(f"    trained member {m} (steps={steps}, hid={hid}) "
              f"loss {loss:.3f}", flush=True)
        ensemble.append(params)
    lw.save_ensemble(outdir / "ensemble.pkl", ensemble)
    return _search_ensemble(outdir, ensemble, label=job, seed=seed)


def run_job(job):
    t0 = time.time()
    print(f"\n=== job {job} ===", flush=True)
    if job == "ymode":
        ensemble = lw.load_ensemble(ANCHOR / "ensemble.pkl")
        for s in (0, 1, 2):
            _search_ensemble(OUT / "sweeps" / f"ymode_s{s}", ensemble,
                             y_mode="sample", seed=s, label=f"ymode_s{s}")
    elif job.startswith("train"):
        steps = {"train1k": 1000, "train10k": 10000, "train30k": 30000}[job]
        _train_and_search(job, steps=steps)
    elif job.startswith("dataseed"):
        seed = int(job[-1])
        outdir = OUT / "sweeps" / job
        outdir.mkdir(parents=True, exist_ok=True)
        episodes = pt.collect(outdir, 300, 600, seed)
        _train_and_search(job, episodes=episodes, seed=seed)
    elif job.startswith("hid"):
        _train_and_search(job, hid=int(job[3:]))
    else:
        raise ValueError(job)
    print(f"=== job {job} done in {round(time.time() - t0)}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(JOBS))
    args = ap.parse_args()
    for job in args.jobs.split(","):
        run_job(job.strip())


if __name__ == "__main__":
    main()
