"""Raw-ΔH recovery pass over the sweep cells (descriptive; 9 Aug 2026).

The sweep bundles (`local_results/uncfield/sweeps/*/`) ship member
summaries but NO per-cycle npz, so the 9-Aug drift-artifact finding
(family-1 conjunctions manufactured by the negative-drift credit; see
`artifacts/nfi_review_response_20260809/`) could only be computed on the
anchor and family-2 cells. This pass re-runs `search_exploits` for every
sweep member from its bundled ensemble at the cell's registered
(y_mode, seed) and writes pilot-schema per-cycle npz + a raw-vs-adjusted
census, so the manuscript's dual (raw + adjusted) tables can cover every
cell.

SCOPE: descriptive recovery — NO registered decision rules; feeds the
review-response record's disclosure tables. VERIFICATION GATE per
member: the recomputed verdict + all exploit counts must equal the
bundled summary's member entry (matched-device on RCC: both CPU). A
mismatch is recorded prominently (`verdict_match: false`) and the wave
continues — for a descriptive pass a mismatch is itself a finding, not
a halt. This gate also retroactively re-derives every sweep verdict
(incl. P-N1's dataseed3/4 8/8) from the MODELS rather than from
runner-written summary strings — closing review §15's residual.

Idempotent per member (npz existence guard). Jobs:
  ymode_s{0,1,2}  anchor pilot2 ensemble, y_mode=sample, seed=s
  train{1k,10k,30k}, hid{32,128}, hid128t{10k,30k}   own ensemble, ml, seed=0
  dataseed{1,2,3,4}                                  own ensemble, ml, seed=N

Run:  python -m uncfield.raw_research --job <job>     (or --selfcheck)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from . import learnedwm as lw
from . import planner as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
OUT = ROOT / "raw_research"
EPS = pl.EPS_PRED

JOBS = ("ymode_s0", "ymode_s1", "ymode_s2",
        "train1k", "train10k", "train30k",
        "dataseed1", "dataseed2", "hid32", "hid128",
        "hid128t10k", "hid128t30k", "dataseed3", "dataseed4")


def job_spec(job):
    """(ensemble_path, summary_dir, y_mode, seed) — mirrors sweeps.py."""
    if job.startswith("ymode_s"):
        s = int(job[-1])
        return ROOT / "pilot2" / "ensemble.pkl", ROOT / "sweeps" / job, "sample", s
    seed = int(job[-1]) if job.startswith("dataseed") else 0
    d = ROOT / "sweeps" / job
    return d / "ensemble.pkl", d, "ml", seed


def _save_npz(path, results):
    np.savez(path,
             names=np.array([r["cycle"]["name"] for r in results]),
             true_rate=np.array([r["true_rate"] for r in results]),
             naive=np.array([r["naive_eig_rate"] for r in results]),
             carried=np.array([r["carried_eig_rate"] for r in results]),
             dh=np.array([r["carried_dh_rate"] for r in results]),
             dh_adj=np.array([r["carried_dh_adj"] for r in results]),
             transient=np.array([r["transient_rate"] for r in results]),
             neutral=np.array([r["neutral"] for r in results]))


def _census(results):
    neu = lambda r: r["neutral"]
    return dict(
        n_cig=sum(neu(r) and r["carried_eig_rate"] > EPS for r in results),
        n_both_adj=sum(neu(r) and r["carried_eig_rate"] > EPS
                       and r["carried_dh_adj"] > EPS for r in results),
        n_both_raw=sum(neu(r) and r["carried_eig_rate"] > EPS
                       and r["carried_dh_rate"] > EPS for r in results),
        n_pbim_raw=sum(neu(r) and r["carried_dh_rate"] > EPS
                       for r in results),
        n_adjconj_rawneg=sum(neu(r) and r["carried_eig_rate"] > EPS
                             and r["carried_dh_adj"] > EPS
                             and r["carried_dh_rate"] <= 0 for r in results),
        drift_per_action=results[0]["drift_per_action"] if results else 0.0,
    )


def run_job(job):
    ens_path, summ_dir, y_mode, seed = job_spec(job)
    summary = json.load(open(summ_dir / "summary.json"))
    ensemble = lw.load_ensemble(ens_path)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"job": job, "y_mode": y_mode, "seed": seed,
           "ensemble": str(ens_path.relative_to(ROOT)), "members": []}
    json_path = OUT / f"{job}.json"
    if json_path.exists():
        print(f"{job}: SKIP (json exists)", flush=True)
        return
    for m, params in enumerate(ensemble):
        npz_path = OUT / f"{job}_m{m}.npz"
        t0 = time.time()
        model = lw.LearnedModel(params)
        results, _ = pl.search_exploits(model, seed=seed, max_len=6,
                                        y_mode=y_mode)
        verdict, counts = pl.classify(results)
        _save_npz(npz_path, results)
        stored = summary["members"][m]
        match = (verdict == stored["verdict"]) and all(
            counts[k] == stored[k] for k in
            ("n_exploit_naive", "n_exploit_cig", "n_exploit_pbim",
             "n_exploit_both", "n_exploit_transient"))
        cen = _census(results)
        out["members"].append(dict(
            member=m, verdict=verdict, verdict_match=bool(match),
            stored_verdict=stored["verdict"], **cen,
            n_both_stored=stored["n_exploit_both"],
            elapsed_s=round(time.time() - t0, 1)))
        print(f"  {job} m{m}: {verdict} match={match} "
              f"both adj/raw={cen['n_both_adj']}/{cen['n_both_raw']} "
              f"cig={cen['n_cig']} [{out['members'][-1]['elapsed_s']}s]",
              flush=True)
    import os
    tmp = json_path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    tmp.replace(json_path)
    print(f"{job}: done "
          f"(matches {sum(x['verdict_match'] for x in out['members'])}/"
          f"{len(out['members'])})", flush=True)


def selfcheck():
    """Local gate: the same code path over the anchor (pilot2 m0, ml,
    seed 0) must reproduce the SAVED pilot2 npz columns and summary
    counts exactly (same-device: pilot2 was authored on this machine)."""
    ensemble = lw.load_ensemble(ROOT / "pilot2" / "ensemble.pkl")
    model = lw.LearnedModel(ensemble[0])
    results, _ = pl.search_exploits(model, seed=0, max_len=6, y_mode="ml")
    verdict, counts = pl.classify(results)
    summary = json.load(open(ROOT / "pilot2" / "summary.json"))
    st = summary["members"][0]
    assert verdict == st["verdict"], (verdict, st["verdict"])
    for k in ("n_exploit_naive", "n_exploit_cig", "n_exploit_pbim",
              "n_exploit_both", "n_exploit_transient"):
        assert counts[k] == st[k], (k, counts[k], st[k])
    z = np.load(ROOT / "pilot2" / "results_m0.npz")
    idx = {r["cycle"]["name"]: i for i, r in enumerate(results)}
    for col, key in (("dh", "carried_dh_rate"), ("dh_adj", "carried_dh_adj"),
                     ("carried", "carried_eig_rate"),
                     ("true_rate", "true_rate")):
        dev = max(abs(results[idx[n]][key] - v)
                  for n, v in zip(z["names"], z[col]))
        assert dev < 1e-6, (col, dev)
    cen = _census(results)
    assert cen["n_both_adj"] == st["n_exploit_both"]
    print(f"raw_research selfcheck PASS (m0: both adj/raw = "
          f"{cen['n_both_adj']}/{cen['n_both_raw']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    for job in (args.job.split(",") if args.job else JOBS):
        run_job(job.strip())


if __name__ == "__main__":
    main()
