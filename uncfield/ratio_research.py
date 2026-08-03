"""Per-member self-consistency ratio re-search (sweep-review B2 repair).

The sweep's `top_self_consistency` column was unusable: it traced
`results[0]` (top by drift-adjusted carried_dh regardless of neutrality or
exploit status — often TV/move-only/legitimately-informative dynamic
cycles) and always replayed under y_mode="ml" even for sample-mode jobs.
This instrument computes the quantity the review prescribed: for each
(job, member), re-search with the job's exact (y_mode, seed), save the
full per-cycle table, and trace the ratio on the member's TOP NEUTRAL
BOTH-EXPLOIT cycle (fallback tiers recorded explicitly):

  fallback 0: top neutral cycle with exploit_both (search order = top
              drift-adjusted carried_dh; the mechanism record's own picks
              use min(carried, dh_adj) — reads must not equate the two)
  fallback 1: TOP-BY-CARRIED-EIG neutral cycle with exploit_carried_eig
              (CIG-ONLY members; review B1)
  fallback 2: none (no neutral exploit — ratio None; the trace dict
              disambiguates "no senses" from "heads license ~0")

Coverage note: JOBCONFIGS spans the original sweep + pilot2 only; the
hid128t10k/hid128t30k interaction arm needs a JOBCONFIGS extension (+
array 44-51) once its ensembles exist — the generalization read must
scope itself accordingly.

Ratio = mean realized EIG per sense / mean own-heads Bayes gain over the
traced loops (mechanism.summarize_trace), traced under the JOB's y_mode.
None when the cycle has no senses or the heads license ~zero gain.

Run:  python -m uncfield.ratio_research --task N      (N in 0..43, slurm array)
      python -m uncfield.ratio_research --selfcheck
Output: local_results/uncfield/ratio_research/<job>_m<member>.json (+ .npz
per-cycle table). Read AFTER instrument review, per standing policy.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import learnedwm as lw
from . import mechanism as mech
from . import planner as pl

OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
ROOT = OUT / "ratio_research"

# (job, ensemble_dir, y_mode, seed) — matches each sweep job's exact search
# configuration (sweeps.py) and the pilot anchor.
JOBCONFIGS = [
    ("pilot2", "pilot2", "ml", 0),
    ("ymode_s0", "pilot2", "sample", 0),
    ("ymode_s1", "pilot2", "sample", 1),
    ("ymode_s2", "pilot2", "sample", 2),
    ("train1k", "sweeps/train1k", "ml", 0),
    ("train10k", "sweeps/train10k", "ml", 0),
    ("train30k", "sweeps/train30k", "ml", 0),
    ("dataseed1", "sweeps/dataseed1", "ml", 1),
    ("dataseed2", "sweeps/dataseed2", "ml", 2),
    ("hid32", "sweeps/hid32", "ml", 0),
    ("hid128", "sweeps/hid128", "ml", 0),
]
N_TASKS = len(JOBCONFIGS) * 4


def pick_target(results):
    """(result, fallback_tier).  Tier 0 (both-exploit): first match in
    search order = top by drift-adjusted carried_dh (NOTE: the mechanism
    record's pick_cycles orders by min(carried, dh_adj) — reads must not
    describe these as "the mechanism record's cycles").  Tier 1
    (CIG-ONLY members): top by carried_eig_rate — dh_adj ordering among
    tier-1 candidates is below-threshold noise (review B1: the search-
    order pick landed at eig-rank 59/146 with a 6.5x ratio error)."""
    tier0 = [r for r in results if r["neutral"] and r["exploit_both"]]
    if tier0:
        return tier0[0], 0
    tier1 = [r for r in results if r["neutral"] and r["exploit_carried_eig"]]
    if tier1:
        return max(tier1, key=lambda r: r["carried_eig_rate"]), 1
    return None, 2


def ratio_on(model, b0, node0, res, y_mode, seed=0):
    cyc = res["cycle"]
    b, node = b0.copy(), node0
    for mv in pl._route_to(node0, cyc["start"]):
        b = model.step(b, mv, node, None)
        node = mv
    loops = mech.trace_cycle(model, b, node, cyc, n_loops=4, y_mode=y_mode,
                             seed=seed)
    s = mech.summarize_trace(loops)
    bg, eig = s.get("mean_bayes_gain"), s.get("mean_eig_per_sense")
    if not s.get("n_senses_per_loop") or bg is None or bg < 1e-4:
        return None, s
    return float(eig / bg), s


def run_task(task):
    job, ens_dir, y_mode, seed = JOBCONFIGS[task // 4]
    member = task % 4
    ensemble = lw.load_ensemble(OUT / ens_dir / "ensemble.pkl")
    model = lw.LearnedModel(ensemble[member])
    results, (b0, _, node0) = pl.search_exploits(model, seed=seed,
                                                 max_len=6, y_mode=y_mode)
    verdict, counts = pl.classify(results)
    target, tier = pick_target(results)
    ratio, trace_summary = (None, None)
    if target is not None:
        ratio, trace_summary = ratio_on(model, b0, node0, target, y_mode,
                                        seed=seed)
    ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{job}_m{member}"
    np.savez_compressed(
        ROOT / f"{tag}.npz",
        names=np.array([r["cycle"]["name"] for r in results]),
        true_rate=np.array([r["true_rate"] for r in results]),
        carried_eig=np.array([r["carried_eig_rate"] for r in results]),
        dh_adj=np.array([r["carried_dh_adj"] for r in results]),
        neutral=np.array([r["neutral"] for r in results]),
        exploit_both=np.array([r["exploit_both"] for r in results]),
    )
    from .sweeps import _code_stamp
    rec = dict(job=job, member=member, y_mode=y_mode, seed=seed,
               stamp=_code_stamp(), n_loops_trace=4,
               verdict=verdict, **counts, fallback_tier=tier,
               target_cycle=None if target is None
               else target["cycle"]["name"],
               target_true=None if target is None
               else float(target["true_rate"]),
               target_eig=None if target is None
               else float(target["carried_eig_rate"]),
               target_dh_adj=None if target is None
               else float(target["carried_dh_adj"]),
               self_consistency=ratio,
               trace=None if trace_summary is None else {
                   k: v for k, v in trace_summary.items()
                   if isinstance(v, (int, float, type(None)))})
    with open(ROOT / f"{tag}.json", "w") as f:
        json.dump(_san(rec), f, indent=1)
    print(f"{tag}: verdict={verdict} tier={tier} "
          f"cycle={rec['target_cycle']} ratio={ratio}", flush=True)
    return rec


def _san(x):
    if isinstance(x, dict):
        return {k: _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_san(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def selfcheck():
    # (1) pick logic on a synthetic table: tier order; tier 0 = first in
    #     search order; tier 1 = MAX by carried_eig_rate (review B1 —
    #     search order among tier-1 candidates is sub-threshold noise).
    mk = lambda n, neu, both, eig, rate: dict(
        cycle=dict(name=n), neutral=neu, exploit_both=both,
        exploit_carried_eig=eig, carried_eig_rate=rate)
    res = [mk("a", False, True, True, 9.0), mk("b", True, False, True, 9.0),
           mk("c", True, True, True, 0.1), mk("d", True, True, True, 5.0)]
    t, tier = pick_target(res)
    assert t["cycle"]["name"] == "c" and tier == 0, (t, tier)
    t, tier = pick_target([mk("a", True, False, True, 0.11),
                           mk("b", True, False, True, 0.23),
                           mk("c", False, False, True, 0.99)])
    assert t["cycle"]["name"] == "b" and tier == 1, "tier-1 must be top-eig"
    t, tier = pick_target([mk("a", False, True, True, 1.0)])
    assert t is None and tier == 2
    # (2) end-to-end smoke on pilot2 m0 at reduced cycle depth (fast).
    ensemble = lw.load_ensemble(OUT / "pilot2" / "ensemble.pkl")
    model = lw.LearnedModel(ensemble[0])
    results, (b0, _, node0) = pl.search_exploits(model, seed=0, max_len=4,
                                                 y_mode="ml")
    target, tier = pick_target(results)
    out = dict(n_results=len(results), tier=tier)
    if target is not None:
        r, s = ratio_on(model, b0, node0, target, "ml")
        out["ratio"] = r
        assert r is None or (np.isfinite(r) and r > 0)
    print(json.dumps(_san(out), indent=1))
    print("ratio_research selfcheck PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        assert args.task is not None and 0 <= args.task < N_TASKS
        run_task(args.task)
