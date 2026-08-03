"""Gamma-discounted potential-shaping rescore (claim-freeze audit,
REQUIRED pre-freeze arm; memo `NFI_ClaimFreeze_Audit_20260802.md` §1-C1).

Our carried_dh with drift subtraction is Ng-form potential shaping with
Phi_t = -H(b_t) at gamma = 1.  The true discounted Ng form is
F_t = gamma*Phi_{t+1} - Phi_t = H_t - gamma*H_{t+1}; on over-contracted
(negative-entropy) beliefs the (1-gamma)*Sigma H correction is NEGATIVE
and the per-loop deficit was estimated at ~0.06-0.30 nats/loop at
gamma=0.99 — straddling EPS_PRED = 0.08, i.e. numerically able to change
the flagship conjunction count.  This instrument replays every cycle's
carried trajectory (exact score_cycle semantics: same warmup, route,
rng creation, burn, and _imagined_step calls — the eig quadrature and the
evidence-stale naive branch are omitted, which is rng-parity-safe since
neither consumes rng), records per-step entropy levels, and recomputes:

  rate_g   steady per-loop rate of F_gamma for gamma in {1.0, 0.995, 0.99}
  adj_g    rate_g - lambda_g * n_actions   (drift baseline recomputed
           under the SAME gamma from move-only cycles — symmetric with
           the registered accounting)
  exploit_dh_g / both_g / verdict_g        (conjunction recount)

VALIDATION GATE (built into every run): the gamma=1.0 steady rate must
reproduce the member's saved npz `dh` column per cycle, and the implied
drift must reproduce `dh_adj` — certifying the replay before the gamma
columns are read.  carried_eig/neutral come FROM THE SAVED TABLE (gamma
touches only the shaping leg).

Scope: pilot2 (the flagship record) has saved per-cycle npz — runnable
now.  Sweep members lack per-cycle tables; their gamma recount joins the
ratio_research npz once that RCC bundle lands (--job mode, later).

Run:  python -m uncfield.gamma_rescore --selfcheck   (member 0, subset)
      python -m uncfield.gamma_rescore --member M | --all
Output: local_results/uncfield/gamma_rescore/pilot2_mM.json.
Read AFTER instrument review (standing policy).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import learnedwm as lw
from . import lgfield as lg
from . import planner as pl

GAMMAS = (1.0, 0.995, 0.99)
OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
ROOT = OUT / "gamma_rescore"


def replay_cycle_gammas(model, b_start, node_start, cycle,
                        n_loops=pl.N_LOOPS, n_burn=pl.N_BURN,
                        y_mode="ml", seed=0):
    """Mirror of planner.score_cycle's carried branch recording per-step
    entropy; returns {gamma: per-loop F array} over the scored window."""
    rng = np.random.default_rng(seed)
    assert cycle["start"] == node_start
    b, node = b_start.copy(), node_start
    for _ in range(n_burn):
        for a in cycle["actions"]:
            b, node = pl._imagined_step(model, b, a, node, y_mode, rng)
    per = {g: [] for g in GAMMAS}
    for _ in range(n_loops):
        hs = [model.entropy(b)]
        for a in cycle["actions"]:
            b, node = pl._imagined_step(model, b, a, node, y_mode, rng)
            hs.append(model.entropy(b))
        for g in GAMMAS:
            per[g].append(sum(hs[i] - g * hs[i + 1]
                              for i in range(len(hs) - 1)))
    return {g: np.array(v) for g, v in per.items()}


def run_member(member, subset=None, tol=1e-6):
    ensemble = lw.load_ensemble(OUT / "pilot2" / "ensemble.pkl")
    model = lw.LearnedModel(ensemble[member])
    saved = np.load(OUT / "pilot2" / f"results_m{member}.npz",
                    allow_pickle=True)
    by_name = {n: i for i, n in enumerate(saved["names"])}
    b0, _, node0 = pl.warmup_state(model, seed=0)
    cycles = pl.enumerate_cycles(max_len=6)
    if subset is not None:
        cycles = cycles[:subset]
    rows, worst_dev = [], 0.0
    for cycle in cycles:
        b, node = b0.copy(), node0
        for mv in pl._route_to(node0, cycle["start"]):
            b = model.step(b, mv, node, None)
            node = mv
        per = replay_cycle_gammas(model, b, node, cycle)
        i = by_name[cycle["name"]]
        rate1 = pl.steady_rate(per[1.0])
        dev = abs(rate1 - float(saved["dh"][i]))
        worst_dev = max(worst_dev, dev)
        assert dev < tol, (cycle["name"], rate1, float(saved["dh"][i]))
        rows.append(dict(
            name=cycle["name"], n_actions=len(cycle["actions"]),
            move_only=cycle["name"].endswith("move_only"),
            neutral=bool(saved["neutral"][i]),
            carried_eig=float(saved["carried"][i]),
            dh_saved=float(saved["dh"][i]),
            dh_adj_saved=float(saved["dh_adj"][i]),
            **{f"rate_g{g}": pl.steady_rate(per[g]) for g in GAMMAS}))
    # drift baselines per gamma from move-only cycles (registered analog)
    out = dict(member=member, n_cycles=len(rows), worst_gate_dev=worst_dev)
    move = [r for r in rows if r["move_only"]]
    for g in GAMMAS:
        lam = float(np.median([r[f"rate_g{g}"] / r["n_actions"]
                               for r in move])) if move else 0.0
        n_both = n_dh = 0
        for r in rows:
            adj = r[f"rate_g{g}"] - lam * r["n_actions"]
            r[f"adj_g{g}"] = adj
            edh = r["neutral"] and adj > pl.EPS_PRED
            eeig = r["neutral"] and r["carried_eig"] > pl.EPS_PRED
            n_dh += edh
            n_both += (edh and eeig)
        out[f"g{g}"] = dict(drift=lam, n_exploit_dh=n_dh,
                            n_exploit_both=n_both)
        # validation leg 2 at gamma=1: adj must reproduce saved dh_adj
        if g == 1.0 and subset is None:
            dev2 = max(abs(r["adj_g1.0"] - r["dh_adj_saved"]) for r in rows)
            out["worst_adj_gate_dev"] = dev2
            assert dev2 < 1e-6, dev2
    out["rows"] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        out = run_member(0, subset=40)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                         indent=1))
        print("gamma_rescore selfcheck PASS (40-cycle gate, worst dev "
              f"{out['worst_gate_dev']:.2e})")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    members = range(4) if args.all else [args.member]
    for m in members:
        out = run_member(m)
        with open(ROOT / f"pilot2_m{m}.json", "w") as f:
            json.dump(out, f, indent=1)
        head = {k: v for k, v in out.items() if k != "rows"}
        print(f"m{m}: {json.dumps(head)}", flush=True)


if __name__ == "__main__":
    main()
