"""P1: PRIME aleatoric-penalized carried-EIG re-score (claim-freeze memo
§6.2 spec — the last pre-freeze fidelity arm; PRIME arXiv 2607.16858).

Records, per (cycle, scored loop, sense action): the per-sense carried
EIG (exact score_cycle replay semantics — same rng creation and
_imagined_step calls as gamma_rescore, both rng-parity-safe omissions)
and TWO aleatoric estimates:

  ale_model  = R_implied = exp(ylv) − c'Σ_z c   (the model's OWN
               irreducible predictive variance — mechanism.head_accounts;
               PRIME's Eq. 3 aleatoric part instantiated on our heads,
               faithful-to-principle)
  ale_data   = per-sensor empirical within-episode variance of real reads
               (their probe estimator at d=1: for static sensors z is
               constant within an episode so this ≈ true R; TV → ~1.0)

The λ/σ₀²/variant grids and the gate variant are POST-HOC re-scores of
this table (registered in the prereg, run by the read script): per sense
u = eig − λ·log(1 + ale/σ₀²), cycle rate = Σu per loop, drift baseline
unchanged (penalty applies to sense actions only; move-only cycles are
penalty-free so λ does not shift the registered drift).

Validation gate (built-in): steady per-loop Σeig must reproduce the
member's saved npz `carried` column per cycle — certifying the replay
before any penalty number is read.

Run:  python -m uncfield.p1_rescore --selfcheck        (m0, 40 cycles)
      python -m uncfield.p1_rescore --member M         (full, ~5 min each)
Output: local_results/uncfield/p1_rescore/pilot2_mM.json (+ ale_data.json
once per run). Read AFTER instrument review + prereg freeze.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np

from . import learnedwm as lw
from . import lgfield as lg
from . import mechanism as mech
from . import planner as pl

OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
ROOT = OUT / "p1_rescore"


def ale_data_table():
    """Per-sensor within-episode variance of real reads, averaged over
    episodes with >= 2 reads of that sensor."""
    with open(OUT / "pilot2" / "episodes.pkl", "rb") as f:
        episodes = pickle.load(f)
    per = {k: [] for k in range(len(lg.SENSORS))}
    for ep in episodes:
        ys = {}
        for step in ep:
            if step["sensed"] is not None:
                ys.setdefault(step["sensed"], []).append(step["y"])
        for k, v in ys.items():
            if len(v) >= 2:
                per[k].append(float(np.var(v, ddof=1)))
    return {k: (float(np.mean(v)) if v else None) for k, v in per.items()}


def replay_cycle_senses(model, b_start, node_start, cycle,
                        n_loops=pl.N_LOOPS, n_burn=pl.N_BURN,
                        y_mode="ml", seed=0):
    """Exact score_cycle carried-branch replay; returns per scored loop a
    list of (sensor_k, eig, ale_model) for each sense action."""
    rng = np.random.default_rng(seed)
    assert cycle["start"] == node_start
    b, node = b_start.copy(), node_start
    for _ in range(n_burn):
        for a in cycle["actions"]:
            b, node = pl._imagined_step(model, b, a, node, y_mode, rng)
    loops = []
    for _ in range(n_loops):
        senses = []
        for a in cycle["actions"]:
            if a >= lg.N_NODES:
                acc = mech.head_accounts(model, b, a)
                senses.append((a - lg.N_NODES, float(acc["eig"]),
                               float(acc["r_implied"])))
            b, node = pl._imagined_step(model, b, a, node, y_mode, rng)
        loops.append(senses)
    return loops


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
    rows, worst = [], 0.0
    for cycle in cycles:
        b, node = b0.copy(), node0
        for mv in pl._route_to(node0, cycle["start"]):
            b = model.step(b, mv, node, None)
            node = mv
        loops = replay_cycle_senses(model, b, node, cycle)
        per_loop_eig = [sum(s[1] for s in lp) for lp in loops]
        steady = pl.steady_rate(np.array(per_loop_eig))
        i = by_name[cycle["name"]]
        dev = abs(steady - float(saved["carried"][i]))
        worst = max(worst, dev)
        assert dev < tol, (cycle["name"], steady, float(saved["carried"][i]))
        rows.append(dict(
            name=cycle["name"], n_actions=len(cycle["actions"]),
            neutral=bool(saved["neutral"][i]),
            carried_saved=float(saved["carried"][i]),
            dh_adj_saved=float(saved["dh_adj"][i]),
            true_rate=float(saved["true_rate"][i]),
            loops=[[[k, e, r] for k, e, r in lp] for lp in loops]))
    from .sweeps import _code_stamp
    return dict(member=member, n_cycles=len(rows), worst_gate_dev=worst,
                stamp=_code_stamp(), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        out = run_member(0, subset=40)
        # penalty unit checks on the recorded table (post-hoc scorer form)
        lam, s0 = 0.5, 0.5
        tv_rows = [r for r in out["rows"] if "tv" in r["name"]
                   and any(lp for lp in r["loops"])]
        pen = lambda ale: lam * np.log1p(max(ale, 0.0) / s0)
        for r in out["rows"][:5]:
            for lp in r["loops"]:
                for k, e, ale in lp:
                    assert (e - 0.0 * pen(ale)) == e          # λ=0 identity
        if tv_rows:
            k, e, ale = tv_rows[0]["loops"][-1][0]
            print(f"TV sense: eig={e:+.3f} ale_model={ale:.3f} "
                  f"penalty@deployed={pen(ale):.3f}")
        print(json.dumps({kk: v for kk, v in out.items() if kk != "rows"},
                         indent=1))
        print("p1_rescore selfcheck PASS (worst gate dev "
              f"{out['worst_gate_dev']:.2e})")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "ale_data.json").exists():
        with open(ROOT / "ale_data.json", "w") as f:
            json.dump(ale_data_table(), f, indent=1)
    out = run_member(args.member)
    with open(ROOT / f"pilot2_m{args.member}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"m{args.member}: gate dev {out['worst_gate_dev']:.2e}, "
          f"{out['n_cycles']} cycles", flush=True)


if __name__ == "__main__":
    main()
