"""CIG-faithful disagreement-kernel arm (claim-freeze audit, strongly
recommended pre-freeze; memo `NFI_ClaimFreeze_Audit_20260802.md` §1-C1).

Implements the ACTUAL CIG estimator (arXiv 2605.20878, Eqs. 5-9, per the
audit deep-read) — not our belief-carrying transplant — on the frozen
pilot2 ensemble, to answer Prop. 2(ii)'s open question: do our exploit
cycles (repeated closed walks = the paradigm prefix-redundant rollout)
collapse to the aleatoric floor under kernel decorrelation, or survive?
Both outcomes publish: collapse => rigorous scoping (CIG evades the
exploit by abandoning state-information accounting); survival => C1
becomes an in-scope refutation.

Estimator (trace-reduced, M members, T-step open-loop trajectory):
  delta_k(t) = mu_k(t) - mean_j mu_j(t)      per-member one-step z-head
               means along the SAME action walk, each member rolling its
               own belief OPEN-LOOP (y=None; CIG never conditions on
               observations), from a common teacher-forced warmup.
  K[j,t]     = (1/M) sum_k delta_k(j) . delta_k(t)        (Eq. 5-6)
  K~         = K + sigma^2 * d * I_T                       (Eq. 7)
  r_t        = 2 log L_tt   (Cholesky K~ = L L')           (Eq. 8)
  floor      = log(sigma^2 * d);  surplus_t = r_t - floor  (>= 0 by PSD)
  sigma^2    post-hoc from ensemble-mean one-step residuals on real
             training episodes (Eq. 9 analog; teacher-forced replay).

Decision quantities per cycle: per-loop AND per-action surplus over burn
vs scored windows, flagship conjunction cycles vs WALK-MATCHED move-only
twins vs the TV cycle.  THREE registered outcomes (arms review, 2 Aug):
(1) collapse-to-floor => scoping conclusion rigorous; (2) exploit-
SPECIFIC survival (exploits above twins per action) => in-scope
refutation; (3) no-contrast insensitivity (twins survive comparably —
open-loop ensemble divergence dominates; no Prop-2(ii) decay because the
kernel rows never actually repeat) => supports the scoping conclusion:
the estimator, as transplanted, is not measuring cycle information
content in this regime.  Scope sentences that must accompany any read:
the kernel is ONE cross-member object (per-member flagship structure is
not testable here), and CIG's open-loop trajectories contain no
observation slots, so scoring sense actions open-loop evaluates the
ensemble outside both CIG's estimator semantics and the model's training
family (measured: sense steps do NOT inflate K_tt — ratio ~1.0 — report
as robustness, not attribution).

Run:  python -m uncfield.cig_kernel --selfcheck | --run
Output: local_results/uncfield/cig_kernel/report.json.
Read AFTER instrument review (standing policy).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np

from . import learnedwm as lw
from . import lgfield as lg
from . import planner as pl

OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
ROOT = OUT / "cig_kernel"
D = lg.DZ


def member_warmups(models, seed=0, t_steps=60):
    """Common real-world warmup teacher-forced through every member
    (identical env/action stream = pl.warmup_state semantics)."""
    env = lg.LGFieldEnv(seed=seed + 10 ** 9)
    rng = np.random.default_rng(seed + 1)
    beliefs = [m.init_belief() for m in models]
    for _ in range(t_steps):
        node = env.node
        a = int(rng.choice(lg.valid_actions(node)))
        _, y, _ = env.step(a)
        beliefs = [m.step(b, a, node, y) for m, b in zip(models, beliefs)]
    return beliefs, env.node


def openloop_means(models, beliefs, node0, actions):
    """Roll each member open-loop (y=None) along the action walk; collect
    per-step z-head means: array (M, T, D)."""
    mus = []
    for m, b in zip(models, beliefs):
        b = b.copy()
        node, out = node0, []
        for a in actions:
            b = m.step(b, a, node, None)
            if a < lg.N_NODES:
                node = a
            mu, _ = m.zstats(b)
            out.append(np.asarray(mu))
        mus.append(np.stack(out))
    return np.stack(mus)


def kernel_rewards(mus, sigma2):
    """CIG Eqs. 5-8 with trace reduction. mus: (M, T, D)."""
    delta = mus - mus.mean(axis=0, keepdims=True)
    m = delta.shape[0]
    k = np.einsum("kjd,ktd->jt", delta, delta) / m
    kt = k + sigma2 * D * np.eye(k.shape[0])
    chol = np.linalg.cholesky(kt)
    r = 2.0 * np.log(np.diag(chol))
    return r - np.log(sigma2 * D)          # surplus over aleatoric floor


def estimate_sigma2(models, episodes, n_ep=20):
    """Eq. 9 analog: mean squared per-dim residual of the ensemble-mean
    one-step z prediction, teacher-forced on real episodes."""
    sq, n = 0.0, 0
    for ep in episodes[:n_ep]:
        beliefs = [m.init_belief() for m in models]
        for step in ep:
            beliefs = [m.step(b, step["action"], step["node"], step["y"])
                       for m, b in zip(models, beliefs)]
            mu = np.mean([np.asarray(m.zstats(b)[0])
                          for m, b in zip(models, beliefs)], axis=0)
            sq += float(np.sum((np.asarray(step["z_after"]) - mu) ** 2))
            n += D
    return sq / max(n, 1)


def cycle_walk(node0, cycle, n_burn, n_loops):
    """(actions, loop-boundary indices for the scored window)."""
    route = pl._route_to(node0, cycle["start"])
    acts = list(route)
    bounds = []
    for i in range(n_burn + n_loops):
        if i >= n_burn:
            bounds.append(len(acts))
        acts.extend(cycle["actions"])
    bounds.append(len(acts))
    return acts, bounds


def score_cycles(models, beliefs, node0, cycles, sigma2,
                 n_burn=pl.N_BURN, n_loops=pl.N_LOOPS):
    rows = []
    for cycle in cycles:
        acts, bounds = cycle_walk(node0, cycle, n_burn, n_loops)
        mus = openloop_means(models, beliefs, node0, acts)
        surplus = kernel_rewards(mus, sigma2)
        loops = [float(np.sum(surplus[bounds[i]:bounds[i + 1]]))
                 for i in range(len(bounds) - 1)]
        burn_end = bounds[0]
        na = len(cycle["actions"])
        steady = float(np.mean(loops[-pl.STEADY:]))
        rows.append(dict(
            name=cycle["name"], n_actions=na,
            burn_surplus_total=float(np.sum(surplus[:burn_end])),
            scored_per_loop=loops,
            scored_steady=steady,
            scored_steady_per_action=steady / na,
            min_surplus=float(np.min(surplus)),
        ))
        assert rows[-1]["min_surplus"] > -1e-8, "PSD violated"
    return rows


def pick_exhibits(member):
    """Flagship conjunction cycles + TV + WALK-MATCHED move-only twins
    (arms review N4: the original bottom-3-dh_adj picks were the most
    entropy-deflating walks, unmatched to exhibit geometry — the control
    for an exhibit is the SAME node walk with sensing removed)."""
    saved = np.load(OUT / "pilot2" / f"results_m{member}.npz",
                    allow_pickle=True)
    names = list(saved["names"])
    neutral = saved["neutral"]
    both = (neutral & (saved["carried"] > pl.EPS_PRED)
            & (saved["dh_adj"] > pl.EPS_PRED))
    order = np.argsort(-saved["dh_adj"])
    exhibits = [names[i] for i in order if both[i]][:3]
    tv = [n for n in names if "tv" in n][:1]

    def twin(name):
        walk = name.split("_")[1]
        for n in names:
            parts = n.split("_")
            if parts[1] == walk and n.endswith("move_only"):
                return n
        return None
    move = [t for t in (twin(n) for n in exhibits + tv) if t]
    if not move:                                # fallback: any move-only
        move = [names[i] for i in order[::-1]
                if names[i].endswith("move_only")][:3]
    return exhibits, tv, list(dict.fromkeys(move))


def run():
    ensemble = lw.load_ensemble(OUT / "pilot2" / "ensemble.pkl")
    models = [lw.LearnedModel(p) for p in ensemble]
    with open(OUT / "pilot2" / "episodes.pkl", "rb") as f:
        episodes = pickle.load(f)
    sigma2 = estimate_sigma2(models, episodes)
    beliefs, node0 = member_warmups(models, seed=0)
    all_cycles = {c["name"]: c for c in pl.enumerate_cycles(max_len=6)}
    report = dict(sigma2=sigma2, members={})
    for m in range(4):
        exhibits, tv, move = pick_exhibits(m)
        wanted = [all_cycles[n] for n in exhibits + tv + move]
        rows = score_cycles(models, beliefs, node0, wanted, sigma2)
        for r, kind in zip(rows, ["exploit"] * len(exhibits)
                           + ["tv"] * len(tv) + ["move_only"] * len(move)):
            r["kind"] = kind
        report["members"][str(m)] = rows
        summary = {k: round(float(np.mean(
            [r["scored_steady"] for r in rows if r["kind"] == k])), 4)
            for k in ("exploit", "tv", "move_only")
            if any(r["kind"] == k for r in rows)}
        print(f"m{m} steady surplus/loop by kind: {summary}", flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("report ->", ROOT / "report.json", flush=True)


def selfcheck():
    out = {}
    # (1) identical members => zero epistemic surplus (exact floor).
    ensemble = lw.load_ensemble(OUT / "pilot2" / "ensemble.pkl")
    m0 = lw.LearnedModel(ensemble[0])
    models = [m0, m0, m0, m0]
    beliefs, node0 = member_warmups(models, seed=0)
    cyc = pl.enumerate_cycles(max_len=4)[0]
    acts, _ = cycle_walk(node0, cyc, 5, 3)
    mus = openloop_means(models, beliefs, node0, acts)
    s = kernel_rewards(mus, 0.05)
    assert np.max(np.abs(s)) < 1e-9, "identical members must sit on floor"
    out["identical_floor"] = float(np.max(np.abs(s)))
    # (2) PSD surplus non-negativity on a real two-pass walk (recorded,
    #     NOT asserted ordered: open-loop member beliefs diverge with
    #     depth, so a genuine second traversal is not row-duplication —
    #     that is precisely the Prop-2(ii) empirical question the run
    #     answers).
    models = [lw.LearnedModel(p) for p in ensemble]
    beliefs, node0 = member_warmups(models, seed=0)
    mus = openloop_means(models, beliefs, node0, acts + acts)
    s = kernel_rewards(mus, 0.05)
    assert np.min(s) > -1e-8
    h = len(acts)
    out["first_half"], out["second_half"] = (float(np.sum(s[:h])),
                                             float(np.sum(s[h:])))
    # (3) PLANTED redundancy, analytic: constant deviation +/-c repeated
    #     T times gives K = c^2 * ones(T,T); the k-th repeat's surplus has
    #     closed form log((k c^2 + r)/((k-1) c^2 + r)) — the 1/k decay to
    #     the floor that Prop 2(ii)'s decorrelation produces. Exact match
    #     validates the kernel + Cholesky + floor machinery end-to-end.
    c2, r, t_rep = 0.7, 0.05 * 1, 8      # D=1 here => floor r = sigma2*D
    mus_syn = np.zeros((2, t_rep, 1))
    mus_syn[0, :, 0], mus_syn[1, :, 0] = np.sqrt(c2), -np.sqrt(c2)
    global D
    d_saved, D = D, 1
    try:
        s_syn = kernel_rewards(mus_syn, 0.05)
    finally:
        D = d_saved
    expected = [np.log((k * c2 + r) / ((k - 1) * c2 + r))
                for k in range(1, t_rep + 1)]
    out["planted_analytic_dev"] = float(np.max(np.abs(s_syn - expected)))
    assert out["planted_analytic_dev"] < 1e-9, out["planted_analytic_dev"]
    assert np.all(np.diff(s_syn) < 0), "repeat surplus must decay"
    print(json.dumps(out, indent=1))
    print("cig_kernel selfcheck PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.run:
        run()
