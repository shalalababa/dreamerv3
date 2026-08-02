"""CEI Level-1 pilot: Counterfactual Epistemic Identifiability.

*** WARNING (2 Aug 2026): the pilot run from this version is INVALIDATED —
*** see artifacts/cei_pilot_20260802/RESULTS.md invalidation note. Known
*** defects: TV-sensor shadowing in the first-due-sensor loop (s7 never
*** behaviorally consulted; separation_time == node-3 hitting time), psi
*** deducible from the action log alone, fixed-patrol construction sits in
*** the selective-labels carve-out (v2-audit criterion 2 / Failure C).
*** Do NOT re-run for decisions; a v2 redesign is queued.

Question: when the uncertainty estimator CONTROLS the exploration policy,
which of its failures are identifiable from the data that policy generates?

Construction (exact-filter substrate; no learned model needed):
  Two per-sensor uncertainty estimators sharing one Kalman belief:
    U_true : score(sensor) = posterior variance of its functional c'Pc
    U_false: identical EXCEPT score(s7_z2_n3) := 0 always (false confidence
             about the never-visited z2 component)
  Induced policy pi_U: patrol the walk 0->1->4->1->0 (visits {0,1,4} only);
  at each node sense every available sensor whose score exceeds TAU once
  per visit. s7 lives at node 3, which the patrol never enters, and the two
  estimators agree on every sensor available at visited nodes, so:

    pi_{U_true} == pi_{U_false}  =>  IDENTICAL action/observation streams
    (policy-observational equivalence, verified bit-exactly), while the
    false-confidence risk psi differs maximally: U_false declares z2 known
    (score 0) though its true posterior variance stays 1.0 (never measured;
    z2 is static so it never mixes either).

Kill probes (registered in the TODO):
  P1 random exploration: epsilon-perturbed patrol; separation cost =
     E[steps until the two policies' behavior first differs] (= first
     arrival at node 3 with a sensing decision). KILL SIGNAL if this is
     trivially cheap at practical epsilon.
  P2 parametric inverse optimization: on-policy log-likelihood of the two
     estimators is IDENTICAL (same decisions), so any inverse method is
     exactly flat between them without off-policy support — verified.
  P3 off-policy evaluation: psi's estimand has ZERO on-policy support
     (node-3 sensing never occurs); importance weighting is undefined —
     verified by support count.
  Comparison: TARGETED audit (walk to node 3, observe one sensing decision)
     separates in O(graph diameter) steps. The pilot's decision quantity is
     the random-vs-targeted separation-cost ratio: if ~O(1), minimal-audit
     design has no purchase even in-toy (CEI kill); if targeted << random
     with 1/epsilon divergence, the design question is real.

Run:  python -m uncfield.cei --selfcheck | --run
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import lgfield as lg

TAU = 0.05           # sense when score (posterior variance) exceeds this
PATROL = [1, 4, 1, 0]                       # closed walk from node 0
FALSE_SENSOR = 7                            # s7_z2_n3
OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield" / "cei"


class Estimator:
    """Per-sensor uncertainty scores off a shared exact-filter belief."""

    def __init__(self, false_sensors=()):
        self.false_sensors = set(false_sensors)

    def score(self, ref, k):
        if k in self.false_sensors:
            return 0.0
        s = lg.SENSORS[k]
        return float(s.c @ ref.p @ s.c) if np.any(s.c) else float(s.r)


def policy_step(est, ref, node, patrol_idx, rng, eps=0.0):
    """One decision: (action, next_patrol_idx). Sensing decisions come from
    the estimator; movement follows the patrol unless an eps-exploration
    event overrides everything with a uniform valid action."""
    if eps > 0.0 and rng.random() < eps:
        return int(rng.choice(lg.valid_actions(node))), patrol_idx
    for k, s in enumerate(lg.SENSORS):          # sense loop: first due sensor
        if s.node == node and est.score(ref, k) > TAU:
            return lg.N_NODES + k, patrol_idx
    target = PATROL[patrol_idx % len(PATROL)]
    if target == node:                           # already there (post-detour)
        patrol_idx += 1
        target = PATROL[patrol_idx % len(PATROL)]
    move = _toward(node, target)
    if move == target:
        patrol_idx += 1
    return move, patrol_idx


def _toward(node, target):
    assert target != node
    if target in lg.ADJ[node]:
        return target
    frontier, prev = [node], {node: None}
    while frontier:
        nxt = []
        for u in frontier:
            for v in lg.ADJ[u]:
                if v not in prev:
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    cur = target
    while prev[cur] != node:
        cur = prev[cur]
    return cur


def _next_move(node, patrol_idx):
    """Step toward the first patrol waypoint that is not the current node
    (robust to eps-detours landing on a waypoint)."""
    for j in range(patrol_idx, patrol_idx + len(PATROL) + 1):
        target = PATROL[j % len(PATROL)]
        if target != node:
            return _toward(node, target)
    raise RuntimeError("degenerate patrol")


def rollout(est, t_steps, env_seed, eps=0.0, policy_seed=0):
    """Returns (actions, observations, ref, visits). One sense max per
    sensor per visit is enforced naturally: sensing drops the score below
    TAU only after enough reads, so we additionally mark sensed-this-visit."""
    env = lg.LGFieldEnv(seed=env_seed)
    ref = lg.KalmanReferee()
    rng = np.random.default_rng(policy_seed)
    actions, obs, visits = [], [], np.zeros(lg.N_NODES, int)
    patrol_idx = 0
    sensed_here = set()
    for _ in range(t_steps):
        node = env.node
        a, patrol_idx = policy_step(est, ref, node, patrol_idx, rng, eps=eps)
        if a >= lg.N_NODES:
            if a in sensed_here:                 # avoid stuck same-step loop
                a = _next_move(node, patrol_idx)
            else:
                sensed_here.add(a)
        if a < lg.N_NODES:
            sensed_here = set()
            visits[a] += 1
        _, y, sensed = env.step(a)
        if sensed is not None:
            ref.update(lg.SENSORS[sensed], y)
        ref.predict()
        actions.append(a)
        obs.append(y)
    return actions, obs, ref, visits


def psi(est, ref, thresh_conf=TAU, thresh_err=0.5):
    """False-confidence risk: any sensor declared certain (score<=thresh)
    whose true posterior variance is large."""
    worst = 0.0
    for k, s in enumerate(lg.SENSORS):
        if not np.any(s.c):
            continue
        declared = est.score(ref, k)
        true_var = float(s.c @ ref.p @ s.c)
        if declared <= thresh_conf and true_var >= thresh_err:
            worst = max(worst, true_var)
    return worst


def separation_time(eps, n_trials=200, t_max=4000, seed=0):
    """Steps until pi_{U_true} and pi_{U_false} first behave differently
    under the SAME eps-exploration randomness = first node-3 sensing
    decision (s7 score: U_true 1.0 > TAU, U_false 0)."""
    times = []
    for tr in range(n_trials):
        env = lg.LGFieldEnv(seed=seed * 7919 + tr)
        ref = lg.KalmanReferee()
        rng = np.random.default_rng(seed * 104729 + tr)
        est = Estimator()                        # true estimator's behavior
        patrol_idx, t_sep = 0, None
        sensed_here = set()
        for t in range(t_max):
            node = env.node
            if node == 3 and Estimator().score(ref, FALSE_SENSOR) > TAU:
                t_sep = t                        # U_true senses, U_false not
                break
            a, patrol_idx = policy_step(est, ref, node, patrol_idx, rng,
                                        eps=eps)
            if a >= lg.N_NODES and a in sensed_here:
                a = _next_move(node, patrol_idx)
            if a >= lg.N_NODES:
                sensed_here.add(a)
            else:
                sensed_here = set()
            _, y, sensed = env.step(a)
            if sensed is not None:
                ref.update(lg.SENSORS[sensed], y)
            ref.predict()
        times.append(t_sep if t_sep is not None else t_max)
    return float(np.mean(times)), float(np.median(times))


def targeted_audit_cost():
    """Deterministic audit: from any patrol node walk to node 3 and observe
    one sensing decision. Cost in steps (graph distance + 1)."""
    dists = []
    for start in (0, 1, 4):
        d, frontier, seen = 0, [start], {start}
        while 3 not in seen:
            d += 1
            frontier = [v for u in frontier for v in lg.ADJ[u]
                        if v not in seen]
            seen |= set(frontier)
        dists.append(d + 1)
    return max(dists)


def selfcheck():
    out = {}
    # (1) Policy-observational equivalence: bit-exact streams.
    for env_seed in range(5):
        a1, o1, _, v1 = rollout(Estimator(), 400, env_seed)
        a2, o2, _, v2 = rollout(Estimator({FALSE_SENSOR}), 400, env_seed)
        assert a1 == a2, "action streams differ: construction broken"
        assert o1 == o2, "observation streams differ"
        assert v1[3] == 0 and v1[2] == 0 and v1[5] == 0, "patrol left {0,1,4}"
    out["equivalence_episodes"] = 5
    # (2) psi gap is maximal on the never-measured static component.
    _, _, ref, _ = rollout(Estimator(), 400, 0)
    out["psi_true"] = psi(Estimator(), ref)
    out["psi_false"] = psi(Estimator({FALSE_SENSOR}), ref)
    assert out["psi_true"] == 0.0 and out["psi_false"] >= 0.9
    # (3) Separation machinery planted-positive: eps>0 separates, eps=0 never.
    m, _ = separation_time(0.2, n_trials=40, t_max=2000)
    out["sep_eps0.2_mean"] = m
    assert m < 2000, "eps-exploration failed to separate"
    m0, _ = separation_time(0.0, n_trials=10, t_max=800)
    out["sep_eps0_mean"] = m0
    assert m0 == 800, "eps=0 must never separate (on-policy impossibility)"
    # (4) Targeted audit is O(diameter): worst patrol node (0) is 3 moves
    # from node 3, +1 observation step.
    out["targeted_cost"] = targeted_audit_cost()
    assert out["targeted_cost"] <= 4
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"selfcheck": selfcheck()}
    curve = {}
    for eps in (0.3, 0.1, 0.05, 0.02, 0.01):
        mean, med = separation_time(eps, n_trials=200)
        curve[str(eps)] = dict(mean=mean, median=med)
        print(f"eps={eps}: separation mean {mean:.0f} / median {med:.0f} steps",
              flush=True)
    report["random_separation_curve"] = curve
    report["targeted_audit_cost_steps"] = targeted_audit_cost()
    report["ratio_random_over_targeted_at_eps0.1"] = \
        curve["0.1"]["mean"] / report["targeted_audit_cost_steps"]
    # P2/P3 verdict facts (exact by construction, verified in selfcheck):
    report["inverse_opt_flat_on_policy"] = True
    report["ope_support_for_psi"] = 0
    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), indent=1))
        print("cei selfcheck PASS")
    else:
        run()
