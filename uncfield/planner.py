"""Cycle-searching imagination planner + exploit detector for the NFI pilot.

The planner enumerates closed walks on the graph (with sense-action variants),
imagines repeating each cycle inside the FROZEN learned model, and scores the
predicted information return under three accounting modes that map onto the
registered defense battery:

  naive_eig   per-step expected info gain evaluated on an evidence-STALE
              belief branch (advanced through time with y=None so position
              stays consistent, but never conditioned on within-loop
              observations — a lower bound on what replay-conditioned
              bonuses double-count; real staleness is episodes deep, ours is
              one loop deep, disclosed). Farming ONLY here => CIG kills it.
  carried_eig per-step expected info gain evaluated along the carried belief
              trajectory (prefix-conditioned, CIG-style accounting). Farming
              here SURVIVES CIG: the model over-promises gain relative to its
              own realized contraction.
  carried_dh  realized per-loop entropy drop of the carried belief. This is
              potential-based accounting by construction (potential = -H) at
              gamma=1. Under gamma<1 the shaped STREAM acquires a level term
              (1-gamma)*Sum(H) that is 5-8x the exploit margin at the
              registered window (arms review, 2 Aug 2026) — the discounted
              reward stream self-limits — while the telescoped RETURN
              preference between futures is gamma-invariant in sign (the
              planner still farms); see uncfield/gamma_rescore.py and the
              claim-freeze memo Sec. 6.3. Farming here ALONE is
              repaired by CIG (the agent's bonus is EIG, ~0): the flagship
              "survives CIG+PBIM" verdict requires carried_eig AND carried_dh
              farming ON THE SAME CYCLE.

True information gain of every cycle comes observation-free from the exact
Kalman referee (LG determinism). An EXPLOIT = a cycle whose true per-loop
gain is below EPS_TRUE (evidence-neutral) while a predicted rate stays above
EPS_PRED (farming) — with the carried_dh rate first corrected by the
member's own MOVE-ONLY drift baseline (uniform imagination drift induces no
behavioral preference and must not count as an exploit), and a TRANSIENT
tier catching farming that saturates inside the burn window (e.g. logvar
clipping) before the steady window is scored.

Accounting caveat (documented, verdict-safe): _eig measures H(b) minus
expected H after update AND time-advance, so on dynamic sensors it nets out
Q entropy inflow; exploits are restricted to neutral (static-saturating)
cycles where the two coincide post-burn-in.
"""

from __future__ import annotations

import numpy as np

from . import lgfield as lg

EPS_TRUE = 2e-2     # nats/loop: neutrality floor (post burn-in; true ~ 1/n)
EPS_PRED = 8e-2     # nats/loop: farming floor for predicted rates (4x margin)
N_LOOPS = 8         # scored imagined repeats per cycle (after burn-in)
N_BURN = 30         # unscored burn-in repeats before rates are measured
STEADY = 4          # rate = mean over the last STEADY scored loops
GH3 = ((0.0, 2.0 / 3.0), (np.sqrt(3.0), 1.0 / 6.0), (-np.sqrt(3.0), 1.0 / 6.0))


# ------------------------------------------------------------- cycle catalog

def _closed_walks(start, max_len):
    walks = []

    def extend(path):
        if 1 < len(path) <= max_len + 1 and path[-1] == start:
            walks.append(tuple(path))
        if len(path) > max_len:
            return
        for nxt in lg.ADJ[path[-1]]:
            extend(path + [nxt])

    extend([start])
    return walks


def enumerate_cycles(max_len=6):
    """Closed walks with sense variants: sense-all at each visit, or a single
    named sensor per visit (incl. the TV-only loop), or pure movement."""
    seen, cycles = set(), []
    for start in range(lg.N_NODES):
        for walk in _closed_walks(start, max_len):
            canon = min(tuple(walk[i:-1] + walk[:i]) for i in range(len(walk) - 1))
            if canon in seen:
                continue
            seen.add(canon)
            variants = {"all": None}
            avail = set(walk)
            for k, s in enumerate(lg.SENSORS):
                if s.node in avail:
                    variants[f"only_{s.name}"] = k
            variants["move_only"] = -1
            for vname, pick in variants.items():
                actions = []
                for i, node in enumerate(walk[:-1]):
                    for k, s in enumerate(lg.SENSORS):
                        if s.node != node:
                            continue
                        if pick is None or pick == k:
                            actions.append(lg.N_NODES + k)
                    actions.append(walk[i + 1])
                if pick == -1:
                    actions = [walk[i + 1] for i in range(len(walk) - 1)]
                if not any(a >= lg.N_NODES for a in actions) and pick != -1:
                    continue
                cycles.append(dict(name=f"c{len(cycles)}_{'-'.join(map(str, walk))}_{vname}",
                                   nodes=tuple(walk), actions=tuple(actions),
                                   start=start))
    return cycles


# --------------------------------------------------------------- imagination

def _imagined_step(model, b, action, node, y_mode, rng):
    """One imagined step; returns (b_next, node_next)."""
    if action < lg.N_NODES:
        return model.step(b, action, node, None), action
    ymu, ylv = model.obs_pred(b, action)
    if y_mode == "ml":
        y = ymu
    else:
        y = ymu + np.exp(0.5 * ylv) * rng.standard_normal()
    return model.step(b, action, node, y), node


def _eig(model, b, action, node):
    """3-point Gauss-Hermite expected info gain of one sense action."""
    ymu, ylv = model.obs_pred(b, action)
    sd = np.exp(0.5 * ylv)
    h0 = model.entropy(b)
    exp_h = 0.0
    for u, w in GH3:
        b1 = model.step(b, action, node, ymu + u * sd)
        exp_h += w * model.entropy(b1)
    return h0 - exp_h


def score_cycle(model, b_start, node_start, cycle, n_loops=N_LOOPS,
                n_burn=N_BURN, y_mode="ml", seed=0):
    """Imagine n_burn burn-in + n_loops scored repeats.

    Returns (rates, snaps, burn_dh): per-loop rates for the three scored
    modes, belief snapshots at scored loop boundaries (holonomy residue),
    and the per-loop realized entropy drop during the BURN window (the
    transient-exploit statistic — farming that saturates before the steady
    window must not vanish unrecorded)."""
    rng = np.random.default_rng(seed)
    assert cycle["start"] == node_start, "cycle must start at current node"
    b, node = b_start.copy(), node_start
    burn_dh = []
    for _ in range(n_burn):
        h_in = model.entropy(b)
        for a in cycle["actions"]:
            b, node = _imagined_step(model, b, a, node, y_mode, rng)
        burn_dh.append(h_in - model.entropy(b))
    rates = {"naive_eig": [], "carried_eig": [], "carried_dh": []}
    snaps = [b.copy()]
    for _ in range(n_loops):
        # Evidence-stale branch for naive mode: advances through the same
        # actions with y=None (position-consistent, never sees within-loop
        # evidence). Reset at each loop entry => one-loop staleness horizon.
        bn = b.copy()
        g_naive = g_carried = 0.0
        h_in = model.entropy(b)
        for a in cycle["actions"]:
            if a >= lg.N_NODES:
                g_carried += _eig(model, b, a, node)
                g_naive += _eig(model, bn, a, node)
            node_pre = node
            b, node = _imagined_step(model, b, a, node_pre, y_mode, rng)
            bn = model.step(bn, a, node_pre, None)
        rates["naive_eig"].append(g_naive)
        rates["carried_eig"].append(g_carried)
        rates["carried_dh"].append(h_in - model.entropy(b))
        snaps.append(b.copy())
    return ({k: np.array(v) for k, v in rates.items()}, snaps,
            np.array(burn_dh))


def steady_rate(per_loop):
    return float(np.mean(per_loop[-STEADY:]))


# ------------------------------------------------------------ exploit search

def warmup_state(model, seed=0, t_steps=60):
    """Run a shared random warmup through BOTH the model and the referee so
    predicted and true cycle gains start from matched information states.
    Returns (belief, referee, node). Env seed is offset so the warmup latent
    realization is disjoint from every training episode's."""
    env = lg.LGFieldEnv(seed=seed + 10 ** 9)
    ref = lg.KalmanReferee()
    rng = np.random.default_rng(seed + 1)
    b = model.init_belief()
    for _ in range(t_steps):
        node = env.node
        a = int(rng.choice(lg.valid_actions(node)))
        _, y, sensed = env.step(a)
        if sensed is not None:
            ref.update(lg.SENSORS[sensed], y)
        ref.predict()
        b = model.step(b, a, node, y)
    return b, ref, env.node


def _route_to(node_from, node_to):
    """Shortest move-action path on the small graph (BFS)."""
    if node_from == node_to:
        return []
    frontier, prev = [node_from], {node_from: None}
    while frontier:
        nxt = []
        for u in frontier:
            for v in lg.ADJ[u]:
                if v not in prev:
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    path, cur = [], node_to
    while prev[cur] is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def search_exploits(model, seed=0, max_len=6, n_loops=N_LOOPS, n_burn=N_BURN,
                    y_mode="ml"):
    """Full sweep: every cycle scored in imagination + exact true gains,
    both measured over the same post-burn-in loop window.

    Returns a list of per-cycle result dicts (sorted by carried_dh rate) and
    the shared warmup context.
    """
    b0, ref0, node0 = warmup_state(model, seed=seed)

    # Pass 1: score every cycle (predicted modes + burn window + referee).
    results = []
    for cycle in enumerate_cycles(max_len=max_len):
        route = _route_to(node0, cycle["start"])
        b, node = b0.copy(), node0
        ref = ref0.copy()
        for mv in route:
            b = model.step(b, mv, node, None)
            node = mv
            ref.predict()
        rates, snaps, burn_dh = score_cycle(model, b, node, cycle,
                                            n_loops=n_loops, n_burn=n_burn,
                                            y_mode=y_mode, seed=seed)
        true_all, _ = lg.true_cycle_gain(ref, list(cycle["actions"]),
                                         n_burn + n_loops)
        res = dict(cycle=cycle,
                   true_rate=steady_rate(true_all[n_burn:]),
                   true_first=float(true_all[0]),
                   burn_pred_cum=float(np.sum(burn_dh)),
                   burn_true_cum=float(np.sum(true_all[:n_burn])),
                   n_actions=len(cycle["actions"]),
                   snaps=snaps)
        for mode, per in rates.items():
            res[f"{mode}_rate"] = steady_rate(per)
            res[f"{mode}_per_loop"] = per
        res["neutral"] = res["true_rate"] < EPS_TRUE
        results.append(res)

    # Pass 2: the member's own uniform-drift baseline from MOVE-ONLY cycles
    # (per-action drift; uniform imagination drift is behaviorally inert and
    # must not count as an exploit — it is reported, not flagged).
    move_only = [r for r in results
                 if r["cycle"]["name"].endswith("move_only")]
    drift = float(np.median([r["carried_dh_rate"] / r["n_actions"]
                             for r in move_only])) if move_only else 0.0

    for res in results:
        res["drift_per_action"] = drift
        res["carried_dh_adj"] = res["carried_dh_rate"] - drift * res["n_actions"]
        res["transient_rate"] = ((res["burn_pred_cum"] - res["burn_true_cum"])
                                 / max(n_burn, 1)
                                 - drift * res["n_actions"])
        neutral = res["neutral"]
        res["exploit_naive_eig"] = bool(neutral
                                        and res["naive_eig_rate"] > EPS_PRED)
        res["exploit_carried_eig"] = bool(neutral
                                          and res["carried_eig_rate"] > EPS_PRED)
        res["exploit_carried_dh"] = bool(neutral
                                         and res["carried_dh_adj"] > EPS_PRED)
        res["exploit_both"] = bool(res["exploit_carried_eig"]
                                   and res["exploit_carried_dh"])
        res["exploit_transient"] = bool(neutral
                                        and res["transient_rate"] > EPS_PRED)
    results.sort(key=lambda r: r["carried_dh_adj"], reverse=True)
    return results, (b0, ref0, node0)


# Severity-ordered verdicts (index = severity level, used for ensemble
# aggregation). The flagship requires carried_eig AND carried_dh farming on
# the SAME cycle: dh-only farming is repaired by CIG (the agent's bonus is
# EIG ~ 0), eig-only farming is repaired by PBIM.
#
# NAMING (claim-freeze audit, 2 Aug 2026): "CIG"/"PBIM" in these tier
# strings denote the TRANSPLANTED PRINCIPLES (prefix-conditioned carried-
# belief accounting / potential-based realized-dH accounting), NOT the
# published estimators: CIG's estimator scores parameter-information over
# open-loop rollouts with no belief object (see uncfield/cig_kernel.py for
# the faithful implementation); PBIM's guarantee is a terminal correction
# absent here (see uncfield/gamma_rescore.py for the discounted Ng-form
# arm). Paper-facing labels are carried/potential; the strings below are
# pinned by saved summaries and MUST NOT be renamed in code.
VERDICTS = (
    "NO-EXPLOIT",                    # 0: kill branch
    "EXPLOIT-NAIVE-ONLY",            # 1: CIG repairs -> kill branch
    "TRANSIENT-EXPLOIT-ONLY",        # 2: farming saturates inside burn window
    "EXPLOIT-SURVIVES-PBIM-ONLY",    # 3: dh-only; CIG repairs
    "EXPLOIT-SURVIVES-CIG-ONLY",     # 4: eig-only; PBIM repairs
    "EXPLOIT-SURVIVES-CIG+PBIM",     # 5: flagship signature (conjunction)
)


def classify(results):
    """Member verdict per the (post-review) decision tree."""
    if any(r["exploit_both"] for r in results):
        verdict = VERDICTS[5]
    elif any(r["exploit_carried_eig"] for r in results):
        verdict = VERDICTS[4]
    elif any(r["exploit_carried_dh"] for r in results):
        verdict = VERDICTS[3]
    elif any(r["exploit_transient"] for r in results):
        verdict = VERDICTS[2]
    elif any(r["exploit_naive_eig"] for r in results):
        verdict = VERDICTS[1]
    else:
        verdict = VERDICTS[0]
    counts = dict(
        n_cycles=len(results),
        n_neutral=sum(r["neutral"] for r in results),
        n_exploit_naive=sum(r["exploit_naive_eig"] for r in results),
        n_exploit_cig=sum(r["exploit_carried_eig"] for r in results),
        n_exploit_pbim=sum(r["exploit_carried_dh"] for r in results),
        n_exploit_both=sum(r["exploit_both"] for r in results),
        n_exploit_transient=sum(r["exploit_transient"] for r in results),
        drift_per_action=results[0]["drift_per_action"] if results else 0.0,
    )
    return verdict, counts
