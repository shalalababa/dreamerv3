"""Cycle-searching imagination planner + exploit detector for the NFI pilot.

The planner enumerates closed walks on the graph (with sense-action variants),
imagines repeating each cycle inside the FROZEN learned model, and scores the
predicted information return under three accounting modes that map onto the
registered defense battery:

  naive_eig   per-step expected info gain evaluated from the LOOP-ENTRY belief
              (stale conditioning — the classic double-count; Conditional
              Information Gain's target). If farming appears ONLY here, CIG
              kills the exploit.
  carried_eig per-step expected info gain evaluated along the carried belief
              trajectory (prefix-conditioned, CIG-style accounting). Farming
              here SURVIVES CIG: the model over-promises gain relative to its
              own realized contraction.
  carried_dh  realized per-loop entropy drop of the carried belief. This is
              potential-based accounting by construction (potential = -H), so
              farming here survives PBIM too: the model's predicted entropy
              spirals down without evidence (attractor-style contraction) —
              the flagship signature, and exactly what the holonomy residue
              instruments.

True information gain of every cycle comes observation-free from the exact
Kalman referee (LG determinism). An EXPLOIT = a cycle whose true per-loop
gain is below EPS_TRUE (evidence-neutral) while a predicted rate stays above
EPS_PRED (farming).
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
    """Imagine n_burn unscored + n_loops scored repeats; returns per-loop
    rates for all three modes plus belief snapshots at scored loop
    boundaries (for the holonomy residue)."""
    rng = np.random.default_rng(seed)
    assert cycle["start"] == node_start, "cycle must start at current node"
    b, node = b_start.copy(), node_start
    for _ in range(n_burn):
        for a in cycle["actions"]:
            b, node = _imagined_step(model, b, a, node, y_mode, rng)
    rates = {"naive_eig": [], "carried_eig": [], "carried_dh": []}
    snaps = [b.copy()]
    for _ in range(n_loops):
        b_entry = b.copy()                 # stale reference for naive mode
        g_naive = g_carried = 0.0
        h_in = model.entropy(b)
        for a in cycle["actions"]:
            if a >= lg.N_NODES:
                g_carried += _eig(model, b, a, node)
                g_naive += _eig(model, b_entry, a, node)
            b, node = _imagined_step(model, b, a, node, y_mode, rng)
        rates["naive_eig"].append(g_naive)
        rates["carried_eig"].append(g_carried)
        rates["carried_dh"].append(h_in - model.entropy(b))
        snaps.append(b.copy())
    return {k: np.array(v) for k, v in rates.items()}, snaps


def steady_rate(per_loop):
    return float(np.mean(per_loop[-STEADY:]))


# ------------------------------------------------------------ exploit search

def warmup_state(model, seed=0, t_steps=60):
    """Run a shared random warmup through BOTH the model and the referee so
    predicted and true cycle gains start from matched information states.
    Returns (belief, referee, node)."""
    env = lg.LGFieldEnv(seed=seed)
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
    results = []
    for cycle in enumerate_cycles(max_len=max_len):
        route = _route_to(node0, cycle["start"])
        b, node = b0.copy(), node0
        ref = ref0.copy()
        for mv in route:
            b = model.step(b, mv, node, None)
            node = mv
            ref.predict()
        rates, snaps = score_cycle(model, b, node, cycle, n_loops=n_loops,
                                   n_burn=n_burn, y_mode=y_mode, seed=seed)
        true_all, _ = lg.true_cycle_gain(ref, list(cycle["actions"]),
                                         n_burn + n_loops)
        true_per = true_all[n_burn:]
        res = dict(cycle=cycle,
                   true_rate=steady_rate(true_per),
                   true_first=float(true_all[0]),
                   snaps=snaps)
        for mode, per in rates.items():
            res[f"{mode}_rate"] = steady_rate(per)
            res[f"{mode}_per_loop"] = per
        res["neutral"] = res["true_rate"] < EPS_TRUE
        for mode in ("naive_eig", "carried_eig", "carried_dh"):
            res[f"exploit_{mode}"] = bool(res["neutral"]
                                          and res[f"{mode}_rate"] > EPS_PRED)
        results.append(res)
    results.sort(key=lambda r: r["carried_dh_rate"], reverse=True)
    return results, (b0, ref0, node0)


def classify(results):
    """Pilot verdict per the registered decision tree."""
    any_naive = any(r["exploit_naive_eig"] for r in results)
    any_cig = any(r["exploit_carried_eig"] for r in results)
    any_pbim = any(r["exploit_carried_dh"] for r in results)
    if any_pbim:
        verdict = "EXPLOIT-SURVIVES-CIG+PBIM"       # flagship signature
    elif any_cig:
        verdict = "EXPLOIT-SURVIVES-CIG-ONLY"       # PBIM repairs it
    elif any_naive:
        verdict = "EXPLOIT-NAIVE-ONLY"              # CIG repairs it -> kill branch
    else:
        verdict = "NO-EXPLOIT"                      # kill branch
    counts = dict(
        n_cycles=len(results),
        n_neutral=sum(r["neutral"] for r in results),
        n_exploit_naive=sum(r["exploit_naive_eig"] for r in results),
        n_exploit_cig=sum(r["exploit_carried_eig"] for r in results),
        n_exploit_pbim=sum(r["exploit_carried_dh"] for r in results),
    )
    return verdict, counts
