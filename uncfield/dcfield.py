"""Discrete-chain uncertainty-field world + exact joint Bayes referee.

FAMILY 2 of the NFI generality factorial (family 1 = lgfield.py). Same
6-node graph, same sensor CATALOG structure (duplicate static pair on the
chord loop, an overlap sensor, a persistently-informative dynamic sensor, a
noisy-TV channel, a lone static sensor), but a genuinely different
mathematics: the latent is a product of 8 two-state Markov chains (chains
0-3 static/identity, 4-7 symmetric flips), observations are SYMBOLS with
deterministic emissions (TV excepted), and the referee is the exact
discrete-Bayes forward filter over the full 256-state joint.

Design theorem (certified at runtime, see true_cycle_gain): with uniform
priors, deterministic emissions and SYMMETRIC dynamic chains, the referee's
posterior-entropy path along any action sequence is independent of the
realized observations — static-chain posteriors are always uniform on a
consistent set whose size depends only on the read history; dynamic-chain
posteriors are always k-step diffusions of a point mass, whose entropy
depends only on k. So "true information gain of a cycle" is observation-free
here exactly as it is in the linear-Gaussian family, by a different
argument. true_cycle_gain re-verifies the branch-entropy equality at every
imagined read (CERT_TOL) instead of assuming it; selfcheck additionally
replays sampled observation paths and asserts identical gain sequences.
"""

from __future__ import annotations

import dataclasses

import numpy as np

N_CHAINS = 8            # chains 0-3 static, 4-7 dynamic symmetric flips
FLIP_P = 0.2            # dynamic-chain per-step flip probability
N_STATES = 2 ** N_CHAINS
N_NODES = 6
N_SYMBOLS = 2
CERT_TOL = 1e-9         # observation-independence certificate tolerance
OBS_KIND = "categorical"

# Same graph as family 1: ring 0-1-2-3-4-5-0 plus chord 1-4.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
ADJ = {i: sorted({b for a, b in EDGES if a == i} | {a for a, b in EDGES if b == i})
       for i in range(N_NODES)}


@dataclasses.dataclass(frozen=True)
class Sensor:
    name: str
    kind: str              # "chain" | "xor" | "tv"
    chains: tuple          # chain indices read (empty for tv)
    node: int


# Catalog mirrors family 1 slot-for-slot:
#   s0/s1 dup pair (z0 @ 1/4)   -> d0/d1 dup pair (chain0 @ 1/4)
#   s2/s3 dup pair (z1 @ 2/5)   -> d2/d3 dup pair (chain1 @ 2/5)
#   s4 overlap z0+z1 @ 0        -> d4 XOR(chain0, chain1) @ 0
#   s5 dynamic z4 @ 0           -> d5 dynamic chain4 @ 0
#   s6 noisy-TV @ 3             -> d6 TV @ 3
#   s7 static z2 @ 3            -> d7 static chain2 @ 3
SENSORS = [
    Sensor("d0_c0_n1", "chain", (0,), 1),
    Sensor("d1_c0_n4", "chain", (0,), 4),
    Sensor("d2_c1_n2", "chain", (1,), 2),
    Sensor("d3_c1_n5", "chain", (1,), 5),
    Sensor("d4_xor01_n0", "xor", (0, 1), 0),
    Sensor("d5_c4_n0", "chain", (4,), 0),
    Sensor("d6_tv_n3", "tv", (), 3),
    Sensor("d7_c2_n3", "chain", (2,), 3),
]
N_SENSORS = len(SENSORS)
N_ACTIONS = N_NODES + N_SENSORS


def action_name(a):
    if a < N_NODES:
        return f"move{a}"
    return SENSORS[a - N_NODES].name


def valid_actions(node):
    acts = [j for j in ADJ[node]]
    acts += [N_NODES + k for k, s in enumerate(SENSORS) if s.node == node]
    return acts


def emit_symbol(sensor, state_bits):
    """Deterministic emission (TV handled by the caller with its own rng)."""
    if sensor.kind == "chain":
        return int(state_bits[sensor.chains[0]])
    if sensor.kind == "xor":
        return int(state_bits[sensor.chains[0]] ^ state_bits[sensor.chains[1]])
    raise ValueError("tv emission is stochastic; sample it at the call site")


def entropy_of(b):
    """Shannon entropy (nats) of a joint belief tensor/vector."""
    p = np.asarray(b, np.float64).ravel()
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log(nz)))


class ChainReferee:
    """Exact forward filter over the full 256-state joint (shape (2,)*8)."""

    def __init__(self):
        self.b = np.full((2,) * N_CHAINS, 1.0 / N_STATES)

    def copy(self):
        r = ChainReferee.__new__(ChainReferee)
        r.b = self.b.copy()
        return r

    def entropy(self):
        return entropy_of(self.b)

    def predict(self):
        for c in range(N_CHAINS):
            if c >= 4:                      # dynamic: symmetric flip
                self.b = (1.0 - FLIP_P) * self.b + FLIP_P * np.flip(self.b, axis=c)
        # static chains: identity (nothing to do)

    def _likelihood(self, sensor, y):
        """(2,)*8 tensor of P(y | state) for a deterministic sensor."""
        shape = [1] * N_CHAINS
        if sensor.kind == "chain":
            c = sensor.chains[0]
            lik = np.zeros((2,) * N_CHAINS)
            idx = [slice(None)] * N_CHAINS
            idx[c] = y
            lik[tuple(idx)] = 1.0
            return lik
        if sensor.kind == "xor":
            c0, c1 = sensor.chains
            bits = np.indices((2,) * N_CHAINS)
            return ((bits[c0] ^ bits[c1]) == y).astype(np.float64)
        raise ValueError(sensor.kind)

    def obs_probs(self, sensor):
        """Predictive symbol distribution under the current belief."""
        if sensor.kind == "tv":
            return np.array([0.5, 0.5])
        return np.array([float(np.sum(self.b * self._likelihood(sensor, y)))
                         for y in range(N_SYMBOLS)])

    def update(self, sensor, y):
        """Condition on observed symbol; returns exact info gain (nats)."""
        if sensor.kind == "tv":                 # state-independent: zero gain
            return 0.0
        assert y is not None, "discrete referee needs the realized symbol"
        h0 = self.entropy()
        post = self.b * self._likelihood(sensor, int(y))
        z = post.sum()
        assert z > 0.0, f"zero-probability observation {y} on {sensor.name}"
        self.b = post / z
        return h0 - self.entropy()

    def eig_of_update(self, sensor):
        """Exact one-step expected info gain by symbol enumeration
        (non-mutating; the branch entropies are also the certificate data)."""
        if sensor.kind == "tv":
            return 0.0, np.array([0.5, 0.5]), None
        q = self.obs_probs(sensor)
        h0 = self.entropy()
        branch_h = []
        for y in range(N_SYMBOLS):
            if q[y] <= 1e-12:
                branch_h.append(None)
                continue
            r = self.copy()
            r.update(sensor, y)
            branch_h.append(r.entropy())
        exp_h = sum(q[y] * h for y, h in enumerate(branch_h) if h is not None)
        return h0 - exp_h, q, branch_h


class ChainFieldEnv:
    """The real environment: joint chain state + agent position."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.bits = self.rng.integers(0, 2, size=N_CHAINS)
        self.node = 0
        return self.node

    def step(self, a):
        """Returns (node, y, sensed_k); y is None for moves. The latent
        advances every step (static chains unaffected by construction)."""
        y, sensed = None, None
        if a < N_NODES:
            assert a in ADJ[self.node], f"invalid move {a} from {self.node}"
            self.node = a
        else:
            k = a - N_NODES
            s = SENSORS[k]
            assert s.node == self.node, f"{s.name} unavailable at {self.node}"
            if s.kind == "tv":
                y = int(self.rng.integers(0, N_SYMBOLS))
            else:
                y = emit_symbol(s, self.bits)
            sensed = k
        flips = self.rng.random(N_CHAINS) < FLIP_P
        flips[:4] = False                       # static chains never move
        self.bits = self.bits ^ flips
        return self.node, y, sensed


def rollout_random(env, referee, t_steps, rng):
    """Random-policy episode; same record schema as family 1 where consumed
    (action / node / y / z_after / true_gain)."""
    recs = []
    for _ in range(t_steps):
        node = env.node
        a = int(rng.choice(valid_actions(node)))
        bits_before = env.bits.copy()
        _, y, sensed = env.step(a)
        gain = 0.0
        if sensed is not None:
            gain = referee.update(SENSORS[sensed], y)
        referee.predict()
        recs.append(dict(node=node, action=a, y=y, sensed=sensed,
                         z=bits_before.astype(np.float64),
                         z_after=env.bits.astype(np.float64), true_gain=gain))
    return recs


def true_cycle_gain(referee, cycle_actions, n_repeats):
    """Exact info gain of repeating an action cycle, per repeat (nats).

    Observation-free by the design theorem; RE-CERTIFIED at every read: all
    reachable observation branches must have equal posterior entropy (spread
    < CERT_TOL), so the canonical branch's gain sequence is every
    realization's. Mirrors lgfield.true_cycle_gain's contract exactly.

    Certificate scope (review n1, 7 Aug): the one-step canonical-path check
    SUFFICES for this catalog because the static block's belief is always
    uniform-on-a-coset (uniform prior, 0/1 likelihood masks, identity
    dynamics) and every shipped sensor is an affine functional of <=2
    chains, so each split's two branch posteriors are related by a bit-flip
    symmetry of the whole generative model (isomorphic subtrees); dynamic
    reads collapse to point masses. Future catalog extensions (non-affine
    multi-chain sensors, non-uniform priors) must NOT rely on this cert
    alone for deep branches.
    """
    ref = referee.copy()
    per = []
    for _ in range(n_repeats):
        g = 0.0
        for a in cycle_actions:
            if a >= N_NODES:
                s = SENSORS[a - N_NODES]
                gain, q, branch_h = ref.eig_of_update(s)
                if branch_h is not None:
                    live = [h for h in branch_h if h is not None]
                    assert max(live) - min(live) < CERT_TOL, (
                        f"observation-independence certificate FAILED on "
                        f"{s.name}: branch entropies {live}")
                    y_canon = int(np.argmax(q))
                    ref.update(s, y_canon)
                g += gain
            ref.predict()
        per.append(g)
    return np.array(per), ref


def _binary_entropy(p):
    p = float(np.clip(p, 1e-15, 1.0 - 1e-15))
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def selfcheck():
    """Referee integrity in the discrete family (mirrors lgfield.selfcheck)."""
    out = {}
    ln2 = np.log(2.0)

    # (1) Chain rule: pure-update gains sum to the batch entropy drop.
    ref = ChainReferee()
    h0 = ref.entropy()
    total = 0.0
    for k in (0, 2, 7, 4):
        s = SENSORS[k]
        y = 0 if s.kind != "xor" else 0
        total += ref.update(s, y)
    out["chain_rule_err"] = abs((h0 - ref.entropy()) - total)
    assert out["chain_rule_err"] < 1e-9

    # (2) Duplicate saturation is EXACT here: first read of chain0 gains ln2,
    #     every later read of either duplicate sensor gains exactly 0.
    ref = ChainReferee()
    g1 = ref.update(SENSORS[0], 1)
    ref.predict()
    g2 = ref.update(SENSORS[1], 1)
    out["dup_gain_first"], out["dup_gain_second"] = g1, g2
    assert abs(g1 - ln2) < 1e-12 and abs(g2) < 1e-12

    # (3) Noisy-TV gain is exactly zero.
    ref = ChainReferee()
    out["tv_gain"] = ref.update(SENSORS[6], 1)
    assert out["tv_gain"] == 0.0

    # (4) Dynamic persistence + closed form: after a read, a k-step-diffused
    #     point mass has read-gain H2((1+(1-2p)^k)/2), > EPS_TRUE forever.
    ref = ChainReferee()
    ref.update(SENSORS[5], 0)
    for k in (1, 4, 8):
        r = ref.copy()
        for _ in range(k):
            r.predict()
        gain, _, _ = r.eig_of_update(SENSORS[5])
        want = _binary_entropy(0.5 * (1.0 + (1.0 - 2.0 * FLIP_P) ** k))
        out[f"dyn_gain_k{k}"] = gain
        assert abs(gain - want) < 1e-9
    assert out["dyn_gain_k4"] > 0.02          # persistently informative

    # (5) XOR structure: fresh pair -> ln2; then a chain0 read identifies
    #     BOTH (ln2 again via the constraint); afterwards every pair sensor
    #     (both duplicates of both chains + XOR itself) gains exactly 0.
    ref = ChainReferee()
    gx = ref.update(SENSORS[4], 1)
    g0 = ref.update(SENSORS[0], 0)
    out["xor_fresh"], out["xor_then_c0"] = gx, g0
    assert abs(gx - ln2) < 1e-12 and abs(g0 - ln2) < 1e-12
    for k in (0, 1, 2, 3, 4):
        y = 0 if SENSORS[k].kind == "chain" and SENSORS[k].chains[0] == 0 else 1
        g = ref.copy().update(SENSORS[k], y if k != 4 else 1)
        out[f"post_pair_gain_{k}"] = g
        assert abs(g) < 1e-12

    # (6) Chord-loop neutrality is exact: sense d0@1, move 4, sense d1,
    #     move 1 — repeat 1 gains ln2, repeats 2+ gain exactly 0.
    ref = ChainReferee()
    cyc = [N_NODES + 0, 4, N_NODES + 1, 1]
    per, _ = true_cycle_gain(ref, cyc, 40)
    out["loop_gain_first"], out["loop_gain_rest_max"] = float(per[0]), float(np.abs(per[1:]).max())
    assert abs(per[0] - ln2) < 1e-12 and np.abs(per[1:]).max() < 1e-12

    # (7) Observation-independence: replay each probe cycle with SAMPLED
    #     observation paths (y ~ referee predictive at every read = the true
    #     marginal law; deterministic emissions make its support the full
    #     consistent set) and assert per-loop gain sequences identical to the
    #     canonical true_cycle_gain path.
    rng = np.random.default_rng(0)
    probes = [
        [N_NODES + 0, 4, N_NODES + 1, 1],                 # duplicate chord
        [N_NODES + 4, N_NODES + 5, 1, 0],                 # xor + dynamic loop
        [N_NODES + 7, N_NODES + 6, 2, 3],                 # static + tv loop
    ]
    max_dev = 0.0
    for cyc in probes:
        canon, _ = true_cycle_gain(ChainReferee(), cyc, 6)
        for _ in range(25):
            ref = ChainReferee()
            per = []
            for _rep in range(6):
                g = 0.0
                for a in cyc:
                    if a >= N_NODES:
                        s = SENSORS[a - N_NODES]
                        q = ref.obs_probs(s)
                        y = int(rng.choice(N_SYMBOLS, p=q / q.sum()))
                        g += ref.update(s, y)
                    ref.predict()
                per.append(g)
            max_dev = max(max_dev, float(np.abs(np.array(per) - canon).max()))
    out["realization_independence_dev"] = max_dev
    assert max_dev < 1e-9

    return out


# Canonical aliases consumed by the world-parameterized planner.
Env = ChainFieldEnv
Referee = ChainReferee


if __name__ == "__main__":
    for k, v in selfcheck().items():
        print(f"  {k}: {v:.3e}" if isinstance(v, float) else f"  {k}: {v}")
    print("dcfield selfcheck PASS")
