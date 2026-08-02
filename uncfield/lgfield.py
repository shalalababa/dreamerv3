"""Linear-Gaussian uncertainty-field environment + exact Kalman referee.

Paper-5 NFI pilot substrate (see research_notes/Pilot_NFI_Design_20260801.md).

World: latent field z in R^8 = 4 STATIC structural components (A=I, Q=0;
once measured to precision, re-measurement gain -> 0: the evidence-neutral
farm targets) + 4 DYNAMIC components (stable rotations, Q>0; legitimately
re-informative). An agent moves on a 6-node graph; each node offers a subset
of sensors, including a DUPLICATE pair reading the same static functional at
two different nodes (so a physical loop re-reads the same evidence) and a
noisy-TV channel (C=0: observation carries no field information).

The exact Kalman filter is the referee. In this LG world the covariance path
is a deterministic function of the action path (never of realized
observations), so true information gain of any action sequence is computable
exactly and observation-free.
"""

from __future__ import annotations

import dataclasses

import numpy as np

DZ = 8                # latent dimension: components 0-3 static, 4-7 dynamic
N_NODES = 6
JITTER = 1e-10        # logdet stabilizer only; never enters the filter

# Graph: ring 0-1-2-3-4-5-0 plus chord 1-4 (tight duplicate-sensor loop).
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
ADJ = {i: sorted({b for a, b in EDGES if a == i} | {a for a, b in EDGES if b == i})
       for i in range(N_NODES)}


def _sensor_row(idx):
    c = np.zeros(DZ)
    for i, v in idx:
        c[i] = v
    return c


@dataclasses.dataclass(frozen=True)
class Sensor:
    name: str
    c: np.ndarray          # (DZ,) observation row
    r: float               # observation noise variance
    node: int              # node where available


# Duplicate pair s0/s1 (z0 @ nodes 1 and 4); duplicate pair s2/s3 (z1 @ 2/5);
# overlap s4 (z0+z1); dynamic s5 (z4); noisy-TV s6 (C=0); static s7 (z2).
SENSORS = [
    Sensor("s0_z0_n1", _sensor_row([(0, 1.0)]), 0.05, 1),
    Sensor("s1_z0_n4", _sensor_row([(0, 1.0)]), 0.05, 4),
    Sensor("s2_z1_n2", _sensor_row([(1, 1.0)]), 0.05, 2),
    Sensor("s3_z1_n5", _sensor_row([(1, 1.0)]), 0.05, 5),
    Sensor("s4_z01_n0", _sensor_row([(0, 1.0), (1, 1.0)]), 0.10, 0),
    Sensor("s5_z4_n0", _sensor_row([(4, 1.0)]), 0.10, 0),
    Sensor("s6_tv_n3", np.zeros(DZ), 1.00, 3),
    Sensor("s7_z2_n3", _sensor_row([(2, 1.0)]), 0.05, 3),
]
N_SENSORS = len(SENSORS)

# Global discrete action space: move-to-node j (valid iff adjacent), then
# sense-k (valid iff SENSORS[k].node == current node).
N_ACTIONS = N_NODES + N_SENSORS


def action_name(a):
    if a < N_NODES:
        return f"move{a}"
    return SENSORS[a - N_NODES].name


def valid_actions(node):
    acts = [j for j in ADJ[node]]
    acts += [N_NODES + k for k, s in enumerate(SENSORS) if s.node == node]
    return acts


def make_dynamics():
    """A (DZ,DZ), Q (DZ,DZ): static identity block + two damped rotations."""
    a = np.eye(DZ)
    q = np.zeros((DZ, DZ))
    for blk, theta in ((4, 0.30), (6, 0.11)):
        rot = 0.90 * np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        a[blk:blk + 2, blk:blk + 2] = rot
        q[blk:blk + 2, blk:blk + 2] = 0.02 * np.eye(2)
    return a, q


A_MAT, Q_MAT = make_dynamics()
P0 = np.eye(DZ)                     # prior covariance
MU0 = np.zeros(DZ)


def entropy(p):
    """Gaussian differential entropy (nats) with jitter-stabilized logdet."""
    sign, logdet = np.linalg.slogdet(p + JITTER * np.eye(DZ))
    assert sign > 0, "covariance lost positive definiteness"
    return 0.5 * (DZ * np.log(2.0 * np.pi * np.e) + logdet)


class KalmanReferee:
    """Exact filter. Covariance path depends only on the action path."""

    def __init__(self):
        self.mu = MU0.copy()
        self.p = P0.copy()

    def copy(self):
        k = KalmanReferee.__new__(KalmanReferee)
        k.mu, k.p = self.mu.copy(), self.p.copy()
        return k

    def predict(self):
        self.mu = A_MAT @ self.mu
        self.p = A_MAT @ self.p @ A_MAT.T + Q_MAT

    def update(self, sensor, y):
        """Returns exact information gain (nats) of this update."""
        c, r = sensor.c, sensor.r
        if not np.any(c):                      # noisy-TV: no field information
            return 0.0
        h_before = entropy(self.p)
        s = float(c @ self.p @ c + r)
        k = (self.p @ c) / s
        if y is not None:
            self.mu = self.mu + k * (y - float(c @ self.mu))
        self.p = self.p - np.outer(k, c @ self.p)
        self.p = 0.5 * (self.p + self.p.T)     # symmetrize
        return h_before - entropy(self.p)

    def gain_of_update(self, sensor):
        """Observation-free gain (the LG determinism fact), non-mutating."""
        k = self.copy()
        return k.update(sensor, None)


class LGFieldEnv:
    """The real environment: latent field + agent position on the graph."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.z = self.rng.multivariate_normal(MU0, P0)
        self.node = 0
        return self.node

    def step(self, a):
        """Returns (node, y, sensed_k): y is None for moves; latent advances
        every step (static components are unaffected by construction)."""
        y, sensed = None, None
        if a < N_NODES:
            assert a in ADJ[self.node], f"invalid move {a} from {self.node}"
            self.node = a
        else:
            k = a - N_NODES
            s = SENSORS[k]
            assert s.node == self.node, f"{s.name} unavailable at {self.node}"
            y = float(s.c @ self.z + self.rng.normal(0.0, np.sqrt(s.r)))
            sensed = k
        w = self.rng.multivariate_normal(np.zeros(DZ), Q_MAT)
        self.z = A_MAT @ self.z + w
        return self.node, y, sensed


def rollout_random(env, referee, t_steps, rng):
    """Random-policy episode; returns per-step records incl. exact gains."""
    recs = []
    for _ in range(t_steps):
        node = env.node
        a = int(rng.choice(valid_actions(node)))
        z_before = env.z.copy()
        _, y, sensed = env.step(a)
        gain = 0.0
        if sensed is not None:
            gain = referee.update(SENSORS[sensed], y)
        referee.predict()
        recs.append(dict(node=node, action=a, y=y, sensed=sensed,
                         z=z_before, z_after=env.z.copy(), true_gain=gain,
                         ref_mu=referee.mu.copy(),
                         ref_logvar=np.log(np.diag(referee.p) + JITTER)))
    return recs


def true_cycle_gain(referee, cycle_actions, n_repeats):
    """Exact info gain of repeating an action cycle, per repeat (nats).

    Observation-free by LG determinism. Returns (per_repeat_gains, final_ref).
    """
    ref = referee.copy()
    per = []
    for _ in range(n_repeats):
        g = 0.0
        for a in cycle_actions:
            if a >= N_NODES:
                g += ref.gain_of_update(SENSORS[a - N_NODES])
                ref.update(SENSORS[a - N_NODES], None)
            ref.predict()
        per.append(g)
    return np.array(per), ref


def selfcheck():
    """Referee integrity: chain rule, duplicate decay, TV-zero, commutator."""
    out = {}

    # (1) Chain rule: sum of per-step gains == batch entropy drop.
    ref = KalmanReferee()
    h0 = entropy(ref.p)
    total = 0.0
    seq = [SENSORS[0], SENSORS[2], SENSORS[7], SENSORS[0], SENSORS[5]]
    for s in seq:
        total += ref.update(s, None)
        # no predict: pure-update chain rule must be exact
    out["chain_rule_err"] = abs((h0 - entropy(ref.p)) - total)
    assert out["chain_rule_err"] < 1e-8

    # (2) Repeated-read gain on a static component decays like 1/n
    #     (Gaussian precision adds linearly; the referee must show it).
    ref = KalmanReferee()
    gains = []
    for _ in range(60):
        gains.append(ref.update(SENSORS[0], None))
        ref.predict()
    out["dup_gain_first"] = gains[0]
    out["dup_gain_last"] = gains[-1]
    assert gains[0] > 0.5 and gains[-1] < 1e-2

    # (3) Noisy-TV sensor has exactly zero referee gain.
    ref = KalmanReferee()
    out["tv_gain"] = ref.update(SENSORS[6], None)
    assert out["tv_gain"] == 0.0

    # (4) Static-pair update commutator vanishes for the exact filter
    #     (pure updates, no predict between).
    ra, rb = KalmanReferee(), KalmanReferee()
    for s in (SENSORS[0], SENSORS[2]):
        ra.update(s, None)
    for s in (SENSORS[2], SENSORS[0]):
        rb.update(s, None)
    out["static_pair_commutator"] = float(np.abs(ra.p - rb.p).max())
    assert out["static_pair_commutator"] < 1e-10

    # (5) Dynamic-pair WITH predicts between differs (physical component >0).
    ra, rb = KalmanReferee(), KalmanReferee()
    for s in (SENSORS[5], SENSORS[0]):
        ra.update(s, None)
        ra.predict()
    for s in (SENSORS[0], SENSORS[5]):
        rb.update(s, None)
        rb.predict()
    out["dynamic_pair_commutator"] = float(np.abs(ra.p - rb.p).max())
    assert out["dynamic_pair_commutator"] > 1e-6

    # (6) Asymptotically evidence-neutral loop: chord loop 1->4->1 sensing the
    #     z0 duplicate pair. Per-repeat true gain ~ 1/n; after 300 burn-in
    #     repeats it must sit below the pilot's neutrality floor 5e-3 nats.
    ref = KalmanReferee()
    cyc = [N_NODES + 0, 4, N_NODES + 1, 1]   # sense s0 @1, move 4, sense s1, move 1
    per, _ = true_cycle_gain(ref, cyc, 300)
    out["loop_gain_first"] = float(per[0])
    out["loop_gain_last"] = float(per[-1])
    assert per[0] > 0.5 and per[-1] < 5e-3

    return out


if __name__ == "__main__":
    for k, v in selfcheck().items():
        print(f"  {k}: {v:.3e}" if isinstance(v, float) else f"  {k}: {v}")
    print("lgfield selfcheck PASS")
