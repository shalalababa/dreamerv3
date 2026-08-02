"""The three killed-branch residues, instrumented inside the NFI pilot.

R1  Commutator / loop-holonomy instrument (reopened commutator branch):
    pairwise update commutators K_{A,B} and per-loop holonomy h(gamma) of the
    learned belief map, measured in head-space (symmetric KL between the
    induced diagonal Gaussians, so the metric is invariant to the arbitrary
    belief embedding). Mechanism hypothesis under test: holonomy magnitude /
    signed potential drift predicts which evidence-neutral cycles farm
    predicted information (and which the planner exploits).

R2  Synergy-horizon environments (higher-order synergy residue): a discrete
    XOR world with an exact enumeration referee where marginal observations
    carry zero information about the target but pairs carry one bit.
    Pilot role: adversarial benchmark — does the learned/predicted gain
    respect I(Z;Y_A)=0 while I(Z;Y_A,Y_B)=1 bit?

R3  H* residue (adaptivity value): closed-loop vs open-loop predicted value
    under the learned model, before and after cycle-free (potential-based)
    accounting. Supporting analysis only, per the corrected triage.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import lgfield as lg


# ------------------------------------------------- R1: commutator / holonomy

def _head_gauss(model, b):
    mu, lv = model.zstats(b)
    return mu, np.exp(lv)


def sym_kl(model, b1, b2):
    """Symmetric KL between the two beliefs' diagonal z-head Gaussians —
    an embedding-invariant belief distance."""
    mu1, v1 = _head_gauss(model, b1)
    mu2, v2 = _head_gauss(model, b2)
    kl12 = 0.5 * np.sum(np.log(v2 / v1) + (v1 + (mu1 - mu2) ** 2) / v2 - 1.0)
    kl21 = 0.5 * np.sum(np.log(v1 / v2) + (v2 + (mu1 - mu2) ** 2) / v1 - 1.0)
    return float(kl12 + kl21)


def pair_commutator(model, b, node, act_a, act_b, y_a, y_b):
    """K_{A,B}: process the same two (action, observation) pieces of evidence
    in both orders from belief b; distance between the results."""
    b_ab = model.step(model.step(b, act_a, node, y_a), act_b, node, y_b)
    b_ba = model.step(model.step(b, act_b, node, y_b), act_a, node, y_a)
    return sym_kl(model, b_ab, b_ba)


def loop_holonomy(model, snaps):
    """Per-loop holonomy of a scored cycle: distance between consecutive
    loop-boundary beliefs (a coherent model on an evidence-saturated neutral
    loop should have h -> 0), plus the signed potential drift dPhi = -dH."""
    hs, dphi = [], []
    for b_prev, b_next in zip(snaps[:-1], snaps[1:]):
        hs.append(sym_kl(model, b_prev, b_next))
        dphi.append(model.entropy(b_prev) - model.entropy(b_next))
    return np.array(hs), np.array(dphi)


def holonomy_report(model, results):
    """Attach holonomy stats to planner results; return the correlation
    between steady holonomy and each farming rate across NEUTRAL cycles —
    the mechanism-hypothesis read."""
    rows = []
    for r in results:
        hs, dphi = loop_holonomy(model, r["snaps"])
        r["holonomy_rate"] = float(np.mean(hs[-4:]))
        r["dphi_rate"] = float(np.mean(dphi[-4:]))
        if r["neutral"]:
            rows.append((r["holonomy_rate"], r["carried_eig_rate"],
                         r["carried_dh_rate"]))
    out = {}
    if len(rows) >= 3:
        arr = np.array(rows)
        with np.errstate(invalid="ignore"):
            out["corr_holonomy_carried_eig"] = float(
                np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            out["corr_holonomy_carried_dh"] = float(
                np.corrcoef(arr[:, 0], arr[:, 2])[0, 1])
    out["n_neutral_cycles"] = len(rows)
    return out


# ------------------------------------------------------ R2: synergy XOR world

class SynergyEnv:
    """Discrete world: X_A, X_B ~ Bernoulli(1/2) iid, target Z = X_A xor X_B.
    Sensors: read_A / read_B (flip noise p), read_XOR (flip noise q).
    Exact referee by enumeration over the 4 latent states."""

    FLIP_P = 0.1

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.integers(0, 2, size=2)

    def sense(self, which):
        if which == "A":
            v = self.x[0]
        elif which == "B":
            v = self.x[1]
        else:
            v = self.x[0] ^ self.x[1]
        flip = self.rng.random() < self.FLIP_P
        return int(v ^ flip)


def synergy_posterior(evidence, flip_p=SynergyEnv.FLIP_P):
    """Exact posterior over (X_A, X_B) given [(which, y), ...]."""
    post = np.full(4, 0.25)
    for which, y in evidence:
        lik = np.empty(4)
        for i, (xa, xb) in enumerate(itertools.product((0, 1), repeat=2)):
            v = {"A": xa, "B": xb, "XOR": xa ^ xb}[which]
            lik[i] = (1 - flip_p) if v == y else flip_p
        post = post * lik
        post = post / post.sum()
    return post


def _entropy_bits(p):
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def z_entropy_bits(post):
    pz1 = post[1] + post[2]                  # states (0,1) and (1,0): Z=1
    return _entropy_bits(np.array([1 - pz1, pz1]))


def exact_info_gains(flip_p=SynergyEnv.FLIP_P, n_mc=20000, seed=0):
    """Referee facts: I(Z; Y_A) = 0 exactly; I(Z; Y_A, Y_B) > 0."""
    rng = np.random.default_rng(seed)
    h0 = 1.0                                  # H(Z) = 1 bit
    ga, gab = [], []
    for _ in range(n_mc):
        env = SynergyEnv(seed=int(rng.integers(1 << 31)))
        ya = env.sense("A")
        ga.append(h0 - z_entropy_bits(synergy_posterior([("A", ya)])))
        yb = env.sense("B")
        gab.append(h0 - z_entropy_bits(
            synergy_posterior([("A", ya), ("B", yb)])))
    return float(np.mean(ga)), float(np.mean(gab))


def synergy_selfcheck():
    ga, gab = exact_info_gains()
    assert abs(ga) < 5e-3, f"I(Z;Y_A) should be 0, got {ga}"
    assert gab > 0.2, f"I(Z;Y_A,Y_B) should be substantial, got {gab}"
    return dict(gain_A_alone=ga, gain_A_then_B=gab)


# ------------------------------------------------------------ R3: H* residue

def adaptivity_residue(model, b0, node0, cycle, n_branches=8, horizon=None,
                       seed=0):
    """Closed-loop vs open-loop predicted value of one cycle pass under the
    learned model, in raw carried-EIG accounting and in cycle-free
    (potential/dH) accounting. In the exact LG referee both are equal for
    covariance objectives (the determinism fact); a learned-model gap is a
    model artifact, and the residue asks whether repaired accounting
    shrinks it."""
    rng = np.random.default_rng(seed)
    horizon = horizon or len(cycle["actions"])
    actions = list(cycle["actions"])[:horizon]

    def rollout(sample):
        b, node = b0.copy(), node0
        g_eig = 0.0
        h_in = model.entropy(b)
        for a in actions:
            if a >= lg.N_NODES:
                mu, lv = model.obs_pred(b, a)
                if sample:
                    y = mu + np.exp(0.5 * lv) * rng.standard_normal()
                else:
                    y = mu                     # ML-observation = open-loop-ish
                g_here = model.entropy(b)
                b = model.step(b, a, node, y)
                g_eig += g_here - model.entropy(b)
            else:
                b = model.step(b, a, node, None)
                node = a
        return g_eig, h_in - model.entropy(b)

    open_eig, open_dh = rollout(sample=False)
    cl_eig, cl_dh = np.mean([rollout(sample=True) for _ in range(n_branches)],
                            axis=0)
    return dict(open_eig=float(open_eig), closed_eig=float(cl_eig),
                gap_eig=float(cl_eig - open_eig),
                open_dh=float(open_dh), closed_dh=float(cl_dh),
                gap_dh=float(cl_dh - open_dh))
