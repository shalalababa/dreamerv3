"""CEI graded-access pipeline (GAP): the access lattice OPERATIONALIZED.
Design: reviews/Algorithm_Ideation_CEI_20260820.md §6 (candidates #1+#2
merged, panel-recommended build); prereg PREREG_cei_gap_20260824.md.

The pipeline is a PROCEDURE, not a new test (the panel's fence): it
decides BETWEEN the passive/NIS family and intervention, and PRICES the
remainder, staging up the CEI access lattice:

  Stage A (L2+L3, free):  replay-coherence screen on the declared
      covariance path (cei2.l2_replay_residual) + the L3 counterfactual
      score-query battery — the SECOND LATTICE COLUMN: the queries
      IDENTIFY the operator's declared noise model exactly (or REFUSE for
      a non-conforming operator) while carrying ZERO falsification power
      about the world (query answers are covariance-only — bit-identical
      across worlds, asserted).  An incoherent operator is FLAGGED here
      at zero forced cost.
  Stage B (L0/L1, free):  count-only power certificate.  From the
      passive log's read count n and the L3-identified lie hypothesis,
      attach the ceiling  power - size <= sqrt(kappa * n / 2)  (the
      KCG/Wald object; kappa = per-read conditional-on-latent KL).  The
      certificate is a NECESSARY-condition screen: if the ceiling forbids
      the target, passivity is certified insufficient.  If it allows the
      target, the realized passive LR test on the log decides; an
      inconclusive realized test still escalates ("vacuity is
      informative" — a vacuous ceiling never certifies success).
  Stage C (L4, priced):  sequential probability-ratio top-up: walk
      (BFS distance) + forced reads under Wald boundaries at the
      registered (delta, gamma); realized E[N] is compared against the
      d(delta,1-delta)/kappa lower bound it should SATURATE, and against
      the fixed-sample m*(power) comparator.

Registered operator suite (all in-house, cei2 classes): U_true (honest),
U_false (coherent liar, rhat7 = 0.01), ParityContractKalman (incoherent
over-contractor).  Registered outcome cells + bars: see the prereg.

Run:  python -m uncfield.cei_gap --selfcheck | --run [--light]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import lgfield as lg
from .cei2 import (CEI2Env, GreedyPolicy, MisKalman, ParityContractKalman,
                   R7_FALSE, R7_TRUE, S7, LR_CONF, false_rhat, loglr,
                   l2_replay_residual, rollout, s7_loglik,
                   targeted_audit_power, true_rhat)

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "local_results" / "uncfield" / "cei_gap")

DELTA = 0.05                 # registered size/error level
TARGET_POWER = 0.95          # registered audit target
GAMMA_PILOT = 0.2            # the paper's pilot lie separation
L2_TOL = 1e-9                # coherence: replay residual below this
L3_TOL = 1e-6                # identification: relative rhat error below
CONF_LR = LR_CONF            # ln 19 — the house confidence criterion
T_ROLLOUT = 1000             # per-operator deployment horizon (E1 form)
# The FLIP arm (review #1): under this policy the passive s7 count is
# DETERMINISTIC at n = ceil(19 * rhat7) (score(s7) = rhat/(rhat+n)
# crosses TAU=0.05 there and z2 is static, so reads stop forever) —
# an honest operator in the R7=1 world can NEVER reach the n >= 207
# certificate flip.  The flip arm is therefore an HONEST operator in
# a HIGH-NOISE world: world R7 = declaration rhat7 = R7_HI, giving
# n ~= 19*48 = 913 passive reads and a ceiling that allows target.
R7_HI = 48.0
T_HI = 1200                  # ~913 consecutive senses finish by ~t=920


# ------------------------------------------------------------ theory bits

def kappa(gamma):
    """Per-read conditional-on-latent KL between the true read law
    N(z, R) and the lie N(z, (1+gamma) R): 0.5*(1/(1+g) - 1 + ln(1+g)).
    Pinned: kappa(0.2) = 0.00783 (rem:constants)."""
    g = float(gamma)
    return 0.5 * (1.0 / (1.0 + g) - 1.0 + np.log(1.0 + g))


def kappa_ratio(rhat, r=R7_TRUE):
    """Per-read conditional KL for an arbitrary assumed/true variance
    pair: D(N(z,r) || N(z,rhat)) = 0.5*(r/rhat - 1 - ln(r/rhat))."""
    x = float(r) / float(rhat)
    return 0.5 * (x - 1.0 - np.log(x))


def d_binary(a, b):
    """Binary relative entropy d(a, b) in nats."""
    return (a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b)))


def power_ceiling(kap, n):
    """The certificate: power - size <= sqrt(KL_n / 2), KL_n <= kappa*n
    (conditioning dominates the marginal by convexity)."""
    return float(min(1.0, np.sqrt(kap * float(n) / 2.0)))


def bfs_dist(a, b):
    front, seen, d = [a], {a}, 0
    while front:
        if b in front:
            return d
        front = [y for x in front for y in lg.ADJ[x] if y not in seen]
        seen |= set(front)
        d += 1
    raise AssertionError("disconnected")


# ------------------------------------------------- Stage A: L2 + L3 screen

class _world_r7:
    """Context manager: set the WORLD's s7 noise (cei2.WORLD_SENSORS)
    for one arm and restore it.  Used by the flip arm (honest operator
    in a high-noise world) and by the L3 world-independence assert."""

    def __init__(self, r7):
        self.r7 = r7

    def __enter__(self):
        import dataclasses
        from . import cei2
        self._saved = cei2.WORLD_SENSORS
        cei2.WORLD_SENSORS = [
            dataclasses.replace(s, r=self.r7) if k == S7 else s
            for k, s in enumerate(cei2.WORLD_SENSORS)]

    def __exit__(self, *exc):
        from . import cei2
        cei2.WORLD_SENSORS = self._saved
        return False


def l3_identify(make_operator, n_conform=8, seed=0):
    """L3 counterfactual score-query battery (covariance-only; no world
    data touches any answer; access scope — DISCLOSED in the prereg —
    includes instantiating the operator at its declared prior).
    (i) IDENTIFY rhat_k for every sensed key from two score queries
    around one update: post = pre - pre^2/(pre + rhat)  =>
    rhat = pre^2/(pre - post) - pre  (score(k) = c'Pc at the prior).
    (ii) CONFORMANCE: replay n_conform random query sequences against
    an exact Kalman at the identified rhat vector, comparing the score
    of EVERY sensed key at every step (review #3: an s7-only
    comparison is blind to incoherence outside the z2 marginal) — an
    operator no single-rhat Kalman explains is REFUSED.
    Returns (rhat7_hat | None, worst deviation, n_queries)."""
    rng = np.random.default_rng(seed)
    sensed = [k for k, s in enumerate(lg.SENSORS) if np.any(s.c)]
    n_q = 0
    fitted = {}
    for k in sensed:                        # identification: 2 queries/key
        o = make_operator()
        pre = o.score(k)
        o.update(k, None)
        post = o.score(k)
        n_q += 2
        denom = pre - post
        if denom <= 1e-15:
            return None, np.inf, n_q
        rhat_k = pre * pre / denom - pre
        if rhat_k <= 0:
            return None, np.inf, n_q
        fitted[k] = rhat_k
    worst = 0.0
    for _ in range(n_conform):
        seq = rng.choice(sensed, size=6)
        op2 = make_operator()
        ref = MisKalman({k: fitted.get(k, s.r)
                         for k, s in enumerate(lg.SENSORS)})
        for k in seq:
            op2.update(int(k), None)
            ref.update(int(k), None)
            op2.predict()
            ref.predict()
            for j in sensed:                # all-key comparison
                worst = max(worst, abs(op2.score(j) - ref.score(j)))
                n_q += 1
    if worst > L3_TOL:
        return None, float(worst), n_q
    return float(fitted[S7]), float(worst), n_q


def stage_a(make_operator, log):
    resid, fitted = l2_replay_residual(log["p_trace"], log["actions"])
    rhat_hat, conform, n_q = l3_identify(make_operator)
    coherent = bool(resid < L2_TOL and rhat_hat is not None)
    return dict(l2_residual=float(resid),
                l3_rhat7=rhat_hat, l3_conform_dev=float(conform),
                l3_queries=int(n_q),
                l2_path_len=len(log["p_trace"]),
                coherent=coherent)


# ---------------------------------------- Stage B: count-only certificate

def stage_b(log, rhat_hat):
    """Free: the passive log's s7 count against the REGISTERED lie
    class.  The audit hypothesis pair is DECLARATION-relative (the
    auditor never sees the true R): H0 world noise = rhat_hat (the
    declaration is true) vs H1 = rhat_hat*(1+GAMMA_PILOT) (an
    over-trust lie at the registered separation — the program's threat
    direction; scope disclosed in the prereg).  kappa is therefore
    kappa(GAMMA_PILOT) for EVERY coherent operator, and the ceiling is
    a property of the read COUNT alone — the count-only certificate.
    The realized passive LR on the log decides only when the ceiling
    allows the target AND the LR clears the confidence criterion;
    otherwise escalate ("vacuity is informative")."""
    n = len(log["s7_ys"])
    kap = kappa(GAMMA_PILOT)
    ceil_ = power_ceiling(kap, n)
    allowed = DELTA + ceil_ >= TARGET_POWER
    r0, r1 = rhat_hat, rhat_hat * (1.0 + GAMMA_PILOT)
    realized = (float(s7_loglik(log["s7_ys"], r0)
                      - s7_loglik(log["s7_ys"], r1)) if n else 0.0)
    decided = abs(realized) >= CONF_LR
    passive_sufficient = bool(allowed and decided)
    need = d_binary(DELTA, 1 - DELTA) / kap
    remainder = (None if passive_sufficient
                 else max(0.0, float(need - n)))
    return dict(n_passive_s7=n, kappa_lie=kap, ceiling=ceil_,
                ceiling_allows_target=bool(allowed),
                realized_loglr=realized, realized_decided=bool(decided),
                passive_sufficient=passive_sufficient,
                priced_remainder_reads=remainder)


# ------------------------------------------------- Stage C: priced top-up

def sprt_reads(r0, r1, delta, n_trials, seed, data_r=None, cap=100000):
    """Wald SPRT on forced s7 reads (fresh z2 ~ N(0,1) per trial; exact
    sequential predictives, incremental — O(n) per trial): H0 r0 vs
    H1 r1, boundaries +-ln((1-d)/d).  Data generated under data_r
    (default r0).  Returns realized read counts + the H0-acceptance
    fraction; err_rate is reported against whichever hypothesis data_r
    is closer to (for a data_r outside the pair, accept_h0_frac is the
    decision record and err_rate is None)."""
    rng = np.random.default_rng(seed)
    assert abs(r1 - r0) > 1e-12 * max(r0, r1), "degenerate SPRT pair"
    a_hi = np.log((1 - delta) / delta)
    counts, accept0 = [], 0
    dr = r0 if data_r is None else data_r
    capped = 0
    for _ in range(n_trials):
        z2 = rng.normal()
        # incremental twin predictives under r0 and r1
        m0 = m1 = 0.0
        v0 = v1 = 1.0
        lr = 0.0
        n = 0
        while abs(lr) < a_hi and n < cap:
            y = z2 + rng.normal(0.0, np.sqrt(dr))
            n += 1
            for which in (0, 1):
                r = r0 if which == 0 else r1
                m, v = (m0, v0) if which == 0 else (m1, v1)
                pv = v + r
                inc = -0.5 * (np.log(2 * np.pi * pv) + (y - m) ** 2 / pv)
                g = v / pv
                m += g * (y - m)
                v *= (1.0 - g)
                if which == 0:
                    m0, v0, lr = m, v, lr + inc
                else:
                    m1, v1, lr = m, v, lr - inc
        counts.append(n)
        capped += int(n >= cap)
        accept0 += int(lr > 0)
    inside = min((r0, r1), key=lambda r: abs(dr - r)) == r0
    err = (None if abs(dr - r0) > 1e-9 and abs(dr - r1) > 1e-9
           else (n_trials - accept0) / n_trials if inside
           else accept0 / n_trials)
    return dict(mean_reads=float(np.mean(counts)),
                median_reads=float(np.median(counts)),
                q90_reads=float(np.quantile(counts, 0.90)),
                accept_h0_frac=accept0 / n_trials,
                err_rate=err, n_capped=capped, n_trials=n_trials)


def fixed_sample_mstar(r0, r1, delta, power, seed, n_mc=200000,
                       lo=1, hi=4000):
    """Smallest m with size <= delta and power >= target for the exact-LR
    fixed-sample test (threshold = the (1-delta) H0 quantile), MC.
    Vectorized over trials; sequential predictive computed iteratively."""
    rng = np.random.default_rng(seed)

    def batch_lr(m, r_data, n):
        z2 = rng.normal(size=n)
        mu = np.zeros(n)
        v = np.ones(n)
        ll0 = np.zeros(n)
        mu1 = np.zeros(n)
        v1 = np.ones(n)
        ll1 = np.zeros(n)
        for _ in range(m):
            y = z2 + rng.normal(0.0, np.sqrt(r_data), size=n)
            for (mm, vv, ll, r) in ((mu, v, ll0, r0), (mu1, v1, ll1, r1)):
                pv = vv + r
                ll += -0.5 * (np.log(2 * np.pi * pv) + (y - mm) ** 2 / pv)
                g = vv / pv
                mm += g * (y - mm)
                vv *= (1.0 - g)
        return ll0 - ll1

    def ok(m):
        lr0 = batch_lr(m, r0, n_mc)          # H0 world
        thr = float(np.quantile(lr0, delta))  # reject H0 when LR small
        lr1 = batch_lr(m, r1, n_mc)          # H1 world
        return float(np.mean(lr1 < thr)) >= power

    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def stage_c(rhat_hat, delta, n_trials, seed, world_r):
    """Priced top-up: SPRT between the DECLARATION (H0: world noise =
    rhat_hat) and the registered lie alternative (H1: rhat_hat*(1+g)),
    with data drawn from the REAL world (s7 true noise world_r).
    Accepting H0 = DECLARATION-CERTIFIED, rejecting =
    DECLARATION-REFUTED.  Wald's error guarantee applies when the world
    lies INSIDE the pair; for a world far outside it (the suite's 100x
    liar) the SPRT is a decision procedure whose speed comes from the
    ACTUAL per-read KL (kappa ~ 47/read), not from the registered
    gamma — disclosed (review #12)."""
    walk = bfs_dist(0, lg.SENSORS[S7].node)
    out = dict(walk_steps=walk)
    out["sprt_declaration"] = sprt_reads(
        rhat_hat, rhat_hat * (1.0 + GAMMA_PILOT), delta, n_trials, seed,
        data_r=world_r)
    return out


# ------------------------------------------------------------ the pipeline

def pipeline(name, make_operator, n_trials, seed, t_steps=T_ROLLOUT,
             world_r=R7_TRUE):
    est = make_operator()
    log = rollout(est, t_steps, env_seed=1)
    a = stage_a(make_operator, log)
    # the access-cost ledger counts what was actually consumed
    # (review #5): L3 queries + the L2 declared-path length are the
    # free tier's itemized reads; forced reads + walk are the priced
    # tier
    rec = dict(operator=name, stage_a=a,
               cost=dict(l3_queries=a["l3_queries"],
                         l2_path_matrices=a["l2_path_len"],
                         forced_reads=0, walk_steps=0))
    if not a["coherent"]:
        rec["verdict"] = "FLAGGED-FREE"
        return rec
    b = stage_b(log, a["l3_rhat7"])
    rec["stage_b"] = b
    if b["passive_sufficient"]:
        rec["verdict"] = ("CLEARED-PASSIVE" if b["realized_loglr"] > 0
                          else "REJECTED-PASSIVE")
        return rec
    c = stage_c(a["l3_rhat7"], DELTA, n_trials, seed, world_r)
    rec["stage_c"] = c
    rec["cost"]["forced_reads"] = c["sprt_declaration"]["mean_reads"]
    rec["cost"]["walk_steps"] = c["walk_steps"]
    # decision by the majority of the audit trials (each trial = one
    # full forced-read audit under fresh z2)
    certified = c["sprt_declaration"]["accept_h0_frac"] >= 0.5
    rec["verdict"] = ("CERTIFIED-PRICED" if certified
                      else "CAUGHT-PRICED")
    return rec


# ------------------------------------------------- certificate soundness

def certificate_soundness(gamma, n_grid, n_mc, seed):
    """MC: sup over LR-threshold tests of (power - size) on n passive
    reads vs the ceiling sqrt(kappa*n/2).  Data marginal over z2 (the
    bound dominates the marginal by convexity)."""
    rng = np.random.default_rng(seed)
    kap = kappa(gamma)
    r1 = (1.0 + gamma) * R7_TRUE
    rows = []
    for n in n_grid:
        def lrs(r_data):
            z2 = rng.normal(size=n_mc)
            mu = np.zeros(n_mc); v = np.ones(n_mc); ll0 = np.zeros(n_mc)
            mu1 = np.zeros(n_mc); v1 = np.ones(n_mc); ll1 = np.zeros(n_mc)
            for _ in range(n):
                y = z2 + rng.normal(0.0, np.sqrt(r_data), size=n_mc)
                for (mm, vv, ll, r) in ((mu, v, ll0, R7_TRUE),
                                        (mu1, v1, ll1, r1)):
                    pv = vv + r
                    ll += -0.5 * (np.log(2 * np.pi * pv)
                                  + (y - mm) ** 2 / pv)
                    g = vv / pv
                    mm += g * (y - mm)
                    vv *= (1.0 - g)
            return ll0 - ll1
        l0, l1 = lrs(R7_TRUE), lrs(r1)
        thr = np.quantile(np.concatenate([l0, l1]),
                          np.linspace(0.001, 0.999, 499))
        gap = np.max(np.mean(l1[None, :] < thr[:, None], 1)
                     - np.mean(l0[None, :] < thr[:, None], 1))
        rows.append(dict(n=int(n), ceiling=power_ceiling(kap, n),
                         best_power_minus_size=float(gap),
                         violated=bool(gap > power_ceiling(kap, n) + 3 *
                                       np.sqrt(0.25 / n_mc))))
    return rows


# ------------------------------------------------------------- entrypoints

def selfcheck():
    out = {}
    out["kappa_pilot"] = kappa(GAMMA_PILOT)
    assert abs(out["kappa_pilot"] - 0.00783) < 2e-4, out["kappa_pilot"]
    out["d_binary"] = float(d_binary(DELTA, 1 - DELTA))
    out["wald_bound_reads"] = out["d_binary"] / out["kappa_pilot"]
    assert abs(out["wald_bound_reads"] - 339) < 3
    out["walk"] = bfs_dist(0, lg.SENSORS[S7].node)
    # L3 battery: identifies both Kalmans exactly, refuses parity
    r_t, _, nq_t = l3_identify(lambda: MisKalman(true_rhat()))
    r_f, _, _ = l3_identify(lambda: MisKalman(false_rhat()))
    r_p, dev_p, _ = l3_identify(
        lambda: ParityContractKalman(false_rhat()))
    assert abs(r_t - R7_TRUE) < L3_TOL and abs(r_f - R7_FALSE) < L3_TOL
    assert r_p is None and dev_p > L3_TOL
    out["l3"] = dict(rhat_true=r_t, rhat_false=r_f, parity_refused=True,
                     n_queries=nq_t)
    # parity refusal is a PROPERTY under the all-key comparison
    # (review #4): no conformance seed lets the double-contraction hide
    for sd in range(20):
        rp, dv, _ = l3_identify(
            lambda: ParityContractKalman(false_rhat()), seed=sd)
        assert rp is None and dv > L3_TOL, sd
    # L3 falsification power is ZERO (review #6): the battery touches
    # no world data, so its full output is bit-identical under
    # different WORLD noise — asserted by actually changing the world
    base = l3_identify(lambda: MisKalman(false_rhat()))
    for wr in (0.01, 48.0):
        with _world_r7(wr):
            assert l3_identify(lambda: MisKalman(false_rhat())) == base
    # ceiling monotone + vacuous exactly when Wald says so
    assert power_ceiling(out["kappa_pilot"], 339) < 1.0 + 1e-9
    # small SPRT sanity at an easy point (pair contains the data law)
    s = sprt_reads(R7_TRUE, 0.01, 0.05, 200, seed=1)
    assert s["err_rate"] <= 0.05 + 0.03 and s["mean_reads"] < 12
    assert s["n_capped"] == 0
    out["sprt_easy"] = s
    # declaration-form SPRT: honest declaration certified, huge lie
    # refuted fast (data OUTSIDE the pair — decision, no err_rate)
    s_lie = sprt_reads(R7_FALSE, R7_FALSE * (1 + GAMMA_PILOT), 0.05,
                       100, seed=4, data_r=R7_TRUE)
    assert s_lie["accept_h0_frac"] < 0.1 and s_lie["mean_reads"] < 12
    assert s_lie["err_rate"] is None
    out["sprt_lie_decl"] = s_lie
    # pipeline staging on the three operators (small n_trials)
    for nm, f in (("U_true", lambda: MisKalman(true_rhat())),
                  ("U_false", lambda: MisKalman(false_rhat())),
                  ("Parity", lambda: ParityContractKalman(false_rhat()))):
        out[f"pipe_{nm}"] = pipeline(nm, f, n_trials=60, seed=2)
    assert out["pipe_U_true"]["verdict"] == "CERTIFIED-PRICED"
    assert out["pipe_U_false"]["verdict"] == "CAUGHT-PRICED"
    assert out["pipe_Parity"]["verdict"] == "FLAGGED-FREE"
    assert out["pipe_U_false"]["cost"]["forced_reads"] < 12
    assert out["pipe_U_true"]["cost"]["forced_reads"] > 100
    # the CLEARED-PASSIVE branch: synthetic long honest log (stage_b
    # only) + the real FLIP arm mechanics (honest operator in the
    # high-noise world; n = ceil(19*R7_HI) deterministic)
    rng = np.random.default_rng(5)
    z2 = rng.normal()
    syn = dict(s7_ys=list(z2 + rng.normal(0, 1.0, size=900)))
    b_long = stage_b(syn, R7_TRUE)
    assert b_long["ceiling_allows_target"]
    out["stage_b_long_synthetic"] = b_long
    with _world_r7(R7_HI):
        flip = pipeline("U_true_hi", lambda: MisKalman(true_rhat()),
                        n_trials=40, seed=6, t_steps=T_HI,
                        world_r=R7_HI)
    assert flip["stage_a"]["coherent"]
    assert abs(flip["stage_a"]["l3_rhat7"] - R7_HI) < L3_TOL * R7_HI
    assert flip["stage_b"]["n_passive_s7"] >= 207
    assert flip["stage_b"]["ceiling_allows_target"]
    out["pipe_flip"] = flip
    # certificate soundness, tiny grid
    rows = certificate_soundness(GAMMA_PILOT, [4, 64], 4000, seed=3)
    assert not any(r["violated"] for r in rows)
    out["cert_rows_small"] = rows
    print("SELFCHECK PASS")
    return out


def run(light=False):
    OUT.mkdir(parents=True, exist_ok=True)
    # review #2: --light writes to its OWN path so a mechanics pass can
    # never consume the registered one-execution guard
    dst = OUT / ("report_light.json" if light else "report.json")
    assert not dst.exists(), f"{dst} exists — the run is ONE execution"
    nt = 500 if light else 4000
    nmc = 20000 if light else 200000
    rep = dict(config=dict(delta=DELTA, target_power=TARGET_POWER,
                           gamma_pilot=GAMMA_PILOT, t_rollout=T_ROLLOUT,
                           n_trials=nt, n_mc=nmc, light=bool(light)),
               kappa_pilot=kappa(GAMMA_PILOT),
               wald_bound_reads=float(d_binary(DELTA, 1 - DELTA)
                                      / kappa(GAMMA_PILOT)))
    # C-GAP-STAGED: the pipeline on the registered suite
    for nm, f in (("U_true", lambda: MisKalman(true_rhat())),
                  ("U_false", lambda: MisKalman(false_rhat())),
                  ("Parity", lambda: ParityContractKalman(false_rhat()))):
        rep[f"pipeline_{nm}"] = pipeline(nm, f, n_trials=nt, seed=11)
    # C-GAP-FLIP: the honest high-noise arm — the count-only ceiling
    # flips to allows-target once the policy's OWN reads suffice
    # (n = ceil(19*R7_HI) deterministic; the verdict row is descriptive
    # because the realized LR at the flip point is stochastic; its
    # stage-C, if reached (~12%), audits the R7_HI world)
    with _world_r7(R7_HI):
        rep["pipeline_U_true_hi"] = pipeline(
            "U_true_hi", lambda: MisKalman(true_rhat()),
            n_trials=nt, seed=16, t_steps=T_HI, world_r=R7_HI)
    # world-side commentary (not an auditor output): the per-read KL
    # the suite liar actually leaks once forced — explains the ~3-read
    # catch vs the 339-read honest certification
    rep["kappa_actual_suite_lie"] = kappa_ratio(R7_FALSE)
    # C-GAP-SATURATE: SPRT at the pilot point vs the bound + fixed-sample
    rep["sprt_pilot"] = sprt_reads(R7_TRUE, (1 + GAMMA_PILOT) * R7_TRUE,
                                   DELTA, nt, seed=12)
    rep["mstar_fixed_pilot"] = fixed_sample_mstar(
        R7_TRUE, (1 + GAMMA_PILOT) * R7_TRUE, DELTA, TARGET_POWER,
        seed=13, n_mc=nmc)
    rep["fixed_over_sequential"] = (rep["mstar_fixed_pilot"]
                                    / rep["sprt_pilot"]["mean_reads"])
    # C-GAP-CERT-SOUND (grid where the ceiling BINDS — review #10:
    # the cap at 1.0 made n=256/653 vacuous)
    rep["certificate_soundness"] = certificate_soundness(
        GAMMA_PILOT, [1, 4, 16, 64, 100, 150], nmc, seed=14)
    # descriptive: the E4 m* ladder under the matched ML criterion
    ladder = {}
    for ratio in (0.01, 0.1, 0.2, 0.5):
        m = 1
        while targeted_audit_power(ratio, m, 4000, seed=15) < TARGET_POWER:
            m += 1
            if m > 200:
                break
        ladder[str(ratio)] = m
    rep["mstar_ml_ladder"] = ladder
    with open(dst, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    sat = rep["sprt_pilot"]["mean_reads"] / rep["wald_bound_reads"]
    print(f"GAP: staged verdicts "
          f"{[rep[f'pipeline_{n}']['verdict'] for n in ('U_true', 'U_false', 'Parity')]}; "
          f"SPRT pilot E[N]={rep['sprt_pilot']['mean_reads']:.1f} "
          f"(x{sat:.3f} of bound {rep['wald_bound_reads']:.1f}); "
          f"fixed m*={rep['mstar_fixed_pilot']}; "
          f"cert violations="
          f"{sum(r['violated'] for r in rep['certificate_soundness'])}")
    print(f"wrote {dst}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--light", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.run:
        run(light=args.light)
    else:
        ap.error("pass --selfcheck or --run")
