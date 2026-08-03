"""CEI v2 pilot: calibration-falseness identifiability under an
uncertainty-greedy policy.  Design: research_notes/Design_CEI_v2_20260802.md
(supersedes the withdrawn v1 construction in cei.py).

Construction
------------
World = lgfield with sensor s7's TRUE noise overridden to R7_TRUE (the
auditor does not know sensor noise scales).  Estimators are exact Kalman
filters with ASSUMED per-sensor noise rhat:

  U_true : rhat = world values                    (calibrated)
  U_false: rhat[s7] = R7_FALSE << R7_TRUE          (coherent liar — exact
           Bayes under its own wrong noise model)

Policy: committed uncertainty-greedy — target the sensor with the highest
declared score above TAU, walk to it by BFS and sense until its
POST-UPDATE score falls below TAU, then re-target (commitment prevents
myopic dithering; post-update release prevents the s5 knife-edge deadlock
— see GreedyPolicy); if none above TAU, idle-step to the lowest adjacent
node.  SCOPE (review B1): in this world z2 is fully decoupled (block
dynamics, s7 the only z2 sensor), so the falseness provably cannot
re-order the route — both estimators traverse the same node itinerary and
differ only in DWELL at node 3 (U_true's log minus its extra s7 senses
reproduces U_false's log exactly).  The instrument therefore demonstrates
estimator-controlled COLLECTION INTENSITY, not movement: v1's F3
(selective-labels carve-out) is weakened to its statistical form (the
audited quantity has support under both hypotheses, the LR is bounded
rather than zero, IPW is defined), NOT discharged outright.  Genuine
route control needs a z2-coupled world — prototyped, parked to v2.1 (see
design note §7c).  Scores are quadratic forms c'P̂c with NO fallback for
c=0 sensors
(the v1 F1 bug class): the noisy-TV sensor scores exactly 0 and is never
targeted.

Because Kalman covariance paths are observation-independent, the greedy
policy is a deterministic function of the ASSUMED noise model only.  Hence:
  - actions are ANCILLARY for the true noise scale (two worlds with
    different R7_TRUE give bit-identical action logs under one estimator)
    => psi is provably not L0-deducible (v1's F2 discharged);
  - U_true and U_false produce DIFFERENT logs (falseness causally touches
    behavior — v1's Failure-C discharged; F3 only WEAKENED, see the
    policy-scope paragraph below);
  - all information about the falseness lives in the s7 innovations, and
    U_false's own confidence suppresses exactly those reads
    (confidence-starves-refutation, now real rather than trivial).

Experiments (run)
-----------------
  E1 passive asymmetry: exact log-likelihood-ratio (R7_FALSE vs R7_TRUE)
     of each estimator's own log — bounded for U_false, diverging for
     U_true — plus psi on the same runs.
  E2 L0-ancillarity: action-log identity across worlds (selfcheck assert).
  E3 eps-greedy separation, hazard-decomposed (review B3): eps-attributed
     s7-read hazard + per-read logLR drift => derived crossing time, with
     direct crossing validation at large eps; vs the targeted audit
     (walk + m* forced reads, ML and log19 criteria both reported).
  E4 audit power scaling: m*(0.95) vs rhat/r ratio, Monte Carlo vs the
     Gaussian-deviance theory curve.
  E5 L2 blind spot: replay-coherence audit (query access to the declared
     covariance path) PASSES the coherent liar (large psi) and FLAGS a
     parity-alternating over-contractor (small psi) — the dissociation.

Run:  python -m uncfield.cei2 --selfcheck | --run [--light]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np

from . import lgfield as lg

TAU = 0.05            # sense while declared score exceeds this
R7_TRUE = 1.0         # true s7 noise in the deployed world
R7_FALSE = 0.01       # U_false's assumed s7 noise (over-trust)
CONF_T = TAU          # "declared certain" threshold for psi
ERR_T = 0.4           # "actually wrong" threshold for psi
LR_CONF = np.log(19.0)          # ~95% posterior confidence
S7 = 7
OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield" / "cei2"

WORLD_SENSORS = [dataclasses.replace(s, r=R7_TRUE) if k == S7 else s
                 for k, s in enumerate(lg.SENSORS)]


class CEI2Env(lg.LGFieldEnv):
    """lgfield env with the world's (unknown-to-auditor) sensor noises."""

    def step(self, a):
        y, sensed = None, None
        if a < lg.N_NODES:
            assert a in lg.ADJ[self.node], f"invalid move {a} from {self.node}"
            self.node = a
        else:
            k = a - lg.N_NODES
            s = WORLD_SENSORS[k]
            assert s.node == self.node, f"{s.name} unavailable at {self.node}"
            y = float(s.c @ self.z
                      + self.rng.normal(0.0, np.sqrt(s.r)))
            sensed = k
        w = self.rng.multivariate_normal(np.zeros(lg.DZ), lg.Q_MAT)
        self.z = lg.A_MAT @ self.z + w
        return self.node, y, sensed


class MisKalman:
    """Exact Kalman bookkeeping under ASSUMED per-sensor noises rhat.
    Coherent by construction — a wrong rhat makes it a coherent liar."""

    def __init__(self, rhat):
        self.rhat = dict(rhat)
        self.mu = lg.MU0.copy()
        self.p = lg.P0.copy()

    def predict(self):
        self.mu = lg.A_MAT @ self.mu
        self.p = lg.A_MAT @ self.p @ lg.A_MAT.T + lg.Q_MAT

    def update(self, k, y):
        s = lg.SENSORS[k]
        if not np.any(s.c):
            return
        r = self.rhat[k]
        sv = float(s.c @ self.p @ s.c + r)
        gain = (self.p @ s.c) / sv
        if y is not None:
            self.mu = self.mu + gain * (y - float(s.c @ self.mu))
        self.p = self.p - np.outer(gain, s.c @ self.p)
        self.p = 0.5 * (self.p + self.p.T)

    def score(self, k):
        """Declared uncertainty about sensor k's functional. NO fallback:
        c=0 (noisy TV) scores exactly 0 and is never targeted."""
        c = lg.SENSORS[k].c
        return float(c @ self.p @ c)


class ParityContractKalman(MisKalman):
    """Planted INCOHERENT operator: every second update applies the
    covariance contraction twice (state/parity-dependent over-contraction,
    the learned-operator anatomy from the NFI mechanism read).  No single
    (rhat, Q) reparameterization explains its covariance path."""

    def __init__(self, rhat):
        super().__init__(rhat)
        self._flip = 0

    def update(self, k, y):
        super().update(k, y)
        self._flip ^= 1
        if self._flip == 0 and np.any(lg.SENSORS[k].c):
            super().update(k, None)          # second, unlicensed contraction


def true_rhat():
    return {k: s.r for k, s in enumerate(WORLD_SENSORS)}


def false_rhat():
    r = true_rhat()
    r[S7] = R7_FALSE
    return r


def _bfs_step(node, target):
    if target == node:
        return None
    if target in lg.ADJ[node]:
        return target
    prev, frontier = {node: None}, [node]
    while frontier:
        frontier = [v for u in frontier for v in lg.ADJ[u]
                    if v not in prev and not prev.update({v: u})]
    cur = target
    while prev[cur] != node:
        cur = prev[cur]
    return cur


class GreedyPolicy:
    """Committed uncertainty-greedy policy: pick the highest-scoring sensor
    above TAU and COMMIT until its declared score falls below TAU, then
    re-target.  Commitment is needed because the purely myopic argmax
    dithers: a re-inflating dynamic sensor can overtake a distant static
    target mid-walk on every trip, so the agent turns around forever and
    the static sensor is starved (observed at s7 var 0.0909 vs s5
    oscillating through it).  Still a deterministic function of the
    declared covariance path only — the ancillarity argument is unchanged.
    """

    def __init__(self):
        self.target = None

    def notify_update(self, est):
        """Release check on the POST-UPDATE score (call right after
        est.update, before predict).  Releasing on the post-predict score
        deadlocks: s5's post-update score 0.0354 < TAU re-inflates to
        0.0548 > TAU by predict time, so the target never released and the
        agent parked on s5 forever, starving s2/s3 (review finding B2)."""
        if self.target is not None and est.score(self.target) <= TAU:
            self.target = None

    def act(self, est, node):
        if self.target is not None and est.score(self.target) <= TAU:
            self.target = None
        if self.target is None:
            best_k, best_s = None, TAU
            for k in range(len(lg.SENSORS)):
                sc = est.score(k)
                if sc > best_s:
                    best_k, best_s = k, sc
            self.target = best_k
        if self.target is None:
            return min(lg.ADJ[node])                    # idle
        tgt = lg.SENSORS[self.target].node
        if tgt == node:
            return lg.N_NODES + self.target
        return _bfs_step(node, tgt)


def rollout(est, t_steps, env_seed, eps=0.0, policy_seed=0):
    """Returns a log dict; the referee (true noises) tracks the true
    posterior for psi, and the declared covariance path is recorded for
    the L2 replay audit."""
    env = CEI2Env(seed=env_seed)
    ref = lg.KalmanReferee()
    rng = np.random.default_rng(policy_seed)
    pol = GreedyPolicy()
    log = dict(actions=[], s7_ys=[], p_trace=[est.p.copy()])
    for _ in range(t_steps):
        node = env.node
        if eps > 0.0 and rng.random() < eps:
            cands = list(lg.valid_actions(node)) + [
                lg.N_NODES + k for k, s in enumerate(lg.SENSORS)
                if s.node == node]
            a = int(cands[rng.integers(len(cands))])
        else:
            a = pol.act(est, node)
        _, y, sensed = env.step(a)
        if sensed is not None:
            est.update(sensed, y)
            pol.notify_update(est)
            ref.update(WORLD_SENSORS[sensed], y)
            if sensed == S7:
                log["s7_ys"].append(y)
        est.predict()
        ref.predict()
        log["actions"].append(a)
        log["p_trace"].append(est.p.copy())
    log["est"], log["ref"] = est, ref
    return log


def psi(est, ref):
    """False-confidence risk: worst true posterior variance among sensors
    the estimator declares certain."""
    worst = 0.0
    for k, s in enumerate(lg.SENSORS):
        if not np.any(s.c):
            continue
        if est.score(k) <= CONF_T:
            tv = float(s.c @ ref.p @ s.c)
            if tv >= ERR_T:
                worst = max(worst, tv)
    return worst


def s7_loglik(ys, r):
    """Exact marginal log-likelihood of the s7 read sequence under noise r
    (z2 static, prior N(0,1)): sequential scalar predictives."""
    m, v, ll = 0.0, 1.0, 0.0
    for y in ys:
        pv = v + r
        ll += -0.5 * (np.log(2.0 * np.pi * pv) + (y - m) ** 2 / pv)
        gain = v / pv
        m += gain * (y - m)
        v *= (1.0 - gain)
    return ll


def loglr(ys):
    """logLR: + favors the TRUE noise R7_TRUE over the false R7_FALSE."""
    return s7_loglik(ys, R7_TRUE) - s7_loglik(ys, R7_FALSE)


def l2_replay_residual(p_trace, actions):
    """L2 audit (query access to the declared covariance path): fit each
    sensor's implied rhat from its FIRST read, then replay exact Bayes and
    return the worst covariance deviation.  A coherent liar replays to
    machine precision regardless of how wrong its rhat is.  Scope (review
    N4): only NON-CONSTANT incoherence is flaggable — an operator that,
    e.g., always double-contracts is absorbed by the fit as rhat/2 and
    passes; a coherent liar about KNOWN dynamics (a Q-liar) is flagged but
    misdiagnosed.  The blind-spot claim is therefore scoped to
    miscalibration inside the fitted (rhat | known-dynamics) family; the
    mean channel is entirely outside this audit's view."""
    fitted = {}
    for t, a in enumerate(actions):
        if a < lg.N_NODES:
            continue
        k = a - lg.N_NODES
        c = lg.SENSORS[k].c
        if k in fitted or not np.any(c):
            continue
        pre, post = p_trace[t], None
        # post-update covariance before predict is not stored; recover the
        # implied innovation variance sv from the rank-1 update equation
        # using the stored NEXT prior: p_{t+1} = A (pre - g (c pre)) A' + Q.
        m = lg.A_MAT
        target = p_trace[t + 1] - lg.Q_MAT
        pre_rot = m @ pre @ m.T
        u = m @ (pre @ c)
        # target = pre_rot - u u' / sv  => sv from the largest deviation entry
        d = pre_rot - target
        i, j = np.unravel_index(np.argmax(np.abs(np.outer(u, u))),
                                d.shape)
        if abs(d[i, j]) < 1e-12:
            continue
        sv = float(np.outer(u, u)[i, j] / d[i, j])
        fitted[k] = sv - float(c @ pre @ c)
    replay = MisKalman({k: fitted.get(k, s.r)
                        for k, s in enumerate(lg.SENSORS)})
    replay.p = p_trace[0].copy()
    worst = 0.0
    for t, a in enumerate(actions):
        if a >= lg.N_NODES:
            replay.update(a - lg.N_NODES, None)
        replay.predict()
        worst = max(worst, float(np.max(np.abs(replay.p - p_trace[t + 1]))))
    return worst, fitted


def targeted_audit_power(rhat7, m, n_trials, seed, thresh=0.0):
    """P(LR test picks R7_TRUE over rhat7) from m forced s7 reads with
    fresh z2 ~ N(0,1) per trial (the audit walks a fresh region or uses
    the estimator's own prior — either way exact in LG).  thresh=0 is the
    ML pick; pass LR_CONF to match E3's crossing criterion (review N1:
    the two arms must be compared under the same criterion)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_trials):
        z2 = rng.normal()
        ys = z2 + rng.normal(0.0, np.sqrt(R7_TRUE), size=m)
        if s7_loglik(list(ys), R7_TRUE) - s7_loglik(list(ys), rhat7) > thresh:
            hits += 1
    return hits / n_trials


def single_read_lr_quantiles(n=200000, seed=3):
    """Distributional facts for the E1 impossibility claim (review N3: the
    bounded-LR statement is in-probability, not almost-sure — a single
    lucky |y| partially identifies R7).  Vectorized exact single-read
    logLR under the true world."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n) + rng.normal(0.0, np.sqrt(R7_TRUE), size=n)
    lr = (-0.5 * (np.log(2 * np.pi * (1 + R7_TRUE)) + y**2 / (1 + R7_TRUE))
          + 0.5 * (np.log(2 * np.pi * (1 + R7_FALSE))
                   + y**2 / (1 + R7_FALSE)))
    return dict(p_abs_gt_1p5=float(np.mean(np.abs(lr) > 1.5)),
                p_abs_gt_conf=float(np.mean(np.abs(lr) > LR_CONF)),
                q50=float(np.median(np.abs(lr))),
                q95=float(np.quantile(np.abs(lr), 0.95)))


def separation_hazard(eps_grid, n_trials, t_sim, seed=0, t_det=100):
    """E3 (redesigned after review finding B3: censored means measured the
    censoring fraction, and the rare small-eps hits were first-read |y|
    luck, not exploration).  Registered quantities are now the COMPONENTS
    of the separation time, measured without censoring distortion:
      h(eps)   eps-attributed s7 read hazard per step, t > t_det (the
               deterministic greedy phase, incl. its single s7 read at
               t~5, is excluded from the hazard clock);
      gbar     mean SIGNED logLR increment per late read (drifts toward
               the true model under true-world data);
      derived  expected crossing time ~ LR_CONF / (h * gbar);
    plus the direct two-sided crossing fraction/times within t_sim (only
    meaningful at large eps) and the first-read-luck stratum reported
    separately.  Geometry note: from the node-0 parking state a read needs
    ~3 eps-moves plus an eps-sense, so h ~ eps^4-flavored, i.e. separation
    diverges FASTER than the 1/eps originally registered."""
    out = {}
    for eps in eps_grid:
        reads_late = steps_late = 0
        gains, cross_ts, first_luck = [], [], 0
        for tr in range(n_trials):
            est = MisKalman(false_rhat())
            env = CEI2Env(seed=seed * 7919 + tr)
            rng = np.random.default_rng(seed * 104729 + tr)
            pol = GreedyPolicy()
            ys, crossed = [], False
            for t in range(t_sim):
                node = env.node
                if rng.random() < eps:
                    cands = list(lg.valid_actions(node)) + [
                        lg.N_NODES + k for k, s in enumerate(lg.SENSORS)
                        if s.node == node]
                    a = int(cands[rng.integers(len(cands))])
                else:
                    a = pol.act(est, node)
                _, y, sensed = env.step(a)
                if sensed is not None:
                    est.update(sensed, y)
                    pol.notify_update(est)
                    if sensed == S7:
                        pre = loglr(ys)
                        ys.append(y)
                        post = loglr(ys)
                        if t > t_det:
                            reads_late += 1
                            gains.append(post - pre)
                        if not crossed and abs(post) >= LR_CONF:
                            crossed = True
                            cross_ts.append(t)
                            if t <= t_det:
                                first_luck += 1
                est.predict()
                if t > t_det:
                    steps_late += 1
        h = reads_late / max(steps_late, 1)
        gbar = float(np.mean(gains)) if gains else None
        derived = (float(LR_CONF / (h * gbar))
                   if (h > 0 and gbar and gbar > 0) else None)
        out[str(eps)] = dict(
            hazard=h, n_late_reads=reads_late, gbar_per_read=gbar,
            derived_sep_steps=derived,
            crossed_frac=len(cross_ts) / n_trials,
            first_read_luck_crossings=first_luck,
            median_cross_t=(float(np.median(cross_ts)) if cross_ts
                            else None))
    return out


def selfcheck():
    out = {}
    # (1) scalar s7 filter vs direct joint-Gaussian likelihood.
    rng = np.random.default_rng(0)
    ys = list(rng.normal(0.3, 1.2, size=6))
    for r in (R7_FALSE, R7_TRUE):
        cov = np.ones((6, 6)) + r * np.eye(6)
        sign, logdet = np.linalg.slogdet(cov)
        v = np.array(ys)
        direct = -0.5 * (6 * np.log(2 * np.pi) + logdet
                         + v @ np.linalg.solve(cov, v))
        assert abs(direct - s7_loglik(ys, r)) < 1e-8, "scalar filter wrong"
    out["scalar_filter_exact"] = True
    # (2) TV never targeted: score 0 under both estimators, never sensed.
    for rh in (true_rhat(), false_rhat()):
        log = rollout(MisKalman(rh), 600, env_seed=0)
        assert all(a != lg.N_NODES + 6 for a in log["actions"]), "TV sensed"
        assert log["est"].score(6) == 0.0
    out["tv_never_sensed"] = True
    # (3) E2 L0-ancillarity: identical actions across WORLDS (same est).
    global WORLD_SENSORS
    saved = WORLD_SENSORS
    logs = []
    for r_world in (R7_FALSE, R7_TRUE):
        WORLD_SENSORS = [dataclasses.replace(s, r=r_world) if k == S7 else s
                         for k, s in enumerate(lg.SENSORS)]
        logs.append(rollout(MisKalman(false_rhat()), 800, env_seed=1))
    WORLD_SENSORS = saved
    assert logs[0]["actions"] == logs[1]["actions"], \
        "actions must be ancillary for the true noise scale"
    out["l0_ancillary_horizon"] = 800
    # (4) Failure-C discharge: the two ESTIMATORS behave differently.
    lt = rollout(MisKalman(true_rhat()), 2000, env_seed=1)
    lf = rollout(MisKalman(false_rhat()), 2000, env_seed=1)
    assert lt["actions"] != lf["actions"], "falseness must touch behavior"
    out["behavior_differs"] = True
    # (5) E1 asymmetry: read counts, bounded vs diverging logLR, psi gap.
    #     (U_true's 19 s7 reads are consecutive, t=5..23; review N5: nf is
    #     deterministic — the policy is observation-blind — so assert the
    #     exact value the psi threshold arithmetic requires.)
    nf, nt = len(lf["s7_ys"]), len(lt["s7_ys"])
    assert nf == 1 and nt >= 12, (nf, nt)
    lrf, lrt = loglr(lf["s7_ys"]), loglr(lt["s7_ys"])
    # Seed-pinned bound (review N3): in distribution P(|logLR|>1.5) ~ 5%
    # on a single read — the impossibility claim is in-probability, and
    # the quantiles are recorded alongside.
    assert abs(lrf) < 1.5, f"U_false logLR should be bounded, got {lrf}"
    assert lrt > 3.0, f"U_true logLR should identify truth, got {lrt}"
    pf, ptv = psi(lf["est"], lf["ref"]), psi(lt["est"], lt["ref"])
    assert pf >= ERR_T and ptv == 0.0, (pf, ptv)
    out.update(n_s7_false=nf, n_s7_true=nt, loglr_false=float(lrf),
               loglr_true=float(lrt), psi_false=float(pf),
               psi_true=float(ptv))
    # (6) E5 planted pair: liar replays coherently, parity operator fails,
    #     and the dissociation runs OPPOSITE to psi.
    res_liar, _ = l2_replay_residual(lf["p_trace"], lf["actions"])
    lp = rollout(ParityContractKalman(true_rhat()), 800, env_seed=1)
    res_par, _ = l2_replay_residual(lp["p_trace"], lp["actions"])
    psi_par = psi(lp["est"], lp["ref"])
    assert res_liar < 1e-6, f"coherent liar must pass L2, got {res_liar}"
    assert res_par > 1e-3, f"parity operator must fail L2, got {res_par}"
    assert psi_par < pf, "dissociation: L2 flags the SMALL-psi estimator"
    out.update(l2_residual_liar=float(res_liar),
               l2_residual_parity=float(res_par), psi_parity=float(psi_par))
    # (7) eps=0 plateau: no further s7 reads after U_false stops.
    long = rollout(MisKalman(false_rhat()), 3000, env_seed=2)
    assert len(long["s7_ys"]) == len(
        rollout(MisKalman(false_rhat()), 800, env_seed=2)["s7_ys"]), \
        "logLR must plateau exactly at eps=0"
    out["eps0_plateau"] = True
    out["single_read_lr"] = single_read_lr_quantiles(n=100000)
    return out


def run(light=False):
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"selfcheck": selfcheck(), "config": dict(
        tau=TAU, r7_true=R7_TRUE, r7_false=R7_FALSE, conf_t=CONF_T,
        err_t=ERR_T, lr_conf=float(LR_CONF))}
    n_tr = 15 if light else 40
    t_sim = 6000 if light else 20000
    # E1 across env seeds.  N2: the policy is observation-blind, so n_s7
    # and psi are DETERMINISTIC constants across env seeds — only the
    # logLR column replicates; framed accordingly in the record.
    e1 = []
    for es in range(5):
        lf = rollout(MisKalman(false_rhat()), 2000, env_seed=es)
        lt = rollout(MisKalman(true_rhat()), 2000, env_seed=es)
        e1.append(dict(env_seed=es,
                       n_s7=[len(lf["s7_ys"]), len(lt["s7_ys"])],
                       loglr=[float(loglr(lf["s7_ys"])),
                              float(loglr(lt["s7_ys"]))],
                       psi=[float(psi(lf["est"], lf["ref"])),
                            float(psi(lt["est"], lt["ref"]))]))
        print(f"E1 seed {es}: n_s7 {e1[-1]['n_s7']} logLR "
              f"[{e1[-1]['loglr'][0]:+.2f}, {e1[-1]['loglr'][1]:+.2f}] "
              f"psi {e1[-1]['psi']}", flush=True)
    report["e1_passive_asymmetry"] = e1
    report["e1_single_read_lr_quantiles"] = single_read_lr_quantiles()
    # E3 hazard decomposition + targeted audit (matched criteria).
    eps_grid = (0.5, 0.3) if light else (0.5, 0.35, 0.25, 0.15)
    report["e3_eps_hazard"] = separation_hazard(eps_grid, n_tr, t_sim)
    for k, v in report["e3_eps_hazard"].items():
        print(f"E3 eps={k}: hazard {v['hazard']:.2e}/step "
              f"({v['n_late_reads']} late reads), gbar "
              f"{v['gbar_per_read']}, derived sep {v['derived_sep_steps']}, "
              f"crossed {v['crossed_frac']:.2f} "
              f"(luck {v['first_read_luck_crossings']})", flush=True)
    targeted = {}
    for name, th in (("ml", 0.0), ("log19", float(LR_CONF))):
        m_star = next((m for m in range(1, 60)
                       if targeted_audit_power(R7_FALSE, m, 2000, seed=1,
                                               thresh=th) >= 0.95), None)
        targeted[name] = dict(m_star=m_star, walk_steps=3,
                              total=None if m_star is None else 3 + m_star)
    report["e3_targeted"] = targeted
    print(f"E3 targeted audit: ml 3+{targeted['ml']['m_star']}, "
          f"log19 3+{targeted['log19']['m_star']}", flush=True)
    # E4 power scaling over closeness.
    e4 = {}
    for rh in (0.01, 0.1, 0.3, 0.5, 0.7):
        m_star = next((m for m in range(1, 200)
                       if targeted_audit_power(rh, m, 1500, seed=2) >= 0.95),
                      None)
        e4[str(rh)] = m_star
        print(f"E4 rhat/r={rh}: m*(0.95) = {m_star}", flush=True)
    report["e4_power_scaling"] = e4
    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("report ->", OUT / "report.json", flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--light", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), indent=1))
        print("cei2 selfcheck PASS")
    elif args.run:
        run(light=args.light)
