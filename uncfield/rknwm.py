"""RKN-class coherent-by-construction belief operator (Track C;
PREREG_trackC_rkn_20260821.md; review #33 adjudicated 21 Aug:
6B/10M/9m ALL applied — including the thesis correction).

The operator is a LEARNED-PARAMETER exact Kalman filter: belief =
[mu (DZ), vec(P) (DZ^2)] propagated by the exact Kalman
predict/update recursions under learned (A_hat, diagonal Q_hat,
c_hat_k, r_hat_k).

THESIS (corrected, #33 F2): coherence removes the
promise-vs-own-license INCONSISTENCY channel (the ratio/min-clip
class) and makes the realized-dH farming CONJUNCTION structurally
unreachable — the covariance path converges to a periodic orbit, so
net realized entropy drop per steady loop is ~0, killing the
carried_dh (PBIM) tier and hence the flagship CIG+PBIM conjunction.
Coherence does NOT prevent honest-wrong carried-EIG farming: a
coherent filter with q_hat > 0 on a truly static latent keeps
promising (and, to itself, realizing per-sense) gain on exhausted
sensors — the coherent-liar cell that CEI's interventional audits
exist to price. Review #33 measured the fire threshold: reaching
NO-EXPLOIT needs q_hat_static <~ 1e-5, ~2.8x the registered adam
budget from init, so CIG-tier occupancy is EXPECTED, not a failure.

Structural degeneracies of this operator class (#33 F12, disclosed
and asserted): the covariance path is observation-free, so
naive_eig == carried_eig exactly (the EXPLOIT-NAIVE-ONLY verdict is
unreachable), and licensed_rate's pure-update Bayes gain exceeds the
predict-inclusive carried_eig structurally (licensed >= carried on
one-hot rows — equality was never the right claim, #33 F5).

Entropy convention: DIAGONAL-SUM surrogate. Diagonal-lens caveat
(selfcheck (4) pins it): on the overlap sensor s4, r_implied =
r_hat + 2*P_01 goes NEGATIVE after one sense and head_accounts
declares INFINITE license on a fully coherent operator — overlap
rows are detected from the cycle's ACTIONS (#33 F4) and excluded
from the one-hot consistency claims.

Train/deploy identity (#33 F1): the trained A is capped by the SAME
power-iteration spectral estimate used inside the loss (computed
once on the final parameters, float32, and STORED as A_eff); the
deployed RKNModel consumes A_eff verbatim — no second, different cap
at load. Training runs in float32 (repo default), deploy math in
float64; both recorded (#33 F14).

Run:  python -m uncfield.rknwm --selfcheck
      python -m uncfield.rknwm --train  --tag rkn_pilot2
      python -m uncfield.rknwm --search --tag rkn_pilot2 [--member M]
      python -m uncfield.rknwm --heldout --tag rkn_pilot2
      python -m uncfield.rknwm --audit  --tag rkn_pilot2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np

from . import lgfield as lg

DZ = lg.DZ
NS = lg.N_SENSORS
OUT = pathlib.Path(__file__).resolve().parent.parent / "local_results" \
    / "uncfield"
N_MEMBERS = 4
AUDIT_SENSORS = [0, 1, 2, 3, 7]   # r-channel audit of STATIC one-hot
#   readouts ONLY (#33 F16): the C-channel and the TV sensor s6 are
#   OUT OF SCOPE (an r-only audit cannot see a C-liar); c_hat row
#   norms are reported per sensor as the disclosure.
AUDIT_DELTA = 0.05
AUDIT_LR_CONF = float(np.log(19.0))
AUDIT_NMAX = 50_000               # #33 F8
AUDIT_TRIALS = 200
AUDIT_KAPPA_T = 4_000
AUDIT_KAPPA_BURN = 200
MAX_CONSISTENCY_ROWS = 400        # #33 F6 cap (disclosed if hit)
SIGMA_PI_ITERS = 12


# ------------------------------------------------------------------ params

def make_true_params():
    """The oracle parameter point: the world's own (A, Q, C, R).
    Disclosure (#33 F18): q is clamped at 1e-12 (log of 0 undefined),
    so the point is the true system up to that clamp."""
    return dict(A=lg.A_MAT.copy(),
                logq=np.log(np.maximum(np.diag(lg.Q_MAT), 1e-12)),
                C=np.stack([s.c for s in lg.SENSORS]),
                logr=np.log(np.array([s.r for s in lg.SENSORS])))


def init_rkn_params(seed):
    rng = np.random.default_rng(seed)
    return dict(A=0.9 * np.eye(DZ),
                logq=np.full(DZ, np.log(0.05)),
                C=0.1 * rng.standard_normal((NS, DZ)),
                logr=np.zeros(NS))


def sigma_max_pi(a, iters=SIGMA_PI_ITERS):
    """Numpy mirror of the in-loss power-iteration spectral estimate
    (same iteration count, same init). A LOWER bound on sigma_max;
    under-converges on near-degenerate top clusters — which is
    exactly why train and deploy must share ONE estimate (#33 F1)."""
    a = np.asarray(a, np.float64)
    v = np.full(DZ, 1.0 / np.sqrt(DZ))
    for _ in range(iters):
        v = a.T @ (a @ v)
        n = np.linalg.norm(v)
        v = v / max(n, 1e-30)
    return float(np.linalg.norm(a @ v))


def _spec_cap(a):
    """Legacy exact-SVD cap — used ONLY for parameter dicts without a
    stored A_eff (the oracle/init points in the battery)."""
    s = np.linalg.norm(a, 2)
    return a * (1.0 / max(1.0, s))


# ------------------------------------------------------------- the operator

class RKNModel:
    """Frozen learned-Kalman operator behind the pilot's numpy API.
    A trained checkpoint carries A_eff (the training-time cap
    applied once to the final A) and is consumed VERBATIM (#33 F1);
    battery parameter points without A_eff get the exact-SVD cap."""

    def __init__(self, params):
        if "A_eff" in params:
            self.A = np.asarray(params["A_eff"], np.float64)
        else:
            self.A = _spec_cap(np.asarray(params["A"], np.float64))
        self.sigma_exact = float(np.linalg.norm(self.A, 2))
        self.q = np.exp(np.asarray(params["logq"], np.float64))
        self.C = np.asarray(params["C"], np.float64)
        self.r = np.exp(np.asarray(params["logr"], np.float64))

    def init_belief(self):
        return np.concatenate([lg.MU0, lg.P0.flatten()]).astype(
            np.float64)

    @staticmethod
    def _unpack(b):
        return b[:DZ].copy(), b[DZ:].reshape(DZ, DZ).copy()

    def step(self, b, action, node, y):
        mu, p = self._unpack(b)
        if action >= lg.N_NODES:
            k = action - lg.N_NODES
            c, r = self.C[k], self.r[k]
            s = float(c @ p @ c + r)
            kg = (p @ c) / s
            if y is not None:
                mu = mu + kg * (y - float(c @ mu))
            p = p - np.outer(kg, c @ p)
            p = 0.5 * (p + p.T)
        mu = self.A @ mu
        p = self.A @ p @ self.A.T + np.diag(self.q)
        p = 0.5 * (p + p.T)
        return np.concatenate([mu, p.flatten()])

    def zstats(self, b):
        mu, p = self._unpack(b)
        return mu, np.log(np.diag(p) + lg.JITTER)

    def obs_pred(self, b, action):
        mu, p = self._unpack(b)
        k = action - lg.N_NODES
        assert 0 <= k < NS, f"obs_pred on a move action {action}"
        c, r = self.C[k], self.r[k]
        return float(c @ mu), float(np.log(c @ p @ c + r))

    def entropy(self, b):
        _, lv = self.zstats(b)
        return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e) + lv))


# ----------------------------------------------------------------- training

def _episodes_to_rkn_arrays(episodes):
    """Direct conversion (#33 F20: skips learnedwm's unused 78 MB xs
    tensor): returns (aidx, zt, ymask, ytrue) float/int arrays."""
    import jax.numpy as jnp
    ai, zt, ym, yt = [], [], [], []
    for recs in episodes:
        ai.append([r["action"] for r in recs])
        zt.append([r["z_after"] for r in recs])
        ym.append([1.0 if r["y"] is not None else 0.0 for r in recs])
        yt.append([r["y"] if r["y"] is not None else 0.0
                   for r in recs])
    return (jnp.asarray(np.array(ai, np.int32)),
            jnp.asarray(np.array(zt, np.float32)),
            jnp.asarray(np.array(ym, np.float32)),
            jnp.asarray(np.array(yt, np.float32)))


def train_rkn(episodes, seed, steps=3000, batch=32, lr=1e-3,
              verbose=False):
    """Fit (A, logq, C, logr) by adam on the learnedwm per-step loss
    (same batch=32 discipline as the GRU anchor, #33 F10), computed
    through the exact Kalman recursions (Joseph form). Float32
    (repo default; recorded). Returns (params incl. stored A_eff,
    final loss, info)."""
    import jax
    import jax.numpy as jnp
    import optax

    aidx, zt, ym, yt = _episodes_to_rkn_arrays(episodes)
    n = aidx.shape[0]
    p0 = {k: jnp.asarray(v) for k, v in init_rkn_params(seed).items()}
    eye = jnp.eye(DZ)

    def _sigma_max(a, iters=SIGMA_PI_ITERS):
        v = jnp.full((DZ,), 1.0 / np.sqrt(DZ))
        for _ in range(iters):
            v = a.T @ (a @ v)
            v = v / jnp.maximum(jnp.linalg.norm(v), 1e-30)
        return jnp.linalg.norm(a @ v)

    def seq_nll(p, a_eff, a_effT, qd, rall, ai, z, m, y):
        # XLA-CPU workaround (jax 0.4.33): transposing the OUTGOING
        # loop-carried covariance inside scan+grad fails to compile;
        # symmetrize the INCOMING carry and build transposes
        # explicitly (a_effT precomputed; ikcT from outer(c, kg))
        def step(carry, inp):
            mu, pp_raw = carry
            pp = 0.5 * (pp_raw + pp_raw.T)
            a, zt_, m_, y_ = inp
            k = jnp.maximum(a - lg.N_NODES, 0)
            c = p["C"][k]
            r = rall[k]
            s = c @ pp @ c + r
            onll = m_ * 0.5 * (jnp.log(s)
                               + (y_ - c @ mu) ** 2 / s
                               + jnp.log(2.0 * jnp.pi))
            kg = (pp @ c) / s
            mu1 = mu + m_ * kg * (y_ - c @ mu)
            ikc = eye - m_ * jnp.outer(kg, c)
            ikcT = eye - m_ * jnp.outer(c, kg)
            p1 = ikc @ pp @ ikcT + (m_ ** 2) * jnp.outer(kg, kg) * r
            mu2 = a_eff @ mu1
            p2 = a_eff @ p1 @ a_effT + qd
            zlv = jnp.log(jnp.diag(p2) + lg.JITTER)
            znll = jnp.sum(0.5 * (zlv + (zt_ - mu2) ** 2
                                  / jnp.exp(zlv)
                                  + jnp.log(2.0 * jnp.pi)))
            return (mu2, p2), znll + onll

        carry0 = (jnp.asarray(lg.MU0), jnp.asarray(lg.P0))
        _, losses = jax.lax.scan(step, carry0, (ai, z, m, y))
        return jnp.mean(losses)

    def batch_loss(p, ai, z, m, y):
        a_eff = p["A"] * (1.0 / jnp.maximum(1.0,
                                            _sigma_max(p["A"])))
        a_effT = jnp.transpose(a_eff)
        qd = jnp.diag(jnp.exp(p["logq"]))
        rall = jnp.exp(p["logr"])
        return jnp.mean(jax.vmap(
            lambda *s: seq_nll(p, a_eff, a_effT, qd, rall, *s))(
            ai, z, m, y))

    opt = optax.adam(lr)
    opt_state = opt.init(p0)

    @jax.jit
    def update(p, s, ai, z, m, y):
        loss, grads = jax.value_and_grad(batch_loss)(p, ai, z, m, y)
        upd, s = opt.update(grads, s)
        return optax.apply_updates(p, upd), s, loss

    rng = np.random.default_rng(seed)
    params, loss = p0, np.nan
    for i in range(steps):
        idx = rng.integers(0, n, size=batch)
        params, opt_state, loss = update(
            params, opt_state, aidx[idx], zt[idx], ym[idx], yt[idx])
        if verbose and (i % 500 == 0 or i == steps - 1):
            print(f"    [rkn seed {seed}] step {i} "
                  f"loss {float(loss):.4f}", flush=True)
    out = {k: np.asarray(jax.device_get(v))
           for k, v in params.items()}
    # #33 F1: ONE cap — the training-time estimate, applied once to
    # the final A (in float32, matching the last update's semantics)
    # and STORED; RKNModel consumes it verbatim.
    sigma_pi = float(_sigma_max(jnp.asarray(out["A"],
                                            jnp.float32)))
    out["A_eff"] = np.asarray(
        out["A"] * (1.0 / max(1.0, sigma_pi)), np.float64)
    info = dict(sigma_pi=sigma_pi,
                sigma_exact=float(np.linalg.norm(out["A_eff"], 2)),
                batch=batch, lr=lr, steps=steps,
                dtype="float32 training / float64 deploy")
    return out, float(loss), info


def save_rkn_ensemble(path, ensemble):
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)


def load_rkn_ensemble(path):
    with open(path, "rb") as f:
        ens = pickle.load(f)
    assert all("logr" in p and "A" in p for p in ens), (
        f"{path} is not an RKN ensemble")
    return ens


# ----------------------------------------------------------- CEI audit math

def kl_gauss_var(r0, r1):
    """KL(N(0,r0) || N(0,r1)) — marginal REFERENCE only; the audit's
    operative kappa is measured empirically (#33 F7)."""
    t = r0 / r1
    return 0.5 * (t - 1.0 - np.log(t))


def d_binary(delta):
    return (delta * np.log(delta / (1.0 - delta))
            + (1.0 - delta) * np.log((1.0 - delta) / delta))


def _declared_path(model, k, T):
    """Deterministic per-read quantities of the DECLARED operator
    (the trained RKNModel itself, #33 F7) under repeated sense-k:
    predictive logvar ylv_t, gain K_t, and c_hat — the mean path is
    then linear in the observed ys and is run trial-vectorized."""
    a = lg.N_NODES + k
    b = model.init_belief()
    ylv, gains = np.empty(T), np.empty((T, DZ))
    for t in range(T):
        _, p = RKNModel._unpack(b)
        c, r = model.C[k], model.r[k]
        s = float(c @ p @ c + r)
        ylv[t] = np.log(s)
        gains[t] = (p @ c) / s
        b = model.step(b, a, lg.SENSORS[k].node, 0.0)
    return ylv, gains


def _audit_sensor(model, k, n_trials, nmax, rng, generator):
    """Trial-vectorized sequential audit of sensor k. Reads generated
    from `generator` in {"true", "declared"}; LLR = truth-side
    loglik minus declared-side loglik; stop at +/- AUDIT_LR_CONF.
    Truth side: exact scalar filter (one-hot STATIC sensor — z
    component constant under the true dynamics; audited from the
    FRESH prior v0 = c'P0c, registered scoping). Declared side: the
    trained operator's own full recursion."""
    s_true = lg.SENSORS[k]
    r_true = float(s_true.r)
    v0 = float(s_true.c @ lg.P0 @ s_true.c)
    ylv_h, gains = _declared_path(model, k, nmax)
    c_hat = model.C[k]
    A = model.A

    z0 = rng.normal(0.0, np.sqrt(v0), n_trials)
    mu_h = np.zeros((n_trials, DZ))
    mu_t = np.zeros(n_trials)
    v_t = np.full(n_trials, v0)
    llr = np.zeros(n_trials)
    alive = np.ones(n_trials, bool)
    n_stop = np.full(n_trials, nmax)
    convicted = np.zeros(n_trials, bool)
    term = np.zeros(n_trials)
    for t in range(nmax):
        if not alive.any():
            break
        ymu_h = mu_h @ c_hat
        sd_h = np.exp(0.5 * ylv_h[t])
        if generator == "true":
            y = z0 + rng.normal(0.0, np.sqrt(r_true), n_trials)
        else:
            y = ymu_h + sd_h * rng.standard_normal(n_trials)
        s_t = v_t + r_true
        lt = -0.5 * (np.log(2 * np.pi * s_t)
                     + (y - mu_t) ** 2 / s_t)
        lh = -0.5 * (2 * np.log(np.sqrt(2 * np.pi) * sd_h)
                     + (y - ymu_h) ** 2 / (sd_h ** 2))
        llr = np.where(alive, llr + lt - lh, llr)
        k_t = v_t / s_t
        mu_t = mu_t + k_t * (y - mu_t)
        v_t = v_t * (1.0 - k_t)
        mu_h = (mu_h + (y - ymu_h)[:, None] * gains[t][None]) @ A.T
        crossed = alive & (np.abs(llr) >= AUDIT_LR_CONF)
        n_stop[crossed] = t + 1
        convicted[crossed] = llr[crossed] > 0
        term[crossed] = llr[crossed]
        alive &= ~crossed
    frac_crossed = float(1.0 - alive.mean())
    stopped = ~alive
    return dict(
        frac_crossed=frac_crossed,
        mean_n=(float(n_stop[stopped].mean()) if stopped.any()
                else float("nan")),
        frac_convicted=(float(convicted[stopped].mean())
                        if stopped.any() else float("nan")),
        mean_terminal_llr=(float(term[stopped].mean())
                           if stopped.any() else float("nan")))


def _kappa_emp(model, k, n_trials, rng, T=AUDIT_KAPPA_T,
               burn=AUDIT_KAPPA_BURN):
    """Empirical per-read LLR drift under the TRUE generator, tail
    mean past the burn-in (the operative kappa, #33 F7)."""
    s_true = lg.SENSORS[k]
    r_true = float(s_true.r)
    v0 = float(s_true.c @ lg.P0 @ s_true.c)
    ylv_h, gains = _declared_path(model, k, T)
    c_hat = model.C[k]
    A = model.A
    z0 = rng.normal(0.0, np.sqrt(v0), n_trials)
    mu_h = np.zeros((n_trials, DZ))
    mu_t = np.zeros(n_trials)
    v_t = np.full(n_trials, v0)
    llr = np.zeros(n_trials)
    llr_burn = None
    for t in range(T):
        ymu_h = mu_h @ c_hat
        sd_h = np.exp(0.5 * ylv_h[t])
        y = z0 + rng.normal(0.0, np.sqrt(r_true), n_trials)
        s_t = v_t + r_true
        lt = -0.5 * (np.log(2 * np.pi * s_t)
                     + (y - mu_t) ** 2 / s_t)
        lh = -0.5 * (2 * np.log(np.sqrt(2 * np.pi) * sd_h)
                     + (y - ymu_h) ** 2 / (sd_h ** 2))
        llr = llr + lt - lh
        k_t = v_t / s_t
        mu_t = mu_t + k_t * (y - mu_t)
        v_t = v_t * (1.0 - k_t)
        mu_h = (mu_h + (y - ymu_h)[:, None] * gains[t][None]) @ A.T
        if t == burn - 1:
            llr_burn = llr.copy()
    return float((llr - llr_burn).mean() / (T - burn))


def audit_member(params, n_trials=AUDIT_TRIALS, nmax=AUDIT_NMAX,
                 seed=0):
    """CEI pricing of one trained operator against ITS OWN declared
    filter (#33 F7): per audited sensor, empirical kappa, the
    information bound d(delta,1-delta)/kappa_emp, simulated SPRT cost
    + power under the TRUE generator, and the false-conviction rate
    under the DECLARED generator (#33 F9). Per-sensor rng (#33 F23).
    """
    model = RKNModel(params)
    rows = []
    for k in AUDIT_SENSORS:
        s = lg.SENSORS[k]
        rng = np.random.default_rng(seed * 7919 + 13 + 1000 * k)
        r_hat = float(model.r[k])
        gamma = r_hat / float(s.r) - 1.0
        row = dict(sensor=s.name, r_true=float(s.r), r_hat=r_hat,
                   gamma_hat=gamma,
                   c_hat_norm=float(np.linalg.norm(model.C[k])),
                   kappa_marginal_ref=kl_gauss_var(float(s.r),
                                                   r_hat))
        kap = _kappa_emp(model, k, max(n_trials // 2, 20), rng)
        row["kappa_emp"] = kap
        bound = (d_binary(AUDIT_DELTA) / kap if kap > 0
                 else float("inf"))
        row["info_bound_reads"] = bound
        if not np.isfinite(bound) or bound > nmax:
            row.update(status="BELOW-AUDIT-RESOLUTION",
                       note=f"kappa_emp {kap:.3e} puts the bound "
                            f"beyond nmax={nmax}: the operator is "
                            f"indistinguishable from truth at this "
                            f"audit budget")
        else:
            tr = _audit_sensor(model, k, n_trials, nmax, rng, "true")
            de = _audit_sensor(model, k, n_trials, nmax, rng,
                               "declared")
            valid = bool(tr["frac_crossed"] >= 0.98
                         and tr["frac_convicted"] >= 0.9
                         and de["frac_convicted"] <= 0.1
                         and 0.5 < tr["mean_n"] / bound < 4.0
                         and 0.5 < kap * tr["mean_n"]
                         / AUDIT_LR_CONF < 3.0)
            row.update(status="PRICED",
                       sprt_mean_reads=tr["mean_n"],
                       sprt_power=tr["frac_convicted"],
                       sprt_crossed_frac=tr["frac_crossed"],
                       false_conviction_rate=de["frac_convicted"],
                       declared_crossed_frac=de["frac_crossed"],
                       wald_ratio=kap * tr["mean_n"] / AUDIT_LR_CONF,
                       criteria_met=valid)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ drivers

def _load_pilot2_episodes():
    with open(OUT / "pilot2" / "episodes.pkl", "rb") as f:
        return pickle.load(f)


def cmd_train(tag, steps, verbose, out_root=None, episodes=None,
              batch=32, lr=1e-3):
    episodes = episodes or _load_pilot2_episodes()
    outdir = (out_root or OUT) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    ens, losses, infos = [], [], []
    for m in range(N_MEMBERS):
        params, loss, info = train_rkn(episodes, seed=m, steps=steps,
                                       batch=batch, lr=lr,
                                       verbose=verbose)
        ens.append(params)
        losses.append(loss)
        infos.append(info)
        print(f"rkn member {m}: final loss {loss:.4f} "
              f"(sigma_pi {info['sigma_pi']:.5f})", flush=True)
    save_rkn_ensemble(outdir / "ensemble.pkl", ens)
    with open(outdir / "train_info.json", "w") as f:
        json.dump(dict(tag=tag, steps=steps, losses=losses,
                       member_info=infos, n_members=N_MEMBERS,
                       data="pilot2/episodes.pkl"
                            if episodes is None else "override",
                       cell="rkn_kalman"), f, indent=1)
    print(f"saved {outdir}/ensemble.pkl")


def cmd_search(tag, member, out_root=None, max_len=6,
               max_rows=MAX_CONSISTENCY_ROWS):
    from . import planner as pl
    from .minclip_rescore import licensed_rate, clip
    from .sweeps import _code_stamp
    from .ratio_research import _san
    outdir = (out_root or OUT) / tag
    ens = load_rkn_ensemble(outdir / "ensemble.pkl")
    members = range(N_MEMBERS) if member < 0 else [member]
    for m in members:
        model = RKNModel(ens[m])
        results, (b0, _, node0) = pl.search_exploits(
            model, seed=0, max_len=max_len, y_mode="ml")
        verdict, counts = pl.classify(results)
        # #33 F12: observation-free covariance => naive == carried,
        # asserted as a structural self-consistency gate
        assert counts["n_exploit_naive"] == counts["n_exploit_cig"], (
            counts)
        # #33 F6: consistency rows = ALL exploit cycles UNION the
        # top-20 by carried (capped, disclosed)
        exploits = [r for r in results if r["exploit_carried_eig"]]
        top20 = sorted(results,
                       key=lambda r: -r["carried_eig_rate"])[:20]
        seen, pool = set(), []
        for r in exploits + top20:
            if r["cycle"]["name"] not in seen:
                seen.add(r["cycle"]["name"])
                pool.append(r)
        truncated = len(pool) > max_rows
        rows = []
        for r in pool[:max_rows]:
            lic, frac_inc, n_sen = licensed_rate(
                model, b0, node0, r["cycle"], "ml", 0)
            car = float(r["carried_eig_rate"])
            rows.append(dict(
                name=r["cycle"]["name"],
                # #33 F4: overlap detection from the ACTIONS
                uses_overlap=any(a == lg.N_NODES + 4
                                 for a in r["cycle"]["actions"]),
                sensors=sorted({lg.SENSORS[a - lg.N_NODES].name
                                for a in r["cycle"]["actions"]
                                if a >= lg.N_NODES}),
                # minclip field set verbatim (#33 F6)
                neutral=bool(r["neutral"]),
                true_rate=float(r["true_rate"]),
                carried=car, licensed=lic,
                clipped=clip(car, lic),
                licensed_minus_carried=(lic - car
                                        if np.isfinite(lic)
                                        else None),
                frac_inconsistent=frac_inc, n_senses=n_sen,
                exploit_carried=bool(r["exploit_carried_eig"]),
                exploit_clipped=bool(r["neutral"]
                                     and clip(car, lic)
                                     > pl.EPS_PRED)))
        rec = dict(tag=tag, member=m, kind="rkn_kalman",
                   stamp=_code_stamp(), verdict=verdict,
                   counts={k: (int(v) if isinstance(v, (bool,
                                                        np.integer,
                                                        int))
                               else float(v))
                           for k, v in counts.items()},
                   n_cycles=len(results),
                   n_neutral=sum(bool(r["neutral"])
                                 for r in results),
                   n_exploit_cig=int(counts["n_exploit_cig"]),
                   n_exploit_pbim=int(counts["n_exploit_pbim"]),
                   n_exploit_both=int(counts["n_exploit_both"]),
                   consistency_rows=rows,
                   consistency_rows_truncated=truncated,
                   naive_equals_carried=True)
        with open(outdir / f"search_m{m}.json", "w") as f:
            json.dump(_san(rec), f, indent=1)
        np.savez_compressed(
            outdir / f"results_m{m}.npz",
            names=np.array([r["cycle"]["name"] for r in results]),
            true_rate=np.array([r["true_rate"] for r in results]),
            carried=np.array([r["carried_eig_rate"]
                              for r in results]),
            dh_adj=np.array([r["carried_dh_adj"] for r in results]),
            neutral=np.array([r["neutral"] for r in results]),
            exploit_both=np.array([r["exploit_both"]
                                   for r in results]))
        print(f"rkn m{m}: verdict {verdict} cig={rec['n_exploit_cig']}"
              f" pbim={rec['n_exploit_pbim']} "
              f"both={rec['n_exploit_both']}", flush=True)


def cmd_heldout(tag, out_root=None, episodes=None):
    from .heldout_eval import collect as ho_collect, eval_exact
    from .learnedwm import DiagExactFilterAdapter
    outdir = (out_root or OUT) / tag
    ens = load_rkn_ensemble(outdir / "ensemble.pkl")
    override = episodes is not None
    episodes = episodes or ho_collect()
    exact = eval_exact(episodes)
    if not override:
        # #33 F24: gate silent lgfield/heldout_eval drift against the
        # published 12-Aug floor (1e-3: cross-host float envelope)
        assert abs(exact[0] - 0.30059) < 1e-3 and \
            abs(exact[1] - (-1.82030)) < 1e-3, exact

    def _model_nll(model):
        onll = znll = 0.0
        n_sense = n_step = 0
        for recs in episodes:
            b = model.init_belief()
            for r in recs:
                if r["sensed"] is not None:
                    ymu, ylv = model.obs_pred(
                        b, lg.N_NODES + r["sensed"])
                    v = np.exp(ylv)
                    onll += 0.5 * (np.log(2 * np.pi * v)
                                   + (r["y"] - ymu) ** 2 / v)
                    n_sense += 1
                b = model.step(b, r["action"], r["node"], r["y"])
                mu, lv = model.zstats(b)
                znll += float(np.sum(
                    0.5 * (np.log(2 * np.pi) + lv
                           + (r["z_after"] - mu) ** 2
                           / np.exp(lv))))
                n_step += 1
        return onll / n_sense, znll / n_step

    # #33 F11: the RKN z-NLL is the DIAGONAL surrogate; eval_exact's
    # floor is full-covariance — emit the diagonal exact floor too
    # and scope the comparison to it
    diag_floor = _model_nll(DiagExactFilterAdapter())
    rows = []
    for m in range(N_MEMBERS):
        o, z = _model_nll(RKNModel(ens[m]))
        rows.append(dict(member=m, obs_nll=o, z_nll=z))
        print(f"rkn m{m}: heldout obs-NLL {o:.4f} z-NLL {z:.4f}",
              flush=True)
    with open(outdir / "heldout.json", "w") as f:
        json.dump(dict(tag=tag,
                       exact_floor_fullcov=exact,
                       exact_floor_diag=dict(obs_nll=diag_floor[0],
                                             z_nll=diag_floor[1]),
                       members=rows,
                       convention="heldout_eval seeds 777123/888777; "
                                  "obs before update, z after step; "
                                  "P-RKN3 comparisons use the DIAG "
                                  "floor (#33 F11)"),
                  f, indent=1)


def cmd_audit(tag, out_root=None, n_trials=AUDIT_TRIALS,
              nmax=AUDIT_NMAX):
    from .ratio_research import _san
    outdir = (out_root or OUT) / tag
    ens = load_rkn_ensemble(outdir / "ensemble.pkl")
    out = []
    for m in range(N_MEMBERS):
        rows = audit_member(ens[m], n_trials=n_trials, nmax=nmax,
                            seed=m)
        out.append(dict(member=m, sensors=rows))
        pr = [r for r in rows if r["status"] == "PRICED"]
        print(f"rkn m{m}: {len(pr)}/{len(rows)} priced; "
              + "; ".join(f"{r['sensor']} g={r['gamma_hat']:+.3f} "
                          f"E[N]={r['sprt_mean_reads']:.0f} "
                          f"bnd={r['info_bound_reads']:.0f} "
                          f"ok={r['criteria_met']}"
                          for r in pr), flush=True)
    with open(outdir / "audit.json", "w") as f:
        json.dump(_san(dict(
            tag=tag, delta=AUDIT_DELTA, lr_conf=AUDIT_LR_CONF,
            nmax=nmax, n_trials=n_trials,
            audit_sensors=[lg.SENSORS[k].name for k in AUDIT_SENSORS],
            scope="r-channel audit of static one-hot readouts from "
                  "the FRESH prior v0=c'P0c (#33 F7 scoping); "
                  "C-channel and the TV sensor OUT OF SCOPE — see "
                  "c_hat_norm rows (#33 F16)",
            members=out)), f, indent=1)


# ---------------------------------------------------------------- selfcheck

def selfcheck():
    from . import planner as pl
    from .learnedwm import DiagExactFilterAdapter
    from . import mechanism as mech
    import tempfile

    # (1) ORACLE EQUIVALENCE over 200 real steps
    true = RKNModel(make_true_params())
    ref = DiagExactFilterAdapter()
    env = lg.Env(seed=123)
    rng = np.random.default_rng(7)
    b_r, b_e = true.init_belief(), ref.init_belief()
    for _ in range(200):
        node = env.node
        a = int(rng.choice(lg.valid_actions(node)))
        _, y, _ = env.step(a)
        b_r = true.step(b_r, a, node, y)
        b_e = ref.step(b_e, a, node, y)
        assert np.abs(b_r - b_e).max() < 1e-8
    mu_r, lv_r = true.zstats(b_r)
    mu_e, lv_e = ref.zstats(b_e)
    assert np.abs(mu_r - mu_e).max() < 1e-8
    assert np.abs(lv_r - lv_e).max() < 1e-4     # log-space amplif.
    assert abs(true.entropy(b_r) - ref.entropy(b_e)) < 1e-4
    for k in (0, 5, 7):
        pr = true.obs_pred(b_r, lg.N_NODES + k)
        pe = ref.obs_pred(b_e, lg.N_NODES + k)
        assert abs(pr[0] - pe[0]) < 1e-8 and abs(pr[1] - pe[1]) < 1e-6

    # (2) STRUCTURAL COHERENCE at random parameter points, any y
    for seed in (0, 1):
        model = RKNModel(init_rkn_params(seed))
        b = model.init_belief()
        rr = np.random.default_rng(seed + 50)
        for _ in range(10):
            b = model.step(b, int(rr.integers(0, lg.N_NODES)), 0,
                           None)
        for k, s in enumerate(lg.SENSORS):
            a = lg.N_NODES + k
            eig = pl._eig(model, b, a, s.node)
            for y in (-2.0, 0.0, 3.7):
                realized = model.entropy(b) - model.entropy(
                    model.step(b, a, s.node, y))
                assert abs(eig - realized) < 1e-9, (k, eig, realized)

    # (3) pure-update commutativity in P
    model = RKNModel(init_rkn_params(2))

    def pure_update(mdl, b, k, y):
        mu, p = RKNModel._unpack(b)
        c, r = mdl.C[k], mdl.r[k]
        s = float(c @ p @ c + r)
        kg = (p @ c) / s
        mu = mu + kg * (y - float(c @ mu))
        p = p - np.outer(kg, c @ p)
        return np.concatenate([mu, (0.5 * (p + p.T)).flatten()])

    b0 = model.init_belief()
    ba = pure_update(model, pure_update(model, b0, 0, 0.3), 7, -0.5)
    bb = pure_update(model, pure_update(model, b0, 7, -0.5), 0, 0.3)
    _, pa = RKNModel._unpack(ba)
    _, pb = RKNModel._unpack(bb)
    assert np.abs(pa - pb).max() < 1e-10

    # (4) diagonal-lens heads consistency + the overlap artifact
    b = true.init_belief()
    for k in AUDIT_SENSORS:
        acc = mech.head_accounts(true, b, lg.N_NODES + k)
        assert abs(acc["r_implied"] - lg.SENSORS[k].r) < 1e-9
        drop = true.entropy(b) - true.entropy(
            pure_update(true, b, k, 0.0))
        assert abs(acc["bayes_gain"] - drop) < 1e-7, (k, acc, drop)
    b1 = true.step(b, lg.N_NODES + 4, 0, 0.5)
    acc = mech.head_accounts(true, b1, lg.N_NODES + 4)
    _, p1 = RKNModel._unpack(b1)
    resid = acc["r_implied"] - lg.SENSORS[4].r
    assert abs(resid - 2.0 * p1[0, 1]) < 1e-9
    assert acc["r_implied"] < 0 and acc["bayes_gain"] == float("inf")

    # (4b) overlap detection from ACTIONS (#33 F4): the name check
    # undercounts by exactly the _all cycles through node 0
    cs = pl.enumerate_cycles(max_len=4)
    n_act = sum(1 for c in cs if any(a == lg.N_NODES + 4
                                     for a in c["actions"]))
    n_name = sum(1 for c in cs if "s4" in c["name"])
    assert n_act > n_name, (n_act, n_name)

    # (5) training sanity + determinism + F1 cap identity
    eps = []
    for e in range(4):
        env = lg.Env(seed=900 + e)
        r = lg.Referee()
        eps.append(lg.rollout_random(env, r, 60,
                                     np.random.default_rng(30 + e)))
    p1_, l1, i1 = train_rkn(eps, seed=0, steps=60, batch=4)
    p2_, l2, _ = train_rkn(eps, seed=0, steps=60, batch=4)
    assert l1 == l2
    for k in p1_:
        assert np.array_equal(p1_[k], p2_[k]), k
    _, l0, _ = train_rkn(eps, seed=0, steps=1, batch=4)
    assert l1 < l0, (l1, l0)
    m1 = RKNModel(p1_)
    assert np.array_equal(m1.A, np.asarray(p1_["A_eff"]))  # F1
    assert "sigma_pi" in i1 and "batch" in i1 and i1["batch"] == 4
    # numpy power-iteration mirror agrees with well-separated SVD
    rr = np.random.default_rng(3)
    for _ in range(3):
        u = np.diag([1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
        q1, _ = np.linalg.qr(rr.standard_normal((DZ, DZ)))
        q2, _ = np.linalg.qr(rr.standard_normal((DZ, DZ)))
        a = q1 @ u @ q2.T
        assert abs(sigma_max_pi(a) - np.linalg.norm(a, 2)) < 1e-6

    # (6) audit machinery (#33 F7/F8/F9/F22 form)
    honest = make_true_params()
    rows = audit_member(honest, n_trials=40, nmax=1500, seed=0)
    assert all(r["status"] == "BELOW-AUDIT-RESOLUTION"
               for r in rows), rows
    near = make_true_params()
    near["logr"] = near["logr"] + np.log(1.01)      # gamma = 0.01
    rows = audit_member(near, n_trials=40, nmax=1500, seed=0)
    assert all(r["status"] == "BELOW-AUDIT-RESOLUTION"
               for r in rows)                        # F22
    liar = make_true_params()
    liar["logr"] = liar["logr"] + np.log(2.0)
    rows = audit_member(liar, n_trials=120, nmax=20000, seed=0)
    for r in rows:
        assert r["status"] == "PRICED", r
        assert r["sprt_crossed_frac"] >= 0.98
        assert r["sprt_power"] > 0.9
        assert r["false_conviction_rate"] <= 0.1     # F9
        assert 0.3 < r["sprt_mean_reads"] / r["info_bound_reads"] < 4
        assert 0.5 < r["wald_ratio"] < 3.0, r
    assert kl_gauss_var(1.0, 1.0) == 0.0

    # (7) END-TO-END tiny driver pass (#33 F13): train 4 members on
    # synthetic data, search (max_len=4), heldout (override
    # episodes), audit (small budget) — all four jsons written
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        cmd_train("e2e", steps=25, verbose=False, out_root=root,
                  episodes=eps, batch=4)
        cmd_search("e2e", member=-1, out_root=root, max_len=4,
                   max_rows=30)
        cmd_heldout("e2e", out_root=root, episodes=eps[:2])
        cmd_audit("e2e", out_root=root, n_trials=10, nmax=800)
        for fn in ("train_info.json", "search_m0.json",
                   "search_m3.json", "heldout.json", "audit.json"):
            with open(root / "e2e" / fn) as f:
                json.load(f)
        with open(root / "e2e" / "search_m0.json") as f:
            sj = json.load(f)
        assert sj["naive_equals_carried"] is True
        assert all(("uses_overlap" in r and "neutral" in r
                    and "exploit_clipped" in r)
                   for r in sj["consistency_rows"])
        with open(root / "e2e" / "heldout.json") as f:
            hj = json.load(f)
        assert "exact_floor_diag" in hj

    print("rknwm selfcheck PASS (oracle equivalence; structural "
          "coherence at random params; commutativity; diagonal-lens "
          "consistency + overlap artifact pinned + action-based "
          "overlap detection; training determinism + A_eff cap "
          "identity + power-iteration mirror; audit: honest and "
          "near-honest BELOW-AUDIT-RESOLUTION, 2x liar PRICED with "
          "power>=0.9, false-conviction<=0.1, Wald in band; "
          "end-to-end tiny driver: all four jsons incl. verbatim "
          "minclip fields and the diag floor)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--tag", default="rkn_pilot2")
    ap.add_argument("--member", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    if args.train:
        cmd_train(args.tag, args.steps, args.verbose,
                  batch=args.batch, lr=args.lr)
    if args.search:
        cmd_search(args.tag, args.member)
    if args.heldout:
        cmd_heldout(args.tag)
    if args.audit:
        cmd_audit(args.tag)


if __name__ == "__main__":
    main()
