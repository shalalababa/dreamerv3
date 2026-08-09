"""L2 representation-adequacy control: full-covariance Cholesky z-head
(ideation fold-in, Research_Ideation_Round_20260809.tex §L2; 9 Aug 2026).

Closes the reviewer objection "the model is not broken, merely
under-equipped — give it the missing covariance parameters and the defect
disappears" (the representation-adequacy sibling of the already-refuted
undertraining objection). Design per the ideation round:

  * Train GRU members whose z-head emits a full Cholesky factor
    (mu, log-diag, strict-lower off-diag: 8+8+28 outputs) with
    full-covariance Gaussian NLL, on the EXISTING pilot2 episodes at the
    registered seeds/steps/batch (0-3 / 3000 / 32). Trunk + obs head init
    identical to the diagonal members (same _glorot keys); only the
    z-head output block is re-initialised.
  * ABLATION against the frozen diagonal thresholds, never a replacement
    scoring functional: the primary readout scores the MARGINAL entropy
    0.5*sum log(2*pi*e*Sigma_ii) — same functional class the thresholds
    were calibrated on. A second, descriptive view scores the JOINT
    log-det entropy (the acquisition objective CIG App. B.2 leaves open).
  * Primary exhibit: the 46 pure-s4 cycles (s4 = z0+z1, the one sensor
    with genuine off-diagonal mass). Outcomes all decisive: carried rate
    falls => read-out expressivity was the binding defect; holds => no
    separate channel; rises toward the blind-oracle ceiling (3.58) =>
    larger design warranted.
  * Correlation-learning diagnostic (a null is ambiguous without it):
    model rho_01 vs the exact referee's rho_01 at the shared warmup
    state — distinguishes "did not learn correlation" from "learned it,
    does not matter".

SCOPE: descriptive control for the NFI manuscript's defenses section; no
registered decision rules. --full additionally runs the whole 626-cycle
search_exploits under both entropy views (verdict + pilot-schema npz).

Run:  python -m uncfield.fullcov --selfcheck
      python -m uncfield.fullcov --run          (train + s4 scoring)
      python -m uncfield.fullcov --run --full   (adds full-library sweeps)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import learnedwm as lw
from . import lgfield as lg
from . import planner as pl

DZ = lg.DZ
N_TRIL = DZ * (DZ - 1) // 2
FULL_OUT = 2 * DZ + N_TRIL                    # mu, ldiag, offdiag = 44
TRIL = np.tril_indices(DZ, -1)
PILOT2 = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield" / "pilot2"
OUT = PILOT2.parent / "fullcov"
SEEDS, STEPS, BATCH = (0, 1, 2, 3), 3000, 32   # pilot2 config.json, frozen


# ------------------------------------------------------------------ full head

def init_params_full(seed, hid=lw.HID):
    """Diagonal-member init with ONLY the z-head output block widened."""
    p = lw.init_params(seed, hid=hid)
    p["zh_W2"] = lw._glorot(jax.random.PRNGKey(seed + 777), (64, FULL_OUT))
    p["zh_b2"] = jnp.zeros(FULL_OUT)
    return p


def is_fullcov(p):
    return p["zh_W2"].shape[1] == FULL_OUT


def z_head_full(p, b):
    h = jnp.tanh(lw._head_features(p, b) @ p["zh_W1"] + p["zh_b1"])
    out = h @ p["zh_W2"] + p["zh_b2"]
    mu = out[..., :DZ]
    # ldiag = log of the Cholesky diagonal; clip mirrors the diagonal
    # head's logvar clamp (marginal variance floor e^LOGVAR_MIN).
    ldiag = jnp.clip(out[..., DZ:2 * DZ], 0.5 * lw.LOGVAR_MIN, 0.5 * lw.LOGVAR_MAX)
    return mu, ldiag, out[..., 2 * DZ:]


def build_chol(ldiag, off):
    L = jnp.zeros((DZ, DZ)).at[jnp.diag_indices(DZ)].set(jnp.exp(ldiag))
    return L.at[TRIL].set(off)


def full_nll(z, mu, ldiag, off):
    """Exact Gaussian NLL with Sigma = L L^T (logdet = 2*sum(ldiag))."""
    L = build_chol(ldiag, off)
    u = jax.scipy.linalg.solve_triangular(L, z - mu, lower=True)
    return (0.5 * DZ * jnp.log(2.0 * jnp.pi) + jnp.sum(ldiag)
            + 0.5 * jnp.sum(u ** 2))


def seq_loss_full(p, xs, a1h, zt, ymask, ytrue):
    """lw.seq_loss with the z-term swapped to the full-covariance NLL."""
    b0 = jnp.zeros(lw.belief_width(p))

    def step(b, inp):
        x, a, z, m, y = inp
        ymu, ylv = lw.obs_head(p, b, a)
        onll = m * lw._gauss_nll(y, ymu, ylv)
        b_next = lw.cell_step(p, b, x)
        mu, ldiag, off = z_head_full(p, b_next)
        return b_next, full_nll(z, mu, ldiag, off) + onll

    _, losses = jax.lax.scan(step, b0, (xs, a1h, zt, ymask, ytrue))
    return jnp.mean(losses)


def batch_loss_full(p, batch):
    return jnp.mean(jax.vmap(lambda *seq: seq_loss_full(p, *seq))(*batch))


def train_model_full(episodes, seed, steps=STEPS, batch=BATCH, hid=lw.HID,
                     lr=1e-3, verbose=True):
    data = lw.episodes_to_arrays(episodes)
    n = data[0].shape[0]
    params = init_params_full(seed, hid=hid)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def update(p, s, batch_):
        loss, grads = jax.value_and_grad(batch_loss_full)(p, batch_)
        upd, s = opt.update(grads, s)
        return optax.apply_updates(p, upd), s, loss

    rng = np.random.default_rng(seed)
    loss = np.nan
    for i in range(steps):
        idx = rng.integers(0, n, size=batch)
        params, opt_state, loss = update(params, opt_state,
                                         tuple(d[idx] for d in data))
        if verbose and (i % 500 == 0 or i == steps - 1):
            print(f"    [full seed {seed}] step {i} loss {float(loss):.4f}",
                  flush=True)
    return params, float(loss)


# ------------------------------------------------------------ frozen wrappers

class FullCovModel(lw.LearnedModel):
    """Pilot numpy API over a full-covariance member. entropy() is the
    MARGINAL functional (inherited diagonal thresholds stay licensed);
    entropy_joint() is the descriptive log-det view."""

    def __init__(self, params):
        assert is_fullcov(params)
        super().__init__(params)
        self._zf = jax.jit(lambda p, b: z_head_full(p, b))

    def _mu_chol(self, b):
        mu, ldiag, off = self._zf(self.p, jnp.asarray(b))
        L = np.zeros((DZ, DZ))
        L[np.diag_indices(DZ)] = np.exp(np.asarray(ldiag))
        L[TRIL] = np.asarray(off)
        return np.asarray(mu), L

    def zstats(self, b):
        mu, L = self._mu_chol(b)
        return mu, np.log((L ** 2).sum(axis=1))      # log marginal variances

    def entropy(self, b):
        _, lv = self.zstats(b)
        return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e) + lv))

    def entropy_joint(self, b):
        _, L = self._mu_chol(b)
        return float(0.5 * DZ * np.log(2.0 * np.pi * np.e)
                     + np.sum(np.log(np.diag(L))))

    def corr(self, b):
        _, L = self._mu_chol(b)
        S = L @ L.T
        d = np.sqrt(np.diag(S))
        return S / np.outer(d, d)


class JointView:
    """Same model, log-det entropy as the scored functional (descriptive)."""

    def __init__(self, model):
        self._m = model

    def __getattr__(self, name):
        return getattr(self._m, name)

    def entropy(self, b):
        return self._m.entropy_joint(b)


# ----------------------------------------------------------------- experiment

def _save_npz(path, results):
    np.savez(path,
             names=np.array([r["cycle"]["name"] for r in results]),
             true_rate=np.array([r["true_rate"] for r in results]),
             naive=np.array([r["naive_eig_rate"] for r in results]),
             carried=np.array([r["carried_eig_rate"] for r in results]),
             dh=np.array([r["carried_dh_rate"] for r in results]),
             dh_adj=np.array([r["carried_dh_adj"] for r in results]),
             transient=np.array([r["transient_rate"] for r in results]),
             neutral=np.array([r["neutral"] for r in results]))


def _s4_cycles():
    cyc = [c for c in pl.enumerate_cycles(max_len=6)
           if c["name"].endswith("only_s4_z01_n0")]
    assert len(cyc) == 46, len(cyc)
    return cyc


def _score_cycles(model, cycles, b0, ref0, node0):
    """Replicates search_exploits' routing for a cycle subset; returns
    {name: steady carried_eig rate}."""
    out = {}
    for cycle in cycles:
        b, node = b0.copy(), node0
        for mv in pl._route_to(node0, cycle["start"]):
            b = model.step(b, mv, node, None)
            node = mv
        rates, _, _ = pl.score_cycle(model, b, node, cycle, seed=0)
        out[cycle["name"]] = pl.steady_rate(rates["carried_eig"])
    return out


def run(full=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ens_path = OUT / "ensemble_full.pkl"
    if ens_path.exists():
        ensemble = lw.load_ensemble(ens_path)
        print("loaded existing ensemble_full.pkl", flush=True)
    else:
        with open(PILOT2 / "episodes.pkl", "rb") as f:
            episodes = pickle.load(f)
        ensemble = []
        for m in SEEDS:
            t0 = time.time()
            params, loss = train_model_full(episodes, seed=m)
            ensemble.append(params)
            print(f"  member {m}: final loss {loss:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        lw.save_ensemble(ens_path, ensemble)

    cycles = _s4_cycles()
    names = [c["name"] for c in cycles]
    record = {"seeds": list(SEEDS), "steps": STEPS, "batch": BATCH,
              "eps_pred": pl.EPS_PRED, "members": []}
    for m, params in enumerate(ensemble):
        model = FullCovModel(params)
        b0, ref0, node0 = pl.warmup_state(model, seed=0)
        P = ref0.p
        rho_true = float(P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))
        rho_model = float(model.corr(b0)[0, 1])
        marg = _score_cycles(model, cycles, b0, ref0, node0)
        joint = _score_cycles(JointView(model), cycles, b0, ref0, node0)
        dz = np.load(PILOT2 / f"results_m{m}.npz")
        didx = {str(n): i for i, n in enumerate(dz["names"])}
        diag = {n: float(dz["carried"][didx[n]]) for n in names}
        ent = dict(
            member=m,
            warmup_rho01_model=rho_model, warmup_rho01_true=rho_true,
            diag_n_exploit=sum(v > pl.EPS_PRED for v in diag.values()),
            diag_max=max(diag.values()),
            marg_n_exploit=sum(v > pl.EPS_PRED for v in marg.values()),
            marg_max=max(marg.values()),
            joint_n_exploit=sum(v > pl.EPS_PRED for v in joint.values()),
            joint_max=max(joint.values()),
            cycles={n: dict(diag=diag[n], full_marginal=marg[n],
                            full_joint=joint[n]) for n in names},
        )
        record["members"].append(ent)
        print(f"m{m}: s4 exploits diag/marg/joint = "
              f"{ent['diag_n_exploit']}/{ent['marg_n_exploit']}/"
              f"{ent['joint_n_exploit']}  max = {ent['diag_max']:.3f}/"
              f"{ent['marg_max']:.3f}/{ent['joint_max']:.3f}  "
              f"rho01 model {rho_model:+.3f} vs true {rho_true:+.3f}",
              flush=True)

        if full:
            for view_name, mdl in (("marginal", model),
                                   ("joint", JointView(model))):
                t0 = time.time()
                results, _ = pl.search_exploits(mdl, seed=0, max_len=6)
                verdict, counts = pl.classify(results)
                _save_npz(OUT / f"full_{view_name}_m{m}.npz", results)
                ent[f"full_{view_name}_verdict"] = verdict
                ent[f"full_{view_name}_counts"] = counts
                print(f"  m{m} full[{view_name}]: {verdict} "
                      f"cig={counts['n_exploit_cig']} "
                      f"both={counts['n_exploit_both']} "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    with open(OUT / "fullcov_results.json", "w") as f:
        json.dump(record, f, indent=1)
    print("written", OUT / "fullcov_results.json", flush=True)


# ------------------------------------------------------------------ selfcheck

def selfcheck():
    rng = np.random.default_rng(0)
    # (1) full_nll matches the closed form computed independently in numpy.
    ldiag = rng.normal(size=DZ) * 0.3
    off = rng.normal(size=N_TRIL) * 0.3
    mu = rng.normal(size=DZ)
    z = rng.normal(size=DZ)
    L = np.zeros((DZ, DZ))
    L[np.diag_indices(DZ)] = np.exp(ldiag)
    L[TRIL] = off
    S = L @ L.T
    ref = 0.5 * (DZ * np.log(2 * np.pi) + np.linalg.slogdet(S)[1]
                 + (z - mu) @ np.linalg.solve(S, z - mu))
    got = float(full_nll(jnp.asarray(z), jnp.asarray(mu),
                         jnp.asarray(ldiag), jnp.asarray(off)))
    assert abs(got - ref) < 1e-5, (got, ref)  # float32
    # (2) zero off-diagonal degenerates EXACTLY to the diagonal NLL with
    # logvar = 2*ldiag (the diagonal head as a special case).
    got0 = float(full_nll(jnp.asarray(z), jnp.asarray(mu),
                          jnp.asarray(ldiag), jnp.zeros(N_TRIL)))
    refd = float(jnp.sum(lw._gauss_nll(jnp.asarray(z), jnp.asarray(mu),
                                       jnp.asarray(2.0 * ldiag))))
    assert abs(got0 - refd) < 1e-5, (got0, refd)  # float32
    # (3) Hadamard: joint entropy <= marginal entropy, equality iff diagonal.
    p = init_params_full(0)
    model = FullCovModel(p)
    b = np.asarray(rng.normal(size=lw.belief_width(p)), np.float32)
    assert model.entropy_joint(b) <= model.entropy(b) + 1e-9
    # (4) API: both views run through score_cycle end-to-end, finite rates.
    cycles = _s4_cycles()
    b0, ref0, node0 = pl.warmup_state(model, seed=0, t_steps=10)
    for mdl in (model, JointView(model)):
        b, node = b0.copy(), node0
        for mv in pl._route_to(node0, cycles[0]["start"]):
            b = mdl.step(b, mv, node, None)
            node = mv
        rates, _, _ = pl.score_cycle(mdl, b, node, cycles[0],
                                     n_loops=2, n_burn=2, seed=0)
        assert np.isfinite(rates["carried_eig"]).all()
    # (5) training smoke: loss decreases on a few real episodes.
    with open(PILOT2 / "episodes.pkl", "rb") as f:
        episodes = pickle.load(f)[:8]
    data = lw.episodes_to_arrays(episodes)
    p0 = init_params_full(0)
    l0 = float(batch_loss_full(p0, data))
    p1, l1 = train_model_full(episodes, seed=0, steps=60, batch=8,
                              verbose=False)
    assert l1 < l0, (l0, l1)
    print(f"fullcov selfcheck PASS (nll exact, diag-degenerate exact, "
          f"Hadamard ok, API ok, smoke loss {l0:.2f}->{l1:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="also run the 626-cycle sweep under both views")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.run:
        run(full=args.full)


if __name__ == "__main__":
    main()
