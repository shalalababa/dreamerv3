"""Worked example for the CEI exposure-budget theorem (12 Aug 2026).

Promoted from the v0.1 math review's verification script (review #19;
exact Gaussian KLs + quadrature, no Monte Carlo). Instantiates
Theory_CEI_Exposure_20260812.tex §6 obligations O4:

  A) the R2 counterexample: the pointwise per-read bound kappa_j <= kappa
     FAILS at a 2-sigma observation (mean-channel term) — recorded as the
     reason the budget theorem must be stated in expectation;
  B) the variance-only channel alone never violates (sup ratio <= 1);
  C) two-block coupled system: exact chain-rule decomposition — at eps=0
     the direct term equals v3.2's KL_n formula and the coupled term
     vanishes; totals never exceed n*kappa; coupled term scales as eps^2
     (small-Delta slope 2, large-Delta saturation);
  D/E) adaptive observation-dependent policies (both selection
     directions): total transcript KL <= kappa * E_P0[N_j] everywhere,
     even where the SELECTED per-read divergence exceeds kappa —
     selection is paid from the budget;
  F) the v0.1 rho functional fails its eps=0 sanity check; the
     normalized cross-covariance does not.

Run:  python -m uncfield.exposure_example            (writes json)
      python -m uncfield.exposure_example --selfcheck
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

OUTDIR = (pathlib.Path(__file__).resolve().parent.parent / "artifacts"
          / "nfi_assessment_response_20260812")


def kl_var(s0, s1):
    return 0.5 * (np.log(s1 / s0) + s0 / s1 - 1)


def kl_gauss1d(m0, s0, m1, s1):
    return 0.5 * (np.log(s1 / s0) + (s0 + (m0 - m1) ** 2) / s1 - 1)


def kl_gauss(S0, S1):
    d = S0.shape[0]
    S1i = np.linalg.inv(S1)
    return 0.5 * (np.trace(S1i @ S0) - d
                  + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])


# A ------------------------------------------------------------------------

def counterexample(v0=1.0, r0=0.01, r1=1.0):
    """Pointwise kappa_j(2,h) after one j-read of value Y, vs kappa."""
    kappa = kl_var(r0, r1)
    rows = []
    for Y in (0.0, 1.0, 2.0, 3.0, 5.0):
        m = {i: Y * v0 / (v0 + r) for i, r in ((0, r0), (1, r1))}
        u = {i: v0 * r / (v0 + r) for i, r in ((0, r0), (1, r1))}
        kj = kl_gauss1d(m[0], u[0] + r0, m[1], u[1] + r1)
        rows.append(dict(Y=Y, kappa_j=kj, violates=bool(kj > kappa)))
    return kappa, rows


# B ------------------------------------------------------------------------

def variance_channel_sup(n_trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    worst = -np.inf
    for _ in range(n_trials):
        v0, r0, r1 = np.exp(rng.uniform(-4, 3, 3))
        n = int(rng.integers(1, 50))
        if abs(np.log(r1 / r0)) < 1e-6:
            continue
        u0 = v0 * r0 / (r0 + n * v0)
        u1 = v0 * r1 / (r1 + n * v0)
        worst = max(worst, kl_var(u0 + r0, u1 + r1) / kl_var(r0, r1))
    return worst


# C ------------------------------------------------------------------------

def coupled_decomposition(v1=1.0, v2=1.0, r0=0.5, r1=1.0, R2=0.3, n=3):
    kappa = kl_var(r0, r1)

    def totals(e, r1_=r1):
        P0m = np.array([[v1, e], [e, v2]])
        H = np.vstack([np.tile([1.0, 0.0], (n, 1)), [[0.0, 1.0]]])

        def Sig(r):
            return H @ P0m @ H.T + np.diag([r] * n + [R2])

        tot = kl_gauss(Sig(r0), Sig(r1_))
        dire = kl_gauss(Sig(r0)[:n, :n], Sig(r1_)[:n, :n])
        return tot, dire, tot - dire

    kln = kl_var(n * v1 + r0, n * v1 + r1) + (n - 1) * kl_var(r0, r1)
    t0, d0, c0 = totals(0.0)
    es = np.array([0.05, 0.1, 0.2, 0.4])
    rows = [totals(e) for e in es]
    slope_e = float(np.polyfit(np.log(es),
                               np.log([r[2] for r in rows]), 1)[0])
    # Delta-scaling at fixed e=0.2
    gs = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0])
    cs2 = []
    for g in gs:
        P0m = np.array([[v1, 0.2], [0.2, v2]])
        H = np.vstack([np.tile([1.0, 0.0], (n, 1)), [[0.0, 1.0]]])

        def Sig(r):
            return H @ P0m @ H.T + np.diag([r] * n + [R2])

        cs2.append(kl_gauss(Sig(r0), Sig(r0 * (1 + g)))
                   - kl_gauss(Sig(r0)[:n, :n], Sig(r0 * (1 + g))[:n, :n]))
    sl_small = float(np.polyfit(np.log(gs[:3]), np.log(cs2[:3]), 1)[0])
    sl_large = float(np.polyfit(np.log(gs[-3:]), np.log(cs2[-3:]), 1)[0])
    return dict(kappa=kappa, kl_n_formula=kln, direct_at_e0=d0,
                coupled_at_e0=c0, budget=n * kappa,
                totals=[dict(e=float(e), total=t, direct=d, coupled=c)
                        for e, (t, d, c) in zip(es, rows)],
                slope_eps=slope_e, slope_delta_small=sl_small,
                slope_delta_large=sl_large)


# D/E ----------------------------------------------------------------------

def adaptive_budget(v1=1.0, v2=1.0, e=0.4, r0=0.5, r1=1.0, R2=0.3):
    """t=1 read j; t=2 read j iff selector(y1) else read k. Exact by
    quadrature. Returns rows for both selection directions."""
    kappa = kl_var(r0, r1)

    def cond_j(y1, r):
        return v1 / (v1 + r) * y1, v1 + r - v1 ** 2 / (v1 + r)

    def cond_k(y1, r):
        return e / (v1 + r) * y1, v2 + R2 - e ** 2 / (v1 + r)

    def total(c, read_j_outside):
        kl1 = kl_var(v1 + r0, v1 + r1)

        def integrand(y1):
            p = norm.pdf(y1, 0, np.sqrt(v1 + r0))
            branch_j = (abs(y1) > c) if read_j_outside else (abs(y1) <= c)
            f = cond_j if branch_j else cond_k
            m0, s0 = f(y1, r0)
            m1, s1 = f(y1, r1)
            return p * kl_gauss1d(m0, s0, m1, s1)

        val = sum(quad(integrand, a, b, limit=200)[0]
                  for a, b in [(-40, -c), (-c, c), (c, 40)])
        pj = 2 * norm.sf(c, 0, np.sqrt(v1 + r0))
        ENj = 1 + (pj if read_j_outside else 1 - pj)
        return kl1 + val, ENj

    rows = []
    for c in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
        tot, ENj = total(c, True)
        m0, s0 = cond_j(c, r0)
        m1, s1 = cond_j(c, r1)
        rows.append(dict(rule="j_if_outside", c=c, total=tot,
                         budget=kappa * ENj, holds=bool(tot <= kappa * ENj + 1e-9),
                         selected_kappa_j=kl_gauss1d(m0, s0, m1, s1)))
    for c in (0.5, 1.5, 3.0):
        tot, ENj = total(c, False)
        rows.append(dict(rule="j_if_inside", c=c, total=tot,
                         budget=kappa * ENj,
                         holds=bool(tot <= kappa * ENj + 1e-9)))
    return kappa, rows


# F ------------------------------------------------------------------------

def rho_sanity():
    P0m = np.eye(2)
    cj, ck = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    Ph = np.linalg.cholesky(P0m)
    outer = float(np.linalg.norm(np.outer(Ph.T @ ck, Ph.T @ cj), 2))
    cross = float(ck @ P0m @ cj)
    return outer, cross


def run():
    kappa_a, ce = counterexample()
    out = dict(
        A_counterexample=dict(kappa=kappa_a, rows=ce),
        B_variance_sup=variance_channel_sup(),
        C_decomposition=coupled_decomposition(),
        DE_adaptive=adaptive_budget()[1],
        F_rho=dict(outer_at_e0=rho_sanity()[0], cross_at_e0=rho_sanity()[1]),
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "exposure_example.json", "w") as f:
        json.dump(out, f, indent=1)
    print("written", OUTDIR / "exposure_example.json")
    return out


def selfcheck():
    kappa, ce = counterexample()
    assert any(r["violates"] for r in ce), "counterexample must fire"
    assert not ce[0]["violates"] and ce[2]["violates"], \
        "violation appears at Y=2 (2-sigma), not at Y=0"
    sup = variance_channel_sup(n_trials=5000)
    assert sup <= 1.0 + 1e-9, sup
    d = coupled_decomposition()
    assert abs(d["direct_at_e0"] - d["kl_n_formula"]) < 1e-9
    assert abs(d["coupled_at_e0"]) < 1e-12
    assert all(r["total"] <= d["budget"] + 1e-12 for r in d["totals"])
    assert 1.9 < d["slope_eps"] < 2.2, d["slope_eps"]
    assert 1.9 < d["slope_delta_small"] < 2.1
    assert d["slope_delta_large"] < 1.6                  # saturation
    kap, rows = adaptive_budget()
    assert all(r["holds"] for r in rows)
    assert max(r.get("selected_kappa_j", 0) for r in rows) > kap, \
        "selected per-read divergence must exceed kappa somewhere"
    outer, cross = rho_sanity()
    assert outer > 0.5 and abs(cross) < 1e-12, (outer, cross)
    print(f"exposure_example selfcheck PASS (counterexample fires at Y=2; "
          f"variance sup {sup:.4f}<=1; budget holds all cells; "
          f"slopes {d['slope_eps']:.2f}/{d['slope_delta_small']:.2f}; "
          f"selected {max(r.get('selected_kappa_j',0) for r in rows):.3f}"
          f">{kap:.3f}=kappa)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    selfcheck() if args.selfcheck else run()


if __name__ == "__main__":
    main()
