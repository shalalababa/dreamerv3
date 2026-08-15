#!/usr/bin/env python3
"""Route-1 simulation suite — selection-induced evaluation bias in RL loops.

Paper 2 Route-1 (GPT publishability review 14 Aug, adjudicated SOUND 15 Aug,
user GO 15 Aug). NON-REGISTERED demonstrative instrument: no estimand is
measured on experiment outputs; every quantity is computed on synthetic
draws with known ground truth. Purpose: (i) demonstrate each bias channel
of the Route-1 taxonomy in isolation and composed, at noise scales
calibrated to the archived Paper-2 case; (ii) show each corrected
estimator drives the bias to zero; (iii) measure size/power of the
diagnostic battery (duplicate-candidate null, permutation null,
cross-fit delta).

Channels (taxonomy doc Route1_Taxonomy_20260815.md):
  A  max-over-noisy-estimates (order-statistic pedestal)
  B  selected-sample reuse (winner's curse proper)
  C  shared-stochasticity leakage (probe shares randomness with outcome)
  D  wrong-null referencing (bars referenced to 0 instead of E[max-mean])

Usage:
  python route1_sims.py --selfcheck     # closed-form + determinism gates
  python route1_sims.py --out results.json

Runtime: ~2-4 min CPU, numpy only, seed fixed (20260815).
"""

import argparse
import json
import math
import sys

import numpy as np

SEED = 20260815
PHI = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------
# closed forms (the selfcheck anchors; verified 2 Jul 2026 synthetic check)
# ----------------------------------------------------------------------

def exp_max_std_normal(k: int, n_mc: int = 2_000_000, rng=None) -> float:
    """E[max of k iid std normals]. Exact for k=1,2; MC otherwise."""
    if k == 1:
        return 0.0
    if k == 2:
        return 1.0 / math.sqrt(math.pi)
    rng = rng or np.random.default_rng(SEED)
    return float(rng.standard_normal((n_mc // k, k)).max(axis=1).mean())


def evpi_two_action_tie(sigma_d: float = 1.0) -> float:
    """True EVPI at the two-action Gaussian tie: sigma_d/sqrt(2*pi)."""
    return sigma_d / math.sqrt(2.0 * math.pi)


def plugin_mean_noisy_tie(sigma_d: float, tau: float, k_ens: int) -> float:
    """Closed form (verified 2 Jul): E[T1-T2] = sqrt(s^2+2 t^2)(1-K^-1/2)/sqrt(2 pi)."""
    return math.sqrt(sigma_d**2 + 2.0 * tau**2) * (1.0 - k_ens**-0.5) / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------
# E1 — channel A: order-statistic pedestal, and average-before-max
# ----------------------------------------------------------------------

def e1_pedestal(rng, n_rep=20_000):
    """k candidates, all true values 0; per-candidate estimate = mean of m
    episodes with noise sd sigma. Reported 'best value' = max of estimates.
    Bias = sigma/sqrt(m) * E[max of k std normals]."""
    out = {}
    sigma = 1.0
    for k in (2, 4, 8, 16, 32, 64):
        row = {}
        for m in (1, 4, 16):
            est = rng.standard_normal((n_rep, k, m)).mean(axis=2) * sigma
            mx = est.max(axis=1)
            pred = sigma / math.sqrt(m) * exp_max_std_normal(k, rng=rng)
            row[f"m{m}"] = {
                "bias_mean": float(mx.mean()),
                "bias_se": float(mx.std(ddof=1) / math.sqrt(n_rep)),
                "closed_form": pred,
            }
        out[f"k{k}"] = row
    return out


# ----------------------------------------------------------------------
# E2 — channel B: selected-sample reuse vs independent re-evaluation
# ----------------------------------------------------------------------

def e2_reuse(rng, n_rep=20_000, m=4, sigma=1.0):
    """Select argmax on m episodes/candidate; report (a) same episodes,
    (b) fresh episodes, (c) split-select (choose on half, score on half).
    All true values 0, so every reported mean is pure bias."""
    out = {}
    for k in (4, 8, 16):
        ep = rng.standard_normal((n_rep, k, m)) * sigma
        sel_all = ep.mean(axis=2).argmax(axis=1)                      # (a)
        reuse = ep.mean(axis=2)[np.arange(n_rep), sel_all]
        fresh = rng.standard_normal((n_rep, m)).mean(axis=1) * sigma  # (b) new draws for winner
        half = m // 2
        sel_h = ep[:, :, :half].mean(axis=2).argmax(axis=1)           # (c)
        split = ep[:, :, half:].mean(axis=2)[np.arange(n_rep), sel_h]
        se = lambda a: float(a.std(ddof=1) / math.sqrt(n_rep))
        out[f"k{k}"] = {
            "reuse_bias": float(reuse.mean()), "reuse_se": se(reuse),
            "fresh_bias": float(fresh.mean()), "fresh_se": se(fresh),
            "split_bias": float(split.mean()), "split_se": se(split),
        }
    return out


# ----------------------------------------------------------------------
# E3 — channel C: shared-stochasticity leakage
# ----------------------------------------------------------------------

def e3_leakage(rng, n_rep=2_000, n_cand=64):
    """Probe shares a randomness component with the outcome it predicts.
    True ranking signal exactly zero. Outcome G = eps_shared + eps_out;
    leaked probe = lam*eps_shared + eps_probe; clean probe independent.
    Report Spearman rho vs shared-variance fraction."""
    from numpy import argsort

    def spearman(a, b):
        ra = argsort(argsort(a, axis=1), axis=1).astype(float)
        rb = argsort(argsort(b, axis=1), axis=1).astype(float)
        ra -= ra.mean(axis=1, keepdims=True)
        rb -= rb.mean(axis=1, keepdims=True)
        num = (ra * rb).sum(axis=1)
        den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
        return num / den

    out = {}
    for frac in (0.0, 0.25, 0.5, 0.75):
        s = math.sqrt(frac)
        o = math.sqrt(1.0 - frac)
        eps_shared = rng.standard_normal((n_rep, n_cand))
        g = s * eps_shared + o * rng.standard_normal((n_rep, n_cand))
        leaked = s * eps_shared + o * rng.standard_normal((n_rep, n_cand))
        clean = rng.standard_normal((n_rep, n_cand))
        out[f"shared{frac}"] = {
            "rho_leaked": float(spearman(leaked, g).mean()),
            "rho_clean": float(spearman(clean, g).mean()),
            "theory_pearson": frac,  # corr = shared variance fraction here
        }
    return out


# ----------------------------------------------------------------------
# E4 — composed channels calibrated to the archived Paper-2 magnitudes
# ----------------------------------------------------------------------

def e4_composed(rng, n_rep=50_000):
    """Archived shape: M=8 candidates, ONE rollout each (m=1), opportunity
    = max - mean, non-negative by construction, headline +1.301.
    Calibrate sigma so that the pure channel-A pedestal matches ~1.25
    (sigma = 1.247 / E[max of 8 std normals]); then show:
      - pedestal alone reproduces the headline scale from a TRUE ZERO;
      - doubling evaluation noise scales it accordingly (archived: x2.21);
      - cross-fit / fresh re-evaluation collapses it to ~0 (archived -0.105).
    Wording discipline: 'consistent with', never 'was exactly this'."""
    k = 8
    emax8 = exp_max_std_normal(k, rng=rng)
    mean_of_max_minus_mean = emax8  # for iid std normals, E[max - mean] = E[max]
    sigma = 1.247 / mean_of_max_minus_mean
    g = rng.standard_normal((n_rep, k)) * sigma          # true values all 0
    opp = g.max(axis=1) - g.mean(axis=1)
    g2 = rng.standard_normal((n_rep, k)) * (2.0 * sigma)
    opp2 = g2.max(axis=1) - g2.mean(axis=1)
    # cross-fit: select on pass 1, evaluate winner on independent pass 2
    sel = g.argmax(axis=1)
    g_fresh = rng.standard_normal((n_rep, k)) * sigma
    xfit = g_fresh[np.arange(n_rep), sel] - g_fresh.mean(axis=1)
    se = lambda a: float(a.std(ddof=1) / math.sqrt(n_rep))
    return {
        "sigma_calibrated": sigma,
        "e_max8_std_normal": emax8,
        "pedestal_opportunity": {"mean": float(opp.mean()), "se": se(opp)},
        "double_noise_opportunity": {"mean": float(opp2.mean()), "se": se(opp2),
                                     "ratio_vs_base": float(opp2.mean() / opp.mean())},
        "crossfit_opportunity": {"mean": float(xfit.mean()), "se": se(xfit)},
        "archived_reference": {"headline": 1.301, "identity_maxminusmean": 1.247,
                               "noisier_variant": 2.881, "crossfit": -0.105},
    }


# ----------------------------------------------------------------------
# E5 — corrected estimators drive every channel to zero
# ----------------------------------------------------------------------

def e5_corrections(rng, n_rep=20_000, k=8, sigma=1.0):
    """Under true zero: (1) average-before-max trend in m; (2) split-select;
    (3) independent re-eval; (4) analytic pedestal subtraction using an
    ESTIMATED sigma (the practical debias when only one pass exists)."""
    out = {}
    for m in (1, 4, 16, 64):
        est = rng.standard_normal((n_rep, k, m)).mean(axis=2) * sigma
        out[f"avgmax_m{m}"] = float(est.max(axis=1).mean())
    ep = rng.standard_normal((n_rep, k, 4)) * sigma
    sel = ep[:, :, :2].mean(axis=2).argmax(axis=1)
    out["split_select"] = float(ep[:, :, 2:].mean(axis=2)[np.arange(n_rep), sel].mean())
    est1 = rng.standard_normal((n_rep, k)) * sigma
    sel1 = est1.argmax(axis=1)
    fresh = rng.standard_normal((n_rep, k)) * sigma
    out["independent_reeval"] = float(fresh[np.arange(n_rep), sel1].mean())
    # analytic debias: subtract sigma_hat * E[max_k std normal], sigma_hat
    # from the cross-candidate spread of the same single pass
    sig_hat = est1.std(axis=1, ddof=1)
    emax = exp_max_std_normal(k, rng=rng)
    debiased = est1.max(axis=1) - sig_hat * emax
    out["analytic_debias"] = float(debiased.mean())
    out["analytic_debias_se"] = float(debiased.std(ddof=1) / math.sqrt(n_rep))
    return out


# ----------------------------------------------------------------------
# E6 — diagnostic battery: size and power
# ----------------------------------------------------------------------

def e6_battery(rng, n_study=400, n_perm=200):
    """Three diagnostics as executable checks, each scored over replicate
    'studies'. Sizes under the clean condition, power under the broken one.

    (a) duplicate-candidate null: within duplicate groups true signal is 0;
        test rank-corr(probe, outcome) > 0 by permutation. Clean probe ->
        size; leaked probe (50% shared variance) -> power.
    (b) permutation null: reference the max-mean statistic to its
        candidate-permutation distribution. With true effect 0 the
        pedestal-referenced excess should cover 0 (size); with a planted
        true best (+1.0) the excess should detect it (power).
    (c) cross-fit delta: in-sample minus cross-fit opportunity, tested
        against 0 via its replicate distribution over episode halves.
        Independent-eval loop -> size; reuse loop -> power."""
    out = {}

    # (a) duplicate-candidate null ------------------------------------
    n_group, group_size = 16, 4
    hits_clean = hits_leak = 0
    for _ in range(n_study):
        shared = rng.standard_normal((n_group, group_size))
        g = shared + rng.standard_normal((n_group, group_size))
        leaked = shared + rng.standard_normal((n_group, group_size))
        clean = rng.standard_normal((n_group, group_size))
        def perm_p(probe):
            gc = g - g.mean(axis=1, keepdims=True)
            pc = probe - probe.mean(axis=1, keepdims=True)
            stat = float((gc * pc).sum())
            null = np.empty(n_perm)
            for i in range(n_perm):
                idx = rng.permuted(np.tile(np.arange(group_size), (n_group, 1)), axis=1)
                null[i] = float((gc * np.take_along_axis(pc, idx, axis=1)).sum())
            return float((null >= stat).mean())
        hits_clean += perm_p(clean) < 0.05
        hits_leak += perm_p(leaked) < 0.05
    out["duplicate_null"] = {"size_clean": hits_clean / n_study,
                             "power_leaked": hits_leak / n_study,
                             "n_study": n_study}

    # (b) permutation-null referencing --------------------------------
    k, sigma = 8, 0.876
    cover_null = detect_true = 0
    for _ in range(n_study):
        for planted, counter in ((0.0, "null"), (1.0, "true")):
            mu = np.zeros(k); mu[0] = planted
            g = mu + rng.standard_normal(k) * sigma
            obs = g.max() - g.mean()
            null = np.empty(n_perm)
            for i in range(n_perm):
                gp = rng.permutation(g - mu) + mu.mean()  # exchangeable no-effect draw
                null[i] = gp.max() - gp.mean()
            excess_p = float((null >= obs).mean())
            if counter == "null":
                cover_null += excess_p >= 0.05
            else:
                detect_true += excess_p < 0.05
    out["perm_null_reference"] = {"size_coverage_null": cover_null / n_study,
                                  "power_planted_1sigma_best": detect_true / n_study,
                                  "n_study": n_study}

    # (c) cross-fit delta ---------------------------------------------
    # Statistic: (in-sample opportunity) - (cross-fit opportunity), averaged
    # over the two disjoint episode half-splits. Threshold calibrated per
    # study by parametric bootstrap of the NO-REUSE null at the study's own
    # sigma-hat (95th percentile), so size is ~nominal by construction.
    m, n_null = 8, 200

    def xfit_delta(ep):
        halves = []
        for h in range(2):
            cols = np.arange(h, ep.shape[1], 2)
            other = np.arange(1 - h, ep.shape[1], 2)
            means = ep[:, cols].mean(axis=1)
            sel = means.argmax()
            in_samp = means.max() - means.mean()
            x_fit = ep[sel, other].mean() - ep[:, other].mean()
            halves.append(in_samp - x_fit)
        return float(np.mean(halves))

    hits_reuse = hits_indep = 0
    for _ in range(n_study):
        ep = rng.standard_normal((k, m)) * sigma
        sig_hat = float(np.sqrt(((ep - ep.mean(axis=1, keepdims=True)) ** 2).sum()
                                / (k * (m - 1))))
        null = np.empty(n_null)
        for i in range(n_null):
            # no-reuse null: a loop whose reported statistic is already
            # cross-fit, simulated at the study's own sigma-hat
            null[i] = _indep_loop_delta(rng.standard_normal((k, m)) * sig_hat)
        thresh = float(np.quantile(null, 0.95))
        hits_reuse += xfit_delta(ep) > thresh                 # reuse loop: reports in-sample
        ep2 = rng.standard_normal((k, m)) * sigma
        hits_indep += _indep_loop_delta(ep2) > thresh         # honest loop: reports cross-fit
    out["crossfit_delta"] = {"power_reuse_loop": hits_reuse / n_study,
                             "size_independent_loop": hits_indep / n_study,
                             "n_study": n_study, "null_calibration": "parametric bootstrap, 95th pct"}
    return out


def _indep_loop_delta(ep):
    """Delta statistic for a loop that already reports cross-fit numbers:
    recomputing the cross-fit answer on the other split changes ~nothing,
    so the diagnostic's delta is pure noise around 0."""
    halves = []
    for h in range(2):
        cols = np.arange(h, ep.shape[1], 2)
        other = np.arange(1 - h, ep.shape[1], 2)
        sel = ep[:, cols].mean(axis=1).argmax()
        reported = ep[sel, other].mean() - ep[:, other].mean()      # cross-fit
        sel_b = ep[:, other].mean(axis=1).argmax()
        recheck = ep[sel_b, cols].mean() - ep[:, cols].mean()       # cross-fit, swapped
        halves.append(reported - recheck)
    return float(np.mean(halves))


# ----------------------------------------------------------------------
# selfcheck
# ----------------------------------------------------------------------

def selfcheck():
    rng = np.random.default_rng(SEED)
    failures = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.5f} want {want:.5f} tol {tol}")
        if not ok:
            failures.append(name)

    # closed forms
    check("E[max2 std normal] exact", exp_max_std_normal(2), 1 / math.sqrt(math.pi), 1e-9)
    check("E[max8 std normal] MC vs literature", exp_max_std_normal(8, rng=rng), 1.4236, 0.01)
    check("EVPI tie sigma=1", evpi_two_action_tie(1.0), 0.39894, 1e-4)
    check("plugin noisy-tie tau=1.066 K=5 (verified 2 Jul)",
          plugin_mean_noisy_tie(1.0, 1.066, 5), 0.3990, 5e-4)
    check("plugin exact-eval K=5 (verified 2 Jul)",
          plugin_mean_noisy_tie(1.0, 0.0, 5), 0.22053, 5e-4)

    # zero-bias legs on small MC
    r2 = e2_reuse(np.random.default_rng(7), n_rep=40_000)["k8"]
    check("fresh re-eval bias ~ 0", r2["fresh_bias"], 0.0, 4 * r2["fresh_se"])
    check("split-select bias ~ 0", r2["split_bias"], 0.0, 4 * r2["split_se"])
    if not r2["reuse_bias"] > 0.5:
        failures.append("reuse bias should be large-positive")
        print("  [FAIL] reuse bias large-positive")
    else:
        print(f"  [PASS] reuse bias large-positive ({r2['reuse_bias']:.3f})")

    # pedestal matches closed form
    r1 = e1_pedestal(np.random.default_rng(11), n_rep=20_000)["k8"]["m1"]
    check("pedestal k8 m1 vs closed form", r1["bias_mean"], r1["closed_form"], 5 * r1["bias_se"])

    # planted-mutant leg: a leaked probe MUST light up the duplicate null,
    # a clean probe must not (rates checked loosely at small n_study)
    b = e6_battery(np.random.default_rng(13), n_study=60, n_perm=120)
    if b["duplicate_null"]["power_leaked"] < 0.8:
        failures.append("duplicate-null power")
        print("  [FAIL] duplicate-null power on leaked probe")
    else:
        print(f"  [PASS] duplicate-null power ({b['duplicate_null']['power_leaked']:.2f})")
    if b["duplicate_null"]["size_clean"] > 0.15:
        failures.append("duplicate-null size")
        print("  [FAIL] duplicate-null size on clean probe")
    else:
        print(f"  [PASS] duplicate-null size ({b['duplicate_null']['size_clean']:.2f})")

    # determinism: same seed twice -> identical result dict
    a = e4_composed(np.random.default_rng(SEED), n_rep=5_000)
    bb = e4_composed(np.random.default_rng(SEED), n_rep=5_000)
    if json.dumps(a, sort_keys=True) != json.dumps(bb, sort_keys=True):
        failures.append("determinism")
        print("  [FAIL] determinism")
    else:
        print("  [PASS] determinism (same seed, identical output)")

    print(f"SELFCHECK {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="route1_sims_results.json")
    args = ap.parse_args()
    if args.selfcheck:
        sys.exit(selfcheck())
    rng = np.random.default_rng(SEED)
    res = {
        "meta": {"seed": SEED, "date": "2026-08-15",
                 "status": "non-registered demonstrative simulation suite",
                 "script": "route1_sims.py"},
        "E1_pedestal": e1_pedestal(rng),
        "E2_reuse": e2_reuse(rng),
        "E3_leakage": e3_leakage(rng),
        "E4_composed_calibrated": e4_composed(rng),
        "E5_corrections": e5_corrections(rng),
        "E6_battery": e6_battery(rng),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
