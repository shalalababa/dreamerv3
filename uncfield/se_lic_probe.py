"""Deployed responsiveness-license probe (the B4-checkpoint deployed read).

BUILD ONLY — not yet executed on registered substrate, not yet in the repo.
Destination at commit time: `uncfield/se_lic_probe.py` (run as
`python -m uncfield.se_lic_probe`).

Revision 2 (28 Aug 2026), after adversarial review #1 (NOT FREEZE-READY,
blocking F1+F2).  Every finding applied; the review file is
`deployed_recon/REVIEW_1.md`.

===========================================================================
WHAT THIS MEASURES
===========================================================================
The NFI repair's core license (research_notes/paper5_uncertainty_field/
repair_ideation_20260825/CONSOLIDATED.md; temp_files/nfi_repair_20260825/
lensA/CANDIDATES.md candidate 1) prices a model's claimed uncertainty
reduction by the mean-responsiveness the model's OWN imagined observations
induce (law of total variance; oracle-free — the C-row cancels).

DreamerV3's deployed claim is not a contraction.  It is

    Var_e[ mu_e(s, a) ]                       (dreamerv3/explore.py:102-116)

the disagreement ensemble's variance about the next postfeat
`z' = concat(deter, softmax(logit))` (agent.py:730-739; 512 + 32*4 = 640
for the SE runs).  The deployed transliteration of the license is a ROUND
TRIP through the model's own observation channel:

    mu_e            member e's claimed next latent      (disag.predict)
    y'_e  = symexp( dec(mu_e).pred() )                  the observation it implies
    q_e   = postfeat( posterior(carry_t, a_t, y'_e) )   the model's own filter
    license_k(h=1) = Var_e[ dec_k(q_e) ]                per obs key
    claim_k        = Var_e[ dec_k(mu_e) ]               the theta_1 numerator

`claim_k` is EXACTLY the existing theta_1 machinery (se_probe.py:208-217
member_vars / se_alea_mask.py:162-172 _proj_vars): the license changes only
WHICH latents are fed to the decoder projection.

===========================================================================
THE SUBSTRATE FACT — REGISTERED DESCRIPTIVE, REPORT IT PER ARM
===========================================================================
`rssm.py:80`: `deter = self._core(deter, stoch, action)` runs BEFORE tokens
are read (rssm.py:81-82).  So `deter_{t+1}` is IDENTICAL for every imagined
observation, and at h=1 `Var_e[q_deter] == 0` EXACTLY (asserted in the
selfcheck, emitted as `lic1_deter_var_max`).

Decomposing the claim itself against the same split (review #1 F1):

    claim_deter_k = Var_e[ dec_k(deter_e, mean_e probs) ]
    claim_stoch_k = Var_e[ dec_k(mean_e deter, probs_e) ]
    deter_frac_k  = claim_deter_k / claim_k

measures 0.88-1.03 on the smoke.  **~90% of the deployed obs-space claim is
Var_e[mu_deter]: member disagreement about a DETERMINISTIC function of
inputs every member already holds.**  That is pure ensemble approximation
error, not epistemic uncertainty about the world, and the h=1 license
correctly prices it at zero.  This number is a registered descriptive of
the read in its own right — report `deter_frac` per arm whatever else
fires, because it may be the most important quantity the read produces.

CONSEQUENCES FOR THE STATISTIC (F1, blocking):
  * The raw ratio `L_full_k(1) = lic_k(1)/claim_k` is NOT a responsiveness
    measure at h=1.  It is dominated by the decoder's per-key deter-vs-stoch
    readout split, and its 3.7x cross-key spread on the smoke tracks
    `stoch_frac` spread (3.4x) rather than anything about observations.
  * THE h=1 PRIMARY IS THEREFORE `L_stochnorm_k = lic_k(1)/claim_stoch_k`,
    which compares like with like on the licensable subspace.
    `L_full` is retained as DESCRIPTIVE only.
  * theta_1 is likewise reported at three levels: `theta1_raw`
    (the published B4 statistic, deter-dominated), `theta1_stoch` (the
    stoch-carried claim), and `theta1_lic(h)`.  The h=1 primary comparison
    is `theta1_lic(1)` vs `theta1_stoch` — both stoch-carried.
  * h >= 2 is unaffected: `deter_{t+2} = _core(deter_{t+1}, stoch_{t+1},
    a_{t+1})` and `stoch_{t+1}` DOES carry the observation, so the deter
    block becomes licensable and `L_full` is the right statistic there.
  * On a simplex a UNIFORM deflation is a share no-op (the 22-Aug M12
    lesson), so `deflation_guard` (spread, dshare) is reported per leg and
    the decision floor is measured in-run, never transported (F4).

===========================================================================
LEGS
===========================================================================
PASS A — the PRIMARY anchor set.  Windows of length `burn_in`, drawn with
    the IDENTICAL first rng call as se_probe/se_alea_mask, so the anchors
    are the se_probe-comparable population and the provenance leg is a real
    comparison (F6b).  Carries: the claim + its deter/stoch decomposition,
    L1 (h=1), the transplant null, and L2.
PASS B — the horizon set.  Windows of length `burn_in + h_max - 1` (the
    tail exists only to supply the recorded action sequence).  Carries L3
    alone.  Disclosed as a different anchor set; every L3 statistic is a
    within-pass ratio.

L1  round-trip license at h=1.  Primary = `L_stochnorm`.
L2  obs-space aleatoric account (gauss arm only).  B4 divided by the
    aleatoric variance PER LATENT DIM and got 1.06x (artifacts/
    b4_alea_read_20260823/RESULTS.md).  L2 does the SAME accounting one
    layer later — Monte-Carlo pushforward of the members' own
    `exp(lv).mean(0)` through the frozen decoder, per obs channel.  It holds
    the instrument fixed and varies only the SPACE.
L3  horizon-persistence license, h in {1,2,4,8,15}.  The p2e objective sums
    disag.reward over an imag_length=15 imagined rollout (agent.py:365-387),
    so the claim being priced is "going here reduces my uncertainty about
    the FUTURE".  Mean-field rollout under the REPLAY-RECORDED actions.
    The persistence statistic is
        persistence_k(h) = L_full_k(h) / L_full_k(1)
    i.e. the decay of a key's OWN license with horizon (F7 iv).
L4  cross-seed min-license — post-hoc, zero extra compute, `--collate`,
    reported at MATCHED n with `n_runs` beside every certificate row (F7 v).

NULL (registered gate, F2).  `pk_null_*`: member deviations transplanted
across anchors — `mu_null[e,b] = mu_bar[b] + (mu_e[perm(b)] - mu_bar[perm(b)])`
with `perm` a checked derangement.  If `L_k` were a state-independent
decode->encode FIDELITY MAP, the null would reproduce the license.  So:

    null_delta_k = |L_null_k - L_k| / L_k   must be >= tau on every
    fire-bearing key, tau = the in-run measured L-floor (F4),
    AND the null's claim scale must match (0.5 <= match_k <= 2.0, F3),

**before any FIRES/FLAT inference is licensed.**  `classify_outcome`
refuses with NULL-GATE-FAILED otherwise.

===========================================================================
REGISTERED EXPECTATIONS  (write the read BEFORE the numbers exist)
===========================================================================
Reference rows to be repriced (artifacts/b4_alea_read_20260823/):
  raw distractor share 0.401 / 0.432 / 0.433 / 0.432   (s120..s123)
  theta_1              4.21  / 5.74  / 5.47  / 5.50    (Stage-1 det band
  4.72-5.64; programme headline 5.26x).  B4's per-dim aleatoric whitening
  moved the share to 0.427/0.456/0.461/0.454 — ratio ~1.06, a NULL.
Decision floor: MEASURED IN-RUN as `dup_floor_L` per leg — `planted_dup0`
is BITWISE identical to `position` in the replay, so a license that prices
them differently is broken, and the size of that difference IS the floor.
The ~0.15% raw-SHARE floor does NOT transport to the L statistic (measured
14.8% / 9.7% / 4.1% at h=1/2/4 on the smoke — 15-28x looser).

  L1 FIRES  — requires ALL THREE: (a) the F1 control, i.e. the effect is on
    `L_stochnorm` and `theta1_lic(1) < theta1_stoch` beyond `dup_floor_L`;
    (b) the F2 null gate PASSES (the effect is state-specific, not a
    fidelity map); (c) `theta1_lic(1)` below the registered bar.  Reading:
    the ensemble's distractor claim, on the subspace an observation can
    actually reach, is NOT LEGIBLE TO THE MODEL'S OWN FILTER.  A drop-in
    oracle-free repair of disag.reward exists.

  L1 ANTI-FIRES — `theta1_lic(1) > theta1_stoch` beyond the floor: the
    license makes the distractor's share LARGER.  **This is what the smoke
    fixture does** (theta1_raw 1.52 -> theta1_lic(1) 4.88), so it is
    pre-registered here rather than discovered.  Reading: the distractor's
    claimed variance is the MOST observation-legible part of the claim —
    it is exactly the channel whose value the next observation reveals —
    and the responsiveness license, applied one step, RATIFIES the misprice
    instead of removing it.  That is a substantive finding about the
    difference between NFI's pathology (claimed contraction no observation
    caused) and Dreamer's (claimed epistemic variance about a channel that
    is resolvable but useless), and it is the single most likely outcome.

  L1 FLAT — `L_stochnorm_k` ~ const across keys, shares unchanged within
    `dup_floor_L`, `deflation_guard.uniform_like` True.  LICENSED WORDING
    (F7 i): "the deployed one-step round-trip license, at THIS ensemble
    unit, does not reprice."  It is *consistent with*, NOT a test of, the
    programme's unified law: this probe is ONE point of the
    self-consistency family (members share enc/dyn/dec, there is no
    ensemble-min across independently trained models in the h=1 leg, and
    none of XEDL's ledger/impossibility/channel-cap layers are present).
    Do NOT write "rules out the self-consistency family at scale".

  L3 FIRES — `persistence_distractor(h)` tracks the AR(1) prediction while
    the real keys do not.  The distractor is a unit-stationary AR(1) with
    theta=0.1 => coefficient 0.9 (embodied/envs/distractor.py:60-63; run
    config `distractor: {theta: 0.1, dim: 8, scale: 1.0, basesd: 1.215}`),
    so responsiveness decays as 0.81^(h-1) and at h=15 is
    **0.81^14 = 5.2%** of h=1 (NOT 0.81^15 — the license at h=1 is the
    re-encoded posterior itself, so the decay runs over h-1 steps).
    Emitted as `predicted_ar1` beside the measured persistence.

  L3 FLAT — the model's own h-step predictions inherit distractor
    responsiveness the true AR(0.9) does not have => the WM has LEARNED A
    WRONG WORLD about the distractor's persistence.  That is the deployed
    instance of NFI's wrong-world class and, by the EDL-solo impossibility
    boundary (CANDIDATES.md candidate 2), unclosable by ANY single-model
    oracle-free rule.  L4's refusal certificate becomes the deliverable.

  L4 — registered prediction: the min does NOT remove the distractor share,
    because every seed shares the defect (raw distractor share 0.40-0.43 in
    4/4 gauss, 0.43-0.49 in 8/8 det).  If so the deployed read CONFIRMS THE
    IMPOSSIBILITY BOUNDARY AT SCALE — a shared-representation ensemble is
    the maximally correlated defect (corrdef, CONSOLIDATED.md follow-up 5)
    — and the min-license LEVEL is the certifiable refusal condition.
    Report at MATCHED n (the min is monotone in n: an 8-seed min is
    mechanically smaller than a 4-seed min), with `n_runs` on every row.
    The corrdef safe/leaking constants (0.043-0.064 vs 0.118-0.607) are
    FAMILY-1-SPECIFIC — re-derive here, never transport.

  L2 reprices where B4's per-dim form did not => the B4 failure was a BASIS
    choice and "representationally entangled, irreparable" softens to
    "reparable in obs space".  L2 flat => the entanglement claim rests on
    two independent bases.

===========================================================================
INSTRUMENT RULES OBSERVED
===========================================================================
* Claim and license are produced by ONE jitted function per pass, so their
  ratio is a within-graph quantity.  B4's Amendment 2 measured +-1-2%
  MIXED-SIGN divergence between two separately-compiled instruments from
  bf16 graph-order noise (embodied/jax/nets.py:12 COMPUTE_DTYPE=bfloat16;
  configs.yaml compute_dtype: bfloat16 — this holds on CPU too, and the B4
  wave ran --platform cpu).  Cross-instrument comparisons use tolerance
  5e-2 PLUS a bootstrapped sampling term (F6), never a bare 1e-4.
* `--platform cpu` is the house default.  The whole family must run on ONE
  device (probe-panels-single-gpu, 16 Aug; rented boxes run five drivers on
  one card model, 22 Aug).  CPU makes that free.
* Deterministic under a fixed seed; asserted end-to-end.  NOTE (F9): the
  anchor posterior's SAMPLED `stoch` IS read — `carry0`/`anchor` take
  `post["stoch"]`, so one Monte-Carlo draw conditions every statistic.  The
  seed is pinned and the soft-vs-hard decode calibration (`cal_*`, emitted
  at the anchor AND at every recorded horizon per F8) is the control.
* Degenerate denominators emit NaN + `degenerate_keys`, never a silent
  zero: a zero claim must not read as a maximal license (F5).
* No mask legs: batch-permuting a pure exogenous AR(1) is law-preserving,
  so its population delta is zero by exchangeability for ANY statistic
  (the 22-Aug M12 cascade).

Run:  python -m uncfield.se_lic_probe --run_logdir <dir> [--platform cpu]
      python -m uncfield.se_lic_probe --selfcheck
      python -m uncfield.se_lic_probe --collate <json> ... --collate_out <p>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np

from uncfield.se_probe import (EXCLUDE, FIRE_KEYS, PLANTED_KEYS, REPO,
                               np_symlog, resolve_ckpt)

SMOKE = (REPO / "local_results" / "uncfield_se_smoke_20260812_155552"
         / "se_smoke0")

# se_alea_mask.py:328-331 (review #21 B3): planted_const is the
# decoder-projection floor DIAGNOSTIC and never enters a fire-bearing
# simplex; its rows are still saved.
FLOOR_DIAGNOSTIC_KEY = "planted_const"

# The distractor's AR(1) coefficient, from the ENV not the model:
# embodied/envs/distractor.py:60-63, `theta: 0.1` => 1 - theta.
AR1_COEF = 0.9


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="",
                   help="default: <run>/se_lic_probe")
    p.add_argument("--ckpt", default="", help="default: <run>/ckpt/<latest>")
    p.add_argument("--n_eval", type=int, default=512,
                   help="S; pinned population (house constant)")
    p.add_argument("--burn_in", type=int, default=16,
                   help="anchor context length; anchor = index burn_in-1")
    p.add_argument("--horizons", default="1,2,4,8,15")
    p.add_argument("--n_alea", type=int, default=16,
                   help="L2 Monte-Carlo pushforward draws")
    p.add_argument("--n_boot", type=int, default=2000,
                   help="anchor bootstrap for the provenance sampling term")
    p.add_argument("--min_frac", type=float, default=0.9,
                   help="refuse if realized S < min_frac * n_eval (F13)")
    p.add_argument("--fire_theta1_bar", type=float, default=2.36,
                   help="registered theta_1 bar (B4 amendment)")
    p.add_argument("--ep_batch", type=int, default=64)
    p.add_argument("--platform", default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match the se_probe/se_alea_mask seed: pass A "
                        "reproduces their pinned replay slice exactly")
    p.add_argument("--policy_actions", action="store_true",
                   help="DESCRIPTIVE VARIANT (labelled, never primary): roll "
                        "L3 under policy-sampled actions.  The anchor action "
                        "a_t stays recorded in both modes.")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--collate", nargs="*", default=None)
    p.add_argument("--collate_out", default="")
    p.add_argument("--expect_n", type=int, default=None,
                   help="L4: refuse a partial collate (review F12)")
    p.add_argument("--n_match", type=int, default=4,
                   help="L4: subset size for the matched-n min (review F7 v)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# pure statistics (no jax; every deciding gate is killed in _fixture_checks)
# --------------------------------------------------------------------------

NAN = float("nan")


def safe_ratio(num, den):
    """Degenerate denominators give NaN, never a silent 0.0 — a zero claim
    must not read as a maximal (or minimal) license (review F5)."""
    return float(num / den) if den and den > 0 and np.isfinite(den) else NAN


def shares_from(perkey, universe):
    tot = float(np.sum([perkey[k] for k in universe]))
    if not (tot > 0) or not np.isfinite(tot):
        return {k: NAN for k in universe}
    return {k: float(perkey[k] / tot) for k in universe}


def degenerate_keys(perkey, universe):
    return [k for k in universe
            if not (perkey.get(k, 0.0) > 0) or not np.isfinite(perkey.get(k, 0.0))]


def theta1(shares, source_key, target_key="distractor"):
    """The pinned misprice statistic: the target channel's share of
    decoder-projected ensemble variance over the SOURCE key's own share
    (se_probe.py:392; se_b4_read forms the ratio).  source_key =
    config.planted.source_key = 'position' for the SE runs."""
    return safe_ratio(shares.get(target_key, NAN), shares.get(source_key, NAN))


def deflation_guard(ratios, share_claim, share_lic, universe, floor):
    """THE M12 GUARD, made a check.

    A license that deflates every key by the SAME factor is invisible on a
    simplex — shares are unchanged and the instrument reports a no-op
    indistinguishable from "no effect".

      spread = max_k L_k / min_k L_k      (1.0 <=> perfectly uniform)
      dshare = max_k |share_lic(k) - share_claim(k)|

    `uniform_like` is judged against the IN-RUN measured floor, never a
    transported constant (review F4)."""
    vals = np.asarray([ratios.get(k, NAN) for k in universe], np.float64)
    good = vals[np.isfinite(vals) & (vals > 0)]
    spread = float(good.max() / good.min()) if good.size else NAN
    ds = [abs(share_lic[k] - share_claim[k]) for k in universe
          if np.isfinite(share_lic.get(k, NAN))
          and np.isfinite(share_claim.get(k, NAN))]
    dshare = float(max(ds)) if ds else NAN
    f = floor if (floor and np.isfinite(floor)) else 0.005
    return dict(spread=spread, dshare=dshare, floor_used=float(f),
                uniform_like=bool(np.isfinite(spread) and np.isfinite(dshare)
                                  and spread < 1.0 + f and dshare < f))


def dup_floor(perkey, source_key="position", dup_key="planted_dup0"):
    """THE VALIDITY GATE AND THE DECISION FLOOR, free and in-band.

    `planted_dup0` is BITWISE identical to the source key in the replay
    (hard-asserted in `measure`), so the two channels differ only through
    independent decoder heads.  Whatever divergence the statistic shows on
    them is its own noise floor.  Applied to SHARES it reproduces the
    0.15% se_probe number; applied to the L RATIOS it is the L-statistic's
    floor, which is 15-28x looser and is what the decision must use (F4)."""
    a, b = perkey.get(dup_key), perkey.get(source_key)
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) \
            or not b > 0:
        return NAN
    return float(abs(a - b) / b)


def null_gate(ratios, null_ratios, tau, gate_keys, scale_ok):
    """THE FIDELITY-MAP GATE (review F2, blocking).

    The transplant null attaches each real member deviation to the WRONG
    anchor.  If `L_k` were a state-independent decode->encode fidelity map
    it would be unchanged, so the license must differ from the null by MORE
    than the statistic's own floor on every fire-bearing key before any
    FIRES/FLAT reading is licensed."""
    per = {}
    for k in gate_keys:
        L, Ln = ratios.get(k, NAN), null_ratios.get(k, NAN)
        per[k] = safe_ratio(abs(Ln - L), L) if np.isfinite(L) \
            and np.isfinite(Ln) else NAN
    ok = bool(scale_ok and np.isfinite(tau)
              and all(np.isfinite(v) and v >= tau for v in per.values()))
    return dict(tau=float(tau) if np.isfinite(tau) else NAN,
                per_key_delta=per, gate_keys=list(gate_keys),
                scale_ok=bool(scale_ok), passed=ok)


def classify_outcome(th1_lic, th1_ref, floor, guard, gate, degen, bar):
    """The F7 outcome table, machine-checkable.  Returns one of
    DEGENERATE / NULL-GATE-FAILED / FIRES / ANTI-FIRES / FLAT / PARTIAL."""
    if degen:
        return "DEGENERATE"
    if not gate.get("passed"):
        return "NULL-GATE-FAILED"
    r = safe_ratio(th1_lic, th1_ref)
    if not np.isfinite(r) or not np.isfinite(floor):
        return "DEGENERATE"
    if r < 1.0 - floor and np.isfinite(th1_lic) and th1_lic < bar:
        return "FIRES"
    if r > 1.0 + floor:
        return "ANTI-FIRES"
    if guard.get("uniform_like") or abs(r - 1.0) <= floor:
        return "FLAT"
    return "PARTIAL"


def horizon_weights(horizons, gamma):
    """Midpoint-rule widths on the sampled horizon grid, discounted.
    NOTE: the SE runs have `agent.horizon = 333` => gamma = 0.997, so over
    imag_length=15 the discount is essentially FLAT.  Weights are recorded
    in the json so this is never implicit."""
    hs, w = list(horizons), {}
    for i, h in enumerate(hs):
        lo = hs[i - 1] if i else hs[0]
        hi = hs[i + 1] if i + 1 < len(hs) else hs[-1]
        w[h] = float(max(0.5 * ((h - lo) + (hi - h)), 1.0) * (gamma ** h))
    return w


def min_license(per_run, universe, horizons, n_match=None):
    """L4: cross-SEED min-license (CANDIDATES.md candidate 3, deployed).

    The 8 disag members share enc/dyn/dec (explore.py:50-59), so they are
    NOT an ensemble in the NFI sense; the deployed ensemble unit is the
    independently trained SEED.  Latent bases are not aligned across seeds,
    but L_k is a per-OBS-KEY quantity and is therefore basis-free.

    MATCHED n (review F7 v): the min is monotone in the number of runs, so
    an 8-seed min is mechanically smaller than a 4-seed min and the two arms
    are not comparable at their natural n.  With `n_match` set, the min is
    averaged over all C(n, n_match) subsets and `n_runs` is stamped on
    every certificate row."""
    out = {}
    n = len(per_run)
    k_match = min(n_match, n) if n_match else n
    subsets = list(itertools.combinations(range(n), k_match))
    if len(subsets) > 70:                       # C(8,4)=70; keep it bounded
        subsets = subsets[:70]
    for h in horizons:
        mat = {k: np.asarray([r["ratios"][str(h)].get(k, NAN)
                              for r in per_run], np.float64)
               for k in universe}
        B = {k: float(np.nanmean([np.nanmin(mat[k][list(s)])
                                  for s in subsets])) for k in universe}
        B_full = {k: float(np.nanmin(mat[k])) for k in universe}
        srt = {k: np.sort(mat[k][np.isfinite(mat[k])]) for k in universe}
        gap = {k: float(srt[k][1] - srt[k][0]) if srt[k].size > 1 else 0.0
               for k in universe}
        rows = []
        for r in per_run:
            relic = {k: B[k] * r["claim"][k] for k in universe}
            sh = shares_from(relic, universe)
            rows.append(dict(run=r["run"], n_runs=n, n_match=k_match,
                             shares=sh, theta1=theta1(sh, r["source_key"]),
                             theta1_raw=r["theta1_raw"]))
        out[str(h)] = dict(
            n_runs=n, n_match=k_match, n_subsets=len(subsets),
            B_matched=B, B_all_runs=B_full,
            argmin={k: per_run[int(np.nanargmin(mat[k]))]["run"]
                    for k in universe},
            min_vs_second_gap=gap, per_run=rows,
            refusal_certificate=dict(
                n_runs=n, n_match=k_match,
                distractor_min_license=B.get("distractor"),
                distractor_min_vs_second_gap=gap.get("distractor"),
                note="corrdef: the min-license LEVEL, not the spread, splits "
                     "safe from leaking; bands must be re-derived on this "
                     "family, never transported.  Compare arms only at "
                     "equal n_match."))
    return out


def _clean(o):
    """JSON sanitiser: bare NaN/Infinity are not valid JSON (review F15)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return _clean(float(o))
    return o


# --------------------------------------------------------------------------
# main measurement
# --------------------------------------------------------------------------

def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax
    import embodied.jax.nets as nn
    from dreamerv3.main import make_agent
    from probing.collect import load_run_config, load_frozen_agent
    from probing import probeset as probeset_mod

    f32 = jnp.float32
    out_dir = args.output or os.path.join(args.run_logdir, "se_lic_probe")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir, False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None, "the license probe requires a disag ensemble"
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    source_key = str(config.planted["source_key"])
    universe = [k for k in obs_keys if k != FLOOR_DIAGNOSTIC_KEY]
    real_keys = [k for k in obs_keys if k not in PLANTED_KEYS]
    gate_keys = sorted(set(FIRE_KEYS).intersection(universe) | {source_key})
    dec_symlog = bool(model.dec.symlog)
    ens = int(config.agent.expl["disag_ens"])
    head = str(config.agent.expl["disag_head"])
    do_l2 = (head == "gauss")
    ddim, nstoch, ncls = (int(model.dyn.deter), int(model.dyn.stoch),
                          int(model.dyn.classes))
    gamma = 1.0 - 1.0 / float(config.agent.horizon)
    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    assert horizons and horizons[0] == 1 and horizons == sorted(set(horizons))
    L = args.burn_in

    # ------------------------------------------------------------- helpers
    def postsplit(pred):
        """(..., 640) -> (deter-hat, soft probs).  se_probe.py:190-199."""
        det = pred[..., :ddim]
        probs = pred[..., ddim:].reshape((*pred.shape[:-1], nstoch, ncls))
        probs = jnp.clip(probs, 1e-6, 1.0)
        return det, probs / probs.sum(-1, keepdims=True)

    def decode(det, probs, B, T):
        """se_probe.py:201-206.  Vec heads are symlog_mse, so `.pred()` is
        the mean IN SYMLOG SPACE (heads.py:127-130)."""
        dec_carry = model.dec.initial(B)
        _, _, recons = model.dec(dec_carry, dict(deter=det, stoch=probs),
                                 jnp.zeros((B, T), bool), training=False)
        return {k: f32(recons[k].pred()) for k in obs_keys}

    def proj(det3, pr4):
        """(E,B,512),(E,B,32,4) -> decoder-projected per-key member variance
        — the theta_1 numerator (se_probe.py:208-217).  E members decoded in
        ONE batched call so claim and license share the decoder graph."""
        E, B = det3.shape[0], det3.shape[1]
        dec = decode(det3.reshape((E * B, ddim))[:, None],
                     pr4.reshape((E * B, nstoch, ncls))[:, None], E * B, 1)
        return {k: dec[k][:, 0].reshape((E, B, -1)).var(0) for k in obs_keys}

    def bcast(x, E):
        return jnp.broadcast_to(x[None], (E, *x.shape))

    def encode_ctx(obs, actctx, reset_):
        B = reset_.shape[0]
        enc_carry, dyn_carry = model.enc.initial(B), model.dyn.initial(B)
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset_, training=False)
        prevact = {k: jnp.concatenate(
            [jnp.zeros_like(v[:, :1]), v[:, :-1]], 1) for k, v in actctx.items()}
        _, _, post = model.dyn.observe(dyn_carry, tokens, prevact, reset_,
                                       training=False)
        return post

    def reencode(y_raw, act1, carry):
        """ONE posterior step on an imagined observation.  The decoder emits
        symlog-space means and the encoder applies `nn.symlog` itself
        (rssm.py:218-219), so the round trip is closed by `nn.symexp`
        (nets.py:63), exact in f32."""
        B = jax.tree.leaves(act1)[0].shape[0]
        zero = jnp.zeros((B,), bool)
        _, _, tok = model.enc(model.enc.initial(B), y_raw, zero,
                              training=False, single=True)
        _, _, feat = model.dyn.observe(carry, tok, act1, zero,
                                       training=False, single=True)
        return (f32(feat["deter"]), jax.nn.softmax(f32(feat["logit"]), -1),
                f32(feat["logit"]))

    def sample_out(xs):
        return jax.tree.map(lambda x: x.sample(nj.seed()), xs)

    def hard_probs(logit):
        """One-hot sample matching `_observe` exactly (rssm.py:87) via the
        PUBLIC OneHot output (rssm.py:174), for the F8 calibration."""
        return f32(embodied.jax.outs.OneHot(
            nn.cast(logit), model.dyn.unimix).sample(nj.seed()))

    # ------------------------------------------------------------- one pass
    def measure(hs, tag):
        h_max = hs[-1]
        # windows: the tail exists only to supply recorded actions, so a
        # single-horizon pass needs NO tail and its rng call is byte-identical
        # to se_probe/se_alea_mask's (review F6b).
        W = L + h_max - 1
        rng = np.random.default_rng(args.seed)
        arrays, _, stats = probeset_mod.collect_windows(
            os.path.join(args.run_logdir, "replay"), W, args.n_eval, rng,
            allow_fewer=True)
        arrays = {k: np.stack(v, 0) for k, v in arrays.items()}
        S = arrays["is_first"].shape[0]
        assert S >= args.min_frac * args.n_eval, (            # review F13
            f"[{tag}] realized S={S} < {args.min_frac}*{args.n_eval} — "
            f"refusing rather than reporting an under-powered population")
        reset = np.zeros((S, L), bool)
        reset[:, 0] = True
        T0 = L - 1
        dup0_equal = bool(np.array_equal(arrays.get("planted_dup0"),
                                         arrays[source_key]))
        assert dup0_equal, "replayed planted_dup0 != source — construction violated"
        print(f"[{tag}] windows: {S} x {W} (ctx {L}, h_max {h_max}) "
              f"from {stats['streams']} streams")

        # normalizers over the CONTEXT window only (review F6b), otherwise
        # the two passes normalize over different amounts of data.
        raw_norms = {}
        for k in obs_keys:
            v = np.asarray(arrays[k][:, :L], np.float32).reshape(S * L, -1)
            raw_norms[k] = (np_symlog(v) if dec_symlog else v).std(0).astype(
                np.float64)
        real_scale = float(np.mean(np.concatenate(
            [raw_norms[k] for k in real_keys])))
        nfloor = 0.05 * real_scale
        norms = {k: np.maximum(raw_norms[k], nfloor) for k in obs_keys}

        def rollout(det_flat, pr_flat, acts, E, B):
            """Mean-field horizon rollout.  The sampled `stoch` returned by
            `dyn.imagine` (rssm.py:100) is DISCARDED and `softmax(logit)`
            propagated — the same soft-probs convention the postfeat target
            (agent.py:733-735) and `postsplit` already use.  This removes a
            sampling-noise floor that would otherwise be read as
            information; the cost is that the rollout is off-distribution
            relative to the one-hot-trained `_core`, which is exactly what
            the per-horizon calibration (review F8) measures.

            `acts[j]` is the action at window index T0+j = the action taken
            AT state T0+j (dv3 post->prev: `prevact[t]=action[t-1]` drives
            the transition into state t; se_probe.py:266-275)."""
            rec = {}
            d, p, lg = det_flat, pr_flat, None
            for h in range(1, h_max + 1):
                if h in hs:
                    rec[h] = (d, p, lg)
                if h == h_max:
                    break
                carry = nn.cast(dict(deter=d, stoch=p))
                if args.policy_actions:
                    a = sample_out(model.pol(model.feat2tensor(
                        dict(deter=d, stoch=p)), 1))
                    a = {k: v.reshape((E * B, -1)) for k, v in a.items()}
                else:
                    a = {k: jnp.broadcast_to(
                        acts[k][:, h][None], (E, B, acts[k].shape[-1])
                    ).reshape((E * B, -1)) for k in act_keys}
                _, (feat, _) = model.dyn.imagine(carry, a, 1, training=False,
                                                 single=True)
                d, lg = f32(feat["deter"]), f32(feat["logit"])
                p = jax.nn.softmax(lg, -1)
            return rec

        def lic_fn(obs, action_dict, reset_, perm):
            B = reset_.shape[0]
            actctx = {k: v[:, :L] for k, v in action_dict.items()}
            post = encode_ctx(obs, actctx, reset_)
            anchor = {"deter": post["deter"][:, -1:],
                      "stoch": post["stoch"][:, -1:]}
            carry0 = {"deter": post["deter"][:, -1],
                      "stoch": post["stoch"][:, -1]}
            feat = model.feat2tensor(anchor)
            actvec = jnp.concatenate(
                [action_dict[k][:, T0:T0 + 1].reshape((B, 1, -1))
                 for k in act_keys], -1)
            act1 = {k: action_dict[k][:, T0] for k in act_keys}
            mu = f32(model.disag.predict(feat, actvec))       # (E,B,1,640)
            out = {}

            det3, pr4 = postsplit(mu[:, :, 0])                # (E,B,..)
            # ---- CLAIM and its deter/stoch decomposition (review F1)
            vclaim = proj(det3, pr4)
            vstoch = proj(bcast(det3.mean(0), ens), pr4)
            vdeter = proj(det3, bcast(pr4.mean(0), ens))
            for k in obs_keys:
                out[f"pk_claim_{k}"] = vclaim[k]
                out[f"pk_claimstoch_{k}"] = vstoch[k]
                out[f"pk_claimdeter_{k}"] = vdeter[k]

            def imagined(d3, p4):
                dec = decode(d3.reshape((ens * B, ddim))[:, None],
                             p4.reshape((ens * B, nstoch, ncls))[:, None],
                             ens * B, 1)
                return ({k: nn.symexp(dec[k][:, 0].reshape((ens, B, -1)))
                         for k in obs_keys}, dec)

            def rt(yr):
                ycat = {k: yr[k].reshape((ens * B, -1)) for k in obs_keys}
                crr = {kk: bcast(vv, ens).reshape((ens * B, *vv.shape[1:]))
                       for kk, vv in carry0.items()}
                aa = {k: bcast(act1[k], ens).reshape((ens * B, -1))
                      for k in act_keys}
                return reencode(ycat, aa, crr)

            y_raw, dec_mu = imagined(det3, pr4)
            q_det, q_pr, q_lg = rt(y_raw)                     # state T0+1
            # structural check, emitted: Var_e[q_deter] must be EXACTLY 0
            out["lic1_deter_var_max"] = jnp.max(
                q_det.reshape((ens, B, ddim)).var(0))[None]

            acts = {k: action_dict[k][:, T0:] for k in act_keys}
            rec = rollout(q_det, q_pr, acts, ens, B)
            for h, (d, p, lg) in rec.items():
                E = ens
                v = proj(d.reshape((E, B, ddim)), p.reshape((E, B, nstoch, ncls)))
                for k in obs_keys:
                    out[f"pk_lic{h}_{k}"] = v[k]
                # F8: soft-vs-hard calibration AT EVERY recorded horizon
                lgh = q_lg if lg is None else lg
                dsoft = decode(d[:, None], p[:, None], E * B, 1)
                dhard = decode(d[:, None], hard_probs(lgh)[:, None], E * B, 1)
                for k in obs_keys:
                    out[f"calh{h}_{k}"] = jnp.abs(
                        dsoft[k] - dhard[k])[:, 0].reshape((E, B, -1)).mean(0)

            # ---- NULL: deviations transplanted across anchors (review F2)
            mu_bar = mu.mean(0)
            mu_null = mu_bar[None] + (mu - mu_bar[None])[:, perm]
            nd3, npr4 = postsplit(mu_null[:, :, 0])
            for k, v in proj(nd3, npr4).items():
                out[f"pk_claimnull_{k}"] = v
            for k, v in proj(bcast(nd3.mean(0), ens), npr4).items():
                out[f"pk_claimnullstoch_{k}"] = v
            y_null, _ = imagined(nd3, npr4)
            n_det, n_pr, _ = rt(y_null)
            for k, v in proj(n_det.reshape((ens, B, ddim)),
                             n_pr.reshape((ens, B, nstoch, ncls))).items():
                out[f"pk_null_{k}"] = v

            # ---- L2: obs-space aleatoric account (gauss only)
            if do_l2:
                lv = f32(model.disag.predict_logvar(feat, actvec))
                alea = jnp.exp(lv).mean(0)
                eps = jax.random.normal(nj.seed(),
                                        (args.n_alea, *mu_bar.shape), f32)
                ad3, apr4 = postsplit((mu_bar[None] + eps * jnp.sqrt(alea)[None]
                                       )[:, :, 0])
                for k, v in proj(ad3, apr4).items():
                    out[f"pk_alea_{k}"] = v
                out["alea_mean"] = alea.mean(-1)[:, 0]
                out["lv_frac_lo"] = (lv <= float(model.disag.logvar_min) + 1e-6
                                     ).astype(f32).mean((0, -1))[:, 0]
                out["lv_frac_hi"] = (lv >= float(model.disag.logvar_max) - 1e-6
                                     ).astype(f32).mean((0, -1))[:, 0]

            # ---- diagnostics
            soft = jax.nn.softmax(f32(post["logit"][:, -1:]), -1)
            d_soft = decode(post["deter"][:, -1:], soft, B, 1)
            d_hard = decode(post["deter"][:, -1:], post["stoch"][:, -1:], B, 1)
            dec_q = decode(q_det[:, None], q_pr[:, None], ens * B, 1)
            for k in obs_keys:
                out[f"cal_{k}"] = jnp.abs(d_soft[k] - d_hard[k])[:, 0]
                out[f"rt_{k}"] = jnp.abs(
                    dec_mu[k][:, 0].reshape((ens, B, -1))
                    - dec_q[k][:, 0].reshape((ens, B, -1))).mean(0)
            out["intrinsic"] = f32(model.disag.reward(feat, actvec))[:, 0]
            return out

        jit_lic = jax.jit(
            lambda p, o, a, r, pm, s: nj.pure(lic_fn)(p, o, a, r, pm, seed=s))
        outs = []
        for i in range(0, S, args.ep_batch):
            sl = slice(i, min(i + args.ep_batch, S))
            b = sl.stop - sl.start
            o = {k: jnp.asarray(arrays[k][sl][:, :L])
                 for k in obs_keys + list(EXCLUDE) if k in arrays}
            a = {k: jnp.asarray(arrays[k][sl]) for k in act_keys}
            pm = np.roll(np.arange(b), max(1, b // 3))
            assert b == 1 or not np.array_equal(pm, np.arange(b)), (
                "transplant permutation is the identity (review F15)")
            _, res = jit_lic(params, o, a, jnp.asarray(reset[sl]),
                             jnp.asarray(pm), args.seed + i)
            outs.append({k: np.asarray(v, np.float64) for k, v in res.items()})
        raw = {k: np.concatenate([o[k] for o in outs], 0) for k in outs[0]}

        def perkey(pre):
            return {k: (raw[f"{pre}{k}"] / (norms[k] ** 2)[None]).mean(-1)
                    for k in obs_keys}
        return dict(tag=tag, S=S, W=W, hs=hs, raw=raw, norms=norms,
                    raw_norms=raw_norms, nfloor=nfloor, streams=stats["streams"],
                    rows={p: perkey(p) for p in
                          ["pk_claim_", "pk_claimstoch_", "pk_claimdeter_",
                           "pk_null_", "pk_claimnull_", "pk_claimnullstoch_"]
                          + [f"pk_lic{h}_" for h in hs]
                          + (["pk_alea_"] if do_l2 else [])})

    # ---- PASS A (primary; identical rng call to se_probe/se_alea_mask)
    A = measure([1], "A/primary")
    # ---- PASS B (horizon sweep)
    B_ = measure(horizons, "B/horizon") if horizons != [1] else A

    def scalars(pas, pre):
        return {k: float(pas["rows"][pre][k].mean()) for k in obs_keys}

    claim = scalars(A, "pk_claim_")
    claim_stoch = scalars(A, "pk_claimstoch_")
    claim_deter = scalars(A, "pk_claimdeter_")
    # review F5: a zero claim on a universe key is a construction failure,
    # not a maximal license — refuse rather than report through it.
    bad = degenerate_keys(claim, universe)
    assert not bad, f"zero/non-finite claim on universe keys {bad}"
    deter_frac = {k: safe_ratio(claim_deter[k], claim[k]) for k in obs_keys}
    stoch_frac = {k: safe_ratio(claim_stoch[k], claim[k]) for k in obs_keys}
    share_claim = shares_from(claim, universe)
    share_stoch = shares_from(claim_stoch, universe)
    th1_raw = theta1(share_claim, source_key)
    th1_stoch = theta1(share_stoch, source_key)

    def leg(pas, h, ref_claim, label):
        lic = scalars(pas, f"pk_lic{h}_")
        sh = shares_from(lic, universe)
        r_full = {k: safe_ratio(lic[k], claim[k]) for k in obs_keys}
        r_sn = {k: safe_ratio(lic[k], ref_claim[k]) for k in obs_keys}
        prim = r_sn if h == 1 else r_full
        fl = dup_floor(prim, source_key)
        return dict(
            h=h, license=lic, shares=sh,
            primary_ratio=label,
            ratios=prim, ratios_full=r_full, ratios_stochnorm=r_sn,
            theta1=theta1(sh, source_key),
            theta1_vs_raw=safe_ratio(theta1(sh, source_key), th1_raw),
            theta1_vs_stoch=safe_ratio(theta1(sh, source_key), th1_stoch),
            dup_floor_L=fl, dup_floor_share=dup_floor(sh, source_key),
            deflation_guard=deflation_guard(prim, share_claim, sh, universe, fl),
            degenerate_keys=degenerate_keys(lic, universe),
            license_over_claim_quantiles={          # review F10, re-derivable
                q: float(np.nanpercentile(
                    [r_full[k] for k in universe], q)) for q in (50, 90, 99)},
            calibration={k: float(pas["raw"][f"calh{h}_{k}"].mean()
                                  / pas["norms"][k].mean()) for k in obs_keys})

    legs = {"1": leg(A, 1, claim_stoch, "L_stochnorm")}
    for h in horizons:
        if h != 1:
            legs[str(h)] = leg(B_, h, claim_stoch, "L_full")

    # ---- the F2 null gate, on the h=1 primary (stoch-normalized both sides)
    nul = scalars(A, "pk_null_")
    nul_claim = scalars(A, "pk_claimnull_")
    nul_stoch = scalars(A, "pk_claimnullstoch_")
    L_null = {k: safe_ratio(nul[k], nul_stoch[k]) for k in obs_keys}
    match = {k: safe_ratio(nul_claim[k], claim[k]) for k in obs_keys}
    scale_ok = all(np.isfinite(match[k]) and 0.5 <= match[k] <= 2.0
                   for k in universe)                          # review F3
    gate = null_gate(legs["1"]["ratios"], L_null,
                     legs["1"]["dup_floor_L"], gate_keys, scale_ok)
    null_block = dict(
        shares=shares_from(nul, universe), L_null=L_null,
        claim_scale_match=match, null_scale_ok=bool(scale_ok),
        gate=gate,
        note="member deviations transplanted across anchors; if L_k were a "
             "state-independent decode->encode fidelity map the null would "
             "REPRODUCE the license, so the gate requires a divergence "
             "above the in-run L-floor on every fire-bearing key")

    outcome = classify_outcome(
        legs["1"]["theta1"], th1_stoch, legs["1"]["dup_floor_L"],
        legs["1"]["deflation_guard"], gate, bool(bad), args.fire_theta1_bar)

    # ---- L3 persistence, on the right object (review F7 iv)
    base = legs["1"]["ratios_full"]
    persistence = {str(h): {k: safe_ratio(legs[str(h)]["ratios_full"][k],
                                          base[k]) for k in universe}
                   for h in horizons}
    predicted_ar1 = {str(h): AR1_COEF ** (2 * (h - 1)) for h in horizons}

    w = horizon_weights(horizons, gamma)
    licsum = {k: float(np.sum([w[h] * legs[str(h)]["license"][k]
                               for h in horizons])) for k in obs_keys}

    l2_block = None
    if do_l2:
        al = scalars(A, "pk_alea_")
        sub = {k: max(claim[k] - al[k], 0.0) for k in obs_keys}
        rat = {k: safe_ratio(claim[k], al[k]) for k in obs_keys}
        zeroed = [k for k in universe if sub[k] <= 0.0]
        l2_block = dict(
            alea_projected=al, subtractive_zeroed_keys=zeroed,
            subtractive_degenerate=bool(
                sum(sub[k] for k in universe) <= 0.0 or sub[source_key] <= 0.0),
            shares_subtractive=shares_from(sub, universe),
            shares_ratio=shares_from(rat, universe),
            theta1_subtractive=theta1(shares_from(sub, universe), source_key),
            theta1_ratio=theta1(shares_from(rat, universe), source_key),
            alea_mean=float(A["raw"]["alea_mean"].mean()),
            lv_frac_lo=float(A["raw"]["lv_frac_lo"].mean()),
            lv_frac_hi=float(A["raw"]["lv_frac_hi"].mean()),
            note="B4 divided by exp(lv) per LATENT dim (ratio 1.06, a null); "
                 "this is the same account per OBS CHANNEL")

    result = dict(
        instrument="se_lic_probe", revision=2,
        run_logdir=os.path.abspath(args.run_logdir),
        ckpt=os.path.abspath(ckpt), seed=args.seed, burn_in=L,
        n_eval_requested=args.n_eval, n_eval=A["S"], n_eval_horizon=B_["S"],
        window_primary=A["W"], window_horizon=B_["W"],
        horizons=horizons, horizon_weights=w, gamma=gamma, ens=ens,
        disag_head=head, dec_symlog=dec_symlog, task=str(config.task),
        train_seed=int(config.seed), source_key=source_key,
        share_universe=universe, gate_keys=gate_keys,
        fire_theta1_bar=args.fire_theta1_bar,
        action_source=("policy_sampled_DESCRIPTIVE" if args.policy_actions
                       else "replay_recorded_PRIMARY"),
        estimand="within-anchor ratio of decoder-projected ensemble variance "
                 "AFTER the model's own decode->re-encode round trip to the "
                 "STOCH-CARRIED claim before it (h=1) / to the full claim "
                 "(h>=2); per obs key; oracle-free",
        # ---- THE SUBSTRATE FACT, registered descriptive
        substrate=dict(
            deter_frac=deter_frac, stoch_frac=stoch_frac,
            deter_frac_universe_mean=float(np.nanmean(
                [deter_frac[k] for k in universe])),
            claim=claim, claim_deter=claim_deter, claim_stoch=claim_stoch,
            lic1_deter_var_max=float(A["raw"]["lic1_deter_var_max"].max()),
            note="deter_frac is the fraction of the deployed obs-space claim "
                 "carried by Var_e[mu_deter] — member disagreement about a "
                 "DETERMINISTIC function of inputs every member already "
                 "holds (rssm.py:80).  That component is architecturally "
                 "unlicensable at h=1 and lic1_deter_var_max is its "
                 "structural witness (must be 0).  REPORT PER ARM."),
        share_claim=share_claim, share_stoch=share_stoch,
        theta1_raw=th1_raw, theta1_stoch=th1_stoch,
        primary=dict(
            statistic="theta1_lic(h=1) vs theta1_stoch, on L_stochnorm",
            theta1_lic=legs["1"]["theta1"],
            theta1_ref=th1_stoch,
            ratio=safe_ratio(legs["1"]["theta1"], th1_stoch),
            floor=legs["1"]["dup_floor_L"],
            outcome=outcome),
        legs=legs,
        horizon_summed=dict(license=licsum, shares=shares_from(licsum, universe),
                            theta1=theta1(shares_from(licsum, universe),
                                          source_key)),
        persistence=dict(measured=persistence, predicted_ar1=predicted_ar1,
                         note=f"AR(1) coefficient {AR1_COEF} from the ENV "
                              f"(distractor.py:60-63); variance "
                              f"responsiveness decays as coef^(2(h-1)); at "
                              f"h=15 that is {AR1_COEF ** 28:.4f}"),
        transplant_null=null_block,
        l2_obs_space_aleatoric=l2_block,
        gates=dict(
            dup_floor_L_h1=legs["1"]["dup_floor_L"],
            dup_floor_share_claim=dup_floor(share_claim, source_key),
            floor_diagnostic_share=safe_ratio(
                claim.get(FLOOR_DIAGNOSTIC_KEY, 0.0), sum(claim.values())),
            share_floor_bound_keys=[k for k in obs_keys if bool(
                (A["raw_norms"][k] < A["nfloor"]).any())],
            norm_floor=float(A["nfloor"]),
            calibration_anchor={k: float(A["raw"][f"cal_{k}"].mean()
                                         / A["norms"][k].mean())
                                for k in obs_keys},
            roundtrip_err={k: float(A["raw"][f"rt_{k}"].mean()
                                    / A["norms"][k].mean()) for k in obs_keys},
            base_intrinsic_mean=float(A["raw"]["intrinsic"].mean()),
            min_frac=args.min_frac,
            cross_instrument_tolerance=5e-2),
    )
    result["provenance"] = _provenance(
        args, ckpt, universe, source_key, share_claim, th1_raw,
        A["rows"]["pk_claim_"], A["S"])

    with open(os.path.join(out_dir, "se_lic_probe.json"), "w") as f:
        json.dump(_clean(result), f, indent=1)
    npz = {}
    for pre, d in A["rows"].items():
        for k in obs_keys:
            npz[f"A_{pre}{k}"] = d[k]
    for h in horizons:
        for k in obs_keys:
            npz[f"B_pk_lic{h}_{k}"] = B_["rows"][f"pk_lic{h}_"][k]
    np.savez(os.path.join(out_dir, "se_lic_probe.npz"),
             intrinsic=A["raw"]["intrinsic"],
             **{f"sharenorms_{k}": A["norms"][k] for k in obs_keys}, **npz)

    print(json.dumps(_clean({
        "OUTCOME": outcome,
        "deter_frac(universe mean)": round(
            result["substrate"]["deter_frac_universe_mean"], 4),
        "lic1_deter_var_max": result["substrate"]["lic1_deter_var_max"],
        "theta1": {"raw": th1_raw, "stoch": th1_stoch,
                   "lic_h1": legs["1"]["theta1"],
                   "ratio_vs_stoch": result["primary"]["ratio"]},
        "L_stochnorm_h1": {k: round(legs["1"]["ratios"][k], 5)
                           for k in universe},
        "dup_floor_L_by_h": {h: legs[h]["dup_floor_L"] for h in legs},
        "null_gate": {"passed": gate["passed"], "tau": gate["tau"],
                      "scale_ok": scale_ok,
                      "per_key_delta": gate["per_key_delta"]},
        "persistence_distractor": {h: persistence[h]["distractor"]
                                   for h in persistence},
        "predicted_ar1": predicted_ar1,
    }), indent=1))
    return result


def _provenance(args, ckpt, universe, source_key, share_claim, th1_raw,
                pk_claim, S):
    """Cross-check the raw claim shares against the run's own se_probe /
    se_alea_mask outputs.

    Pass A draws windows of length `burn_in` with the identical first rng
    call, so this is a genuine same-population comparison (review F6b).
    The tolerance is 5e-2 PLUS a bootstrapped sampling term (review F6):
    anchor-set sampling alone consumes ~76% of a bare 5e-2 budget, so a
    fixed threshold would fire on noise about 2.4% of the time per cell.

    CKPT-IDENTITY + PAIRING GUARD (se_mask.py:269-279 / the 22-Aug B1 s53
    lesson): a reference json from a different checkpoint, seed, or
    population is not a reference — skip it with a status."""
    mine = os.path.basename(os.path.normpath(ckpt))
    ref, skipped = {}, {}
    for name in ("se_probe", "se_alea_mask"):
        p = os.path.join(args.run_logdir, name, f"{name}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            j = json.load(f)
        theirs = os.path.basename(os.path.normpath(j.get("ckpt", "")))
        why = None
        if theirs != mine:
            why = f"ckpt {theirs} != {mine}"
        elif int(j.get("seed", -1)) != args.seed:
            why = f"seed {j.get('seed')} != {args.seed}"
        elif int(j.get("n_eval", -1)) != S:
            why = f"n_eval {j.get('n_eval')} != {S}"
        if why:
            skipped[name] = why
            continue
        ref[name] = (j["key_share"] if name == "se_probe"
                     else {k: v["raw"] for k, v in j["shares"].items()})
    if not ref:
        return dict(status="no ckpt/seed/n_eval-matched reference",
                    skipped=skipped)

    rng = np.random.default_rng(args.seed + 991)
    idx = np.arange(S)
    boots = {k: np.empty(args.n_boot) for k in universe}
    for i in range(args.n_boot):
        s = rng.integers(0, S, S)
        den = pk_claim[source_key][s].mean()
        for k in universe:
            boots[k][i] = pk_claim[k][s].mean() / den if den > 0 else NAN
    se = {k: float(np.nanstd(boots[k]) / max(
        abs(np.nanmean(boots[k])), 1e-12)) for k in universe}

    out = dict(status="compared", base_tolerance=5e-2, n_boot=args.n_boot,
               ckpt=mine, skipped=skipped, rel_se=se, refs={})
    for name, sh in ref.items():
        rows = {}
        for k in universe:
            if k not in sh or not sh.get(source_key):
                continue
            here = share_claim[k] / share_claim[source_key]
            there = sh[k] / sh[source_key]
            dev = abs(here - there) / max(abs(there), 1e-12)
            tol = 5e-2 + 2.0 * se[k]
            rows[k] = dict(dev=float(dev), tol=float(tol),
                           pass_=bool(dev <= tol))
        out["refs"][name] = dict(
            per_key=rows,
            max_dev=float(max((r["dev"] for r in rows.values()), default=0.0)),
            theta1_ref=safe_ratio(sh.get("distractor", NAN),
                                  sh.get(source_key, NAN)),
            theta1_here=th1_raw,
            pass_=bool(rows and all(r["pass_"] for r in rows.values())))
    return out


# --------------------------------------------------------------------------
# L4 collate
# --------------------------------------------------------------------------

def collate(paths, out_path, n_match=None, expect_n=None):
    assert paths, "collate needs at least one json (review F15)"
    per_run, universe, horizons = [], None, None
    seen = set()
    for p in paths:
        with open(p) as f:
            j = json.load(f)
        assert j.get("instrument") == "se_lic_probe", p
        assert j.get("revision", 0) >= 2, f"pre-revision-2 output: {p}"
        rd = j["run_logdir"]
        assert rd not in seen, f"duplicate run_logdir in collate: {rd}"
        seen.add(rd)
        if universe is None:
            universe, horizons = j["share_universe"], j["horizons"]
        assert j["share_universe"] == universe, ("universe mismatch", p)
        assert j["horizons"] == horizons, ("horizon grid mismatch", p)
        per_run.append(dict(run=rd, claim=j["claim"] if "claim" in j
                            else j["substrate"]["claim"],
                            theta1_raw=j["theta1_raw"],
                            source_key=j["source_key"],
                            disag_head=j["disag_head"],
                            ratios={h: j["legs"][h]["ratios"] for h in j["legs"]}))
    arms = sorted({r["disag_head"] for r in per_run})
    assert len(arms) == 1, f"refusing a mixed-arm collate: {arms}"   # F12
    if expect_n is not None:
        assert len(per_run) == expect_n, (
            f"expected {expect_n} runs for arm {arms[0]}, got {len(per_run)} "
            f"— a partial collate is n-biased (review F12)")
    res = dict(instrument="se_lic_collate", n_runs=len(per_run),
               expected_n=expect_n, n_match=n_match, arm=arms[0],
               share_universe=universe, horizons=horizons,
               min_license=min_license(per_run, universe, horizons, n_match),
               note="L4 cross-SEED min-license; registered prediction is that "
                    "the min does NOT remove the distractor share (every seed "
                    "shares the defect) => the deployed read confirms the "
                    "impossibility boundary at scale and the min-license "
                    "LEVEL is the refusal certificate.  Compare arms only at "
                    "equal n_match.")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_clean(res), f, indent=1)
    return res


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _fixture_checks():
    uni = ["distractor", "position", "velocity", "planted_dup0"]
    claim = {"distractor": 0.40, "position": 0.10, "velocity": 0.30,
             "planted_dup0": 0.10}
    sc = shares_from(claim, uni)
    assert abs(theta1(sc, "position") - 4.0) < 1e-12

    # (1) UNIFORM DEFLATION MUST BE DETECTED AS A NO-OP  (M12)
    lic_u = {k: 0.03 * v for k, v in claim.items()}
    su, ru = shares_from(lic_u, uni), {k: 0.03 for k in uni}
    gu = deflation_guard(ru, sc, su, uni, 0.05)
    assert gu["uniform_like"] and gu["spread"] < 1.001 and gu["dshare"] < 1e-9
    assert abs(theta1(su, "position") - theta1(sc, "position")) < 1e-9, \
        "a uniform deflation must leave theta_1 exactly unchanged"

    # (2) DIFFERENTIAL DEFLATION MUST NOT BE CALLED UNIFORM
    lic_d = dict(lic_u); lic_d["distractor"] *= 0.05
    sd = shares_from(lic_d, uni)
    rd = {k: lic_d[k] / claim[k] for k in uni}
    gd = deflation_guard(rd, sc, sd, uni, 0.05)
    assert not gd["uniform_like"] and gd["spread"] > 15 and gd["dshare"] > 0.2
    assert theta1(sd, "position") < 0.5 * theta1(sc, "position")

    # (3) DUP-FLOOR both directions, on shares AND on L ratios
    assert dup_floor({"planted_dup0": 0.095127, "position": 0.094986}) < 0.005
    assert dup_floor({"planted_dup0": 0.20, "position": 0.10}) > 0.5
    assert dup_floor({"planted_dup0": NAN, "position": 0.1}) != \
        dup_floor({"planted_dup0": NAN, "position": 0.1})       # nan, not 0

    # (4) F5: degenerate denominators are NaN, never a silent zero
    assert not np.isfinite(safe_ratio(1.0, 0.0))
    assert not np.isfinite(theta1({"distractor": 0.4, "position": 0.0},
                                  "position"))
    assert degenerate_keys({"a": 0.0, "b": 1.0}, ["a", "b"]) == ["a"]
    assert all(not np.isfinite(v) for v in
               shares_from({k: 0.0 for k in uni}, uni).values())

    # (5) F2 NULL GATE: a pure per-key FIDELITY MAP must FAIL the gate
    L = {"distractor": 0.30, "position": 0.10, "velocity": 0.20,
         "planted_dup0": 0.10}
    fidelity = {k: v * 1.002 for k, v in L.items()}     # null reproduces L
    g_bad = null_gate(L, fidelity, 0.05, uni, True)
    assert not g_bad["passed"], g_bad
    statedep = {k: v * (0.4 if k == "distractor" else 1.5) for k, v in L.items()}
    g_ok = null_gate(L, statedep, 0.05, uni, True)
    assert g_ok["passed"], g_ok
    assert not null_gate(L, statedep, 0.05, uni, False)["passed"], \
        "a scale-mismatched null must not pass the gate (F3)"

    # (6) F7 outcome table, machine-checked
    guard_u = dict(uniform_like=True)
    guard_n = dict(uniform_like=False)
    assert classify_outcome(1.0, 4.0, 0.05, guard_n, g_ok, False, 2.36) == "FIRES"
    assert classify_outcome(4.88, 1.52, 0.05, guard_n, g_ok, False, 2.36) \
        == "ANTI-FIRES", "the smoke's own behaviour must be pre-registered"
    assert classify_outcome(4.0, 4.0, 0.05, guard_u, g_ok, False, 2.36) == "FLAT"
    assert classify_outcome(1.0, 4.0, 0.05, guard_n, g_bad, False, 2.36) \
        == "NULL-GATE-FAILED", "no FIRES reading without the null gate"
    assert classify_outcome(1.0, 4.0, 0.05, guard_n, g_ok, True, 2.36) \
        == "DEGENERATE"
    # a FIRES-shaped ratio above the theta_1 bar is only PARTIAL
    assert classify_outcome(3.0, 5.0, 0.05, guard_n, g_ok, False, 2.36) \
        == "PARTIAL"

    # (7) horizon weights + the AR(1) prediction arithmetic (F7 iv)
    w = horizon_weights([1, 2, 4, 8, 15], 1.0 - 1.0 / 333)
    assert abs(w[4] - 3.0 * (1 - 1 / 333) ** 4) < 1e-9
    assert abs(AR1_COEF ** (2 * (15 - 1)) - 0.05233) < 1e-4, \
        "0.81^14 = 5.2%, not 0.81^15"

    # (8) L4: shared defect survives; one sane member closes; matched n
    mk = lambda rr, nm: dict(run=nm, claim=claim, theta1_raw=4.0,
                             source_key="position", disag_head="gauss",
                             ratios={"1": rr})
    shared = [mk({k: 0.9 for k in uni}, f"r{i}") for i in range(3)]
    m = min_license(shared, uni, [1])["1"]
    assert abs(m["per_run"][0]["theta1"] - 4.0) < 1e-9
    idio = shared[:2] + [mk({**{k: 0.9 for k in uni}, "distractor": 0.01},
                            "sane")]
    m2 = min_license(idio, uni, [1])["1"]
    assert m2["B_all_runs"]["distractor"] == 0.01
    assert m2["argmin"]["distractor"] == "sane"
    assert m2["per_run"][0]["theta1"] < 0.2 * 4.0
    # matched-n: with n_match=2 the sane member is absent from some subsets,
    # so the averaged min is strictly LOOSER than the all-runs min
    m3 = min_license(idio, uni, [1], n_match=2)["1"]
    assert m3["n_match"] == 2 and m3["B_matched"]["distractor"] > \
        m3["B_all_runs"]["distractor"], (m3["B_matched"], m3["B_all_runs"])

    # (9) provenance ckpt/seed/n_eval identity guard
    import tempfile
    td = tempfile.mkdtemp(prefix="lic_prov_")
    os.makedirs(os.path.join(td, "se_probe"))
    pj = os.path.join(td, "se_probe", "se_probe.json")
    fake = argparse.Namespace(run_logdir=td, seed=0, n_boot=50)
    rows = {k: np.full(32, claim[k]) for k in uni}
    with open(pj, "w") as f:
        json.dump(dict(ckpt="/x/ckpt/OTHER", seed=0, n_eval=32, key_share=sc), f)
    assert _provenance(fake, "/y/ckpt/MINE", uni, "position", sc, 4.0, rows,
                       32)["status"].startswith("no ckpt")
    with open(pj, "w") as f:
        json.dump(dict(ckpt="/x/ckpt/M", seed=7, n_eval=32, key_share=sc), f)
    assert "seed" in _provenance(fake, "/y/ckpt/M", uni, "position", sc, 4.0,
                                 rows, 32)["skipped"]["se_probe"]
    with open(pj, "w") as f:
        json.dump(dict(ckpt="/x/ckpt/M", seed=0, n_eval=32, key_share=sc), f)
    pok = _provenance(fake, "/y/ckpt/M", uni, "position", sc, 4.0, rows, 32)
    assert pok["status"] == "compared" and pok["refs"]["se_probe"]["pass_"]

    # (10) JSON sanitiser
    assert _clean({"a": NAN, "b": [float("inf"), 1.0]}) == \
        {"a": None, "b": [None, 1.0]}
    json.dumps(_clean({"a": NAN}))          # must not raise / emit bare NaN

    print("  [fixtures] 10 mutant families killed (M12 uniform no-op vs "
          "differential; dup-floor 3 ways; F5 degenerate=NaN; F2 fidelity-map "
          "null FAILS the gate + F3 scale; F7 outcome table incl. ANTI-FIRES "
          "and NULL-GATE-FAILED; 0.81^14; L4 shared/idiosyncratic/matched-n; "
          "provenance ckpt+seed+n_eval; json sanitiser)")


def selfcheck():
    print("se_lic_probe selfcheck (revision 2)")
    _fixture_checks()

    smoke = os.environ.get("LIC_SMOKE_DIR", str(SMOKE))
    assert os.path.isdir(os.path.join(smoke, "ckpt")), (
        f"smoke ckpt not found at {smoke} (set LIC_SMOKE_DIR)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_selfcheck_out")
    argv = ["--run_logdir", smoke, "--n_eval", "24", "--burn_in", "8",
            "--horizons", "1,2,4", "--n_alea", "8", "--ep_batch", "12",
            "--n_boot", "200", "--platform", "cpu", "--output", out]
    args = parse_args(argv)
    r1 = run(args)
    uni = r1["share_universe"]

    # ---- F1: the structural premise, machine-checked
    assert r1["substrate"]["lic1_deter_var_max"] < 1e-12, (
        "Var_e[q_deter] must be EXACTLY 0 at h=1 (rssm.py:80) — if this is "
        "nonzero the whole stoch-normalization argument is wrong",
        r1["substrate"]["lic1_deter_var_max"])
    assert all(np.isfinite(r1["substrate"]["deter_frac"][k]) for k in uni)
    assert r1["legs"]["1"]["primary_ratio"] == "L_stochnorm"
    assert r1["legs"]["2"]["primary_ratio"] == "L_full"
    # stoch-normalized ratios must be strictly larger than the full ones
    # (claim_stoch <= claim), i.e. the F1 correction actually bites
    for k in uni:
        assert r1["legs"]["1"]["ratios_stochnorm"][k] >= \
            r1["legs"]["1"]["ratios_full"][k] - 1e-9, k

    # ---- F6b: pass A is the se_probe-comparable draw
    assert r1["window_primary"] == r1["burn_in"], (
        "the primary pass must draw at burn_in length so its rng call is "
        "identical to se_probe/se_alea_mask")
    assert r1["window_horizon"] == r1["burn_in"] + max(r1["horizons"]) - 1

    # ---- simplex validity everywhere
    for name, sh in [("claim", r1["share_claim"]), ("stoch", r1["share_stoch"])] \
            + [(f"h{h}", r1["legs"][h]["shares"]) for h in r1["legs"]] \
            + [("null", r1["transplant_null"]["shares"])]:
        assert set(sh) == set(uni), (name, sorted(sh))
        assert abs(sum(sh.values()) - 1.0) < 1e-9, (name, sum(sh.values()))
        assert all(0.0 <= v <= 1.0 and np.isfinite(v) for v in sh.values()), name

    # ---- F4: the decision floor is measured, finite, and USED
    for h in r1["legs"]:
        fl = r1["legs"][h]["dup_floor_L"]
        assert np.isfinite(fl) and fl >= 0, (h, fl)
        assert r1["legs"][h]["deflation_guard"]["floor_used"] == fl or fl == 0
    assert r1["primary"]["floor"] == r1["legs"]["1"]["dup_floor_L"]
    # the L-floor must be looser than the raw-share floor (the transported
    # ~1% constant is the thing review F4 rejected)
    assert r1["gates"]["dup_floor_L_h1"] > r1["gates"]["dup_floor_share_claim"]

    # ---- F2/F3: the null gate is present, typed, and consumed
    g = r1["transplant_null"]["gate"]
    assert set(g["gate_keys"]) == set(r1["gate_keys"])
    assert isinstance(g["passed"], bool)
    assert g["tau"] == r1["legs"]["1"]["dup_floor_L"] or not np.isfinite(g["tau"])
    assert "null_scale_ok" in r1["transplant_null"]
    assert all(np.isfinite(v) for v in
               r1["transplant_null"]["claim_scale_match"].values())
    assert r1["primary"]["outcome"] in (
        "FIRES", "ANTI-FIRES", "FLAT", "PARTIAL", "NULL-GATE-FAILED",
        "DEGENERATE")
    if not g["passed"]:
        assert r1["primary"]["outcome"] == "NULL-GATE-FAILED"

    # ---- F8: calibration at every recorded horizon
    for h in r1["legs"]:
        cal = r1["legs"][h]["calibration"]
        assert set(cal) >= set(uni) and all(
            np.isfinite(v) and v >= 0 for v in cal.values()), h

    # ---- F7 iv: persistence on the right object + the AR(1) reference
    for h in r1["horizons"]:
        assert str(h) in r1["persistence"]["measured"]
        assert str(h) in r1["persistence"]["predicted_ar1"]
    assert r1["persistence"]["measured"]["1"]["distractor"] == 1.0
    assert abs(r1["persistence"]["predicted_ar1"]["2"] - 0.81) < 1e-9

    # ---- mechanics: not the identity, not a uniform no-op everywhere
    assert not all(abs(r1["legs"][h]["ratios"][k] - 1.0) < 1e-6
                   for h in r1["legs"] for k in uni), "round trip = identity"
    assert not all(r1["legs"][h]["deflation_guard"]["uniform_like"]
                   for h in r1["legs"]), "uniform at every horizon = share no-op"

    # ---- npz rows for the reader's paired bootstrap
    z = np.load(os.path.join(out, "se_lic_probe.npz"))
    for k in uni:
        for pre in ["A_pk_claim_", "A_pk_claimstoch_", "A_pk_claimdeter_",
                    "A_pk_null_", "A_pk_lic1_", "B_pk_lic2_"]:
            v = np.asarray(z[f"{pre}{k}"])
            assert np.all(np.isfinite(v)) and np.all(v >= 0), (pre, k)

    # ---- L2 iff gauss, and never a silent NaN
    l2 = r1["l2_obs_space_aleatoric"]
    assert (l2 is not None) == (r1["disag_head"] == "gauss")
    if l2 is not None:
        assert "subtractive_degenerate" in l2 and \
            "subtractive_zeroed_keys" in l2
        assert l2["subtractive_degenerate"] or np.isfinite(
            l2["theta1_subtractive"])

    # ---- json is valid strict JSON (no bare NaN/Infinity)
    with open(os.path.join(out, "se_lic_probe.json")) as f:
        json.loads(f.read(), parse_constant=lambda c: (_ for _ in ()).throw(
            AssertionError(f"bare {c} in json (review F15)")))

    # ---- DETERMINISM
    r2 = run(args)
    for h in r1["legs"]:
        for k in uni:
            assert r1["legs"][h]["ratios"][k] == r2["legs"][h]["ratios"][k]
        assert r1["legs"][h]["theta1"] == r2["legs"][h]["theta1"]
    assert r1["primary"] == r2["primary"]
    assert r1["substrate"]["deter_frac_universe_mean"] == \
        r2["substrate"]["deter_frac_universe_mean"]

    # ---- L4 collate: identity at n=1-per-run, and the F12 refusals
    p1 = os.path.join(out, "se_lic_probe.json")
    import shutil
    p2 = os.path.join(out, "copy2.json")
    shutil.copy(p1, p2)
    with open(p2) as f:
        j2 = json.load(f)
    j2["run_logdir"] = j2["run_logdir"] + "_seed2"
    with open(p2, "w") as f:
        json.dump(j2, f)
    c = collate([p1, p2], os.path.join(out, "se_lic_collate.json"), n_match=2)
    assert c["n_runs"] == 2 and c["arm"] == r1["disag_head"]
    m1 = c["min_license"]["1"]
    for k in uni:
        assert abs(m1["B_all_runs"][k] - r1["legs"]["1"]["ratios"][k]) < 1e-12
        assert m1["per_run"][0]["n_runs"] == 2
    try:
        collate([p1, p1], "", n_match=2); raise SystemExit("dup not caught")
    except AssertionError as e:
        assert "duplicate run_logdir" in str(e)
    try:
        collate([p1, p2], "", expect_n=4); raise SystemExit("n not caught")
    except AssertionError as e:
        assert "expected 4 runs" in str(e)

    print("se_lic_probe selfcheck PASS "
          f"(arm={r1['disag_head']}, S={r1['n_eval']}, h={r1['horizons']}; "
          f"F1 Var_e[q_deter]=0 verified, primary=L_stochnorm; "
          f"F2 null gate typed+consumed; F4 L-floor measured and looser than "
          f"the share floor; F6b pass A at burn_in length; F8 per-horizon "
          f"calibration; strict-JSON; deterministic; L4 matched-n + refusals)")
    return r1


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.collate is not None:
        print(json.dumps(_clean(collate(
            args.collate, args.collate_out, n_match=args.n_match,
            expect_n=args.expect_n)), indent=1))
    else:
        run(args)


if __name__ == "__main__":
    main()
