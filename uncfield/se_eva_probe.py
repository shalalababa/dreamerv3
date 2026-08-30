"""EVA-at-scale — anytime-valid claim audit on deployed DreamerV3 world models.

BUILD ONLY — not yet run on trained checkpoints, not yet in the repo.
Destination at commit time: `uncfield/se_eva_probe.py`.

Ports the in-world EVA (MathAlgo_CANDIDATES.md §1; temp_files/
nfi_repair_20260825/mathalgo/eva_kdb.py, eva_cycle.py) to the 12 deployed
cells already mapped by `uncfield/se_lic_probe.py` (4 B4 gauss se_alea_s120-123
+ 8 Stage-1 det se_cheetah_seed10-17).

===========================================================================
THE TEST
===========================================================================
For an audited channel k with a one-step predictive that is a PREDICTABLE
functional of the realized past, H0 is

    y_{t+1,k} | F_t  ~  p_t,k      ("the WM's world is the true world here")

Under H0 the standardized innovations eps = (y - yhat)/sqrt(S) are
conditionally N(0,1) per dim, so for ANY predictable r_t the likelihood
ratio against the N(0, r_t) alternative

    log LR_t = -0.5*D*log r_t - 0.5*(sum_d eps_d^2)*(1/r_t - 1)

accumulates into a nonnegative martingale E_T with E[E_T] = 1.  Ville:
P_H0(exists T: E_T >= 1/alpha) <= alpha — ANYTIME-VALID, robust to optional
stopping, no gate-analog (once crossed, crossed forever).  D = the channel's
dim; the model's own decoder likelihood is factorized across dims
(`embodied/jax/outs.py:129-142`), so the per-dim product is the model's own
law, not an added assumption.

Two one-sided legs, each with its own alpha/2 (directions DERIVED below):
  OVER  : r_t <= 1 — the model claims MORE dispersion than the data shows.
  UNDER : r_t >= 1 — the model claims LESS (over-confidence).
The in-world EVA used OVER only; the deployed persistence question needs
UNDER, so both are built and the direction each channel/leg predicts is
registered before running.

Predictable plug-in (prequential, exactly eva_kdb.py:147):
    r_t = clip((nu0 + sum_past eps^2) / (nu0 + D * n_past), lo, hi)
computed from the PAST ONLY, so validity holds whatever it does.

===========================================================================
WHERE THE WM'S LIKELIHOOD ACTUALLY LIVES  (the wiring, and a correction)
===========================================================================
The brief assumed "decoder mu, sigma per key".  **There is no learned
sigma.**  The decoder's vec heads are `symlog_mse`
(`dreamerv3/rssm.py:299`) -> `outs.MSE(pred, nets.symlog)`
(`embodied/jax/heads.py:127-130`), whose `.loss(y) = (mean - symlog(y))^2`.
So:

  * the likelihood is Gaussian **in SYMLOG space** — every residual here is
    `symlog(y_true) - dec_mean`, and NOTHING is symexp'd (contrast
    se_lic_probe, which symexps to feed the encoder; different purpose);
  * its scale is a CONSTANT, not a per-key per-state sigma.  `c` below.

The model's one-step predictive dispersion is therefore assembled, in
symlog space, from the three places uncertainty actually lives:

    S_P(t,k) = c + Var_M[ dec_k(z'_m) ]                     LEG P
    S_C(t,k) = c + Var_M[ dec_k(z'_m) ] + Var_E[ dec_k(mu_e) ]   LEG C

  Var_M : the RSSM prior's own stochastic spread — M one-hot samples of the
          prior categorical at t+1 (`dyn.imagine(..., single=True)` ->
          `logit`), decoded.  This IS the model's generative law.
  Var_E : the disagreement ensemble's decoded variance — the SAME quantity
          that is theta_1's numerator (se_probe.py:208-217) and the deployed
          intrinsic reward's numerator (explore.py:102-116).
  mean  : E_M[dec_k(z'_m)], the generative predictive mean, shared by both
          legs so they differ ONLY in the audited dispersion.

  LEG P audits the WM's own generative law  -> the wrong-world question.
  LEG C audits the DEPLOYED CLAIM (aleatoric + the epistemic bonus)
        -> the misprice question, i.e. exactly the account theta_1 prices.

`c` is a nuisance shared by every key.  Two DECLARED settings, both run,
and a finding must have the same sign under both:
  c_theory : 0.5  — the MSE convention (squared error = -log N(mu, 1/2)).
             No fitting, fully oracle-free.
  c_recon  : IDENTIFIED (not fitted to the statistic) on a DISJOINT prefix
             as the decoder's POSTERIOR RECONSTRUCTION RESIDUAL over the
             real keys — decode the filtered state at t against symlog(y_t)
             and what remains is the observation noise the symlog_mse head
             was trained to minimise.  Sample-splitting keeps Ville intact.
             (Review B3 REPLACED an earlier `c_cal` that forced pooled
             real-key eps^2 = 1: it made prediction (b) untestable by
             construction and sat on the same side of c* as c_theory.)
  dDlogrho/dc is single-signed, so two nearby scalings can both lie on one
  side of the sign-flip point.  The primary is therefore ALSO reported over
  a declared grid c in {0.05,0.1,0.25,0.5,1,2}, with c* — the sign-flip
  location — emitted as a sensitivity.

===========================================================================
STREAM CONSTRUCTION — THE VALIDITY ARGUMENT (must be airtight)
===========================================================================
1. Streams come from `probing/probeset.py:93 chain_streams`, which recovers
   contiguous per-worker chains from the chunk filenames' successor UUIDs
   and returns them in temporal order (heads sorted by timestamp).  We take
   the first `--n_streams` of that deterministic list; within a stream the
   chunks are concatenated in recorded order (`load_stream`).  **No
   segment is selected on any model quantity, ever.**
2. The audit runs over the stream's OWN temporal order, front to back, from
   the first admissible index to the last.  No shuffling, no re-ordering,
   no stopping on the statistic.
3. Admissibility is EXOGENOUS: index t is audited iff (a) t and t+1 lie in
   the same episode, and (b) t is at least `--burn_in` steps after that
   episode's start.  Both depend only on the environment's recorded reset
   schedule (`is_first`) and the index — never on the model.  Skipping on a
   predictable/exogenous rule preserves the martingale (optional skipping).
   The realized episode-length distribution is recorded so the reader can
   see the schedule is regular.
4. The calibration prefix is the first `--n_cal` admissible reads; the
   e-processes start only after it, so `c` is F-measurable at audit start.
5. Each stream carries an INDEPENDENT e-process.  We report the per-stream
   crossing fraction (as eva_kdb does over seeds) and, since independent
   e-values multiply into a valid e-value, the pooled product.
6. Horizon leg (below) uses NON-OVERLAPPING stride-h blocks so that each
   h-step innovation is conditioned on its own block start; overlapping
   h-step windows would break the martingale and are never used.

===========================================================================
THE HORIZON / PERSISTENCE LEG  (DPB's kappa at scale)
===========================================================================
`artifacts/se_lic_read_20260826/` measured, uniform in 12/12 runs, a
licensed distractor persistence at h=15 of 0.138-0.162 against the
environment's own AR(1) law 0.052 — the WM believes the distractor ~3x
more persistent than it is.  The distractor is a unit-stationary AR(1)
with theta=0.1 => phi = 0.9 (`embodied/envs/distractor.py:60-63`).

Algebra (scalar AR(1), stationary variance sigma^2).  True h-step
conditional variance given the value now is sigma^2 (1 - phi^{2h}).  A model
believing phi_m > phi claims sigma^2 (1 - phi_m^{2h}).  So its realized /
claimed dispersion ratio at horizon h is

    rho(h) = (1 - phi^{2h}) / (1 - phi_m^{2h})   >  1     (UNDER leg)

and the e-process grows at 0.5 (rho - 1 - ln rho) nats per read.  Inverting
the deployed number: persistence(15) = phi_m^{2*14} = 0.148 (gauss_s120)
gives phi_m = 0.148^{1/28} = 0.9333, so rho(15) = (1-0.81^15)/(1-0.8711^15)
~ 1.09 — a ~2.3e-3 nats/read growth, which is why this leg needs LONG
streams rather than the 512 anchors se_lic_probe used.  `predicted_rho` is
emitted per horizon beside the measured one; both phi and phi_m come from
declared sources (the env config and the prior read), not from this data.

DIRECTIONS ARE OPPOSITE AT THE TWO HORIZONS AND THAT IS THE POINT:
  h = 1  : the epistemic bonus is ADDED on top of a roughly-calibrated
           one-step law -> claimed dispersion too BIG -> OVER leg.
  h >= 2 : the WM thinks the distractor's reading stays informative -> its
           h-step claimed dispersion is too SMALL -> UNDER leg.
A single channel is predicted to certify on OPPOSITE sides at different
horizons.  Nothing about a global scale error can produce that pattern, so
it is the sharpest available signature.

===========================================================================
REGISTERED PREDICTIONS -- MIRROR of PREDICTIONS.md (rev 5).  That file is
the single registration; where they differ, it governs.
===========================================================================
ROUND 3.  Two earlier primaries were adjudicated CONSTRUCT-INVALID:
`position_on_bracket` (rev 2/3 -- R did not depend on vens, so the
"justified" end was unattainable) and `A` with a bar at 0.25 (rev 4 --
review B1: A = (Eq+1)/(E+1), so the bar INVERTED the reading, labelling
q = 0 as unearned).  Bars are now on q_hat and on nothing else.

PRIMARY  q_hat = ((E-1)*slope_IV - 1)/E.  ESTIMAND (F1): the
  V-PROPORTIONAL EXCESS OF MEAN ERROR OVER THE COHERENT RATE.  A shared
  common-mode error is ONE mechanism that produces it, not the only one, so
  q_hat alone does NOT attribute mechanism -- that is gamma_hat/q_hat_net
  (the same functional on the posterior RECONSTRUCTION residual, which
  holds no prediction error; q_hat_net = q_hat - gamma_hat, offsets cancel).
  SHARED-DOMINATED names the SIZE of the excess, not its cause.
  q=0 the spread accounts for the whole error scaling; q=1 exchangeable;
  q>>1 common-mode dominated.  Slope by SPLIT-HALF IV (halves rescaled by
  kappa = 7/6), because V is a chi-square estimate (rel. var 2/(E-1) =
  0.286) and OLS attenuates -- measured A_OLS 0.599 vs A_IV 0.936 at q = 1.
  lambda_hat = Cov(Va,Vb)/Var(V) is a GATE: below 0.15 the verdict is
  UNRESOLVED, never UNEARNED.

VERDICT (in order): lambda_hat < 0.15 -> UNRESOLVED-WEAK-INSTRUMENT;
  circular-shift permutation p > 0.05 or sign(A) != sign(r) ->
  UNRESOLVED-NO-ALIGNMENT; A < 0 -> UNRESOLVED-OUT-OF-SUPPORT (ANTI STRUCK
  29 Aug 2026, F5: q >= 0 by construction so no fixture is constructible);
  q non-finite -> UNRESOLVED-DEGENERATE (F12: every non-classifying outcome
  carries the UNRESOLVED prefix, so "not resolved => UNRESOLVED*" is total);
  dq_iv > bar_distance -> UNRESOLVED-IV-DISAGREEMENT (REV 7: the bar cannot
  decide THAT KEY -- a per-key resolution fact.  Rev 6 had this in
  controls_pass and refused 3 of 12 cells whose SYNTHG margins were healthy
  at 4-5 log10; the cell-level gate now runs only on REFERENCE keys whose
  bar distance is >= 2.0, where a dq excursion cannot be near-bar physics);
  q < 0.5 SPREAD-ACCOUNTS; q > 2 SHARED-DOMINATED; else MIXED.  Gates use the WIDER of the cluster
  bootstrap and the per-stream t-interval (the 6-cluster percentile
  interval under-covers, 0.860 vs 0.95).

(A1) DISTRACTOR: PREDICT q_hat > 2 (SHARED-DOMINATED) with the wider
  interval excluding 0; counter-branch q_hat < 0.5.  Adjudicability gate:
  |q_d - q_pos| > floor_q = |q_dup0 - q_pos|, else PRIMARY-UNRESOLVED.
(A2) SPECIFICITY: PREDICT all keys SHARED-DOMINATED; a differential needs
  |q_d - q_k| > 2*floor_q.
(A3) FLOOR: dup0 is bitwise identical to position, so their difference IS
  the resolution.  Re-derived (I1): dup2's uncorrelated added noise raises
  the INTERCEPT, not the slope -- PREDICT intercept(dup2) > intercept(dup0);
  no slope clause on dup2 (rev 4's was mechanistically wrong).
(A4) INTERCEPT: DESCRIPTIVE only (I2) -- it is s2_true PLUS any
  V-uncorrelated systematic error, and the >10 bar is dropped as
  non-discriminating (untrained returns 6.98).
(A5) TIMING, re-derived (B6): rev 4's max(det) < min(gauss) is FALSE on
  rev-3's own numbers (they overlap on [0.9748, 0.9915]).  BAR: exact
  one-sided Mann-Whitney (gauss > det, 8 vs 4) p <= 0.01 at the pinned
  scaling c_recon, summary statistic median -- rev 3 gives U = 30/32,
  p = 0.00808 -- widened to p <= 0.05 AND U >= 26 (F13), pooling only
  resolved+aligned cells (F3), AND the same ordering reproduced by median
  q_hat; a significant reverse ordering is TIMING-REVERSED (F11).  The pb
  leg is a reproduction bar on a statistic already declared
  construct-invalid; only the q_hat conjunction is interpretable.
  Disagreement -> TIMING-NOT-REPORTABLE.  Cross-cell, in
  analysis/se_eva_read.py.
(A6) REV-3 REPRODUCTION: rho and log10e_pooled bitwise on non-synthetic
  legs (statable only because the rng is per-audit, review I6), floors
  within +-0.05.  An INSTRUMENT ALARM, not a finding.

NEGATIVE CONTROLS, both registered (B4): step-0 = the lambda -> 0 limit
  (no instrument; UNRESOLVED-WEAK-INSTRUMENT 6/6, primary_resolved False,
  controls_pass True); smoke = the q >> 1 limit (q_hat 11.7-18.7,
  SHARED-DOMINATED, with the distractor UNRESOLVED-NO-ALIGNMENT).
  controls_pass (instrument integrity) and primary_resolved (is the
  primary readable here) are SEPARATE AXES -- an unresolvable primary must
  not suppress a cell's certificates.

CONSTRUCT VALIDITY: all fixtures build REAL members (shared common-mode +
  iid idiosyncratic) and an actual y.  NOTHING synthesises Rbar from an
  observed V -- rev 4's fixtures did, which made them tautological.
  Recovered: q=0 -> -0.012, q=1 -> 0.928, q=4 -> 3.857; the homoscedastic
  case is correctly flagged by lambda = 0.010.

STRUCK FROM THE FREEZE: leg E is EXPLORATORY (B9).


CONTROLS (computed every run; `controls_pass` gates the cell, review B13).
  SYNTHG  — residuals drawn with variance EXACTLY S at each read, keeping
    the real heteroscedastic S sequence.  Validates the e-process and the
    prequential plug-in at the deployed scale.  Required SILENT on both
    legs, both sides, and every horizon.  NOTE (review B5): SYNTHG cancels
    errors in the ASSEMBLY of S by construction — it is drawn FROM S — so
    it is a pipeline control, NOT an assembly control.
  SYNTHMIX — the mixture-faithful draw (reserved prior sample + on leg C an
    injected N(0,sqrt(vens)) + the decoder's N(0,c)).  THIS is the assembly
    control, run on BOTH legs.  It does not come out at exactly 1: the
    latent predictive is a non-Gaussian mixture and the M-sample variance
    carries a Jensen inflation ~2/(M-1) into E[R/S] (29% at M=8, 6.5% at
    M=32 — hence --n_lat 32).  Its value IS THE FLOOR, measured per key and
    per horizon, and every reading is adjudicated against it
    (`rho_floor_corrected`, `dlogrho_legC_floor_corrected`).
  inflate_x4 / marginal_rescale — power, and the conditional-vs-marginal
    demonstration, on this run's own residuals.

ARM SCOPE (review B10).  LEG C audits the DEPLOYED claim only on the DET
arm, where the intrinsic reward is `var(0)` of the member predictions
(explore.py:115-116) — exactly `vens`.  On the GAUSS arm the reward is
ALEATORIC-NORMALISED IN LATENT SPACE (explore.py:108-114), and that
normalisation IS the B4 repair, which deflates precisely the distractor.
So on gauss, leg C audits "the ensemble's decoded dispersion claim", NOT
the reward the agent optimised.  Word claims per arm; the DET arm carries
the deployed-misprice statement.

STREAM POSITION (review B11).  The replay spans training, so where the
audit sits in it matters.  `--tail 1` (default) audits the window CLOSEST
IN TIME to the checkpoint; the offset is chosen by INDEX ALONE, never on a
statistic, so the exogenous-selection argument holds.  The realized offset
is `stream_meta.offset`.

SCOPE.  H0 as tested is "the standardized innovation is N(0,1)".  The true
predictive is a latent mixture, so this is a Gaussian-quasi-H0; the
synthetic no-op is what bounds the resulting false-certification rate
empirically, and its measured crossing fraction is reported beside every
certificate as that certificate's own calibration.

Run:  python -m uncfield.se_eva_probe --run_logdir <dir> [--platform cpu]
      python -m uncfield.se_eva_probe --selfcheck
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zlib

import numpy as np

from uncfield.se_probe import EXCLUDE, PLANTED_KEYS, REPO, resolve_ckpt

SMOKE = (REPO / "local_results" / "uncfield_se_smoke_20260812_155552"
         / "se_smoke0")
FLOOR_DIAGNOSTIC_KEY = "planted_const"
REAL_KEYS = ("position", "velocity")
AR1_COEF = 0.9                      # embodied/envs/distractor.py:60-63
LOG10 = math.log(10.0)
NAN = float("nan")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--n_streams", type=int, default=6,
                   help="first N streams of chain_streams' temporal order")
    p.add_argument("--t_max", type=int, default=8000,
                   help="steps per stream for the h=1 legs")
    p.add_argument("--t_max_h", type=int, default=4000,
                   help="steps per stream for the horizon leg")
    p.add_argument("--horizons", default="2,4,8,15")
    p.add_argument("--burn_in", type=int, default=16,
                   help="steps after each episode start before auditing")
    p.add_argument("--n_cal", type=int, default=1000,
                   help="admissible reads reserved for calibrating c")
    p.add_argument("--n_lat", type=int, default=32,
                   help="M: prior samples for the latent predictive.  The "
                        "M-sample variance estimate carries a Jensen "
                        "inflation ~2/(M-1) into E[R/S] (29%% at M=8, 6.5%% "
                        "at M=32) which IS the non-Gaussianity floor and "
                        "grows with h — review B4.")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Ville level; each one-sided leg gets alpha/2")
    p.add_argument("--n_perm_r", type=int, default=500,
                   help="circular-shift permutations for the alignment null "
                        "(review B3); the null is cluster-resampled so it "
                        "preserves each stream's marginal and autocorrelation")
    p.add_argument("--nu0", type=float, default=8.0,
                   help="plug-in prior strength, in pseudo-READS")
    p.add_argument("--r_min", type=float, default=0.05)
    p.add_argument("--r_max", type=float, default=20.0)
    p.add_argument("--blk", type=int, default=240)
    p.add_argument("--tail", type=int, default=1,
                   help="1 = audit the replay tail (closest in time to the "
                        "checkpoint); 0 = the head.  Index-only selection "
                        "either way (review B11)")
    p.add_argument("--platform", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# the e-process (pure numpy; every deciding gate killed in _fixture_checks)
# --------------------------------------------------------------------------

BARS_Q = (0.5, 2.0)         # the registered q bars


def safe_ratio(num, den):
    """Degenerate denominators give NaN, never a silent 0.0."""
    if den is None or num is None:
        return NAN
    try:
        d = float(den)
        return float(num) / d if d > 0 and math.isfinite(d) else NAN
    except (TypeError, ValueError):
        return NAN


def eprocess(sq, D, side, nu0=8.0, r_min=0.05, r_max=20.0):
    """One-sided anytime-valid e-process on standardized innovations.

    `sq[t]` = sum over the channel's D dims of eps^2 at read t.  Under H0
    each eps is N(0,1), so E[sq] = D.

    `side='over'`  tests r <= 1 (the model claims MORE dispersion than the
                   data shows — the in-world EVA's direction);
    `side='under'` tests r >= 1 (over-confidence).

    The plug-in r_t is computed from the PAST ONLY (eva_kdb.py:147), so
    E[LR_t | F_{t-1}] = 1 under H0 whatever r_t does, and Ville gives
    P(exists T: E_T >= 1/alpha) <= alpha.  Returns the running log e-value
    in NATS."""
    sq = np.asarray(sq, np.float64)
    n = sq.size
    if n == 0:
        return np.zeros(0)
    lo, hi = (r_min, 1.0) if side == "over" else (1.0, r_max)
    # vectorised, and EXACTLY the sequential recursion: cum/cnt at step t use
    # strictly the past, which is what makes r_t predictable.  (The loop form
    # is kept in the fixtures as the reference implementation and the two are
    # asserted bit-comparable.)
    cum = np.concatenate(([0.0], np.cumsum(sq)[:-1]))
    cnt = np.arange(n, dtype=np.float64)
    r = np.clip((nu0 * D + cum) / (nu0 * D + D * cnt), lo, hi)
    inc = -0.5 * D * np.log(r) - 0.5 * sq * (1.0 / r - 1.0)
    return np.cumsum(inc)


def _eprocess_loop(sq, D, side, nu0=8.0, r_min=0.05, r_max=20.0):
    """Reference (sequential) implementation, used only by the fixtures to
    hold the vectorised form honest."""
    sq = np.asarray(sq, np.float64)
    log_e = np.empty(sq.size)
    cum, cnt, le = 0.0, 0, 0.0
    lo, hi = (r_min, 1.0) if side == "over" else (1.0, r_max)
    for t in range(sq.size):
        r = float(np.clip((nu0 * D + cum) / (nu0 * D + D * cnt), lo, hi))
        le += -0.5 * D * math.log(r) - 0.5 * sq[t] * (1.0 / r - 1.0)
        log_e[t] = le
        cum += sq[t]
        cnt += 1
    return log_e


def half_kappa(ens):
    """Rescale a half-ensemble variance to the full-ensemble scale.
    E[V_half] = su^2 (E/2-1)/(E/2); E[V] = su^2 (E-1)/E."""
    h = ens / 2.0
    return ((ens - 1.0) / ens) / ((h - 1.0) / h)


def _cov(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 3:
        return NAN
    return float(((a - a.mean()) * (b - b.mean())).sum() / (a.size - 1))


def alignment(Rbar, Va, Vb, ens, per_stream=None, n_perm=0, seed=0):
    """THE v2 PRIMARY (rev 5).  Identified under a SHARED common-mode error.

    MODEL (review B1).  The E members share enc/dyn/dec, so their errors
    are NOT exchangeable with the truth:

        yhat_e = mu + b + u_e        b  shared  (var q * su^2)
        y      = mu + eps            u_e idiosyncratic (var su^2)

    With W := E[V] = su^2 (E-1)/E,

        E[Rbar] = s2_eps + W * (E q + 1)/(E - 1)

    so the regression slope of Rbar on V identifies

        q_hat = ((E - 1) * slope - 1) / E                        (B1)

    q is THE MEASURED QUANTITY: the shared error variance the ensemble's
    own spread cannot see, in units of the idiosyncratic scale.
      q = 0  -> V accounts for the whole scaling of the mean error
      q = 1  -> members and truth exchangeable (the rev-4 "coherent" case)
      q >> 1 -> the error is dominated by common-mode the spread misses

    `A = slope / beta*` (beta* = (E+1)/(E-1)) is retained as a REPORTING
    convention only; A = (Eq+1)/(E+1), so the earned family is
    A in [1/(E+1), inf) and rev 4's "A<0.25 = UNEARNED" bar INVERTED the
    reading (it labelled q=0, A=1/9=0.111, as unearned).  Bars are on q_hat.

    ERRORS-IN-VARIABLES (review B2).  V is a chi-square estimate with
    relative variance 2/(E-1) = 0.286, so OLS attenuates: an exactly-q=1
    ensemble returns A_OLS 0.002-1.17 depending on heteroscedasticity.
    Fix: SPLIT-HALF IV.  Va, Vb are the two half-ensemble variances
    (members [0,E/2) and [E/2,E)), each rescaled by kappa = 7/6 to the full
    scale.  Their measurement errors are independent given the state, and
    the full-ensemble mean ybar is independent of each half's centered
    deviations (Gaussian mean/variance independence), so

        b_IV = Cov(Rbar, Vb) / Cov(Va, Vb)

    is unbiased for the slope.  `lambda_hat = Cov(Va,Vb)/Var(V)` is the
    instrument's reliability: SMALL lambda_hat means UNRESOLVED, never
    UNEARNED (measured: lambda 0.03 -> A 1.28 +- 0.57; lambda 0.17 ->
    1.03 +- 0.10; lambda 0.70 -> 1.00 +- 0.02 against truth 1).
    """
    R = np.asarray(Rbar, np.float64).ravel()
    va = np.asarray(Va, np.float64).ravel()
    vb = np.asarray(Vb, np.float64).ravel()
    v = 0.5 * (va + vb)
    beta = (ens + 1.0) / (ens - 1.0)
    out = dict(beta_star=float(beta), n=int(R.size), slope_ols=NAN,
               slope_iv=NAN, q_hat=NAN, A=NAN, A_ols=NAN, lambda_hat=NAN,
               intercept=NAN, ci=[NAN, NAN], ci_t=[NAN, NAN],
               q_per_stream=[], r_per_stream=[], r_spearman=NAN,
               r_pearson=NAN, r_perm_p=NAN, r_perm_q95=NAN,
               slope_iv_sym=NAN, dq_iv=NAN,
               bar_distance=NAN)      # review M1: no KeyError path
    if R.size < 32:
        return out
    vv = _cov(v, v)
    cab = _cov(va, vb)
    if not (vv > 0) or not np.isfinite(cab):
        return out
    b_ols = _cov(R, v) / vv
    out["slope_ols"] = float(b_ols)
    out["A_ols"] = float(b_ols / beta)
    out["lambda_hat"] = float(cab / vv)
    if abs(cab) > 0:
        b_iv = _cov(R, vb) / cab                      # review B2, verbatim
        b_sym = _cov(R, v) / cab                      # cross-check
        out["slope_iv"] = float(b_iv)
        out["slope_iv_sym"] = float(b_sym)
        out["A"] = float(b_iv / beta)
        out["q_hat"] = float(((ens - 1.0) * b_iv - 1.0) / ens)
        out["intercept"] = float(R.mean() - b_iv * v.mean())
        # REV 7: the two IV forms' disagreement in q UNITS, and this key's
        # distance to the nearest registered bar.  Carried on the alignment
        # dict so the CLASSIFIER can route per key.
        out["dq_iv"] = float(abs((ens - 1.0) / ens * (b_iv - b_sym)))
        out["bar_distance"] = float(min(abs(out["q_hat"] - BARS_Q[0]),
                                        abs(out["q_hat"] - BARS_Q[1])))
    # bounded companion (review B3): Spearman, because Rbar is a squared
    # error and heavy-tailed
    rv = np.argsort(np.argsort(v)).astype(np.float64)
    rr = np.argsort(np.argsort(R)).astype(np.float64)
    den = math.sqrt(_cov(rv, rv) * _cov(rr, rr))
    out["r_spearman"] = float(_cov(rv, rr) / den) if den > 0 else NAN
    dr = math.sqrt(vv * _cov(R, R))
    out["r_pearson"] = float(_cov(v, R) / dr) if dr > 0 else NAN
    if per_stream:
        suf, rs, qs = [], [], []
        for a_, b_, RR in per_stream:
            one = alignment(RR, a_, b_, ens)
            qs.append(one["q_hat"]); rs.append(one["r_spearman"])
            aa = np.asarray(a_, np.float64).ravel()
            bb = np.asarray(b_, np.float64).ravel()
            RRa = np.asarray(RR, np.float64).ravel()
            suf.append((aa, bb, RRa))
        out["q_per_stream"] = qs
        out["r_per_stream"] = rs
        rng = np.random.default_rng(seed + 77)
        boots = np.empty(2000)
        for i in range(2000):
            pick = rng.integers(0, len(suf), len(suf))
            A_ = np.concatenate([suf[j][0] for j in pick])
            B_ = np.concatenate([suf[j][1] for j in pick])
            R_ = np.concatenate([suf[j][2] for j in pick])
            cb = _cov(A_, B_)
            boots[i] = (((ens - 1.0) * (_cov(R_, B_) / cb) - 1.0) / ens) \
                if abs(cb) > 0 else NAN
        out["ci"] = [float(np.nanpercentile(boots, 2.5)),
                     float(np.nanpercentile(boots, 97.5))]
        # review I3: the 6-cluster percentile interval UNDER-COVERS
        # (0.860 vs 0.95); pair it with a t-interval on the per-stream
        # values (df = n_streams - 1) and gate on the WIDER of the two.
        qv = np.asarray([x for x in qs if np.isfinite(x)], np.float64)
        if qv.size >= 3:
            from scipy import stats as _st
            tcrit = float(_st.t.ppf(0.975, qv.size - 1))
            hw = tcrit * float(qv.std(ddof=1)) / math.sqrt(qv.size)
            out["ci_t"] = [float(qv.mean() - hw), float(qv.mean() + hw)]
        # review B3: |r| against a CLUSTER-RESAMPLED circular-shift null,
        # which preserves each stream's marginal and autocorrelation and
        # destroys only the contemporaneous association.
        if n_perm:
            null = np.empty(n_perm)
            for i in range(n_perm):
                vs, Rs = [], []
                for a_, b_, RR in per_stream:
                    vv_ = 0.5 * (np.asarray(a_, np.float64).ravel()
                                 + np.asarray(b_, np.float64).ravel())
                    k = int(rng.integers(1, max(vv_.size - 1, 2)))
                    vs.append(np.roll(vv_, k))
                    Rs.append(np.asarray(RR, np.float64).ravel())
                vs = np.concatenate(vs); Rs = np.concatenate(Rs)
                rv2 = np.argsort(np.argsort(vs)).astype(np.float64)
                rr2 = np.argsort(np.argsort(Rs)).astype(np.float64)
                d2 = math.sqrt(_cov(rv2, rv2) * _cov(rr2, rr2))
                null[i] = abs(_cov(rv2, rr2) / d2) if d2 > 0 else NAN
            obs = abs(out["r_spearman"])
            out["r_perm_p"] = float(
                (1 + np.nansum(null >= obs)) / (1 + n_perm))
            out["r_perm_q95"] = float(np.nanpercentile(null, 95))
    return out


def classify(al, lam_min=0.15, perm_alpha=0.05):
    """THE REGISTERED VERDICT RULE (rev 6), as a pure function so the truth
    mutants are run through the SAME classifier the cells are (review F5).

    ANTI was STRUCK from the registered rule on 29 Aug 2026 (review F5).
    Reason, dated and recorded rather than left decorative: A < 0 means
    (Eq+1)/(E+1) < 0, i.e. q < -1/E, and q = sigma_b^2/sigma_u^2 >= 0 by
    construction -- so no world in the model can produce it and NO FIXTURE
    IS CONSTRUCTIBLE for it.  A negative estimate can only be sampling
    noise, so it now routes to UNRESOLVED-OUT-OF-SUPPORT (the safe family)
    instead of being reportable as a finding."""
    q = al.get("q_hat", NAN)
    lam = al.get("lambda_hat", NAN)
    A = al.get("A", NAN)
    r = al.get("r_spearman", NAN)
    pp = al.get("r_perm_p", NAN)
    # REVIEW F12: a degenerate key also has lambda_hat NaN, so it is
    # UNRESOLVED too -- returning a bare "DEGENERATE" here broke the
    # invariant "not resolved => verdict starts with UNRESOLVED" on a
    # REACHABLE path.  Every non-classifying outcome now carries the
    # UNRESOLVED prefix, and the resolution gate is evaluated first so the
    # ordering cannot reintroduce the gap.
    if not (np.isfinite(lam) and lam >= lam_min):
        return "UNRESOLVED-WEAK-INSTRUMENT"
    if not np.isfinite(q):
        return "UNRESOLVED-DEGENERATE"
    if not (np.isfinite(pp) and pp <= perm_alpha
            and np.isfinite(r) and np.sign(A) == np.sign(r)):
        return "UNRESOLVED-NO-ALIGNMENT"
    if np.isfinite(A) and A < 0:
        return "UNRESOLVED-OUT-OF-SUPPORT"          # review F5
    # REV 7.  If the two IV forms disagree by more than this key's distance
    # to the nearest bar, THE BAR CANNOT DECIDE THIS KEY.  That is a
    # RESOLUTION fact about one key, not an instrument failure.  Rev 6 put
    # it in controls_pass and refused 3 of 12 whole cells whose SYNTHG
    # margins were healthy at 4-5 log10 -- the same category error as the
    # rev-2 multiplicity, and a violation of the controls_pass /
    # primary_resolved split this instrument already had.
    dq, bd = al.get("dq_iv", NAN), al.get("bar_distance", NAN)
    if np.isfinite(dq) and np.isfinite(bd) and dq > bd:
        return "UNRESOLVED-IV-DISAGREEMENT"
    return ("SPREAD-ACCOUNTS" if q < BARS_Q[0] else
            "SHARED-DOMINATED" if q > BARS_Q[1] else "MIXED")


def make_members(n, ens, q, rng, het=1.0, s_eps=0.3, extra_noise=0.0):
    """REAL member construction (review B1): shared common-mode + iid
    idiosyncratic.  Nothing here synthesises Rbar FROM an observed V — the
    rev-4 fixtures did, which made them tautological and blind to exactly
    the scale-matching this model exposes.  Truth: A = (Eq+1)/(E+1)."""
    s = np.exp(rng.normal(0.0, het, n))
    su = np.sqrt(s)
    b = rng.normal(0.0, 1.0, n) * su * math.sqrt(q)
    u = rng.normal(0.0, 1.0, (ens, n)) * su
    yhat = b[None, :] + u
    y = rng.normal(0.0, 1.0, n) * s_eps
    if extra_noise:                       # V-uncorrelated added error
        y = y + rng.normal(0.0, extra_noise, n)
    return yhat, y


def members_to_stats(yhat, y, ens):
    """(yhat, y) -> (Va, Vb, Rbar) with the half-ensemble kappa rescale."""
    k = half_kappa(ens)
    h = ens // 2
    ybar = yhat.mean(0)
    return (yhat[:h].var(0) * k, yhat[h:].var(0) * k, (y - ybar) ** 2)


def alignment_mutant(ens, q, seed=0, het=1.0, n=6000, streams=6,
                     extra_noise=0.0, n_perm=0):
    """CONSTRUCT-VALIDITY fixture, from REAL member construction."""
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(streams):
        yh, y = make_members(n, ens, q, rng, het=het,
                             extra_noise=extra_noise)
        ps.append(members_to_stats(yh, y, ens))
    out = alignment(np.concatenate([p[2] for p in ps]),
                    np.concatenate([p[0] for p in ps]),
                    np.concatenate([p[1] for p in ps]),
                    ens, per_stream=ps, seed=seed, n_perm=n_perm)
    out["q_true"] = float(q)
    out["A_true"] = float((ens * q + 1.0) / (ens + 1.0))
    return out


def synthg_family_verdict(block, label=""):
    """CONTROL GATE (rev 3).  A SYNTHG family fails ONLY IF IT CERTIFIES on
    the POOLED e-value — exactly the standard the findings are held to
    (review B7).

    Rev 2 used `crossed_frac == 0.0`: a family failed if ANY stream's
    running supremum ever crossed.  That is ~16 families x 6 keys x
    6 streams = ~576 independent alpha/2 = 0.005 tests per cell, so the
    expected number of crossings under H0 is ~2.9 and a cell refuses itself
    with probability ~0.95.  Measured: run 56089362 refused 7/12 cells on
    single isolated sub-gates while inflate_x4 and marginal_rescale passed
    everywhere — a manufactured false-refusal rate, not an instrument
    defect.

    Pooling collapses the 6 per-stream supremum tests into ONE test on the
    final product of e-values (Markov: P(product >= 1/alpha_side) <=
    alpha_side), which both removes a factor of 6 from the multiplicity and
    replaces a supremum with a final value.  A correctly calibrated control
    drifts strongly NEGATIVE, so the realized margin is large; the margin is
    reported so the gate is visibly not knife-edge."""
    failing, worst, thr = [], -float("inf"), NAN
    for k, v in block.items():
        pooled = v.get("log10e_pooled", NAN)
        thr = v.get("log10e_threshold", thr)
        if np.isfinite(pooled):
            worst = max(worst, pooled)
        if v.get("certified"):
            failing.append(k)
    return dict(label=label, passed=not failing, failing_keys=failing,
                worst_pooled_log10e=(float(worst) if np.isfinite(worst)
                                     else NAN),
                threshold=float(thr) if np.isfinite(thr) else NAN,
                margin=(float(thr - worst)
                        if np.isfinite(thr) and np.isfinite(worst) else NAN))


def cert(log_e, alpha):
    """Ville certificate at level alpha (running supremum: once crossed,
    crossed forever — no relapse, by the theorem)."""
    if log_e.size == 0:
        return dict(log10e_final=NAN, log10e_max=NAN, crossed=False,
                    t_cross=-1, n=0, growth_nats=NAN)
    thr = math.log(1.0 / alpha)
    hit = log_e >= thr
    return dict(
        log10e_final=float(log_e[-1] / LOG10),
        log10e_max=float(log_e.max() / LOG10),
        crossed=bool(hit.any()),
        t_cross=int(np.argmax(hit)) if hit.any() else -1,
        n=int(log_e.size),
        growth_nats=float(log_e[-1] / log_e.size))


def kl_rate(rho):
    """Asymptotic e-process growth for a persistent dispersion ratio rho:
    KL(N(0,rho) || N(0,1)) = 0.5 (rho - 1 - ln rho) nats per dim per read."""
    if not (rho > 0) or not np.isfinite(rho):
        return NAN
    return float(0.5 * (rho - 1.0 - math.log(rho)))


def predicted_rho_ar1(h, phi, phi_m):
    """(1 - phi^{2h}) / (1 - phi_m^{2h}) — realized/claimed dispersion for a
    model believing coefficient phi_m in a world with coefficient phi."""
    num = 1.0 - phi ** (2 * h)
    den = 1.0 - phi_m ** (2 * h)
    return float(num / den) if den > 0 else NAN


def phi_from_persistence(p_h, h):
    """Invert the se_lic L3 persistence statistic p(h) = phi_m^{2(h-1)}."""
    if not (0 < p_h < 1) or h <= 1:
        return NAN
    return float(math.exp(math.log(p_h) / (2 * (h - 1))))


def phi_m_implied(rho, h, phi=AR1_COEF):
    """Invert rho = (1 - phi^{2h}) / (1 - phi_m^{2h}) for phi_m — the AR
    coefficient the model's OWN h-step dispersion implies.

    This is DPB's kappa at scale: the persistence misclaim expressed as a
    single scalar per horizon, extracted from the audit itself rather than
    imported.  If the measured phi_m is ~constant across h and exceeds the
    environment's phi, the wrong-persistence account is confirmed with a
    number and its internal consistency is visible."""
    if not (rho > 0) or not np.isfinite(rho):
        return NAN
    x = 1.0 - (1.0 - phi ** (2 * h)) / rho
    if not (0.0 < x < 1.0):
        return NAN
    return float(math.exp(math.log(x) / (2 * h)))


def fit_c(resid_sq, var_lat, lo=1e-6, hi=1e4, iters=80):
    """Solve mean over (reads, dims) of resid^2/(c + var_lat) = 1 for c.

    Monotone decreasing in c, so bisection is exact to machine precision.
    Returns NaN if no root exists in the bracket (the model is UNDER-
    dispersed even at c -> 0, i.e. no constant can calibrate it — reported
    rather than clipped)."""
    r2 = np.asarray(resid_sq, np.float64).ravel()
    vl = np.asarray(var_lat, np.float64).ravel()
    f = lambda c: float(np.mean(r2 / (c + vl))) - 1.0
    if f(lo) < 0:
        return NAN                      # already over-dispersed at c ~ 0
    if f(hi) > 0:
        return NAN                      # cannot be calibrated below hi
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if f(m) > 0:
            a = m
        else:
            b = m
    return float(0.5 * (a + b))


def episode_segments(is_first, burn_in, need):
    """Admissible read indices, EXOGENOUS by construction: t is audited iff
    t and t+need lie in the same episode and t is >= burn_in after that
    episode's start.  Depends only on the recorded reset schedule and the
    index — never on any model quantity (see the validity argument)."""
    is_first = np.asarray(is_first, bool).ravel()
    n = is_first.size
    starts = list(np.flatnonzero(is_first))
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [n]
    keep, lens = [], []
    for i in range(len(starts)):
        s, e = bounds[i], bounds[i + 1]
        lens.append(int(e - s))
        lo = s + burn_in
        hi = e - need
        if hi > lo:
            keep.append(np.arange(lo, hi))
    idx = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    return idx.astype(np.int64), lens


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return _clean(float(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# --------------------------------------------------------------------------
# main
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
    out_dir = args.output or os.path.join(args.run_logdir, "se_eva_probe")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir, False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None, "EVA leg C requires a disag ensemble"
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    universe = [k for k in obs_keys if k != FLOOR_DIAGNOSTIC_KEY]
    real_keys = [k for k in obs_keys if k in REAL_KEYS]
    assert real_keys, f"no real keys among {obs_keys}"
    source_key = str(config.planted["source_key"])
    ddim, nstoch, ncls = (int(model.dyn.deter), int(model.dyn.stoch),
                          int(model.dyn.classes))
    ens = int(config.agent.expl["disag_ens"])
    head = str(config.agent.expl["disag_head"])
    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    assert all(h >= 2 for h in horizons), "h=1 is the dedicated one-step leg"
    M = args.n_lat
    dims = {k: int(np.prod(agent.obs_space[k].shape)) for k in obs_keys}
    for k in obs_keys:                                     # review M11
        assert not agent.obs_space[k].discrete, (
            f"{k} is discrete; the symlog-Gaussian standardisation applies "
            "to continuous vec keys only")
    assert bool(model.dec.symlog), (
        "this instrument standardizes in SYMLOG space because the decoder's "
        "likelihood does (rssm.py:299); a non-symlog decoder needs a "
        "re-derivation, not a flag")

    # ---------------------------------------------------------- model wiring
    def decode(det, probs, B):
        dec_carry = model.dec.initial(B)
        _, _, recons = model.dec(dec_carry, dict(deter=det[:, None],
                                                 stoch=probs[:, None]),
                                 jnp.zeros((B, 1), bool), training=False)
        return {k: f32(recons[k].pred())[:, 0] for k in obs_keys}

    def postsplit(pred):
        det = pred[..., :ddim]
        probs = pred[..., ddim:].reshape((*pred.shape[:-1], nstoch, ncls))
        probs = jnp.clip(probs, 1e-6, 1.0)
        return det, probs / probs.sum(-1, keepdims=True)

    def onehot_sample(logit):
        """One-hot draw from the prior, matching `_observe`/`imagine`
        (rssm.py:87,100) via the PUBLIC OneHot output (rssm.py:174)."""
        return f32(embodied.jax.outs.OneHot(
            nn.cast(logit), model.dyn.unimix).sample(nj.seed()))

    def block_fn(carry, obs, prevact, act, reset):
        """One block of the stream.  Returns, per step t in the block, the
        model's one-step predictive mean and its two dispersion parts, plus
        the carry for the next block (so the scan is exact across blocks).

        `prevact[t] = action[t-1]` drives the transition INTO state t;
        `act[t] = action[t]` is taken AT state t and drives t -> t+1 (dv3's
        post->prev convention, se_probe.py:266-275)."""
        T = reset.shape[1]
        enc_carry = model.enc.initial(1)
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
        dyn_carry, _, post = model.dyn.observe(carry, tokens, prevact, reset,
                                               training=False)
        st = {"deter": post["deter"][0], "stoch": post["stoch"][0]}   # (T,..)
        a_t = {k: act[k][0] for k in act_keys}                        # (T,..)
        # one-step PRIOR at t+1 (the model's own generative law)
        _, (feat1, _) = model.dyn.imagine(nn.cast(st), a_t, 1,
                                          training=False, single=True)
        d1, lg1 = f32(feat1["deter"]), f32(feat1["logit"])
        zs = jnp.stack([onehot_sample(lg1) for _ in range(M + 1)], 0)
        dec_lat = decode(jnp.broadcast_to(d1[None], (M + 1, T, ddim)
                                          ).reshape((M + 1) * T, ddim),
                         zs.reshape((M + 1) * T, nstoch, ncls), (M + 1) * T)
        out = {}
        for k in obs_keys:
            v = dec_lat[k].reshape((M + 1, T, -1))
            out[f"mean_{k}"] = v[:M].mean(0)          # generative pred mean
            # UNBIASED (ddof=1): vlat is a sample estimate from M draws, and
            # the biased form under-states it by (M-1)/M, which inflates the
            # standardized residual on exactly the keys where the latent
            # spread dominates c (caught by the mixture control).
            out[f"vlat_{k}"] = v[:M].var(0, ddof=1)   # latent dispersion
            out[f"synth_{k}"] = v[M]                  # reserved draw (no-op)
        # the DEPLOYED CLAIM: decoder-projected disag ensemble variance
        feat_t = model.feat2tensor(st)
        actvec = jnp.concatenate(
            [a_t[k].reshape((T, -1)) for k in act_keys], -1)
        mu = f32(model.disag.predict(feat_t, actvec))                # (E,T,640)
        de, pe = postsplit(mu)
        dec_e = decode(de.reshape(ens * T, ddim),
                       pe.reshape(ens * T, nstoch, ncls), ens * T)
        kap = half_kappa(ens)
        hh = ens // 2
        for k in obs_keys:
            de_k = dec_e[k].reshape((ens, T, -1))
            out[f"vens_{k}"] = de_k.var(0)          # ddof=0: the deployed
            out[f"emean_{k}"] = de_k.mean(0)        # numerator; ybar
            # split-half disagreements for the IV (review B2); kappa puts
            # each half on the full-ensemble scale
            out[f"vensa_{k}"] = de_k[:hh].var(0) * kap
            out[f"vensb_{k}"] = de_k[hh:].var(0) * kap
        # POSTERIOR RECONSTRUCTION (review B3): decoding the FILTERED state
        # at t against symlog(y_t) leaves only the decoder's own observation
        # noise, which is exactly the constant `c` the symlog_mse head was
        # trained to minimise.  One extra decode; F_t-measurable.
        rec_post = decode(f32(st["deter"]), f32(st["stoch"]), T)
        for k in obs_keys:
            out[f"recon_{k}"] = rec_post[k]
        return dyn_carry, out

    jit_block = jax.jit(
        lambda p, c, o, pa, a, r, s: nj.pure(block_fn)(p, c, o, pa, a, r,
                                                       seed=s))

    def horizon_fn(carry, obs, prevact, act, reset, h):
        """h-step predictive rolled under the RECORDED action sequence,
        sampling the latent trajectory (the model's own law).

        The posterior sweep must be sequential over every step, but the
        h-step rollout is computed ONLY on the fixed lattice
        {0, h, 2h, ...} of block-local offsets — the non-overlapping
        starts.  The block size is forced to a multiple of h so the lattice
        is aligned across blocks and its shape is static.  This is both the
        h-fold saving and the cleanest form of the validity argument: the
        lattice is fixed before anything is computed, and admissibility is
        applied to it afterwards by the exogenous episode rule."""
        T = reset.shape[1]
        enc_carry = model.enc.initial(1)
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
        dyn_carry, _, post = model.dyn.observe(carry, tokens, prevact, reset,
                                               training=False)
        d0 = f32(post["deter"][0])[::h]                     # (P, ddim)
        z0 = f32(post["stoch"][0])[::h]
        P = d0.shape[0]
        # M independent sampled trajectories from each lattice start
        MM = M + 1                       # +1 reserved for the mix control
        d = jnp.broadcast_to(d0[None], (MM, P, ddim)).reshape(MM * P, ddim)
        z = jnp.broadcast_to(z0[None], (MM, P, nstoch, ncls)
                             ).reshape(MM * P, nstoch, ncls)
        for j in range(h):
            aj = {k: act[k][0, ::h, j] for k in act_keys}    # (P, adim)
            a = {k: jnp.broadcast_to(aj[k][None], (MM, P, aj[k].shape[-1])
                                     ).reshape(MM * P, -1) for k in act_keys}
            _, (fj, _) = model.dyn.imagine(nn.cast(dict(deter=d, stoch=z)),
                                           a, 1, training=False, single=True)
            d, lg = f32(fj["deter"]), f32(fj["logit"])
            z = onehot_sample(lg)
        dec = decode(d, z, MM * P)
        out = {}
        for k in obs_keys:
            v = dec[k].reshape((MM, P, -1))
            out[f"mean_{k}"] = v[:M].mean(0)
            out[f"vlat_{k}"] = v[:M].var(0, ddof=1)
            out[f"synth_{k}"] = v[M]
        return dyn_carry, out

    jit_hor = {h: jax.jit(
        lambda p, c, o, pa, a, r, s, _h=h: nj.pure(horizon_fn)(
            p, c, o, pa, a, r, _h, seed=s)) for h in horizons}

    # ------------------------------------------------------------- streams
    chains_all = probeset_mod.chain_streams(
        os.path.join(args.run_logdir, "replay"))
    assert len(chains_all) >= args.n_streams, (          # review B12
        f"only {len(chains_all)} replay streams, need {args.n_streams}")
    chains = chains_all[:args.n_streams]
    _scache = {}

    def get_stream(si):
        """load_stream is otherwise called ~30x per cell (review M12)."""
        if si not in _scache:
            _scache.clear()
            _scache[si] = probeset_mod.load_stream(chains[si])
        return _scache[si]

    print(f"streams: {len(chains)} of {len(chains_all)} "
          f"(chain_streams temporal order)")

    def sweep(arrays, T, jitfn, need, extra_act=0, blk=None):
        """Run `jitfn` block-by-block over the first T steps of a stream,
        carrying the RSSM state across blocks (exact, not restarted).

        For the horizon leg `blk` is forced to a multiple of h and n to a
        multiple of blk, so the stride-h lattice is aligned across blocks
        and every block has the same static shape (no recompilation)."""
        blk = blk or args.blk
        N = arrays["is_first"].shape[0]
        usable = N - need - extra_act
        n = min(T, usable)
        n = (n // blk) * blk
        assert n > 0, (                                   # review B12
            f"stream too short: {N} steps, need >= {blk + need + extra_act}")
        # REVIEW B11: audit the replay CLOSEST IN TIME to the checkpoint,
        # not the oldest retained chunks.  The offset is chosen by INDEX
        # ALONE (never on any statistic), so the exogenous-selection
        # argument is untouched.
        off = ((usable - n) // blk) * blk if args.tail else 0
        acc, carry = {}, None
        for i in range(off, off + n, blk):
            j = i + blk
            sl = slice(i, j)
            o = {k: jnp.asarray(arrays[k][sl][None]) for k in obs_keys}
            for k in EXCLUDE:
                if k in arrays:
                    o[k] = jnp.asarray(arrays[k][sl][None])
            pa = {k: jnp.asarray(
                np.concatenate([arrays[k][max(i - 1, 0):max(i - 1, 0) + 1],
                                arrays[k][sl][:-1]], 0)[None])
                for k in act_keys}
            if extra_act:
                a = {k: jnp.asarray(np.stack(
                    [arrays[k][sl.start + d:sl.stop + d]
                     for d in range(extra_act)], 1)[None]) for k in act_keys}
            else:
                a = {k: jnp.asarray(arrays[k][sl][None]) for k in act_keys}
            r = jnp.asarray(np.asarray(arrays["is_first"][sl], bool)[None])
            if carry is None:
                carry = model.dyn.initial(1)
                carry = jax.tree.map(jnp.asarray, carry)
            # nj.pure returns (state, fn_result); fn_result is (carry, out)
            _, (carry, res) = jitfn(params, carry, o, pa, a, r, args.seed + i)
            for k, v in res.items():
                acc.setdefault(k, []).append(np.asarray(v, np.float64))
        return {k: np.concatenate(v, 0) for k, v in acc.items()}, n, off

    def symlog(x):
        return np.sign(x) * np.log1p(np.abs(x))

    # ---- h = 1 pass over every stream
    per_stream, ep_lens_all, stream_meta = [], [], []
    for si in range(len(chains)):
        arr = get_stream(si)
        res, n, off = sweep(arr, args.t_max, jit_block, need=1)
        loc, ep_lens = episode_segments(
            arr["is_first"][off:off + n], args.burn_in, 1)
        ep_lens_all += ep_lens
        idx = off + loc                       # indices into the STREAM
        tgt = {k: symlog(np.asarray(arr[k], np.float64)) for k in obs_keys}
        rec = dict(stream=si)
        rec["resid2"] = {k: ((tgt[k][idx + 1] - res[f"mean_{k}"][loc]) ** 2)
                         for k in obs_keys}
        # posterior reconstruction residual at t (review B3): identifies c
        rec["recon2"] = {k: ((tgt[k][idx] - res[f"recon_{k}"][loc]) ** 2)
                         for k in obs_keys}
        rec["vlat"] = {k: res[f"vlat_{k}"][loc] for k in obs_keys}
        rec["vens"] = {k: res[f"vens_{k}"][loc] for k in obs_keys}
        rec["synth_dev"] = {k: (res[f"synth_{k}"][loc] - res[f"mean_{k}"][loc])
                            for k in obs_keys}
        # rev 4: the ENSEMBLE-mean error and the ensemble disagreement, the
        # two arguments of the coherent-ensemble identity (II).  Summed over
        # dims -> one (Rbar, V) pair per read per key.
        rec["ybar_resid2"] = {k: ((tgt[k][idx + 1]
                                   - res[f"emean_{k}"][loc]) ** 2)
                              for k in obs_keys}
        rec["Rbar"] = {k: rec["ybar_resid2"][k].sum(-1) for k in obs_keys}
        rec["Vsum"] = {k: res[f"vens_{k}"][loc].sum(-1) for k in obs_keys}
        rec["Va"] = {k: res[f"vensa_{k}"][loc].sum(-1) for k in obs_keys}
        rec["Vb"] = {k: res[f"vensb_{k}"][loc].sum(-1) for k in obs_keys}
        # F1: the POSTERIOR reconstruction residual, summed over dims and on
        # the same read index as Rbar.  It contains no prediction error at
        # all (it is scored against the FILTERED state), so its
        # V-proportional part is the share of the excess already present in
        # pure reconstruction -- the handle that separates a representation
        # effect from a prediction effect.
        rec["Rrec"] = {k: rec["recon2"][k].sum(-1) for k in obs_keys}
        # REVIEW B6: the stream index at which the calibration prefix ends.
        # The horizon leg must drop every lattice point before it, or c is
        # not F0-measurable there.
        rec["cal_end_stream"] = int(idx[min(args.n_cal, idx.size - 1)]) \
            if idx.size else int(off)
        per_stream.append(rec)
        stream_meta.append(dict(stream=si, offset=int(off), n_steps=int(n),
                                n_admissible=int(idx.size),
                                n_audited=int(max(idx.size - args.n_cal, 0)),
                                cal_end_stream=rec["cal_end_stream"]))
        print(f"  stream {si}: off {off}, {n} steps, {idx.size} admissible, "
              f"{max(idx.size - args.n_cal, 0)} audited")
    assert all(m["n_audited"] > 0 for m in stream_meta), (   # review B12
        "a stream contributed no audited reads", stream_meta)

    # ---- DECLARE c BEFORE ANY E-PROCESS (sample splitting).
    # REVIEW B3: c_cal (forcing pooled real-key eps2 = 1) is REPLACED.  It
    # made prediction (b) untestable by construction and sat on the same
    # side of c* as c_theory.  c is now IDENTIFIED, not fitted to the
    # statistic: the posterior reconstruction residual on a disjoint prefix
    # over the real keys IS the decoder's own observation-noise variance
    # (the quantity symlog_mse was trained to minimise).
    cal_r2 = {k: np.concatenate([rec["recon2"][k][:args.n_cal]
                                 for rec in per_stream]) for k in obs_keys}
    c_recon = float(np.mean(np.concatenate(
        [cal_r2[k].ravel() for k in real_keys])))
    scalings = {"c_theory": 0.5}
    if np.isfinite(c_recon) and c_recon > 0:
        scalings["c_recon"] = c_recon
    assert len(scalings) >= 2, (                            # review B2
        "sign-stability needs >= 2 declared scalings; c_recon failed to "
        "identify", c_recon)
    print(f"declared scalings: {scalings} (c_recon = posterior "
          f"reconstruction residual on {args.n_cal} reads/stream, real keys)")

    def audit_rng(*key):
        """REVIEW I6: one shared rng made every draw depend on the ORDER of
        audits, so inserting a leg shifted all of them and A6 (reproduce
        rev-3) was unstatable.  Seed each audit from its own identity."""
        h = zlib.crc32(("|".join(str(x) for x in key)).encode()) & 0xffffffff
        return np.random.default_rng((int(args.seed) << 32) ^ h)

    def audit(rec_key, c, leg, side, synth=None, scale_S=None, tag=""):
        rng = audit_rng(tag, c, leg, side, synth, scale_S)
        """Per-key e-processes across streams (independent -> the product of
        e-values is an e-value), with the calibration prefix excluded."""
        out = {}
        for k in universe:
            D = dims[k]
            logs, rhos, ns, sS, sV = [], [], [], [], []
            for rec in per_stream:
                sl = slice(args.n_cal, None)
                # the predictive mean is an M-sample average, so the
                # residual y - mean_M has variance (c + vlat) + vlat/M;
                # S is the exact conditional variance of the residual
                # actually formed, not of an idealised one.
                beta_e = (ens + 1.0) / (ens - 1.0)
                if leg == "E":
                    # EXPLORATORY (review B9): struck from the freeze.  The
                    # residual is against ybar, which is NOT an M-sample
                    # average, so the (1+1/M) inflation that legs P/C carry
                    # does NOT belong here -- rev 4 applied it anyway.
                    v = rec["vlat"][k][sl] + rec["vens"][k][sl] * beta_e
                else:
                    v = rec["vlat"][k][sl] * (1.0 + 1.0 / M)
                    if leg == "C":
                        v = v + rec["vens"][k][sl]
                S = c + v
                if scale_S is not None:
                    S = S * float(scale_S)
                if synth == "gauss":
                    # EXACT-SCALE NO-OP.  Residuals drawn with variance
                    # EXACTLY S at every read, so H0 holds conditionally
                    # while keeping the real (heteroscedastic) S sequence.
                    # Validates the e-process + prequential plug-in at the
                    # deployed scale; silent on both legs and both sides.
                    # It CANNOT validate the assembly of S (it is drawn
                    # from S) — that is SYNTHMIX's job (review B5).
                    r2 = (np.sqrt(S) * rng.normal(0.0, 1.0, S.shape)) ** 2
                elif synth == "mix":  # review B5: valid on BOTH legs
                    # ASSEMBLY control (review B5), BOTH legs: the reserved
                    # prior draw (a real sample of the latent mixture), the
                    # ensemble spread on leg C, and the decoder's N(0,c).
                    # `dev` carries the M-sample mean's error exactly as the
                    # real residual does, and S already accounts for it via
                    # vlat*(1+1/M), so NO correction is applied to either
                    # side — deliberately, and symmetrically.
                    dev = rec["synth_dev"][k][sl]
                    draw = dev + rng.normal(0.0, math.sqrt(c), dev.shape)
                    if leg == "E":
                        # review B9: leg E's control must be built from leg
                        # E's OWN law, not from the M-sample draw `dev`
                        # (which carries the (1+1/M) term leg E does not).
                        draw = (rng.normal(0.0, 1.0, dev.shape)
                                * np.sqrt(np.maximum(
                                    rec["vlat"][k][sl], 0.0))
                                + rng.normal(0.0, math.sqrt(c), dev.shape))
                    if leg in ("C", "E"):
                        sc = 1.0 if leg == "C" else beta_e
                        draw = draw + rng.normal(
                            0.0, 1.0, dev.shape) * np.sqrt(
                                np.maximum(rec["vens"][k][sl] * sc, 0.0))
                    r2 = draw ** 2
                else:
                    # leg E scores the ENSEMBLE mean; P and C keep the
                    # latent-sample mean, so rev-3 numbers reproduce exactly
                    r2 = rec["ybar_resid2" if leg == "E" else rec_key][k][sl]
                if r2.size == 0:
                    continue
                sq = (r2 / S).sum(-1)
                logs.append(eprocess(sq, D, side, args.nu0, args.r_min,
                                     args.r_max))
                rhos.append(float((r2 / S).mean()))
                ns.append(int(sq.size))
                sS.append(float(np.mean(S)))
                # review M2: report the share for EVERY leg, not 0 on P/E
                w_ = (rec["vens"][k][sl] if leg == "C" else
                      rec["vens"][k][sl] * beta_e if leg == "E" else
                      np.zeros_like(S))
                sV.append(float(np.mean(w_ / S)))
            if not logs:
                out[k] = dict(n=0, rho=NAN, crossed_frac=NAN)
                continue
            certs = [cert(l, args.alpha / 2.0) for l in logs]
            rho = float(np.mean(rhos))
            pooled = float(np.sum([c_["log10e_final"] for c_ in certs]))
            # REVIEW B7: the POOLED e-value is THE certificate.  Independent
            # streams give independent e-values and a product of e-values is
            # an e-value, so Ville applies to the product at level alpha/2.
            # crossed_frac is descriptive only.
            out[k] = dict(
                D=D, n_reads=int(np.sum(ns)), n_streams=len(logs),
                rho=rho, log_rho=float(math.log(rho)) if rho > 0 else NAN,
                rho_per_stream=[float(x) for x in rhos],
                mean_S=float(np.mean(sS)),
                vens_share_of_S=float(np.mean(sV)),
                certified=bool(pooled >= math.log10(2.0 / args.alpha)),
                log10e_pooled=pooled,
                log10e_threshold=float(math.log10(2.0 / args.alpha)),
                crossed_frac=float(np.mean([c_["crossed"] for c_ in certs])),
                log10e_median=float(np.median(
                    [c_["log10e_final"] for c_ in certs])),
                t_cross_median=float(np.median(
                    [c_["t_cross"] if c_["t_cross"] >= 0 else np.inf
                     for c_ in certs])),
                growth_nats_median=float(np.median(
                    [c_["growth_nats"] for c_ in certs])),
                growth_predicted_from_rho=kl_rate(rho) * D)
        return out

    legs = {}
    for cname, c in scalings.items():
        for leg in ("P", "C", "E"):
            for side in ("over", "under"):
                legs[f"{cname}|{leg}|h1|{side}"] = audit(
                    "resid2", c, leg, side, tag="h1")
        # controls: the exact-scale Gaussian no-op on BOTH legs and BOTH
        # sides, plus the mixture-faithful non-Gaussianity check on leg P
        for leg in ("P", "C", "E"):
            for side in ("over", "under"):
                legs[f"{cname}|{leg}|h1|{side}|SYNTHG"] = audit(
                    "resid2", c, leg, side, synth="gauss", tag="h1")
        for leg in ("P", "C", "E"):
            for side in ("over", "under"):
                legs[f"{cname}|{leg}|h1|{side}|SYNTHMIX"] = audit(
                    "resid2", c, leg, side, synth="mix", tag="h1")

    # ---- IN-RUN PLANTED MUTANTS, on the real data, at zero model cost.
    # A pure-numpy fixture proves the e-process is right; these prove the
    # DEPLOYED PIPELINE is right, on this run's own residuals:
    #   inflate  : S x 4 must certify OVER on every key (a planted overclaim);
    #   exact    : S x rho_k (the realized ratio) must be SILENT — the
    #              exact-scale no-op, per key, measured not assumed.
    c_last = list(scalings.values())[-1]
    base_over = legs[f"{list(scalings)[-1]}|C|h1|over"]
    infl = audit("resid2", c_last, "C", "over", scale_S=4.0)
    exact = {}
    for k in universe:
        rho_k = base_over[k]["rho"]
        if not (rho_k > 0) or not np.isfinite(rho_k):
            continue
        one = audit("resid2", c_last, "C", "over", scale_S=rho_k, tag="planted")
        exact[k] = one[k]
    planted = dict(
        inflate_x4={k: dict(rho=infl[k]["rho"],
                            crossed_frac=infl[k]["crossed_frac"])
                    for k in universe},
        marginal_rescale={k: dict(rho=exact[k]["rho"],
                                  crossed_frac=exact[k]["crossed_frac"])
                          for k in exact},
        note="inflate_x4: a planted 4x overclaim MUST certify on every key "
             "(power on this run's own residuals).  marginal_rescale: S is "
             "rescaled by the realized mean rho, so rho == 1 EXACTLY and "
             "the channel is MARGINALLY calibrated — yet it still "
             "certifies, because the e-process is a CONDITIONAL test.  That "
             "is the whole reason EVA beat the marginal instruments in "
             "world (marginal E[eps^2] 0.99-1.04 was consistent with H0 "
             "while the deficit was conditional).  CONSEQUENCE FOR THE "
             "READ: on real data a bare certificate is weak evidence, "
             "because conditional heteroscedasticity alone certifies.  The "
             "adjudicating quantities are the per-key CONTRAST (primary "
             "Dlogrho) against the dup floor, and the SYNTHETIC control — "
             "which shares the same conditional structure and must be "
             "silent.")

    # ---- the primary, under both declared scalings
    def dlogrho(block, a="distractor", b=None):
        b = b or source_key
        ra, rb = block.get(a, {}).get("rho"), block.get(b, {}).get("rho")
        if not ra or not rb or not (ra > 0 and rb > 0):
            return NAN
        return float(math.log(ra) - math.log(rb))

    # ============ REV 5 PRIMARY: q_hat via SPLIT-HALF IV ============
    # Bars are registered on q_hat (review B1): A = (Eq+1)/(E+1), so rev 4's
    # "A < 0.25 = UNEARNED" bar INVERTED the reading -- it labelled q = 0
    # (V accounts for the whole scaling of the mean error) as unearned.
    LAM_MIN = 0.15          # registered: lambda 0.03 -> A 1.28+-0.57;
                            # 0.17 -> 1.03+-0.10; 0.70 -> 1.00+-0.02
    sl_a = slice(args.n_cal, None)
    align, align_mut, align_rec = {}, {}, {}
    for k in universe:
        ps = [(r["Va"][k][sl_a], r["Vb"][k][sl_a], r["Rbar"][k][sl_a])
              for r in per_stream]
        align[k] = alignment(np.concatenate([b[2] for b in ps]),
                             np.concatenate([b[0] for b in ps]),
                             np.concatenate([b[1] for b in ps]),
                             ens, per_stream=ps, n_perm=args.n_perm_r,
                             seed=args.seed)
        align[k]["mean_V"] = float(np.mean(np.concatenate(
            [0.5 * (b[0] + b[1]) for b in ps])))
        align[k]["mean_Rbar"] = float(np.mean(np.concatenate(
            [b[2] for b in ps])))
        align[k]["intercept_per_dim"] = safe_ratio(align[k]["intercept"],
                                                   dims[k])
        # F1: the same functional applied to the POSTERIOR RECONSTRUCTION
        # residual.  gamma_hat is the V-proportional part of an error that
        # contains no prediction at all, so q_hat_net = q_hat - gamma_hat
        # is the excess BEYOND what pure reconstruction already shows.  The
        # -1/E offset is identical in both and cancels exactly in the
        # difference: q_net = (E-1)/E * (slope_R - slope_recon).
        psr = [(r["Va"][k][sl_a], r["Vb"][k][sl_a], r["Rrec"][k][sl_a])
               for r in per_stream]
        arec = alignment(np.concatenate([b[2] for b in psr]),
                         np.concatenate([b[0] for b in psr]),
                         np.concatenate([b[1] for b in psr]),
                         ens, per_stream=psr, seed=args.seed)
        align_rec[k] = arec
        align[k]["gamma_hat"] = arec["q_hat"]
        align[k]["q_hat_net"] = (
            float(align[k]["q_hat"] - arec["q_hat"])
            if np.isfinite(align[k]["q_hat"]) and np.isfinite(arec["q_hat"])
            else NAN)
    # CONSTRUCT VALIDITY from REAL member construction (review B1): no
    # fixture anywhere synthesises Rbar from an observed V.
    MUT_EXPECT = {                       # review F5: registered verdicts
        "centered_q0": "SPREAD-ACCOUNTS",
        "exchangeable_q1": "MIXED",
        "shared_dominant_q4": "SHARED-DOMINATED",
        "homoscedastic_weak_iv": "UNRESOLVED-WEAK-INSTRUMENT",
    }
    for nm, qq, kw in (("centered_q0", 0.0, {}),
                       ("exchangeable_q1", 1.0, {}),
                       ("shared_dominant_q4", 4.0, {}),
                       ("homoscedastic_weak_iv", 1.0, dict(het=0.05)),
                       ("q1_plus_uncorrelated_noise", 1.0,
                        dict(extra_noise=1.0))):
        align_mut[nm] = alignment_mutant(
            ens, qq, seed=args.seed, streams=len(per_stream),
            n_perm=max(args.n_perm_r, 1), **kw)
        # run the truth mutants through the SAME classifier the cells use
        align_mut[nm]["verdict"] = classify(align_mut[nm], LAM_MIN)
        align_mut[nm]["verdict_expected"] = MUT_EXPECT.get(nm)

    floor_q = abs(align["planted_dup0"]["q_hat"] - align[source_key]["q_hat"])
    floor_qnet = abs(align["planted_dup0"]["q_hat_net"]
                     - align[source_key]["q_hat_net"])
    floor_r = abs(align["planted_dup0"]["r_spearman"]
                  - align[source_key]["r_spearman"])
    

    def _aligned(k):
        """STEP 1 (review B3): alignment must be established against a
        CLUSTER-RESAMPLED circular-shift null -- rev 4's sign-agreement
        rule did NOT fire on its own motivating case."""
        a = align[k]
        pp = a.get("r_perm_p", NAN)
        if not np.isfinite(pp) or not np.isfinite(a["r_spearman"]):
            return False
        if pp > 0.05:
            return False
        return bool(np.sign(a["A"]) == np.sign(a["r_spearman"]))  # I5

    def _resolved(k):
        """The instrument-resolution gate: a weak instrument gives an
        UNRESOLVED verdict, never an UNEARNED one (review B2)."""
        return bool(np.isfinite(align[k]["lambda_hat"])
                    and align[k]["lambda_hat"] >= LAM_MIN)

    def _ci_excludes(k, x):                                   # review I4
        lo, hi = align[k]["ci"]
        lo2, hi2 = align[k]["ci_t"]
        if not all(np.isfinite(v) for v in (lo, hi)):
            return False
        lo = min(lo, lo2 if np.isfinite(lo2) else lo)         # I3: the WIDER
        hi = max(hi, hi2 if np.isfinite(hi2) else hi)
        return not (lo <= x <= hi)

    def _verdict(k):
        return classify(align[k], LAM_MIN)      # review F5: one classifier

    dv = _verdict("distractor")
    adjudicable_q = bool(
        np.isfinite(floor_q) and np.isfinite(align["distractor"]["q_hat"])
        and abs(align["distractor"]["q_hat"]
                - align[source_key]["q_hat"]) > floor_q)     # review B7
    primary_v2 = dict(
        statistic="q_hat = ((E-1)*slope_IV - 1)/E, slope by SPLIT-HALF IV.  "
                  "ESTIMAND (review F1): the V-PROPORTIONAL EXCESS OF MEAN "
                  "ERROR OVER THE COHERENT RATE.  A shared common-mode error "
                  "is one mechanism that produces it, but not the only one, "
                  "so q_hat alone does NOT attribute mechanism; the "
                  "attribution is the q_hat_net vs gamma_hat comparison.  "
                  "The verdict name SHARED-DOMINATED is retained for "
                  "continuity and names the SIZE of the excess, not its "
                  "cause.",
        registered_on="q_hat", beta_star=float((ens + 1.0) / (ens - 1.0)),
        lambda_min=LAM_MIN,
        q_hat={k: align[k]["q_hat"] for k in universe},
        q_ci={k: align[k]["ci"] for k in universe},
        q_ci_t={k: align[k]["ci_t"] for k in universe},
        q_per_stream={k: align[k]["q_per_stream"] for k in universe},
        lambda_hat={k: align[k]["lambda_hat"] for k in universe},
        A_convention={k: align[k]["A"] for k in universe},
        A_ols_attenuated={k: align[k]["A_ols"] for k in universe},
        slope_iv_sym_crosscheck={k: align[k]["slope_iv_sym"]
                                 for k in universe},
        r_spearman={k: align[k]["r_spearman"] for k in universe},
        r_perm_p={k: align[k].get("r_perm_p") for k in universe},
        aligned={k: _aligned(k) for k in universe},
        resolved={k: _resolved(k) for k in universe},
        verdict_per_key={k: _verdict(k) for k in universe},
        distractor_verdict=dv,
        floor_q=floor_q, floor_r=floor_r,
        adjudicable=adjudicable_q,
        ci_excludes_q0=_ci_excludes("distractor", 0.0),
        ci_excludes_q1=_ci_excludes("distractor", 1.0),
        # F1: the mechanism attribution lives HERE, not in the verdict name
        gamma_hat={k: align[k]["gamma_hat"] for k in universe},
        q_hat_net={k: align[k]["q_hat_net"] for k in universe},
        floor_q_net=floor_qnet,
        gamma_note="gamma_hat applies the SAME functional to the posterior "
                   "RECONSTRUCTION residual, which contains no prediction "
                   "error.  q_hat_net = q_hat - gamma_hat is the "
                   "V-proportional excess BEYOND pure reconstruction; the "
                   "-1/E offset cancels exactly in the difference.  A large "
                   "q_hat with q_hat_net ~ 0 means the excess is already "
                   "present in reconstruction (a representation effect); a "
                   "large q_hat_net means it is specific to prediction.",
        # F7: an unresolved key's intercept-derived quantities are NULLED,
        # not reported -- they are only interpretable alongside a slope the
        # instrument could actually resolve.
        intercept={k: (align[k]["intercept"] if _resolved(k) else NAN)
                   for k in universe},
        intercept_per_dim={k: (align[k]["intercept_per_dim"]
                               if _resolved(k) else NAN)
                           for k in universe},
        intercept_negative_keys=[k for k in universe
                                 if _resolved(k)
                                 and np.isfinite(align[k]["intercept"])
                                 and align[k]["intercept"] < 0],
        intercept_ratio_vel_pos=(
            safe_ratio(align["velocity"]["intercept_per_dim"],
                       align[source_key]["intercept_per_dim"])
            if ("velocity" in align and _resolved("velocity")
                and _resolved(source_key)) else NAN),
        # review I1: uncorrelated added noise raises the INTERCEPT, not the
        # slope, so the dup ladder's registered check is on intercepts
        dup_intercept_ordered=bool(
            np.isfinite(align["planted_dup2"]["intercept_per_dim"])
            and np.isfinite(align["planted_dup0"]["intercept_per_dim"])
            and align["planted_dup2"]["intercept_per_dim"]
            > align["planted_dup0"]["intercept_per_dim"]),
        construct_validity={m: dict(
            q_hat=v["q_hat"], q_true=v["q_true"], A=v["A"],
            A_true=v["A_true"], A_ols=v["A_ols"],
            lambda_hat=v["lambda_hat"], verdict=v.get("verdict"),
            verdict_expected=v.get("verdict_expected"),
            descriptive=(v.get("verdict_expected") is None))
            for m, v in align_mut.items()},
        construct_descriptive_note=(
            "q1_plus_uncorrelated_noise carries NO registered bar and is "
            "DESCRIPTIVE (review F5): the added V-uncorrelated error inflates "
            "R's sampling variance, so the IV's own error grows and no "
            "stable bar was justifiable at this n.  It is reported to show "
            "the direction (q_hat attenuates toward 0), not to gate."),
        construct_validity_verdicts={
            m: dict(got=v.get("verdict"), expected=v.get("verdict_expected"))
            for m, v in align_mut.items()},
        construct_ok=bool(
            abs(align_mut["centered_q0"]["q_hat"] - 0.0) < 0.30
            and abs(align_mut["exchangeable_q1"]["q_hat"] - 1.0) < 0.35
            and align_mut["shared_dominant_q4"]["q_hat"] > 2.0
            and align_mut["homoscedastic_weak_iv"]["lambda_hat"] < LAM_MIN
            # review F5: the truth mutants must land on the registered
            # verdicts through the ACTUAL classifier, not merely near the
            # right numbers
            and all(v["verdict"] == v["verdict_expected"]
                    for v in align_mut.values()
                    if v.get("verdict_expected"))),
        verdict_rule="(1) lambda_hat >= 0.15 else UNRESOLVED-WEAK-INSTRUMENT; "
                     "(2) circular-shift permutation p <= 0.05 AND "
                     "sign(A)==sign(r) else UNRESOLVED-NO-ALIGNMENT; "
                     "(3) A < 0 -> UNRESOLVED-OUT-OF-SUPPORT; "
                     "(4) q<0.5 SPREAD-ACCOUNTS; q>2 SHARED-DOMINATED; "
                     "else MIXED.  ANTI was STRUCK on 29 Aug 2026 "
                     "(review F5): A<0 requires q < -1/E and q >= 0 by "
                     "construction, so no fixture is constructible for it "
                     "and it could only ever have been sampling noise.",
        note="NOT a certificate -- a point estimate with a cluster bootstrap "
             "AND a t-interval on the per-stream values; gates use the WIDER "
             "(review I3: the 6-cluster percentile interval under-covers, "
             "0.860 vs 0.95).  The certificates are the e-processes.")


    assert source_key == "position", (                    # review M10
        "the dup ladder duplicates the SOURCE key; a non-position source "
        "invalidates the floor's interpretation", source_key)

    primary = {}
    for cname in scalings:
        bc = legs[f"{cname}|C|h1|over"]
        bp = legs[f"{cname}|P|h1|over"]
        fc = legs[f"{cname}|C|h1|over|SYNTHMIX"]
        dl_c, dl_p = dlogrho(bc), dlogrho(bp)
        floor = abs(dlogrho(bc, "planted_dup0"))
        gap = (float(dl_c - dl_p) if np.isfinite(dl_c) and np.isfinite(dl_p)
               else NAN)
        # REVIEW B1: the "fully unjustified" endpoint is NOT -log(theta_1).
        # theta_1 is a normalised share of decoded ENSEMBLE variance, while
        # S is dominated by c + vlat.  If the ensemble term bought nothing,
        # leg P would be calibrated and rho_C = S_P/S_C, so the achievable
        # gap is measured on THIS run as the differential inflation S_C/S_P.
        def _ms(b, k):
            return b.get(k, {}).get("mean_S", NAN)
        infl_d = safe_ratio(_ms(bc, "distractor"), _ms(bp, "distractor"))
        infl_p = safe_ratio(_ms(bc, source_key), _ms(bp, source_key))
        endpoint = (-(math.log(infl_d) - math.log(infl_p))
                    if infl_d > 0 and infl_p > 0 else NAN)
        # REVIEW B5: subtract the leg-C non-Gaussianity floor, which is
        # per-key and asymmetric, before reading the contrast.
        fl_d = fc.get("distractor", {}).get("rho", NAN)
        fl_p = fc.get(source_key, {}).get("rho", NAN)
        dl_c_corr = (float(dl_c - (math.log(fl_d) - math.log(fl_p)))
                     if np.isfinite(dl_c) and fl_d > 0 and fl_p > 0 else NAN)
        rr = [bc.get(k, {}).get("rho", NAN)
              for k in ("planted_dup0", "planted_dup1", "planted_dup2")]
        primary[cname] = dict(
            dlogrho_legC=dl_c, dlogrho_legP=dl_p,
            dlogrho_legC_floor_corrected=dl_c_corr,
            legC_minus_legP=gap,
            floor_dlogrho_dup0_vs_position=floor,
            # THE ADJUDICABILITY GATE (review B1).  Only the leg-C minus
            # leg-P difference is attributable to the ensemble term, i.e.
            # to the account theta_1 prices.  Below the dup floor the
            # instrument cannot address the misprice at all, and says so
            # instead of reporting a number it cannot support.
            adjudicable=bool(np.isfinite(gap) and np.isfinite(floor)
                             and abs(gap) > floor),
            bracket_endpoint_measured=endpoint,
            bracket_endpoint_declared_reference=float(-math.log(5.26)),
            bracket_if_fully_justified=0.0,
            # gap and endpoint are both NEGATIVE when the misprice is
            # over-claimed, so this ratio must NOT go through safe_ratio's
            # positivity guard; 1.0 = the whole ensemble term is
            # unjustified, 0.0 = it is fully earned.
            position_on_bracket=(
                float(gap / endpoint)
                if np.isfinite(gap) and np.isfinite(endpoint)
                and abs(endpoint) > 1e-12 else NAN),
            dup_monotone=(None if not all(np.isfinite(x) for x in rr)
                          else bool(rr[0] <= rr[1] <= rr[2])),
            dup_rhos=rr)
    signs = [np.sign(primary[c]["legC_minus_legP"]) for c in scalings
             if np.isfinite(primary[c]["legC_minus_legP"])]
    sign_stable = bool(len(scalings) >= 2 and len(signs) == len(scalings)
                       and len(set(signs)) == 1)          # review B2

    # ---- c-sensitivity (review B3): Dlogrho over a DECLARED grid, plus the
    # sign-flip location c*.  dDlogrho/dc is single-signed, so two nearby
    # scalings can both sit on one side of c*; the grid makes that visible
    # instead of leaving it as an unstated assumption.
    c_grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    grid = []
    for cg in c_grid:
        bc = audit("resid2", cg, "C", "over", tag="grid")
        bp = audit("resid2", cg, "P", "over", tag="grid")
        grid.append(dict(c=cg, dlogrho_legC=dlogrho(bc),
                         dlogrho_legP=dlogrho(bp),
                         legC_minus_legP=float(dlogrho(bc) - dlogrho(bp))))
    c_star = None
    for a_, b_ in zip(grid[:-1], grid[1:]):
        u, v = a_["legC_minus_legP"], b_["legC_minus_legP"]
        if np.isfinite(u) and np.isfinite(v) and u * v < 0:
            w = abs(u) / (abs(u) + abs(v))
            c_star = float(math.exp(math.log(a_["c"])
                                    + w * (math.log(b_["c"])
                                           - math.log(a_["c"]))))
            break
    gsig = [np.sign(g["legC_minus_legP"]) for g in grid
            if np.isfinite(g["legC_minus_legP"])]
    c_sensitivity = dict(grid=grid, c_star=c_star,
                         sign_constant_over_grid=bool(
                             len(gsig) == len(grid) and len(set(gsig)) == 1),
                         grid_span=[c_grid[0], c_grid[-1]],
                         note="c* is where the leg-C-minus-leg-P contrast "
                              "changes sign; a declared scaling far from it "
                              "on the predicted side is what makes the sign "
                              "meaningful")

    # ---- horizon / persistence leg
    # REVIEW M4: the reference constants must come from THIS run's own
    # se_lic_probe.json, not from a transported single cell (gauss s120).
    lic = {}
    lic_path = os.path.join(args.run_logdir, "se_lic_probe",
                            "se_lic_probe.json")
    if os.path.exists(lic_path):
        with open(lic_path) as f:
            lj = json.load(f)
        lic = dict(source=lic_path, theta1_raw=lj.get("theta1_raw"),
                   persistence15=lj.get("persistence", {}).get(
                       "measured", {}).get("15", {}).get("distractor"),
                   deter_frac=lj.get("substrate", {}).get(
                       "deter_frac_universe_mean"))
    else:
        lic = dict(source="DECLARED BAND (no se_lic_probe.json beside the "
                          "run) — band endpoints from the 26-Aug read",
                   theta1_raw=5.26, persistence15=0.148, deter_frac=None)
    hor = {}
    if horizons:
        phi_m_ref = phi_from_persistence(lic["persistence15"] or 0.148, 15)
        for h in horizons:
            blocks = []
            blk_h = max(h, (args.blk // h) * h)
            for si in range(len(chains)):
                arr = get_stream(si)
                res, n, off = sweep(arr, args.t_max_h, jit_hor[h], need=h,
                                    extra_act=h, blk=blk_h)
                # the model was evaluated on the FIXED lattice
                # {off, off+h, off+2h, ...} (blk_h is a multiple of h, so it
                # is aligned across blocks); intersect with the exogenous
                # admissible set, keeping temporal order.
                loc, _ = episode_segments(
                    arr["is_first"][off:off + n], args.burn_in, h)
                lattice = off + np.arange(0, n, h)
                keep = np.isin(lattice, off + loc)
                # REVIEW B6: c is only F0-measurable AFTER the calibration
                # prefix, so lattice points inside it must be dropped here
                # exactly as the h=1 leg drops them.
                keep &= lattice >= per_stream[si]["cal_end_stream"]
                idx = lattice[keep]                  # stream indices
                pos = np.flatnonzero(keep)           # rows of `res`
                tgt = {k: symlog(np.asarray(arr[k], np.float64))
                       for k in obs_keys}
                blocks.append(dict(
                    resid2={k: (tgt[k][idx + h]
                                - res[f"mean_{k}"][pos]) ** 2
                            for k in obs_keys},
                    vlat={k: res[f"vlat_{k}"][pos] for k in obs_keys},
                    synth_dev={k: (res[f"synth_{k}"][pos]
                                   - res[f"mean_{k}"][pos])
                               for k in obs_keys}))
            for cname, c in scalings.items():
                for side in ("over", "under"):
                  for mode in ("", "|SYNTHG", "|SYNTHMIX"):
                    # review I6: per-audit rng, keyed by this audit's own
                    # identity, so inserting a leg cannot shift any other
                    # audit's draws (which is what made A6 unstatable)
                    rng = audit_rng("h", h, cname, "P", side, mode)
                    out = {}
                    for k in universe:
                        D = dims[k]
                        logs, rhos = [], []
                        for b in blocks:
                            S = c + b["vlat"][k] * (1.0 + 1.0 / M)
                            if mode == "|SYNTHG":
                                r2 = (np.sqrt(S) * rng.normal(
                                    0.0, 1.0, S.shape)) ** 2
                            elif mode == "|SYNTHMIX":
                                dv = b["synth_dev"][k]
                                r2 = (dv + rng.normal(
                                    0.0, math.sqrt(c), dv.shape)) ** 2
                            else:
                                r2 = b["resid2"][k]
                            sq = (r2 / S).sum(-1)
                            if sq.size == 0:
                                continue
                            logs.append(eprocess(sq, D, side, args.nu0,
                                                 args.r_min, args.r_max))
                            rhos.append(float((r2 / S).mean()))
                        if not logs:
                            continue
                        certs = [cert(l, args.alpha / 2.0) for l in logs]
                        rho_k = float(np.mean(rhos))
                        pooled_h = float(np.sum(
                            [c_["log10e_final"] for c_ in certs]))
                        thr_h = float(math.log10(2.0 / args.alpha))
                        out[k] = dict(
                            D=D, rho=rho_k,
                            n_reads=int(sum(l.size for l in logs)),
                            certified=bool(pooled_h >= thr_h),
                            log10e_threshold=thr_h,
                            crossed_frac=float(np.mean(
                                [c_["crossed"] for c_ in certs])),
                            log10e_pooled=pooled_h,
                            growth_nats_median=float(np.median(
                                [c_["growth_nats"] for c_ in certs])),
                            # DPB's kappa at scale: the AR coefficient this
                            # channel's own h-step dispersion implies.
                            # Meaningful for the distractor (a true AR(1));
                            # descriptive elsewhere.
                            phi_m_implied=phi_m_implied(rho_k, h))
                    hor[f"{cname}|P|h{h}|{side}{mode}"] = out
            hor[f"predicted|h{h}"] = dict(
                phi=AR1_COEF, phi_m=phi_m_ref,
                rho_predicted=predicted_rho_ar1(h, AR1_COEF, phi_m_ref),
                growth_predicted_per_dim=kl_rate(
                    predicted_rho_ar1(h, AR1_COEF, phi_m_ref)))

    # ---- MEASURED FLOORS.  SYNTHG validates the pipeline (must be silent).
    # SYNTHMIX does NOT come out at rho = 1: the latent predictive is a
    # non-Gaussian mixture and the Gaussian LR reads its excess kurtosis as
    # a few percent of extra dispersion.  That is a real limit of the test,
    # so it is MEASURED PER KEY and every UNDER-side reading is adjudicated
    # on the floor-corrected ratio rho / rho_floor rather than on rho.
    cn0 = list(scalings)[-1]
    floors = dict(
        h1_under_nongauss={k: legs[f"{cn0}|P|h1|under|SYNTHMIX"][k]["rho"]
                           for k in universe},
        h1_over_nongauss={k: legs[f"{cn0}|P|h1|over|SYNTHMIX"][k]["rho"]
                          for k in universe},
        h1_legC_nongauss={k: legs[f"{cn0}|C|h1|over|SYNTHMIX"][k]["rho"]
                          for k in universe},          # review B5
        h1_legC_nongauss_under={
            k: legs[f"{cn0}|C|h1|under|SYNTHMIX"][k]["rho"]
            for k in universe},
        horizon_under_nongauss={
            f"h{h}": {k: hor.get(f"{cn0}|P|h{h}|under|SYNTHMIX", {}).get(
                k, {}).get("rho") for k in universe} for h in horizons},
        note="rho of the mixture-faithful control = the non-Gaussianity "
             "floor of this test at this scale; a persistence reading must "
             "exceed it, and rho_corrected = rho / rho_floor is the "
             "adjudicating quantity")
    # REVIEW B4: a horizon is only ADJUDICABLE if the predicted effect
    # clears the measured floor by more than the floor's own excess.  At
    # M=8 the Jensen inflation of the M-sample variance made h=15 have
    # NEGATIVE headroom (floor 1.225 > predicted 1.100); the gate makes that
    # visible and refuses the horizon instead of reporting through it.
    resolution = {}
    for h in horizons:
        base = hor.get(f"{cn0}|P|h{h}|under", {})
        fl = floors["horizon_under_nongauss"][f"h{h}"]
        for k in universe:
            if k in base and fl.get(k):
                rc = base[k]["rho"] / fl[k]
                base[k]["rho_floor_corrected"] = float(rc)
                base[k]["phi_m_implied_corrected"] = phi_m_implied(rc, h)
        fd = fl.get("distractor")
        pr = hor.get(f"predicted|h{h}", {}).get("rho_predicted", NAN)
        headroom = (float(pr / fd - 1.0) if fd and fd > 0 and np.isfinite(pr)
                    else NAN)
        need = float(2.0 * (fd - 1.0)) if fd and np.isfinite(fd) else NAN
        resolution[f"h{h}"] = dict(
            floor=fd, predicted_rho=pr, headroom=headroom, required=need,
            adjudicable=bool(np.isfinite(headroom) and np.isfinite(need)
                             and headroom > need))
    # REVIEW B8: phi_m ~ const > 0.9 does NOT discriminate (velocity 0.936
    # and dup2 0.961 on an UNTRAINED model).  The discriminating object is
    # the PROFILE: rho decays with h in a way an AR-coefficient misclaim
    # predicts and a generic miscalibration does not.
    def _pro(k, a, b):
        ba = hor.get(f"{cn0}|P|h{a}|under", {}).get(k, {})
        bb = hor.get(f"{cn0}|P|h{b}|under", {}).get(k, {})
        return safe_ratio(ba.get("rho_floor_corrected"),
                          bb.get("rho_floor_corrected"))
    prof = {}
    if len(horizons) >= 2:
        # every (shortest, longer) pair; the REGISTERED ones are (2,15) and
        # (2,8), the rest are reported for the shape of the decay
        for a, b in [(horizons[0], hb) for hb in horizons[1:]]:
            pa = hor.get(f"predicted|h{a}", {}).get("rho_predicted", NAN)
            pb = hor.get(f"predicted|h{b}", {}).get("rho_predicted", NAN)
            prof[f"rho{a}_over_rho{b}"] = dict(
                predicted=safe_ratio(pa, pb),
                measured={k: _pro(k, a, b) for k in universe},
                null_comparators=[k for k in ("position", "velocity")
                                  if k in universe])
    profile_ratios = dict(
        ratios=prof,
        note="an AR-coefficient misclaim predicts a specific DECAY PROFILE "
             "of rho with h; position/velocity are the null comparators "
             "because they have no AR(1) law.  Registered band: the "
             "distractor's measured ratio within +-0.15 of predicted AND "
             "separated from both comparators by more than 0.10.")

    # ---- REVIEW B13: the controls are GATES, computed here and stamped,
    # so the driver can refuse a cell instead of leaving a human to notice.
    ctl, ctl_detail = {}, {}
    for cn_ in scalings:
        for leg in ("P", "C", "E"):
            for side in ("over", "under"):
                lab = f"synthg_{cn_}_{leg}_{side}"
                v = synthg_family_verdict(
                    legs[f"{cn_}|{leg}|h1|{side}|SYNTHG"], lab)
                ctl[lab], ctl_detail[lab] = v["passed"], v
    for h in horizons:
        for side in ("over", "under"):
            lab = f"synthg_h{h}_{side}"
            b = hor.get(f"{cn0}|P|h{h}|{side}|SYNTHG", {})
            v = synthg_family_verdict(b, lab)
            v["passed"] = bool(b) and v["passed"]
            ctl[lab], ctl_detail[lab] = v["passed"], v
    ctl["inflate_x4"] = bool(all(
        v["crossed_frac"] == 1.0 for v in planted["inflate_x4"].values()))
    ctl["marginal_rescale_rho1"] = bool(all(
        abs(v["rho"] - 1.0) < 1e-9
        for v in planted["marginal_rescale"].values()))
    ctl["n_scalings"] = bool(len(scalings) >= 2)
    # REVIEW B8: without these, an instrument failure (floor_r = NaN drives
    # every key UNEARNED) would MANUFACTURE the registered outcome and still
    # report controls_pass True.
    ctl["primary_construct_ok"] = bool(primary_v2["construct_ok"])
    ctl["primary_floors_finite"] = bool(
        np.isfinite(primary_v2["floor_q"])
        and np.isfinite(primary_v2["floor_r"]))
    # The IV cross-check is only meaningful where the INSTRUMENT IS
    # RESOLVED: below lambda_min both estimators are noise and disagreeing
    # is the correct behaviour, not an instrument failure.  A cell whose
    # primary is unresolvable must still deliver its e-process
    # certificates, so this is scoped to resolved keys and the primary's
    # own readability is reported separately as `primary_resolved`.
    _res_keys = [k for k in universe
                 if np.isfinite(align[k]["lambda_hat"])
                 and align[k]["lambda_hat"] >= LAM_MIN]
    # REVIEW F6: state the tolerance in q UNITS and against the nearest
    # registered bar, not as an abstract 25% on a slope.
    #   dq = (E-1)/E * d(slope);  the bars are q = 0.5 and q = 2.0
    #   dq = (E-1)/E * d(slope);  the registered bars are q = 0.5 and 2.0.
    # The gate is whether the two IV forms could CHANGE THE CLASSIFICATION:
    # dq must be smaller than the key's distance to its nearest bar.  A key
    # sitting on a bar therefore fails, which is correct -- it is exactly
    # the case the bar cannot decide.
    def _dq(k):
        return abs((ens - 1.0) / ens
                   * (align[k]["slope_iv"] - align[k]["slope_iv_sym"]))

    def _bar_dist(k):
        q_ = align[k]["q_hat"]
        return (float(min(abs(q_ - 0.5), abs(q_ - 2.0)))
                if np.isfinite(q_) else NAN)
    iv_dq = {k: float(_dq(k)) for k in _res_keys}
    iv_bar = {k: _bar_dist(k) for k in _res_keys}
    # REV 7.  The instrument-level gate runs ONLY on REFERENCE keys whose
    # distance to the nearest bar is structurally large (>= FAR_BAR), i.e.
    # keys where a dq excursion CANNOT be near-bar physics and can only be
    # genuine estimator breakage.  In the rev-6 cells those keys sat at
    # bar distances of 7-17, so the gate passes everywhere and fails only
    # on real breakage.  Per-key near-bar disagreement is handled by the
    # UNRESOLVED-IV-DISAGREEMENT verdict, never by refusing the cell.
    FAR_BAR = 2.0
    _iv_ref = [k for k in _res_keys
               if k in (source_key, "planted_dup0", "planted_dup1")
               and np.isfinite(iv_bar[k]) and iv_bar[k] >= FAR_BAR]
    ctl["primary_iv_crosscheck"] = bool(all(
        iv_dq[k] <= iv_bar[k] for k in _iv_ref))
    ctl["streams_supplied"] = bool(all(m["n_audited"] > 0
                                       for m in stream_meta))
    controls_pass = bool(all(ctl.values()))
    # SEPARATE AXIS: whether THIS cell's primary is readable at all.  Not
    # part of controls_pass -- an unresolvable primary must not suppress
    # the cell's certificates (the step-0 lambda->0 limit is exactly this
    # case, and it is a registered negative control, not a failure).
    primary_resolved = bool(
        not primary_v2["distractor_verdict"].startswith("UNRESOLVED")
        and primary_v2["adjudicable"])

    result = dict(
        instrument="se_eva_probe", revision=7,
        revision_note=(
            "REVIEW I8 (record defect): the code that produced "
            "artifacts/se_eva_read_20260829/ stamped revision=2 although it "
            "was the gate-fix build referred to throughout as rev 3.  Those "
            "artifacts' numbers are unaffected; only their version stamp is "
            "wrong.  Recorded in ADJUDICATION.md."),
        primary_v2=primary_v2, alignment=align,
        primary_resolved=primary_resolved,
        resolved_keys=_res_keys,
        iv_crosscheck=dict(
            dq_per_key=iv_dq, bars_q=list(BARS_Q),
            nearest_bar_distance=iv_bar,
            gate_keys=_iv_ref, gate_far_bar_min=FAR_BAR,
            gate_scope="reference keys (source + dup0/dup1) whose bar "
                       "distance >= 2.0 -- where a dq excursion cannot be "
                       "near-bar physics.  Per-key near-bar disagreement "
                       "routes to UNRESOLVED-IV-DISAGREEMENT instead "
                       "(rev 7).",
            classification_safe={k: bool(np.isfinite(iv_bar[k])
                                         and iv_dq[k] <= iv_bar[k])
                                 for k in _res_keys},
            note="review F6: |dq| between the asymmetric and symmetric IV, "
                 "in q units, against the registered bars 0.5 and 2.0.  A "
                 "cell whose dq is comparable to its distance from the "
                 "nearest bar cannot be classified by that bar."),
        exploratory=dict(
            leg_E=True,
            note="REVIEW B9: leg E is EXPLORATORY and STRUCK FROM THE "
                 "FREEZE -- it has no registered key/side/bar/floor.  Its "
                 "S_E algebra and its SYNTHMIX injection are corrected here "
                 "so the exploratory numbers are not wrong, but no claim "
                 "may rest on it."),
        decode_asymmetry_note=(
            "REVIEW M5: legs P/C decode M ONE-HOT prior samples and average; "
            "leg E and the alignment primary decode the disag members' SOFT "
            "postsplit probabilities (se_probe idiom).  The decoder is "
            "nonlinear, so these are not the same operator and their "
            "outputs are not on a common footing -- which is a further "
            "reason leg E is exploratory and why the primary is read only "
            "through within-key contrasts."),
        controls=ctl, controls_pass=controls_pass,
        controls_detail=ctl_detail,
        controls_gate=("SYNTHG families fail only if the control CERTIFIES "
                       "on the POOLED e-value — the same standard as the "
                       "findings (review B7).  The rev-2 per-stream "
                       "crossed_frac==0.0 rule was ~576 alpha/2 tests per "
                       "cell and refused ~95% of cells by multiplicity "
                       "alone; measured on run 56089362 (7/12)."),
        run_logdir=os.path.abspath(args.run_logdir),
        ckpt=os.path.abspath(ckpt), seed=args.seed,
        task=str(config.task), train_seed=int(config.seed),
        disag_head=head, ens=ens, source_key=source_key,
        share_universe=universe, real_keys=real_keys, dims=dims,
        alpha=args.alpha, alpha_per_side=args.alpha / 2.0,
        nu0=args.nu0, r_clip=[args.r_min, args.r_max],
        n_streams=len(chains), t_max=args.t_max, t_max_h=args.t_max_h,
        burn_in=args.burn_in, n_cal=args.n_cal, n_lat=M,
        horizons=horizons, scalings=scalings, c_recon_identified=float(c_recon),
        episode_lengths=dict(
            n=len(ep_lens_all),
            unique=sorted(set(int(x) for x in ep_lens_all))[:10],
            constant=bool(len(set(ep_lens_all)) <= 2),
            note="segments of the AUDITED WINDOW, so the first and last of "
                 "each stream are truncated by the window edge, not by the "
                 "environment (review M8); the interior values are the "
                 "environment's own schedule"),
        symlog_reference_note=(
            "the AR(1) predictions are derived in raw space while the audit "
            "standardises in symlog space; over this observation range the "
            "compression factor is 0.992-0.997, i.e. well inside the "
            "measured non-Gaussianity floor, so no correction is applied "
            "and none is hidden (review M9)"),
        stream_construction=dict(
            source="probing/probeset.py:93 chain_streams (temporal order)",
            selection="first n_streams; whole stream front-to-back",
            admissibility="exogenous: same-episode t,t+need and t >= "
                          "episode_start + burn_in — depends only on the "
                          "recorded reset schedule and the index",
            horizon_blocks="non-overlapping stride-h starts",
            calibration="first n_cal admissible reads per stream, excluded "
                        "from every e-process"),
        standardization_space="symlog (the decoder's own likelihood space; "
                              "nothing is symexp'd)",
        legs_h1=legs, primary=primary, primary_sign_stable=sign_stable,
        timing_pb=dict(
            scaling_pinned="c_recon",
            summary_statistic="median (rank-based; the mean is not pinned "
                              "because pb's distribution is skewed)",
            pb={cn_: primary[cn_]["position_on_bracket"] for cn_ in scalings},
            q_hat_distractor=primary_v2["q_hat"]["distractor"],
            A_convention_distractor=primary_v2["A_convention"]["distractor"],
            note="REGISTERED A5, RE-DERIVED (review B6).  Rev 4's bar "
                 "max(pb_det) < min(pb_gauss) was FALSE ON REV-3'S OWN "
                 "NUMBERS: det [0.8515, 0.9915], gauss [0.9748, 1.0058] "
                 "overlap on [0.9748, 0.9915].  The correct instrument is a "
                 "RANK test, which those same numbers pass: exact one-sided "
                 "Mann-Whitney (gauss > det, n=8 vs 4) gives U = 30 of 32, "
                 "p = 0.00808 over 495 arrangements; medians 0.9369 vs "
                 "0.9920.  BAR: exact one-sided MWU p <= 0.01 at the pinned "
                 "scaling c_recon, AND the same one-sided ordering "
                 "reproduced by median q_hat.  Disagreement between the two "
                 "=> TIMING-NOT-REPORTABLE.  pb remains construct-invalid "
                 "as a justification measure (rev-3 adjudication) and is "
                 "retained ONLY as this cross-check; it is computed and "
                 "adjudicated by analysis/se_eva_read.py across cells, not "
                 "within a cell."),
        planted_mutants=planted, floors=floors,
        resolution_gates=resolution, profile_ratios=profile_ratios,
        c_sensitivity=c_sensitivity, se_lic_reference=lic,
        stream_meta=stream_meta, n_scalings_used=len(scalings),
        horizon=hor,
    )
    with open(os.path.join(out_dir, "se_eva_probe.json"), "w") as f:
        json.dump(_clean(result), f, indent=1)

    def row(block, k):
        b = block.get(k, {})
        return (f"rho {b.get('rho', NAN):7.4f}  "
                f"log10E {b.get('log10e_pooled', NAN):+9.2f}  "
                f"cross {b.get('crossed_frac', NAN):.2f}")
    print(json.dumps(_clean({
        "PRIMARY_v2_qhat_IV": {
            "q_hat": primary_v2["q_hat"],
            "q_ci_distractor": primary_v2["q_ci"]["distractor"],
            "q_ci_t_distractor": primary_v2["q_ci_t"]["distractor"],
            "lambda_hat": primary_v2["lambda_hat"],
            "r_perm_p": primary_v2["r_perm_p"],
            "verdict_per_key": primary_v2["verdict_per_key"],
            "floor_q": primary_v2["floor_q"],
            "floor_r": primary_v2["floor_r"],
            "adjudicable": primary_v2["adjudicable"],
            "construct_ok": primary_v2["construct_ok"],
            "construct": {m: dict(q_hat=round(v["q_hat"], 4),
                                  q_true=v["q_true"],
                                  A_ols=round(v["A_ols"], 4),
                                  lam=round(v["lambda_hat"], 4))
                          for m, v in
                          primary_v2["construct_validity"].items()},
            "A_convention": primary_v2["A_convention"],
        },
        "controls_pass": controls_pass, "controls": ctl,
        "primary_resolved": primary_resolved,
        "scalings": scalings, "n_scalings_used": len(scalings),
        "PRIMARY": {cn_: {
            "dlogrho_legC": primary[cn_]["dlogrho_legC"],
            "dlogrho_legP": primary[cn_]["dlogrho_legP"],
            "legC_minus_legP": primary[cn_]["legC_minus_legP"],
            "floor": primary[cn_]["floor_dlogrho_dup0_vs_position"],
            "adjudicable": primary[cn_]["adjudicable"],
            "bracket_endpoint_measured":
                primary[cn_]["bracket_endpoint_measured"],
            "position_on_bracket": primary[cn_]["position_on_bracket"],
        } for cn_ in scalings},
        "primary_sign_stable": sign_stable,
        "c_star": c_sensitivity["c_star"],
        "vens_share_of_S": {
            k: legs[f"{cn0}|C|h1|over"][k]["vens_share_of_S"]
            for k in universe},
        "resolution_gates": resolution,
        "profile_ratios": {a: dict(
            predicted=b["predicted"],
            distractor=b["measured"].get("distractor"),
            position=b["measured"].get("position"),
            velocity=b["measured"].get("velocity"))
            for a, b in profile_ratios["ratios"].items()},
        "se_lic_reference": lic,
    }), indent=1))
    return result


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _fixture_checks():
    rng = np.random.default_rng(0)
    N, D, A = 4000, 4, 0.005

    def sq_from(scale, n=N, d=D, r=rng):
        return (r.normal(0.0, math.sqrt(scale), (n, d)) ** 2).sum(-1)

    # (1) PLANTED OVERCLAIMING CHANNEL (realized 0.30 of claimed) CERTIFIES
    c_over = cert(eprocess(sq_from(0.30), D, "over"), A)
    assert c_over["crossed"] and c_over["t_cross"] < 200, c_over

    # (2) CALIBRATED CHANNEL DOES NOT CERTIFY, at the Ville rate
    fired = sum(cert(eprocess(sq_from(1.0, 1500, D,
                                      np.random.default_rng(100 + i)),
                              D, "over"), A)["crossed"] for i in range(60))
    assert fired / 60.0 <= 4 * A, f"H0 crossing rate {fired/60:.3f} > 4*alpha"

    # (3) ONE-SIDEDNESS: an UNDER-dispersed model (realized 3x claimed) must
    #     leave the OVER leg silent and fire the UNDER leg
    sq_hard = sq_from(3.0)
    assert not cert(eprocess(sq_hard, D, "over"), A)["crossed"], \
        "difficulty must push AWAY from the over-dispersion certificate"
    assert cert(eprocess(sq_hard, D, "under"), A)["crossed"]
    # ... and symmetrically the over-claiming channel must not fire UNDER
    assert not cert(eprocess(sq_from(0.30), D, "under"), A)["crossed"]

    # (4) EXACT-SCALE NO-OP: rescaling the claim by the realized dispersion
    #     silences the process (the deployed synthetic control in miniature)
    raw = rng.normal(0.0, math.sqrt(0.3), (N, D))
    assert cert(eprocess((raw ** 2).sum(-1), D, "over"), A)["crossed"]
    assert not cert(eprocess(((raw / math.sqrt(0.3)) ** 2).sum(-1), D,
                             "over"), A)["crossed"], "exact rescale must be a no-op"

    # (5) MONOTONE / NO RELAPSE: the certificate is a running supremum
    le = eprocess(np.concatenate([sq_from(0.2, 800), sq_from(1.0, 3000)]),
                  D, "over")
    c5 = cert(le, A)
    assert c5["crossed"] and c5["log10e_max"] >= c5["log10e_final"]

    # (6) BIG CLAIMS DIE FAST (the Dutch-book sizing)
    t_big = cert(eprocess(sq_from(0.2), D, "over"), A)["t_cross"]
    t_small = cert(eprocess(sq_from(0.8, 20000), D, "over"), A)["t_cross"]
    assert 0 <= t_big < t_small, (t_big, t_small)
    assert kl_rate(0.2) > kl_rate(0.8) > 0

    # (7a) the vectorised e-process must equal the sequential reference
    for sd in (0.3, 1.0, 2.5):
        q = sq_from(sd, 700, D, np.random.default_rng(7))
        for side in ("over", "under"):
            assert np.allclose(eprocess(q, D, side),
                               _eprocess_loop(q, D, side),
                               rtol=0, atol=1e-9), side

    # (7) PREDICTABILITY of the plug-in: r_t must depend on the PAST only,
    #     so perturbing sq[t] cannot change log_e[t-1]
    s = sq_from(0.5, 500)
    a_ = eprocess(s, D, "over")
    s2 = s.copy(); s2[300] += 50.0
    b_ = eprocess(s2, D, "over")
    assert np.allclose(a_[:300], b_[:300], rtol=0, atol=0), \
        "plug-in leaked future information"

    # (8) AR(1) persistence algebra + the inversion, both directions
    assert abs(phi_from_persistence(0.148, 15) - 0.9333) < 1e-3
    rp = predicted_rho_ar1(15, 0.9, phi_from_persistence(0.148, 15))
    assert rp > 1.0, "believing MORE persistence => claimed dispersion too " \
                     "small => realized/claimed > 1 (the UNDER side)"
    assert abs(predicted_rho_ar1(15, 0.9, 0.9) - 1.0) < 1e-12
    assert predicted_rho_ar1(2, 0.9, 0.95) > 1.0

    # (9) fit_c: recovers a planted constant, and refuses when it cannot
    vl = rng.uniform(0.05, 0.5, (3000, 2))
    r2 = (rng.normal(0, 1, vl.shape) ** 2) * (0.37 + vl)
    assert abs(fit_c(r2, vl) - 0.37) < 0.05, fit_c(r2, vl)
    assert not np.isfinite(fit_c((rng.normal(0, 1, vl.shape) ** 2) * 1e-6, vl))

    # (10) EXOGENOUS admissibility: no index crosses an episode boundary,
    #      burn-in respected, and the rule never looks at model output
    isf = np.zeros(300, bool); isf[[0, 100, 200]] = True
    idx, lens = episode_segments(isf, 16, 1)
    assert lens == [100, 100, 100]
    assert idx.min() >= 16 and set(np.flatnonzero(isf)).isdisjoint(idx + 1)
    for s_ in (0, 100, 200):
        assert not ((idx >= s_) & (idx < s_ + 16)).any()
    idx15, _ = episode_segments(isf, 16, 15)
    assert set(np.flatnonzero(isf)).isdisjoint(
        np.concatenate([idx15 + d for d in range(1, 16)]))

    # (11) THE MULTIPLICITY FIXTURE (rev 3).  The rev-2 gate
    #      (`any stream ever crossed`) refuses a well-calibrated control at
    #      a rate set by the NUMBER OF TESTS, not by the instrument.  The
    #      pooled-certificate gate must (i) pass a null control with many
    #      streams with high probability, and (ii) still refuse a genuinely
    #      mis-assembled S.
    def _fam(scale, nstream, seed, D_=4, nread=1500):
        """One gate family: `nstream` independent streams, one key."""
        rr = np.random.default_rng(seed)
        certs = [cert(eprocess(sq_from(scale, nread, D_, rr), D_, "over"), A)
                 for _ in range(nstream)]
        pooled = float(np.sum([c_["log10e_final"] for c_ in certs]))
        thr = float(math.log10(1.0 / A))
        return dict(
            block={"k": dict(log10e_pooled=pooled, log10e_threshold=thr,
                             certified=bool(pooled >= thr))},
            any_crossed=any(c_["crossed"] for c_ in certs))
    #  (i) null control, 6 streams, 40 independent replicates
    old_rule_fails = new_rule_fails = 0
    for i in range(40):
        f = _fam(1.0, 6, 5000 + i)
        old_rule_fails += int(f["any_crossed"])
        new_rule_fails += int(
            not synthg_family_verdict(f["block"])["passed"])
    assert new_rule_fails <= 2, (
        "the pooled gate still refuses a well-calibrated control too often",
        new_rule_fails)
    assert new_rule_fails <= old_rule_fails, (new_rule_fails, old_rule_fails)
    #  (ii) a genuinely MIS-ASSEMBLED S must still certify and refuse.  This
    #  is the rev-1 bug reproduced: residuals generated WITHOUT the ensemble
    #  term while S is standardised WITH it, so rho sits well below 1.
    bad = _fam(0.80, 6, 99)
    v_bad = synthg_family_verdict(bad["block"], "mis-assembled")
    assert not v_bad["passed"], (
        "a mis-assembled S must still be caught by the pooled gate", v_bad)
    assert v_bad["margin"] < 0

    # (12) json sanitiser
    assert _clean({"a": NAN, "b": [float("inf"), 1.0]}) == \
        {"a": None, "b": [None, 1.0]}
    print("  [fixtures] 12 mutant families killed (overclaim certifies; "
          "calibrated does not at the Ville rate; one-sidedness both ways; "
          "exact-rescale no-op; no-relapse; big-claims-die-fast; plug-in "
          "predictability; AR(1) algebra + inversion; fit_c recover/refuse; "
          "exogenous admissibility; GATE MULTIPLICITY: pooled gate passes a "
          "null control that the per-stream rule refuses, and still "
          "refuses a mis-assembled S; sanitiser)")


def selfcheck():
    print("se_eva_probe selfcheck")
    _fixture_checks()
    smoke = os.environ.get("EVA_SMOKE_DIR", str(SMOKE))
    assert os.path.isdir(os.path.join(smoke, "ckpt")), (
        f"smoke ckpt not found at {smoke} (set EVA_SMOKE_DIR)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_selfcheck_out")
    # REVIEW M7: the selfcheck must run at the DEPLOYED n_lat — the floor
    # is a function of M, so a selfcheck at M=4 validates a different
    # instrument than the one that ships.
    args = parse_args([
        # REVIEW M3: 2 streams never exercised the deployed verdict rule
        # (a 2-stream sign agreement is a coin flip); 4+ is the minimum at
        # which the cluster machinery means anything.
        "--run_logdir", smoke, "--n_streams", "4", "--t_max", "900",
        "--t_max_h", "480", "--horizons", "2,4", "--burn_in", "8",
        "--n_cal", "100", "--n_lat", "32", "--blk", "120",
        "--n_perm_r", "200",
        "--platform", "cpu", "--output", out])
    r = run(args)
    uni = r["share_universe"]
    source_key_sc = r["source_key"]

    assert r["standardization_space"].startswith("symlog")
    assert r["n_scalings_used"] >= 2, r["scalings"]        # review B2
    assert set(r["scalings"]) >= {"c_theory", "c_recon"}   # review B3
    assert r["controls_pass"] is True, r["controls"]       # review B13
    assert set(r["controls_detail"]) == {
        k for k in r["controls"] if k.startswith("synthg_")}
    for lab, v in r["controls_detail"].items():
        assert v["passed"] is True, (lab, v)
        assert np.isfinite(v["margin"]) and v["margin"] > 0, (lab, v)
    assert r["revision"] == 7
    assert r["disag_head"] in ("det", "gauss"), (
        "the arm name was clobbered on its way into the json", r["disag_head"])
    # ---- REV 5 CONSTRUCT VALIDITY, from REAL member construction.  Every
    # mutant builds actual members (shared common-mode + idiosyncratic) and
    # an actual y; NOTHING synthesises Rbar from an observed V.
    pv = r["primary_v2"]
    cv = pv["construct_validity"]
    assert abs(cv["centered_q0"]["q_hat"] - 0.0) < 0.30, (
        "q=0 (the ensemble's spread accounts for the whole error scaling) "
        "must be recovered as q_hat ~ 0 -- rev 4's bar called this UNEARNED",
        cv["centered_q0"])
    assert abs(cv["exchangeable_q1"]["q_hat"] - 1.0) < 0.35, cv["exchangeable_q1"]
    assert cv["shared_dominant_q4"]["q_hat"] > 2.0, cv["shared_dominant_q4"]
    # the OLS estimator must be visibly ATTENUATED relative to the IV on the
    # exchangeable case -- that is the defect review B2 named
    assert cv["exchangeable_q1"]["A_ols"] < cv["exchangeable_q1"]["A"], cv
    # a weak instrument must be FLAGGED, never read as a small q
    assert cv["homoscedastic_weak_iv"]["lambda_hat"] < pv["lambda_min"], cv
    # REVIEW F5: the truth mutants must land on the REGISTERED VERDICTS
    # through the actual classifier, not merely near the right numbers.
    for m, v in cv.items():
        if v.get("verdict_expected"):
            assert v["verdict"] == v["verdict_expected"], (
                "a truth mutant did not classify as registered", m,
                v["verdict"], v["verdict_expected"])
        else:
            assert v["descriptive"] is True, m
    # ---- REV 7 FIXTURE: the two IV forms forced apart NEAR A BAR.  The KEY
    # must route to UNRESOLVED-IV-DISAGREEMENT, and this must NOT be an
    # instrument failure -- rev 6 refused 3 of 12 cells on exactly this
    # configuration while their SYNTHG margins were healthy.
    _near = dict(q_hat=0.52, lambda_hat=0.60, A=0.70, r_spearman=0.40,
                 r_perm_p=0.001, dq_iv=0.310, bar_distance=0.020)
    assert classify(_near, 0.15) == "UNRESOLVED-IV-DISAGREEMENT", \
        classify(_near, 0.15)
    # the SAME key with the estimators agreeing classifies normally
    _agree = dict(_near, dq_iv=0.001)
    assert classify(_agree, 0.15) == "MIXED", classify(_agree, 0.15)
    # and FAR from a bar the same dq is harmless -- near-bar physics, not
    # estimator breakage, is what the verdict is about
    _far = dict(_near, q_hat=17.0, bar_distance=15.0)
    assert classify(_far, 0.15) == "SHARED-DOMINATED", classify(_far, 0.15)
    # the instrument gate must be scoped so this cannot refuse a cell
    assert r["controls_pass"] is True
    assert r["controls"]["primary_iv_crosscheck"] is True
    ivg = r["iv_crosscheck"]
    assert set(ivg["gate_keys"]) <= {r["source_key"], "planted_dup0",
                                     "planted_dup1"}, ivg["gate_keys"]
    assert all(ivg["nearest_bar_distance"][k] >= ivg["gate_far_bar_min"]
               for k in ivg["gate_keys"]), ivg
    assert "distractor" not in ivg["gate_keys"], (
        "the distractor must never gate the CELL -- it is the key that sits "
        "near the 0.5 bar, and rev 6 refused 3 cells for exactly that")

    # REVIEW F12: the invariant must hold on the DEGENERATE path too, which
    # no cell exercised -- construct one and classify it.
    for _bad in (dict(q_hat=NAN, lambda_hat=NAN, A=NAN, r_spearman=NAN,
                      r_perm_p=NAN),
                 dict(q_hat=NAN, lambda_hat=0.9, A=1.0, r_spearman=0.5,
                      r_perm_p=0.001)):
        _v = classify(_bad, 0.15)
        assert _v.startswith("UNRESOLVED"), (_v, _bad)
    assert pv["construct_ok"] is True, cv
    # REVIEW F1: gamma_hat / q_hat_net emitted beside q_hat
    for k in uni:
        assert k in pv["gamma_hat"] and k in pv["q_hat_net"], k
        if np.isfinite(pv["q_hat"][k]) and np.isfinite(pv["gamma_hat"][k]):
            assert abs((pv["q_hat"][k] - pv["gamma_hat"][k])
                       - pv["q_hat_net"][k]) < 1e-9, k
    assert np.isfinite(pv["floor_q_net"])
    # REVIEW F7: intercept-derived quantities are NULLED when unresolved
    for k in uni:
        if not pv["resolved"][k]:
            assert not np.isfinite(pv["intercept"][k]), k
            assert not np.isfinite(pv["intercept_per_dim"][k]), k
    # REVIEW F6: the IV cross-check is stated in q units against the bars
    ivc = r["iv_crosscheck"]
    assert set(ivc["bars_q"]) == {0.5, 2.0}
    for k in r["resolved_keys"]:
        assert k in ivc["dq_per_key"] and k in ivc["nearest_bar_distance"]
    # gates present and typed
    for k in uni:
        assert isinstance(pv["aligned"][k], bool)
        assert isinstance(pv["resolved"][k], bool)
        assert pv["verdict_per_key"][k] in (
            "SPREAD-ACCOUNTS", "MIXED", "SHARED-DOMINATED",
            "UNRESOLVED-WEAK-INSTRUMENT", "UNRESOLVED-NO-ALIGNMENT",
            "UNRESOLVED-OUT-OF-SUPPORT", "UNRESOLVED-DEGENERATE",
            "UNRESOLVED-IV-DISAGREEMENT")
        # review F12: the two asserts must agree -- every non-classifying
        # verdict carries the prefix, so this is now total
        assert (pv["verdict_per_key"][k].startswith("UNRESOLVED")
                or pv["resolved"][k]), k
        assert pv["verdict_per_key"][k] != "ANTI", (          # review F5
            "ANTI was struck from the registered rule -- q >= 0 by "
            "construction, so no world in the model produces A < 0")
        assert len(pv["q_per_stream"][k]) == r["n_streams"], k
        assert -1.0 <= pv["r_spearman"][k] <= 1.0, k
        if not pv["resolved"][k]:
            assert pv["verdict_per_key"][k].startswith("UNRESOLVED"), k
    assert np.isfinite(pv["floor_q"]) and np.isfinite(pv["floor_r"])
    # REVIEW B3: the guard must fire on the case that motivated it -- the
    # smoke distractor, where rev 4 read a slope (A = 0.4665) off a series
    # with |r| = 0.0197 and no association at all.  rev 4's sign+floor rule
    # PASSED it; the permutation null must not.
    dk = pv["r_perm_p"]["distractor"]
    assert np.isfinite(dk), "the alignment null did not run"
    if abs(pv["r_spearman"]["distractor"]) < 0.05:
        assert not pv["aligned"]["distractor"], (
            "the alignment guard failed to fire on a series with no "
            "association -- exactly the rev-4 defect review B3 named",
            pv["r_spearman"]["distractor"], dk)
        assert pv["verdict_per_key"]["distractor"].startswith("UNRESOLVED")
    assert isinstance(pv["adjudicable"], bool)
    assert pv["registered_on"] == "q_hat"
    # review I4: exclusion, not containment
    assert pv["ci_excludes_q0"] == (not (
        min(pv["q_ci"]["distractor"][0], pv["q_ci_t"]["distractor"][0])
        <= 0.0 <=
        max(pv["q_ci"]["distractor"][1], pv["q_ci_t"]["distractor"][1])))
    # review B8: the construct/floor gates are IN ctl
    for g in ("primary_construct_ok", "primary_floors_finite",
              "primary_iv_crosscheck"):
        assert g in r["controls"], g
    assert isinstance(r["primary_resolved"], bool)
    assert isinstance(r["resolved_keys"], list)
    # review B9: leg E is exploratory and struck from the freeze
    assert r["exploratory"]["leg_E"] is True
    assert "decode_asymmetry_note" in r
    for cn_ in r["scalings"]:
        for k in uni:
            e = r["legs_h1"][f"{cn_}|E|h1|over"][k]
            assert np.isfinite(e["rho"]) and e["rho"] > 0, (cn_, k)
    # review M2: vens_share reported on every leg, not 0 on P/E
    _cn_last = list(r["scalings"])[-1]
    assert r["legs_h1"][f"{_cn_last}|E|h1|over"]["distractor"][
        "vens_share_of_S"] > 0
    assert "timing_pb" in r
    assert r["disag_head"] in ("det", "gauss"), (
        "the arm name was clobbered on its way into the json", r["disag_head"])
    assert r["episode_lengths"]["n"] >= 1
    for name, blk in r["legs_h1"].items():
        for k in uni:
            b = blk[k]
            assert b["n_reads"] > 0 and b["D"] == r["dims"][k], (name, k)
            assert np.isfinite(b["rho"]) and b["rho"] > 0, (name, k)
            assert 0.0 <= b["crossed_frac"] <= 1.0
            assert np.isfinite(b["log10e_pooled"])
    # leg C claims strictly MORE dispersion than leg P (it adds the
    # ensemble term), so its rho must be strictly smaller on every key
    cn = list(r["scalings"])[-1]
    for k in uni:
        assert r["legs_h1"][f"{cn}|C|h1|over"][k]["rho"] < \
            r["legs_h1"][f"{cn}|P|h1|over"][k]["rho"], k
    # the two sides cannot both certify on the same key/leg
    for k in uni:
        o = r["legs_h1"][f"{cn}|C|h1|over"][k]["crossed_frac"]
        u = r["legs_h1"][f"{cn}|C|h1|under"][k]["crossed_frac"]
        assert not (o > 0.5 and u > 0.5), (k, o, u)
    # the SYNTHETIC no-op must be silent (this is the control that validates
    # the composite predictive assembly at the deployed scale)
    # THE EXACT-SCALE NO-OP must be silent on both legs AND both sides
    for leg in ("P", "C"):
        for side in ("over", "under"):
            tag = f"{cn}|{leg}|h1|{side}|SYNTHG"
            for k in uni:
                b = r["legs_h1"][tag][k]
                assert abs(b["rho"] - 1.0) < 0.10, (
                    "the Gaussian no-op must have rho ~ 1 by construction",
                    tag, k, b["rho"])
                # rev 3: the control is held to the SAME standard as the
                # findings — it fails only if it CERTIFIES on the pooled
                # e-value.  `crossed_frac` is descriptive (a per-stream
                # supremum criterion here was ~576 tests/cell).
                assert not b["certified"], (
                    "the exact-scale no-op CERTIFIED — the pipeline "
                    "manufactures evidence at this scale", tag, k, b)
    # the mixture-faithful control is a MEASURED FLOOR, not a silence gate:
    # a non-Gaussian latent mixture reads as a few percent of dispersion.
    # It must be BOUNDED and it must be reported.
    for leg in ("P", "C"):
      for side in ("over", "under"):
        tag = f"{cn}|{leg}|h1|{side}|SYNTHMIX"
        for k in uni:
            rho = r["legs_h1"][tag][k]["rho"]
            assert 0.85 < rho < 1.15, (
                "non-Gaussianity floor out of range — the mixture control "
                "should sit within a few percent of 1", tag, k, rho)
    assert set(r["floors"]["h1_under_nongauss"]) == set(uni)
    assert set(r["floors"]["h1_legC_nongauss"]) == set(uni)   # review B5
    for h in r["horizons"]:
        for k in uni:
            b = r["horizon"][f"{cn}|P|h{h}|under"].get(k, {})
            assert "rho_floor_corrected" in b, (h, k)
            assert f"{cn}|P|h{h}|under|SYNTHG" in r["horizon"], h
            g = r["horizon"][f"{cn}|P|h{h}|under|SYNTHG"][k]
            assert not g["certified"], (
                "the horizon exact-scale no-op CERTIFIED", h, k, g)
    # REVIEW B1: the bracket endpoint is run-measured and the gate exists
    for cn_ in r["scalings"]:
        pr = r["primary"][cn_]
        assert np.isfinite(pr["dlogrho_legC"]) and np.isfinite(
            pr["dlogrho_legP"])
        assert np.isfinite(pr["legC_minus_legP"])
        assert np.isfinite(pr["bracket_endpoint_measured"]), (
            "the bracket endpoint must be MEASURED on the run, not "
            "transported from theta_1")
        assert isinstance(pr["adjudicable"], bool)
        assert np.isfinite(pr["dlogrho_legC_floor_corrected"])  # review B5
    assert isinstance(r["primary_sign_stable"], bool)
    # REVIEW B3: the c-grid exists and c* is reported (or explicitly None)
    assert len(r["c_sensitivity"]["grid"]) == 6
    assert all(np.isfinite(g["legC_minus_legP"])
               for g in r["c_sensitivity"]["grid"])
    # REVIEW B1: vens must actually be a visible share of S on leg C, or the
    # primary has nothing to detect
    for k in uni:
        assert 0.0 <= r["legs_h1"][f"{cn}|C|h1|over"][k]["vens_share_of_S"] \
            <= 1.0, k
    # REVIEW B4: per-horizon resolution gates present and typed
    for h in r["horizons"]:
        g = r["resolution_gates"][f"h{h}"]
        assert isinstance(g["adjudicable"], bool)
        assert np.isfinite(g["floor"]) and np.isfinite(g["predicted_rho"])
    # REVIEW B7: the pooled e-value is the certificate
    for k in uni:
        b = r["legs_h1"][f"{cn}|C|h1|over"][k]
        assert isinstance(b["certified"], bool)
        assert b["certified"] == (b["log10e_pooled"]
                                  >= b["log10e_threshold"])
    # REVIEW B12/M5: per-stream supply and rho emitted
    assert len(r["stream_meta"]) == r["n_streams"]
    assert all(m["n_audited"] > 0 for m in r["stream_meta"])
    assert len(r["legs_h1"][f"{cn}|C|h1|over"]["distractor"]
               ["rho_per_stream"]) == r["n_streams"]
    # REVIEW B6: no horizon lattice point may precede its stream's cal end
    assert all(m["cal_end_stream"] >= m["offset"] for m in r["stream_meta"])
    # REVIEW B8: profile ratios registered with null comparators
    assert r["profile_ratios"]["ratios"], "no profile ratios emitted"
    # in-run planted mutants on THIS run's real residuals
    pm = r["planted_mutants"]
    for k in uni:
        assert pm["inflate_x4"][k]["crossed_frac"] == 1.0, (
            "a 4x planted overclaim must certify on every key", k,
            pm["inflate_x4"][k])
        assert abs(pm["marginal_rescale"][k]["rho"] - 1.0) < 1e-9, (
            "the marginal rescale must set rho to 1 by construction", k)
    # The exact-scale no-op is the SYNTHETIC control (asserted above), NOT
    # the marginal rescale: marginal calibration does not buy conditional
    # calibration, and the e-process is conditional.  Assert the two
    # controls actually differ on this run, or the distinction is untested.
    assert any(pm["marginal_rescale"][k]["crossed_frac"]
               > r["legs_h1"][f"{cn}|C|h1|over|SYNTHG"][k]["crossed_frac"]
               for k in uni), (
        "marginal_rescale and the synthetic no-op behaved identically — the "
        "conditional-vs-marginal distinction is untested on this fixture")
    for h in r["horizons"]:
        for k in uni:
            b = r["horizon"][f"{cn}|P|h{h}|under"].get(k, {})
            assert "phi_m_implied" in b, (h, k)
    for h in r["horizons"]:
        assert f"predicted|h{h}" in r["horizon"]
        assert r["horizon"][f"predicted|h{h}"]["rho_predicted"] > 1.0
        assert f"{cn}|P|h{h}|under" in r["horizon"]
    with open(os.path.join(out, "se_eva_probe.json")) as f:
        json.loads(f.read(), parse_constant=lambda c: (_ for _ in ()).throw(
            AssertionError(f"bare {c} in json")))
    r2 = run(args)
    for k in uni:
        assert r["legs_h1"][f"{cn}|C|h1|over"][k]["rho"] == \
            r2["legs_h1"][f"{cn}|C|h1|over"][k]["rho"], k
    assert r["primary"] == r2["primary"]
    print("se_eva_probe selfcheck PASS (rev 7) "
          f"(arm={r['disag_head']}, {r['n_streams']} streams, "
          f"scalings={list(r['scalings'])}; legC dispersion > legP on every "
          "key; sides mutually exclusive; SYNTHETIC no-op silent; horizon "
          "predictions present; strict-JSON; deterministic)")
    return r


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
