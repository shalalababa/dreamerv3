# PREREG — PHI-SWEEP: the decisive falsifier for theta_1 = Lambda = 1/(1-phi^2)

Registered 30 Aug 2026. Paper 5 (uncertainty field), EVM program.
Status: FULL registration discipline — this wave adjudicates a paper
claim, so predictions, decision rule and reader are frozen BEFORE any
compute.

Upstream record: `research_notes/paper5_uncertainty_field/EVM_Rescue_20260830.md`
(panel verdict) and the independent verification
`temp_files/evm_rescue_20260830/VERIFICATION.md` (Part 3, "The falsifier
— re-derived independently"). Reader: `uncfield/se_phisweep_read.py`.
Wave: `ops/waves/phisweep/spec.yaml`.

---

## 1. WHAT IS ON TRIAL

The EVM rescue panel certified `theta_1(distractor) = 5.2579`, BCa95
[5.02, 5.46] on the Stage-1 det arm (n=8, seeds 10–17, s=1.0), against
the environment constant `Lambda = 1/(1-phi^2) = 5.2632` at the deployed
`phi = 0.9` — a 0.100% (0.043 SE) agreement.

The panel ALSO established that this is **not derived**:

- Lambda enters the probe's normalization only as a *level* multiplying
  two phi-free constants (`d_pos = 0.00205`, CV 8.8% over 24 runs and 5
  scales; `N_dist`, the symlog channel variance). A scaling law cannot
  produce a level identity.
- The supporting `s^1.9 ~ s^2` "par-pricing" scaling was a two-point
  endpoint artifact (OLS to the deployment point: 1.346 ± 0.104).
- In the probe's own (symlog) space the leverage at s=1.0 is 5.017, i.e.
  1.98 SE from the measurement — **the phi=0.9 data cannot choose
  between the raw and symlog conventions** (tension T3, left OPEN).
- The claimed cross-instrument corroboration was circular (T8).

So `theta_1 = Lambda` currently stands as an OBSERVED
coincidence-candidate. Its own coincidence-risk statement, quantified by
the panel: the above-onset theta_1 band is [4.85, 8.41] (log-width
0.551), and a landing within the measurement's actual resolution
(det-arm relative SE 2.31%) is not a 0.1% coincidence.

**This wave moves the claim to a place where the law can be wrong.** It
is the single registered experiment that can promote the
Phase-Transition frame (held in reserve per the panel's frame decision)
or retire the identity to a printed near-miss.

---

## 2. DESIGN

### 2.1 Factors

| factor | levels |
|---|---|
| `phi` (OU persistence) | **0.80**, **0.95** |
| seeds | 4 per arm — **140–143** (phi=0.80), **144–147** (phi=0.95) |
| task | `dmc_cheetah_run` (Stage-1 pin) |
| steps | 5e5 (Stage-1 pin) |
| objective | p2e disagreement, reward-free (Stage-1 pin) |

8 runs, ~7 h each, ~56 GPU-h.

Seeds 140–147 are fresh: the SE-family program has consumed 0–17, 20–39,
40–51, 52–67, 68–75, 96–111, 120–123, 132–139. Nothing above 139 is
registered anywhere in `prereg/`, `ops/waves/` or `STUDY_LEDGER.md`.

`phi` is set through the producer's OU knob **`theta = 1 - phi`**
(`embodied/envs/distractor.py:23,63`: `_ar = 1 - theta`), i.e.
`DISTRACTOR_THETA = 0.20` and `0.05`. The distractor stays
unit-stationary by construction (`_noisesd = sqrt(1 - _ar^2)`), so
changing phi moves **persistence only**, not the emitted amplitude.

**Producer delta (registered here, applied 30 Aug):**
`scripts/uncfield_se.sbatch` did not pass `--distractor.theta`. Two
lines added — the flag itself and a BIND-CHECK `want()` entry — both
defaulting to `0.1`, which is the `configs.yaml:44` default and the
value every prior SE consumer ran. The delta is therefore bitwise-inert
for all earlier waves, and the phi axis is BIND-CHECKed (the 21-Aug
NOBOOT stale-producer incident class: a silently-unread override cost
4 runs / ~25 GPU-h).

### 2.2 The s/s_0 CONTROL — and why it is convention-free

theta_1 is a **scale plateau above onset**, not a scale-free constant:
the verified dose curve (phi=0.9) is

| s | 0.10 | 0.25 | 0.30 | 0.34 | 0.38 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| theta_1 | 0.113 | 0.465 | 4.849 | 7.832 | 8.219 | 8.408 | 6.557 | 5.258 |

and the onset moves with phi: `s_0(phi) = 0.275 * sqrt((1-phi^2)/0.19)`
→ 0.3785 (phi=0.80), 0.275 (phi=0.90, measured 0.275 [0.27, 0.28]),
0.1970 (phi=0.95). A fixed s=1.0 would therefore put the three arms at
**different positions on the dose curve** (s/s_0 = 2.64 / 3.64 / 5.08),
confounding the phi effect with a dose effect.

**Registered convention.** Each arm runs at

  `s_arm = k * s_0(phi)`,  `k = 1.0 / s_0(0.90)`,

with the SAME `s_0` convention used for all three arms, so that the
reference point is exactly the existing `s = 1.0, phi = 0.9` Stage-1
data. Because `s_0(phi) ∝ sqrt(1-phi^2)` under every convention in the
panel's bracket (the 0.275 multiplier cancels in `k * s_0`), this
reduces to

  **`s_arm = sqrt((1 - phi^2) / 0.19)`**

— a control that does **not** depend on which onset accounting
(beta=1.0, beta=1.1, or the encoder-price-only bracket) is right. That
robustness is registered as a design property, not discovered later.

PINNED SCALES (exact, to be passed verbatim; reader tolerance 1e-6):

| phi | theta = 1-phi | **s_arm** | s/s_0 | emitted stationary sd = s*1.215 |
|---|---|---|---|---|
| 0.80 | 0.20 | **1.3764944032233706** | 3.6364 | 1.67244 |
| 0.90 | 0.10 | 1.0 (EXISTING data, seeds 10–17) | 3.6364 | 1.21500 |
| 0.95 | 0.05 | **0.7163503994113789** | 3.6364 | 0.87037 |

All three arms therefore sit at the identical dose-curve position
`s/s_0 = 3.6364` — the position the flagship measurement occupies.

No floor risk: at phi=0.95 the distractor's symlog per-dim sd is
sqrt(0.31912) = 0.565, far above `norm_floor = 0.05 * real_scale`
(`uncfield/se_probe.py:187`).

---

## 3. PRE-STATED PREDICTIONS (frozen)

Symlog values are MC (N=8e6, pair-difference estimator of
`E[Var(symlog(X_{t+1})|X_t)]` against `Var(symlog(X))`, stationary
`N(0, (s*1.215)^2)`), **evaluated at each arm's own controlled scale**
— the symlog leverage is scale-dependent, so quoting the s=1.0 numbers
would be wrong for this design. The estimator reproduces the panel's
independent MC at every published anchor (5.2539/5.0146/4.8088 at
phi=0.9 s=0.1/1.0/2.0; 2.6905 and 9.6516 at s=1.0 for phi=0.80/0.95).

### 3.1 The four accounts

| account | phi=0.80 (s=1.3765) | phi=0.90 (s=1.0, EXISTING) | phi=0.95 (s=0.7164) |
|---|---|---|---|
| **H_LAW-raw** — theta_1 = 1/(1-phi^2) | **2.7778** | 5.2632 (measured 5.2579) | **10.2564** |
| **H_LAW-symlog** — theta_1 = symlog leverage at the arm's own s | **2.6610** | 5.0146 | **9.8347** |
| **H_RIVAL-A** — theta_1 follows the dose curve in s/s_0 units | **5.258** | 5.258 | **5.258** |
| **H_RIVAL-B** — theta_1 follows the dose curve in RAW s units | **4.115** † | 5.258 | **6.744** |

† EXTRAPOLATED: s=1.3765 is outside the measured dose window
[0.10, 1.00]; taken off the log-slope of the last measured segment
(0.75→1.00, exponent −0.7675). Flagged as the weakest of the four
predictions and never used to fire anything on its own.
H_RIVAL-B at phi=0.95 (s=0.7164) is a log-interpolation *inside* the
measured window (between 8.408@0.50 and 6.557@0.75) and is sound.

**For the record, the uncontrolled design's rival number.** Had the
sweep run at a fixed s=1.0, H_RIVAL-A would predict theta_1 ≈ **6.686**
at phi=0.80 (log-interpolation of the dose curve at s/s_0 = 2.6418) —
the "~7" in the panel note — and would require an off-curve
extrapolation to s/s_0 = 5.08 at phi=0.95. The controlled design is
registered instead because it converts H_RIVAL-A into a **flat null**
(5.258 in every arm), which is both sharper and free of interpolation
uncertainty.

### 3.2 The convention-free reformulation (why the primary is a RATIO)

The raw and symlog conventions differ by a near-constant multiplicative
offset across the three arms:

| phi | symlog / raw | offset |
|---|---|---|
| 0.80 | 0.95796 | −4.20% |
| 0.90 | 0.95277 | −4.72% |
| 0.95 | 0.95888 | −4.11% |

So the **ratio** `L(phi) = theta_1(phi) / theta_1(0.90)` is essentially
convention-free:

| phi | L under raw | L under symlog | gap |
|---|---|---|---|
| 0.80 | **0.52778** | 0.53065 | 0.54% |
| 0.95 | **1.94872** | 1.96121 | 0.64% |

Both gaps are far below the ratio's own SE (~4%). PRIMARY-1 therefore
tests the LAW on `L(phi)`, where the open T3 convention question cannot
contaminate it; the convention question is tested separately as
PRIMARY-2 on the absolute level.

Rival predictions for `L`: H_RIVAL-A → **1.000** in both arms;
H_RIVAL-B → **0.7825** (phi=0.80) and **1.2827** (phi=0.95).

### 3.3 Power

Realized Stage-1 det-arm dispersion (seeds 10–17): mean 5.257875,
sd 0.343507, **per-run CV 6.533%**, relative SE 2.310% at n=8. At n=4
the arm's relative SE is **3.267%**.

Separations at n=4 (in units of the arm's own SE):

| arm | Lambda_raw | vs H_RIVAL-A | vs H_RIVAL-B | vs H_LAW-symlog |
|---|---|---|---|---|
| phi=0.80 | 2.7778 (SE 0.0907) | **27.3 σ** | **14.7 σ** | 1.29 σ |
| phi=0.95 | 10.2564 (SE 0.3350) | **14.9 σ** | **10.5 σ** | 1.26 σ |

**PRIMARY-1 is enormously over-powered; PRIMARY-2 is under-powered, and
that is registered up front** (§4.2).

---

## 4. DECISION RULE (frozen, pre-stated)

### 4.1 PRIMARY-1 — the LAW (registered, fires)

**Estimand.** Per arm, `L_hat = mean_i theta_1_i / 5.257875`, where the
denominator is the registered Stage-1 det-arm mean (a FIXED registered
constant, not re-estimated).

**Interval.** 95% CI in log space,

  `CI = exp( log(L_hat) ± t_{0.975,3} * sqrt( s_log^2 / 4 + 0.02310^2 ) )`

with `s_log` the sd of `log(theta_1)` over the arm's 4 runs,
`t_{0.975,3} = 3.18245`, and `0.02310` the registered relative SE of the
reference arm. The t-critical is applied to the whole SE (conservative).

**Per-arm verdict.**

| condition | verdict |
|---|---|
| `L_raw(phi)` ∈ CI and `1.000` ∉ CI and `L_B(phi)` ∉ CI | **LAW-CONFIRMED** |
| `L_raw(phi)` ∉ CI | **LAW-KILLED** |
| `L_raw(phi)` ∈ CI but a rival is also in CI | **AMBIGUOUS** |

**Wave outcome cell** (`outcome_cell` in the read json):

- **LAW-CONFIRMED** — both arms LAW-CONFIRMED.
- **LAW-REFUTED** — either arm LAW-KILLED.
- **LAW-SPLIT** — one arm CONFIRMED, the other AMBIGUOUS.
- **NOT-ADJUDICABLE** — otherwise, or on any validity trip in §5.

**What a fire licenses.** LAW-CONFIRMED licenses exactly: *"theta_1
tracks 1/(1-phi^2) across a 3.7x range of the environment constant, at
matched dose-curve position"*. It does NOT license the mechanism (the
panel showed no derivation route exists through the probe's
normalization), and the paper must continue to print that.
**LAW-REFUTED retires `theta_1 = Lambda` to a printed phi=0.9
near-miss**, and the Phase-Transition frame stays unpromoted per the
panel's frame decision.

### 4.2 PRIMARY-2 — the T3 convention adjudication (registered as
UNDER-POWERED; reported, does not gate PRIMARY-1)

**Estimand.** `D = mean over the 8 NEW runs of theta_1_i /
Lambda_raw(phi_i)`. H_LAW-raw predicts **1.0000**; H_LAW-symlog predicts
**0.9584** (the mean of 0.95796 and 0.95888 — the two new arms agree to
0.1%). 95% CI as in §4.1 (t_{0.975,7} = 2.36462, SE = s_log/sqrt(8)).

Separation 4.16% against a 2.31% SE = **1.80 σ**. This is pre-declared
**insufficient**: the registered verdict is

- **CONVENTION-RAW** if 1.0000 ∈ CI and 0.9584 ∉ CI,
- **CONVENTION-SYMLOG** if 0.9584 ∈ CI and 1.0000 ∉ CI,
- **CONVENTION-NOT-ADJUDICATED** otherwise (the expected outcome).

**SUPPLEMENTARY (declared supplementary, never a fire):** the same
statistic pooled over all 16 runs including the Stage-1 det arm (whose
own ratio is 5.257875/5.2632 = 0.99899 raw, 0.95277 symlog) gives
SE 1.633% against a 4.4% separation = 2.69 σ. It is *supplementary*
because it re-uses the arm that generated the observation; it is
reported with that caveat printed inline.

### 4.3 SECONDARY — onset: **DEFERRED, out of scope**

The registered arms all sit at `s/s_0 = 3.6364`, deliberately far above
onset, so this wave carries **no** power on the onset predictions
(0.3785 at phi=0.80, 0.1970 at phi=0.95). A real onset test needs a
sub-onset scale ladder per arm: 2 arms x 3 scales x 3 seeds = 18 runs
≈ 126 GPU-h, i.e. **2.3x the cost of this wave**, which fails the
cheapness bar. DEFERRED; if PRIMARY-1 fires it becomes the natural
follow-up (and only then).

**In scope instead — the weak onset-consistency datum (descriptive, no
fire):** if the s/s_0 control is right, every arm should show a
plateau-grade theta_1 with a non-binding normalizer floor and no
sub-onset collapse (the dose curve's sub-onset values are 0.113 and
0.465). The reader reports `above_onset_consistent` per run
(theta_1 > 1.0 and floor not binding) and per arm. A LAW-CONFIRMED
outcome at phi=0.80 predicts 2.78, which is above 1.0 and therefore
still distinguishable from a sub-onset collapse — the two failure modes
do not alias.

---

## 5. VALIDITY GATES (house pattern; all FATAL unless marked)

FATAL identity gates — a defective run refuses the WHOLE read (B1
pattern, not a silent drop); the refusal happens before the output json
is written, so the one-execution guard survives a repair-and-rerun.

1. **Arm pins.** `distractor.theta == 1 - phi` exactly for the arm the
   seed belongs to; `distractor.scale == s_arm` to 1e-6.
2. **Seed↔arm map.** seeds 140–143 ⇒ phi=0.80; 144–147 ⇒ phi=0.95.
   Any other seed, or a seed in the wrong arm, is FATAL.
3. **Stage-1 pin block** (verbatim from the B1/onset readers): task
   `dmc_cheetah_run`; expl `mode=p2e`, `disag_bootstrap=True`,
   `disag_ens=8`, `disag_scale=1000.0`, `disag_bootstrap_prob=0.8`;
   planted `source_key=position`, `basesd=0.0976`; distractor `dim=8`,
   `basesd=1.215`; gates and mod off on both wrappers; penalty
   machinery inert (`penalty.scale=0.0`, `penalty_mix=False`).
4. **Probe checkpoint identity.** `ckpt_final_ok` (se_read's
   final-checkpoint gate) plus the newest-ckpt cross-check (the A1
   append-verify lesson): the probed ckpt must be the newest directory
   under `<run>/ckpt`. A stale `ckpt/latest` pointer is FATAL.
5. **theta_1 provenance.** The stored `channels.distractor.vs_source`
   must agree with the recomputation from `se_probe_dims.npz` to
   1e-4 relative.

Report-only / annotating gates:

6. **Fit counters** (`fit_counters`, expect 5e5): any
   TRUNCATION-SUSPECT run annotates the primary statement (7-Aug
   standing disclosure rule), never silently.
7. **Degeneracy.** Per the B1 gate against the Stage-1 level reference
   (`stage1_reference_gated` over seeds 10–17, scale 1.0):
   `real_mean < ref/10` or `intrinsic_mean < ref/10` marks a run
   collapsed. ≥2 collapsed ⇒ GLOBAL-COLLAPSE, not adjudicable.
   Any collapsed run is excluded from the arm mean and flagged.
8. **Minimum panel.** ≥3 usable runs in EVERY arm, ≥7 usable overall;
   otherwise NOT-ADJUDICABLE. A usable panel below 8 annotates the
   primary as a FLAGGED SUBSET with the missing seeds listed.
9. **Normalizer floor — DIRECTIONAL rule (the onset reader's FLIPPED
   pattern).** A binding floor inflates the effective normalizer and
   therefore **deflates** theta_1. At phi=0.80 the law predicts the
   *lowest* value of the three arms, so deflation is
   **anti-conservative**: a LAW-CONFIRMED verdict with any floored run
   in the phi=0.80 arm is **NOT-ADJUDICABLE**. At phi=0.95 the law
   predicts the *highest* value, so deflation is conservative and a
   confirm stands with the floor disclosed.
10. **T6 NORMALIZATION GUARD (new, registered here).** The panel's T6
    showed theta_1's arm-invariance across the det/gauss reward arms was
    a *cancellation*: the absolute dist/pos disagreement ratio rose
    1.64x while the position key's symlog normalizer fell 1.56x. The
    same masking could operate across phi arms. The reader therefore
    reports, per arm: `d_pos` (the phi-free instrument floor, registered
    at 0.002016 with CV 8.8% over 24 runs and 5 scales), `d_dist`, the
    distractor and position raw normalizers, and the unnormalized
    dist/pos ensemble-variance ratio. **Pre-stated threshold:** if the
    arm-to-arm relative spread of `d_pos` exceeds **17.6%** (2x the
    registered 8.8% CV), the primary statement is annotated
    **NORMALIZATION-CONFOUNDED** — a disclosure, not a veto, because the
    law's estimand is theta_1 itself, but the paper must then print the
    absolute decomposition beside the ratio.

---

## 6. WHAT IS **NOT** CLAIMED WHATEVER HAPPENS

Frozen so a fire cannot be over-read:

- No mechanism. The panel established that Lambda has no route into the
  probe's normalization that does not pass through the phi-free constant
  `d_pos`. A confirmed phi-scaling makes the coincidence far harder to
  dismiss; it does not supply the derivation.
- No promotion of any Tier-3 refuted item (sqrt(Lambda) ~ rho,
  theta_1 = rho^2, s^1.9, the c-bar amplitude match, "theta_1 is not a
  space invariant", the EVA innovation-space provenance claim). Those
  stay refuted regardless of this wave.
- No onset claim (§4.3).
- No cross-family or cross-task generality: one task
  (`dmc_cheetah_run`), one objective (p2e), one dose position.
- The variance-ledger flagship (c* invariant, the a/b explanation of the
  certified 2.1x, the two-sided phi ladder) is untouched by this wave in
  either direction.

---

## 7. FROZEN ARTEFACTS

| item | path | state at registration |
|---|---|---|
| this prereg | `prereg/PREREG_phisweep_20260830.md` | frozen |
| reader | `uncfield/se_phisweep_read.py` | written + selfchecked BEFORE compute; ONE execution; explicit `--output` |
| wave spec | `ops/waves/phisweep/spec.yaml` | frozen |
| producer | `scripts/uncfield_se.sbatch` | `DISTRACTOR_THETA` added, default-inert, BIND-CHECKed |
| MC for the symlog predictions | reproduced in the reader's `SYMLOG_MC` constant block with the generating parameters recorded | frozen |

Read invocation (ONE execution, explicit output):

```
python -m uncfield.se_phisweep_read \
    --runs "<runroot>/se_phi*_s14*" \
    --stage1_runs "<stage1 bundle>/se_cheetah_seed*" \
    --output artifacts/phisweep_read_<date>
```
