# EVA-at-scale — ADJUDICATION against the registered predictions

Run 56095696 (rev 3), 12/12 cells, `controls_pass: True` everywhere,
41 min. Data: `artifacts/se_eva_read_20260829/` (8 det `se_cheetah_seed10-17`
+ 4 gauss `se_alea_s120-123`); 6 streams/cell, 6784 audited reads/stream,
40 704 read-key rows per h=1 leg. Adjudicated against `PREDICTIONS.md`
rev 2 **as written**. Where a prediction fails, it fails.

---

## 0. VERDICT TABLE

| # | prediction | verdict | decisive number |
|---|---|---|---|
| **PRIMARY** | `position_on_bracket` locates the θ₁ misprice on a measured bracket | **CONSTRUCT-INVALID** | the bracket's "fully justified" end is **unattainable for any vens > 0**; `pb ≈ 1` is the generic value, not evidence |
| **(a)** | distractor certifies **OVER** at h=1; ρ_d < ρ_p | **FAILED, 12/12** | OVER pooled log10E = **0.0**; distractor certifies **UNDER** at 1.64e4–3.76e4; ρ_d = 1.85–2.41 **>** ρ_p = 0.033–0.073 |
| **(b)** | distractor \|gap\| ≥ 2×floor **AND** \|Δlogρ(vel,pos)\| ≤ 1×floor | **FAILED, 12/12** (conjunction) | first half met (gap/floor 3.2–658); second half missed by **200–38 845×** (\|Δlogρ(vel,pos)\| = 3.26–3.72) |
| **(c)** | dup0 ≈ position; ρ_dup0 ≤ ρ_dup1 ≤ ρ_dup2 | **MET 11/12; 12th inside its own floor** | ρ_dup0 vs ρ_position agree to 3–4 s.f.; floor 8.7e-5–0.0163; `dup_monotone` True 11/12 (det_s16 inverts by 0.00036 = 0.5 %, inside its floor of 0.0163) |
| **(d)** | distractor profile ratio within ±0.15 of predicted **AND** > 0.10 from both nulls | **FAILED, 12/12** | ρ(2)/ρ(15): predicted 1.28–1.35, measured **0.924–1.067**; \|distractor − velocity\| < 0.10 in **11/12** |
| **(4)** | arm split (B10) | **no repair signal** | θ₁ det 5.258 vs gauss 5.230; `position_on_bracket` gauss 0.991 vs det 0.925 (gauss no better) |
| controls | SYNTHG / inflate / marginal / scalings / supply | **ALL PASS, 12/12** | worst SYNTHG margin **+2.39** log10 |

**Two of five registered predictions fail outright, one is met (11/12
strictly), one is a control, and the primary is invalid.** The instrument worked; the
hypotheses did not.

---

## 1. THE PRIMARY IS CONSTRUCT-INVALID — withdraw the 85–101 % reading

Both your reading and mine were that `position_on_bracket` = 0.85–1.01
means "the distractor-vs-position excess is 85–101 % unjustified". **That
is wrong, and the error is structural, not statistical.**

`R(t)`, the squared residual, is computed against the **generative
predictive mean, which both legs share by design** (docstring: "they differ
ONLY in the audited dispersion"). Only `S` differs: `S_C = S_P + vens`. So

    rho^C / rho^P  =  E[R/S_C] / E[R/S_P]

is a **pure S-ratio effect**. `R` does not depend on `S` at all, so no
value of this statistic can distinguish "the extra dispersion was needed"
from "it was not". Two consequences:

1. **The `pb = 0` ("fully justified") end is unattainable.** It requires
   `rho^C = rho^P` with `vens > 0`, which is impossible. A bracket with an
   unreachable endpoint is not a bracket.
2. **`pb ≈ 1` is the generic value**, obtained whenever `vens(t)` is
   uncorrelated with `R(t)`. Simulated with `R` constructed **independent
   of `vens` by construction**:

   | relationship | gap | endpoint | `pb` |
   |---|---|---|---|
   | vens uncorrelated with R | −0.0665 | −0.0677 | **0.983** |
   | vens tracks R | −0.0918 | −0.0677 | 1.356 |
   | vens anti-tracks R | −0.0462 | −0.0677 | 0.682 |

   `pb ≈ 1` appears in the case where the ensemble term is, by
   construction, telling us nothing — and also where it would be
   "justified". The statistic cannot separate them.

The registered derivation also assumed leg P is calibrated (`rho^P = 1`).
Measured `rho^P` is 0.03–0.08 on position and ~2 on the distractor. **The
endpoint's premise fails in the data as well as in principle.**

**Adjudication: the primary is withdrawn as a measure of justification.**
This is not post-hoc bar-bending — the argument is checkable from the code
alone and would have held whatever the numbers came back as. It should have
been caught at design time; review B1 corrected the endpoint's *magnitude*
(rightly) and I did not then re-examine whether the endpoint had *content*.

**What survives, EXPLORATORY only** (derived after seeing data; needs its
own registration before it can carry anything): `pb − 1` is a **timing**
statistic — the correlation between where the ensemble puts its extra
dispersion and where the error actually is. Measured: det **0.852–0.992**,
gauss **0.975–1.006**. Read against the table above, that places `vens(t)`
between uncorrelated and mildly **anti**-correlated with realized error.
That is a genuinely interesting shape — it is what "the bonus is not
tracking the difficulty" would look like — but it is not what was
registered and it is not adjudicated here.

**The honest replacement question**, answerable from what was measured:
*is the distractor's total claimed dispersion right-sized?* → `rho_d^C`.
Answer in §2.

---

## 2. (a) DISTRACTOR OVER AT h=1 — FAILED, and in the opposite direction

Registered: certificate on the **OVER** leg, `rho_d < rho_p`, Δlogρ < 0.

Measured at `c_recon` (the identified scaling), 12/12:

| quantity | distractor | position | dup0 | velocity |
|---|---|---|---|---|
| ρ | **1.845 – 2.414** | 0.033 – 0.073 | 0.033 – 0.073 | ~1.85 |
| OVER pooled log10E | **≈ 0.0 (silent)** | **1.20e5 – 1.66e5** | ~1.3e5 | −8.7 |
| UNDER pooled log10E | **1.64e4 – 3.76e4** | 0.0 | 0.0 | 1.88e4 |

The distractor is **UNDER-dispersed by 1.85–2.41×** — its claimed
predictive dispersion is too *small* even after the ensemble bonus is added
— and it certifies on the side the registered mechanism said was
structurally protected. `rho_d/rho_p` = **29.1–70.2**, the reverse of the
registered inequality.

**The mechanism story behind (a) is refuted.** It reasoned from
`deter_frac` = 0.968–0.983 ("the epistemic claim is approximation error, so
it will not be matched by realized error") to an over-claim in innovation
space. The inference does not go through: the ensemble term is only
**4.8–7.1 %** of `S` on the distractor (`vens_share_of_S`), so its
composition cannot determine which side of 1 `ρ` falls on. `S` is
dominated by `c + vlat`, and those are what the audit actually sees.

**Scale caveat, stated because it limits the claim.** At `c_theory` the
distractor is over-dispersed (ρ = 0.287) and certifies OVER. The *side* is
not scale-robust; the *ordering* is: at both declared scalings and across
the whole c-grid, ρ_position < ρ_velocity < ρ_distractor, with
**ρ_d/ρ_p = 29.1–70.2 (c_recon) and 81.1–207.3 (c_theory)**. `sign_constant_over_grid`
is True in 12/12 with no `c*` in [0.05, 2].

---

## 3. (b) REAL CHANNELS — FAILED on the conjunction, and the position
## blowout is a likelihood finding, not a composition finding

Bar: distractor `|gap| ≥ 2×floor` **and** `|Δlogρ(velocity, position)| ≤
1×floor`.

- First half **met** 12/12 (`gap/floor` 3.2 – 658).
- Second half **missed by 2–4 orders of magnitude**:
  `|Δlogρ(vel,pos)|` = **3.26 – 3.72** against floors of 8.7e-5 – 0.0163,
  i.e. **200× to 38 845×** the bar.

The two "real" channels are not comparable to each other at all, so the
λ*-control as designed cannot function. **FAILED.**

### Is the position blowout "the composition story in innovation space"?

**No — and the instrument's own leg split is what rules it out.**

Position certifies OVER at log10E 1.20e5–1.66e5 with ρ = 0.033–0.073, i.e.
the model's claimed one-step dispersion on position is **13.8–30.3× too
large**. But on position `vens_share_of_S` is only **0.30–1.19 %**. The
ensemble term — the thing `deter_frac` describes — is essentially absent
from that certificate. It is a **leg-P** phenomenon (the `c + vlat`
likelihood), which leg C merely inherits.

The actual cause is more mundane and more checkable: **the decoder's
likelihood is homoscedastic across channels whose realized one-step
predictability differs by ~31×.** `symlog_mse` gives one constant `c` for
every key (`rssm.py:299` → `heads.py:127-130`). `c_recon` (identified from
the posterior reconstruction residual pooled over position *and* velocity)
comes out 0.016–0.037, which is simultaneously ~15–30× too large for
position and ~2× too small for the distractor and velocity. Pooling is
dominated by velocity's 9 dims with ~60× larger squared residuals, so the
single constant fits neither real channel.

So: a real, certificate-backed finding — but about the **observation
likelihood**, not about the disagreement ensemble, and therefore not
evidence for the composition/`deter_frac` story. Keeping those separate is
the point of having run leg P and leg C rather than one audit.

---

## 4. (c) DUP LADDER — MET 11/12 (the 12th inside its own floor)

`planted_dup0` is bitwise identical to `position` in the replay, and the
audit reproduces that to 3–4 significant figures on 12/12 cells
(e.g. det_s10 ρ 0.0603 vs 0.0601; det_s12 0.0505 vs 0.0504; gauss_s122
0.0612 vs 0.0614). Measured floor `|Δlogρ(dup0,position)|` =
**8.7e-5 – 0.0163**.

Graded control monotone in **11/12**: ρ_dup0 ≤ ρ_dup1 < ρ_dup2 with dup2
(50 % of basesd added) separating cleanly — det_s10 0.0603 / 0.0612 /
0.1496; gauss_s120 0.0334 / 0.0337 / 0.1057. **det_s16 inverts**: dup0
0.07368 vs dup1 0.07332, a gap of 0.00036 (**0.5 % relative**) against that
cell's own measured floor of **0.0163** — the inversion sits a factor of
~45 inside the instrument's resolution there. Reported as a strict 11/12
rather than waved through; dup2 separation is intact (0.1283).

This is the only registered prediction that is met, and
it is what licenses quoting any of the other numbers: the instrument's
resolution is **measured on this data**, not assumed.

---

## 5. (d) PERSISTENCE — FAILED, 12/12; the profile is flat

Bar: distractor floor-corrected ratio within **±0.15** of predicted **and**
separated from **both** nulls by **> 0.10**.

| ratio | predicted | distractor | position | velocity | band | separation |
|---|---|---|---|---|---|---|
| ρ(2)/ρ(8) | 1.16–1.19 | 0.942–1.058 | 0.591–0.975 | 0.941–1.150 | **2/12** in ±0.15 | **fails 11/12** (\|d−vel\| = 0.000–0.141) |
| ρ(2)/ρ(15) | 1.28–1.35 | 0.924–1.067 | 0.418–0.975 | 0.881–1.217 | **0/12** | fails |

Both halves fail on ρ(2)/ρ(15) (band 0/12); on ρ(2)/ρ(8) the band holds in
only 2/12 and the separation fails in 11/12.
The null comparators move **more** than the distractor, which is the
opposite of the registered discriminant. **FAILED.**

### Diagnosis: a horizon-independent level error swamps the horizon signal

The distractor's floor-corrected `rho(h)` is **flat**, 12/12:

```
cell            h2      h4      h8     h15      predicted: 1.43 1.35 1.22 1.10
det_s10       2.253   2.137   2.129   2.131
det_s15       1.968   1.982   1.969   1.990
gauss_s121    2.251   2.201   2.230   2.109
```

Within-cell spread across h is ≤ 0.12 while the AR(1) prediction requires a
fall of 0.33. An AR-coefficient misclaim **must** shrink with h (true and
claimed conditional variances both converge to the stationary variance). A
constant ≈ 2.0–2.25 is not a persistence error — it is a **stationary-scale
error**: the model's decoded distractor predictive is uniformly ~2.1× too
narrow at every horizon. That level error is ~9× the size of the predicted
persistence effect, and the profile ratio (built to be level-free) is then
dominated by whatever else varies with h.

**Three candidate explanations, adjudicated:**

1. **Different estimand — yes, and this is the main one.** L3 measured
   `L_full(h)/L_full(1)`: the decay, *in the model's own imagination*, of
   the round-trip-recoverable share of the ensemble's decoded variance.
   EVA measures realized-vs-claimed dispersion of the *generative
   predictive* against *real replay*. A model can believe the distractor's
   information persists (L3) while its h-step predictive spread is
   uniformly too narrow (EVA). Both were measured; they are not proxies for
   one another, and the registered prediction (d) treated them as if they
   were.
2. **Imagination-vs-replay dissociation — yes, and it connects to the
   protocol finding.** The in-world work found the ML-imagination battery
   *understates* the real farming surface. Here the polarity is reversed:
   the imagination-side instrument (se_lic L3) found horizon structure the
   replay-side instrument does not see. Combined, the two readings support
   the general claim — **imagination-side and replay-side accounts of the
   same WM diverge, and the direction is instrument-specific** — while
   refusing the specific claim that L3's persistence number would show up
   in innovations.
3. **The exogenous-episode filter — ruled out.** The filter removes
   **1.7–1.9 %** of steps (admissible 98.07–98.28 % of `n_steps`, measured
   per stream). Audited segment lengths vary (169–1001; the short ones are
   window-edge truncations, `episode_lengths.constant` is False), but at
   h ≤ 15 the loss is under 2 % either way. It cannot flatten a profile.

**Resolution-gate caveat, in the failure's favour and reported anyway:**
h=15 was **adjudicable in only 2/12 cells** (det_s17, gauss_s120); in the
other 10 the measured floor left insufficient headroom. So the ρ(2)/ρ(15)
row rests on an unresolved endpoint in 10/12 cells. My registered profile
bar did not require both endpoints to pass their own resolution gates —
a design gap. It does not rescue (d): ρ(2)/ρ(8), whose endpoints are
adjudicable 12/12, fails the separation criterion on its own.

---

## 6. (4) ARM SPLIT — a small consistent difference, but NOT the B4 repair

| quantity | det (n=8) | gauss (n=4) |
|---|---|---|
| `vens_share_of_S` distractor | 0.0655 [0.0591, 0.0709] | **0.0536** [0.0482, 0.0589] |
| `vens_share_of_S` position | 0.0089 [0.0075, 0.0119] | **0.0051** [0.0030, 0.0074] |
| ρ distractor (leg C) | 2.048 [1.845, 2.374] | 2.223 [1.854, 2.414] |
| bracket endpoint | −0.0599 | −0.0503 |
| `position_on_bracket` | 0.925 [0.852, 0.992] | 0.991 [0.975, 1.006] |
| θ₁ (from each run's `se_lic_probe.json`) | 5.258 [4.715, 5.640] | 5.230 [4.210, 5.744] |
| `deter_frac` | 0.975 | 0.979 |

The gauss arm carries a **~18 % smaller** distractor `vens_share_of_S` and
a **~43 % smaller** position share, with non-overlapping ranges on the
latter. But:

**This does not test the B4 repair, and must not be reported as if it did.**
Per the registered B10 scope, `vens` here is the **raw** decoded ensemble
variance from `disag.predict`, which returns the mu block for *both* heads
(`explore.py:65-71`). The gauss arm's aleatoric normalisation happens in
the **reward** (`explore.py:108-114`), in latent space, and is **not** in
`S_C`. So the contrast measures whether gauss *training* (a sigma head on a
stop-gradiented trunk) shifted the ensemble's decoded dispersion — a check
on `explore.py`'s design claim that mu training is det-identical. It
shifted it slightly; the claim is approximately but not exactly held.

**On the repair question itself the arms are indistinguishable**: θ₁ is
5.258 vs 5.230, and every leg-C quantity that could show a distractor
deflation shows none (gauss ρ_d is if anything *higher*, 2.223 vs 2.048).
That is an independent, innovation-space corroboration of the B4 read's own
verdict (share ratio 1.06, no repair) — obtained from a different estimand
and a different data path.

---

## 7. WHAT EVM'S PAPER CAN NOW SAY — with certificates attached

1. **The deployed observation likelihood is mis-calibrated across channels,
   certified anytime-valid on 12/12 checkpoints.** A single homoscedastic
   decoder scale serves channels whose realized one-step predictability
   differs by ~31×: position over-dispersed 13.8–30.3× (pooled
   log10E 1.20e5–1.66e5, α/2 = 0.005), distractor and velocity
   under-dispersed ~1.9–2.4× (pooled log10E 1.6e4–3.8e4). Ville, so
   robust to optional stopping and with no relapse.
2. **The distractor's h-step predictive is uniformly ~2.1× too narrow, flat
   in h ∈ {2,4,8}, certified UNDER in 12/12** — a stationary-scale error,
   distinct from and larger than any persistence error.
3. **Instrument resolution is measured, not assumed**: a bitwise-duplicate
   channel reproduces its source to 3–4 s.f. (floor 0.0001–0.0163) with the
   graded dup1/dup2 ladder monotone 12/12.
4. **The imagination-side and replay-side accounts of the same WM
   diverge**, in both directions across the two reads — a protocol claim
   now supported at deployed scale by two instruments with different
   estimands.
5. **The B4 gauss arm shows no distractor deflation in innovation space**,
   corroborating the B4 read from an independent estimand.

## …AND WHAT IT CANNOT SAY

1. **It cannot say EVA certifies the θ₁ = 5.26× misprice.** `vens` is
   0.3–7.1 % of `S`; the resolvable differential is ~5–6 %; and the
   statistic built to locate it (`position_on_bracket`) is
   construct-invalid. **No certificate attaches to the misprice.**
2. **It cannot say the distractor's disagreement claim is over-dispersed.**
   Registered prediction (a) failed in the opposite direction at the
   identified scaling.
3. **It cannot say the L3 ~3× persistence miscalibration is confirmed on
   real data.** Registered prediction (d) failed; the h-profile is flat and
   the nulls move more.
4. **It cannot make any statement about the deployed gauss *reward*** from
   leg C (B10 scope): leg C is the deployed claim only on the det arm.
5. **It cannot report which side of 1 a channel falls on as
   scale-free.** The side flips with `c` (distractor: over at `c_theory`,
   under at `c_recon`). Only the ordering ρ_pos < ρ_vel < ρ_dist and the
   ratio ρ_d/ρ_p (29–70× at `c_recon`, 81–207× at `c_theory`) are stable
   in sign across the declared grid — the ratio's *magnitude* is not.
6. **It cannot use `position_on_bracket` at all**, in either direction.

---

## 8. WHAT WOULD HAVE TO CHANGE BEFORE ANY RE-REGISTRATION

- **Replace the primary.** A justification test must let the data move
  something that `S` does not determine. Two candidates, neither yet
  registered: (i) `rho^C` itself against 1, per key, which is a genuine
  right-sizing statement but inherits the `c` scale problem; (ii) the
  **timing** statistic `pb − 1` (the vens(t)–R(t) correlation), which is
  scale-robust and is the only thing in this read that behaved like a
  property of the ensemble rather than of the likelihood.
- **The shared-`c` homoscedastic likelihood is the binding constraint on
  everything here.** Any per-channel claim needs either a per-channel scale
  (which the model does not have, so it would have to be identified and
  declared per key, costing the cross-channel comparison) or a statistic
  built to be invariant to it.
- **Require both endpoints of a profile ratio to pass their own resolution
  gates** before the ratio is adjudicable.
- The four EVA legs, the controls, and the driver need no change:
  `controls_pass` held 12/12, SYNTHG margins ≥ **+2.39** log10 (worst
  family across all 12 cells), the planted
  4× overclaim certified on every key, and the dup floor behaved exactly as
  constructed.
