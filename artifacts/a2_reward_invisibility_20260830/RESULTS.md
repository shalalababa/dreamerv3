# A2 SCARECROW — REWARD-INVISIBILITY CERTIFICATE

**ENGINEERING-MODE / DESCRIPTIVE.** This certifies an architectural
fact about the A2 apparatus. It is **not a registered primary**, has
no prereg, and changes no registered verdict. It upgrades "the
scarecrow fence is reward-invisible" from a claim about the code to a
claim with a measured bound attached.

Run 2026-08-30 13:18–13:40 CST. CPU only, saved data only, one
execution of `a2_reward_invisibility.py` on the pre-stated plan in
`PRESTATE.md` (written before any statistic was computed).

Companion: `artifacts/a2_scarecrow_read_20260824/RESULTS.md` (the
registered occupancy read: scare Δ = −0.0632, p = .0143).

---

## 0. Verification chain

- Data: `local_results/a2_scarecrow_20260824_184000_light/runroot/
  sc_*_s*/scores.jsonl` — 20 runs × 496 episodes, from the light
  bundle pulled local per the bundle NOTES.
- **40/40 files (20 `scores.jsonl` + 20 `config.yaml`) hashed against
  the committed manifest `manifests/a2_scarecrow_20260824_184000_light.sha256`
  (commit 3e24defd) — 0 mismatches**, before any number was read.
- Per-run asserts inside the reader: 496 episodes each (20/20),
  strictly increasing `step` (20/20), arm pins (20/20, §1).
- Repo untouched: the only files written are this artifact directory.

---

## 1. Level 1 — reward-FUNCTION invisibility (exact, not statistical)

The fence is the distractor wrapper's amplitude-modulation triple.
ctrl carries `distractor.scale 0.35891393, mod_key ''`; scare carries
`scale 1.0, mod_key position, mod_lo −0.32203162, mod_hi 0.19711795`;
scare2 is the same with `scale 2.0`. The modulation multiplies the
emitted OU amplitude by `m = clip((x − mod_lo)/(mod_hi − mod_lo),0,1)`
— a spatial gradient in the **observation**.

Three code facts fix the reward channel:

| fact | location |
|---|---|
| the distractor wrapper writes only `obs['distractor']`; `reward` is explicitly excluded from everything it touches | `embodied/envs/distractor.py:105,108` (write), `:73` (exclusion) |
| the only wrapper that writes `obs['reward']` is `RegionPenalty`, and it is **constructed only when `penalty.scale != 0`** | `dreamerv3/main.py:275`; `embodied/envs/regionpenalty.py:62` |
| in `reward_free` arms the reward head is **not an optimizer target at all** — enforced by a tripwire assert, not by convention | `dreamerv3/agent.py:114`, `:172` |

Measured per-run pins (20/20 asserts passed, `arm_pins` in the JSON):

| arm | `RegionPenalty` constructed | `expl.penalty_mix` | `distractor.mod_key == position` | reward in objective |
|---|---|---|---|---|
| ctrl s76–79 | no | false | no | **no** (`reward_free`) |
| scare s80–83 | no | false | yes | **no** (`reward_free`) |
| scare2 s84–87 | no | false | yes | **no** (`reward_free`) |
| pen1 s88–91 | yes, λ = 0.5 | true | no | **yes** |
| pen2 s92–95 | yes, λ = 2.0 | true | no | **yes** |

**So at the level of the reward function the certificate is exact, not
probabilistic: the scarecrow arms differ from control in one
observation-side constant, the reward map is byte-identical, and in
all three of ctrl/scare/scare2 the reward signal is not connected to
the objective in the first place** (p2e explorers; `self.rew` absent
from `self.modules`, asserted at `agent.py:172`).

What remains to be measured is the weaker and more interesting
statement: whether the **realized reward stream** the agent collects
is also unchanged. It is not exactly unchanged. §3–§5.

---

## 2. Pre-stated statistics and tests (from `PRESTATE.md`)

- **T1** final-window mean return — mean `episode/score` over the last
  100 of 496 episodes.
- **T2** full-trajectory AUC — trapezoidal area of score against
  `step`, divided by the step span.
- **T3** final-window distribution — two-sample KS D on the arm's
  pooled 4 × 100 final-window episode scores.

Exact permutation, run-level exchangeability. 4v4 → C(8,4) = 70
assignments, two-sided floor 2/70 = .0286. Pooled 8v4 →
C(12,4) = 495, two-sided floor 2/495 = .0040.

## 3. Per-arm and per-run distributions

| arm | T1 mean ± sd | T2 mean ± sd | per-run T1 |
|---|---|---|---|
| ctrl | **50.83 ± 1.93** | **41.83 ± 2.78** | 52.99 / 48.97 / 49.45 / 51.90 |
| scare | 43.70 ± 5.42 | 38.18 ± 2.52 | 50.64 / 43.08 / 43.66 / 37.42 |
| scare2 | 42.62 ± 9.78 | 34.75 ± 3.76 | 43.44 / 45.41 / 52.47 / 29.15 |
| pen1 (λ 0.5) | −12.23 ± 5.16 | −22.59 ± 1.31 | −11.16 / −6.28 / −18.79 / −12.70 |
| pen2 (λ 2.0) | −24.28 ± 2.98 | −60.87 ± 9.17 | −25.10 / −24.04 / −20.40 / −27.58 |

Within-episode spread, ctrl: all-episode sd 20.9–25.6, final-100 sd
22.8–28.5. **The scare/scare2 arm means sit inside one within-arm sd
of the treated arms** (scare sd 5.42 vs |Δ| 7.12; scare2 sd 9.78 vs
|Δ| 8.21).

## 4. The contrasts

| contrast | T1 Δ (p₂ / p₁) | T2 Δ (p₂ / p₁) | T3 KS D (p) | pre-stated expectation | outcome |
|---|---|---|---|---|---|
| scare − ctrl | **−7.12** (.0857 / .0429) | **−3.65** (.1429 / .0714) | 0.158 (.1143) | indistinguishable | **not separated** |
| scare2 − ctrl | −8.21 (.1143 / .0571) | −7.08 (.0571 / .0286) | 0.180 (.1714) | indistinguishable | **not separated** |
| {scare ∪ scare2} − ctrl (8v4) | −7.67 (.0727 / .0222) | −5.36 (**.0242** / .0121) | 0.169 (.0626) | indistinguishable | **T2 separates, uncorrected** |
| **pen1 − ctrl** | **−63.06 (.0286 = FLOOR)** | **−64.41 (.0286 = FLOOR)** | **1.000 (.0286 = FLOOR)** | distinguishable | **FIRES on all 3** |
| **pen2 − ctrl** | **−75.11 (.0286 = FLOOR)** | **−102.70 (.0286 = FLOOR)** | **1.000 (.0286 = FLOOR)** | distinguishable | **FIRES on all 3** |
| scare2 − scare | −1.09 (.9714) | −3.43 (.1714) | 0.045 (1.0000) | descriptive | no dose separation in return |

**Positive control (the gate on the whole certificate): PASSES.**
Both penalty arms separate from ctrl at the exact 4v4 two-sided floor
on all three pre-stated statistics, with **KS D = 1.000** — the
final-window return distributions are completely disjoint. The
instrument reads the reward channel and fires when the reward channel
is touched. Permutation-null sd for scale: 23.97 (pen1 T1), 3.79
(scare T1) — pen1's Δ is 2.63 permutation-null sd's out and saturates
the 70-assignment resolution; scare's Δ is 1.88 out and does not.

**Multiplicity over the 9 invisibility tests** (3 contrasts × T1/T2/T3),
Holm step-down: smallest adjusted p = **.218**. **No invisibility test
survives correction.** The single sub-.05 raw value (pooled T2,
p = .0242) is one of nine.

## 5. Effect-size bounds

Δ(scare − ctrl), 95% intervals. The permutation-inverted interval is
the honest one (exact, conservative — with 70 assignments α = .05
rejects only at 2/70 = .0286, so coverage ≥ 95%). The n = 4 percentile
bootstrap is included as pre-stated but **under-covers badly at n = 4**
and should not be quoted.

| contrast | stat | Δ | perm-inverted 95% CI | Welch 95% CI | bootstrap (under-covers) |
|---|---|---|---|---|---|
| scare − ctrl | T1 | −7.12 | **[−15.53, +1.65]** | [−15.32, +1.07] | [−11.84, −2.08] |
| scare − ctrl | T2 | −3.65 | **[−9.63, +1.93]** | [−8.25, +0.95] | [−6.79, −0.60] |
| scare2 − ctrl | T1 | −8.21 | [−23.80, +3.45] | [−23.45, +7.03] | [−17.34, −0.76] |
| scare2 − ctrl | T2 | −7.08 | [−14.74, +0.54] | [−12.93, −1.23] | [−11.02, −3.23] |
| pooled − ctrl | T1 | −7.67 | [−16.11, +0.84] | [−13.97, −1.37] | [−13.01, −2.90] |
| pooled − ctrl | T2 | −5.36 | [−10.06, **−0.67**] | [−9.69, −1.03] | [−8.74, −2.16] |

**Footprint ratio.** |Δ| and the widest permutation-CI bound as a
percentage of the penalty arms' reward-channel footprint:

| stat | scare point | scare CI bound | vs pen1 (63.06 / 64.41) | vs pen2 (75.11 / 102.70) |
|---|---|---|---|---|
| T1 final-window | 7.12 | 15.53 | 11.3% point, **≤ 24.6% CI** | 9.5% point, **≤ 20.7% CI** |
| T2 AUC | 3.65 | 9.63 | 5.7% point, **≤ 15.0% CI** | 3.6% point, **≤ 9.4% CI** |

### The certificate sentence

> **The fence moves occupancy by −6.32 pp (p = .0143, complete
> separation) while moving final-window return by −7.12 points, 95%
> exact-permutation CI [−15.53, +1.65] on a control mean of 50.83, and
> trajectory-AUC return by −3.65, CI [−9.63, +1.93] on 41.83. Neither
> is separable from zero at the 4v4 exact floor (p = .086, .143;
> Holm-adjusted ≥ .218 across all nine invisibility tests), while both
> penalty arms separate at the floor on all three statistics with
> KS D = 1.000. The scarecrow's reward-channel footprint is bounded at
> ≤ 24.6% of the λ = 0.5 penalty arm's footprint on final-window
> return and ≤ 15.0% on trajectory AUC, with point estimates of 11.3%
> and 5.7%.**

### Occupancy→return conversion (derived)

| arm | occupancy Δ | T1 Δ | as % of ctrl T1 | T1 points per pp occupancy |
|---|---|---|---|---|
| scare | −6.32 pp | −7.12 | −14.0% | 1.13 |
| scare2 | −7.36 pp | −8.21 | −16.2% | 1.12 |

The two doses give the same conversion rate to two decimals — the
residual return decrement scales with the displacement, which is the
signature of a **behavioral** cost, not a reward-function change.

---

## 6. What the certificate does and does not license

**Licensed.**

1. "The scarecrow adds no reward term." **Exact** — code path plus
   20/20 config pins (§1). Not a statistical claim.
2. "In the scarecrow arms the reward signal is not in the objective at
   all." **Exact** — `reward_free` p2e, reward head excluded from the
   optimizer, tripwire-asserted (`agent.py:172`).
3. "The fence's reward-channel footprint is an order of magnitude
   below an explicit penalty's." **Measured** — ≤ 24.6% (T1) /
   ≤ 15.0% (T2) at the CI bound; 11.3% / 5.7% at the point estimate.
4. "The instrument would have detected a reward-channel change."
   **Measured** — both positive controls at the exact floor on all
   three statistics, KS D = 1.000.

**NOT licensed.**

5. "Return is unchanged." **False as stated.** Both scarecrow arms
   run below control on every statistic, in the same direction, and
   the pooled 8v4 AUC contrast reaches p = .0242 uncorrected
   (Holm .218). The measured decrement is 14–16% of control return.
   The defensible phrasing is *bounded*, not *zero*.
6. Zero-cost framing. The A2 read's "at zero reward-channel
   footprint" (Consequences bullet) is **stronger than this
   measurement supports** and should be revised to
   "at ≤ 1/4 of the explicit penalty's reward-channel footprint, with
   a measured 14% return decrement that is not separable from seed
   noise at n = 4."
7. A like-for-like ratio against the penalty arms. `RegionPenalty`
   **replaces** rather than adds the reward (`regionpenalty.py:62`),
   so pen `episode/score` is a penalty return, not a task return on
   the ctrl/scare scale. The positive control is therefore
   separation-by-construction: it proves detection capability, not an
   independent effect, and the footprint ratio is an
   order-of-magnitude bound. This caveat was stated in `PRESTATE.md`
   before computing.
8. Any registered claim. Engineering-mode, no prereg, n = 4/arm.

**Mechanism reading.** The fence does not enter the reward; it changes
where the agent goes; the agent — which is not optimizing task reward
at all — consequently collects 14% less of it. That chain is exactly
what "reward-invisible control surface" should mean, and the honest
tool claim is: **a control surface that costs 14% of incidental return
per 6.3 pp of displacement, bought without writing anything into the
reward function.**

---

## 7. Exploratory diagnostic (NOT in the pre-stated plan, no verdict)

Quartile means (124 episodes each), and scare−ctrl by quartile:

| arm | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| ctrl | 34.30 | 44.60 | 42.93 | 50.02 |
| scare | 29.39 | 35.57 | 41.14 | 44.07 |
| scare2 | 28.66 | 34.33 | 32.44 | 40.32 |
| pen1 | −49.14 | −20.29 | −15.36 | −12.16 |
| pen2 | −166.76 | −37.42 | −27.80 | −24.73 |

| gap | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| scare − ctrl | −4.91 (.229) | −9.03 (.086) | −1.79 (.571) | −5.95 (.057) |
| scare2 − ctrl | −5.64 (.143) | −10.27 (.057) | −10.50 (.029) | −9.69 (.057) |

The decrement is present in Q1 (steps ~16k–140k) and does not
monotonically grow. This is a diagnostic only; it is post-hoc, it is
not in `PRESTATE.md`, and it bears on no statement above.

---

## Files

- `PRESTATE.md` — the analysis plan, written before compute.
- `a2_reward_invisibility.py` — the reader (one execution).
- `a2_reward_invisibility.json` — all numbers, per-run and per-contrast.

Python 3 / numpy 1.26.4 / scipy 1.17.1, single CPU, ~3 s.
