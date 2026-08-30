# PREREG AMENDMENT 1 — A3 calibration pilot, 30 Aug 2026

**Parent:** `PREREG_a3_calib_20260830.md`. **Reader:**
`uncfield/se_a3cal_amend1_read.py` (new; selfcheck PASS, 7 scenarios).
**Artifacts:** `artifacts/a3_calib_read_20260830/` (parent read, as
executed) and `artifacts/a3_calib_amend1_20260830/` (this read).

## 0. Disclosure, stated first

**This amendment was written AFTER seeing the pilot data.** It changes
how the A3 ramp constants are derived and therefore which environment
A3 runs in. The parent reader has already executed; it is EVIDENCE and
has not been edited. The new reader IMPORTS it, so replay ingestion and
the Stage-1 / non-circularity config pins are literally the same code.

**Why this is not the move this program has previously rejected.** The
WB rev-1 failer-only extension was refused because the post-data change
was outcome-coupled: it selected on the hypothesis. This is the
opposite case. The pilot fires no hypothesis (parent §0). A3 has
produced no data. The defect corrected here is a property of
environment geometry and of the ramp algebra, computable before any A3
run and invariant to anything A3 could show. Proceeding unamended would
not have been discipline — it would have spent ~50 GPU-h to buy an
uninterpretable result.

## 1. What the pilot revealed

Under the A1 range rule (`mod_lo`/`mod_hi` = the observed min/max of
`position[0]`), NEITHER candidate reproduced cheetah's delivered
manipulation:

| | cheetah (registered) | finger_spin | hopper_hop |
|---|---|---|---|
| m̄ (delivered dose) | **0.35891393** | 0.4375 | 0.1352 |
| amplitude ratio | **1.8575×** | 5.1787× | 4.7984× |

The A1 range rule is a HEURISTIC that happened to deliver cheetah's
dose on cheetah. It does not transport: it pins the ramp to the
extremes of a distribution whose SHAPE differs between environments.

**Instrument lesson, recorded and not used as the basis of any
decision:** the two natural dose statistics — m̄ and the amplitude
ratio — DIVERGE on this data, and the parent's registered rule used the
ratio, which selected `hopper_hop`: the environment that fails the
integrity gate registered below. A criterion can be pre-specified,
faithfully applied, and still select the wrong arm when it measures one
projection of a two-dimensional quantity. Worth an appendix line.

**A coverage-floor gate was considered and REJECTED on the data.**
Cheetah's own 5–95% visitation coverage of its ramp is **45.8%**
(computed from the smoke replay). A floor at half of cheetah's (22.9%)
does NOT exclude hopper, whose coverage is 25.1%. Tuning that floor
downward until hopper failed would have been fitting the gate to the
candidates. It is not adopted.

## 2. The amendment: solve, do not choose

`m = clip((x − mod_lo)/(mod_hi − mod_lo), 0, 1)` carries **two** free
knobs against **two** targets, and the gate threshold — the median of
visitation — does not move with them. The system is therefore
generically exactly determined. So the amendment does NOT adjudicate m̄
against the ratio; it **satisfies both simultaneously** by solving for
the ramp endpoints. The choice between the two statistics never has to
be made.

`solve_ramp` is a fixed grid plus local descent, no rng, reproducible
bit-for-bit from the same replay.

## 3. Registered validity gate G6 (cheetah-anchored)

Cheetah's ramp clipped **exactly nothing**: its endpoints were the
observed extremes, so `m` traverses [0,1] across the visited support
and the gradient is followable EVERYWHERE. Amendment-3 §2 makes this a
design requirement in as many words — "a followable gradient exists
everywhere, with no boundary discontinuity (removing M3's
snap-transition confound)". A ramp narrower than the support would clip
and create a saturated region where the gradient VANISHES: the
snap-transition confound rebuilt.

**G6 requires a solution that (a) contains the entire visited support
(no clipping) and (b) meets BOTH cheetah targets within 2%.** The
standard is a property of cheetah's construction, not a threshold
fitted to the candidates.

## 4. Adjudication (executed)

| | solved ramp | m̄ achieved | ratio achieved | verdict |
|---|---|---|---|---|
| finger_spin | [−4.5895, +7.4692] | 0.359484 (**0.16%**) | 1.8637× (**0.33%**) | **VALID** |
| hopper_hop | [−1.2124, 0.0] | 0.320928 (10.58%) | 1.5535× (16.36%) | **EXCLUDED** |

Hopper can match m̄ **or** the ratio, never both, while covering its own
support: its optimum pins against `hi = xmax = 0.0` and stalls there.
`position[0]` is bounded above by exactly 0 with 96% of visitation
crushed into the bottom quarter of its range, so the geometry — not the
run — forbids the solution. A synthetic hopper-like support fails G6 at
10.57%/17.40% in selfcheck, confirming the exclusion is geometric.

**SELECTED: `dmc_finger_spin`.**

## 5. The fair-transplant caveat is DELETED

The parent recorded that neither candidate reproduced cheetah's 1.857×
contrast, and proposed carrying that as a permanent scope caveat on
A3's generality claim. **That caveat is withdrawn.** The retuned finger
ramp delivers cheetah's dose to 0.16% and cheetah's contrast to 0.33%,
so A3 transplants the same manipulation strength and the generality
test is clean. No new compute was required: this is a design-time
computation on already-collected visitation.

## 6. A3-main pins (registered)

```
TASK=dmc_finger_spin
SOURCE_KEY=position
BASESD_PLANTED=0.54193702
BASESD_N=1.35803102
DISTRACTOR_MOD_KEY=position
DISTRACTOR_MOD_INDEX=0
DISTRACTOR_MOD_LO=-4.589493315966835
DISTRACTOR_MOD_HI=7.469163326854745
GATE_THRESHOLD=-0.21752405166625977
FLAT_SCALE=0.35948428173662794
```

## 7. Instrument validation

Run against the cheetah smoke replay
(`local_results/uncfield_se_smoke_20260812_155552/se_smoke0`), the
ingestion path shared with the parent reader reproduces **all four**
registered cheetah constants to <1e-6 — `mod_lo` −0.32203162, `mod_hi`
+0.19711795, threshold −0.13009691, `flat_scale` 0.35891393 — and
amendment-3 §2's split means (0.2512 / 0.4666) and E[m²] ratio (3.195)
exactly. The instrument is provably the registered one. This control
runs inside `--selfcheck` whenever the smoke replay is present.
