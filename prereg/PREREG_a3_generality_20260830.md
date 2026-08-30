# PREREG — A3 GENERALITY LEG: does the A1 avoidance result transplant?

Registered 30 Aug 2026. Paper 5 (uncertainty field), Track A of
`Plan_FollowupPrograms_20260821.md`. Status: **FULL registration
discipline** — this wave adjudicates a paper claim, so predictions,
decision rule and reader are frozen BEFORE any compute.

**Parent registrations.** `PREREG_nfi_avoidance_20260821.md` (A1 — the
result on trial for generality, and the source of the primary
statistic), `PREREG_a3_calib_20260830.md` (the calibration pilot) and
`PREREG_a3_calib_amend1_20260830.md` (**Amendment 1 §6 is the authority
for every constant in this document**). Upstream estimand:
`PREREG_nfi_scale_exhibit_amend2_20260814.md` §2 (occupancy) and
`amend3` §2 (the ramp construction).

**Instruments.** Reader `uncfield/se_a3gen_read.py` (new, frozen,
selfchecked before this compute). Wave `ops/waves/a3_generality/spec.yaml`.
Producer `scripts/uncfield_se.sbatch` — **NO producer delta** (§8.4).

---

## 1. WHAT IS ON TRIAL

A1 registered, on fresh cheetah seeds, that *agents whose epistemic
objective overprices a spatially-structured zero-information channel
AVOID the region where that channel's amplitude is highest*. The
registered primary fired: occupancy hetero2 **0.4599** vs flat2
**0.5197**, Δ = **−0.0598**, one-sided exact permutation
**p = .0143 = 1/70**, the C(8,4) floor — the third complete separation
in three independent waves (`artifacts/a1_avoidance_read_20260822/`).

That result is **scoped to one environment**. This wave asks the single
next question a reviewer asks: *is it a property of the objective, or a
property of cheetah?* It transplants the A1 design to
**`dmc_finger_spin`** with a re-calibrated planted quadruple, chosen and
solved by the pre-registered calibration pilot so that the **delivered
dose and the delivered amplitude contrast match cheetah's** — a fair
generality test moves the environment and holds the manipulation
strength fixed, or it confounds generality with dose.

**This registration covers the GENERALITY leg ONLY** (§8.2).

---

## 2. DESIGN

### 2.1 Factors

| factor | level |
|---|---|
| task | `dmc_finger_spin` (Amendment-1 G6 selection) |
| arms | **hetero** (solved mod quadruple, `scale` 1.0) vs **flat** (empty mod, `scale` = FLAT_SCALE) |
| seeds | **210–213** hetero, **214–217** flat |
| steps | 5e5 (A1 pin) |
| objective | p2e disagreement, reward-free (Stage-1 pin) |
| distractor | dim 8, `basesd` 1.35803102, `theta` 0.1 (φ = 0.9) |
| planted | `source_key` position, `basesd` 0.54193702 |
| gates | ALL training-time gates OFF, both wrappers |
| penalty | inert (`scale` 0.0, `penalty_mix` False) |

8 runs, ~7.3 h each, ~58 GPU-h, **all on one instance** (the 1-Aug rule:
GPU jobs are not job-to-job deterministic across hosts, and a 4-vs-4
permutation is precisely where a device confound is indistinguishable
from the effect).

**Arms are never seed-paired.** Seeds 210–217 are fresh: verified
30 Aug that `SEED=21x`, `seed: 21x` and "seeds 21x" occur nowhere in
`prereg/`, `ops/waves/` or `STUDY_LEDGER.md` (the only 21x hits in
those trees are decimal fragments of `1.215` and `0.215`). The SE
family's registered seeds run no higher than 147 (enumerated in
`PREREG_phisweep_20260830.md` §2.1), and the only seeds at or above 200
anywhere in the repo are the A3 calibration pilot's **200/201** and
Paper 2's unrelated **202**.

### 2.2 Pin provenance — every constant traces to Amendment 1 §6

| constant | value | role |
|---|---|---|
| `TASK` | `dmc_finger_spin` | environment (G6 selection) |
| `BASESD_PLANTED` | `0.54193702` | Phase-A random-policy inventory |
| `BASESD_N` | `1.35803102` | Phase-A inventory, all-key mean |
| `DISTRACTOR_MOD_KEY/INDEX` | `position` / `0` | A1 protocol |
| `DISTRACTOR_MOD_LO` | `-4.589493315966835` | solved ramp foot (G6, no clipping) |
| `DISTRACTOR_MOD_HI` | `7.469163326854745` | solved ramp shoulder (G6, no clipping) |
| `FLAT_SCALE` | `0.35948428173662794` | realized m̄ of the solved ramp = the amplitude-matched comparator |
| `GATE_THRESHOLD` | `-0.21752405166625977` | **READER constant**: the region split |

**`GATE_THRESHOLD` is not a producer knob.** It is the pilot visitation
median and it defines the high-amplitude REGION the reader splits
occupancy on. Like A1, this wave trains with every gate OFF; the wave
spec passes `GATE_KEY=` and `GATE_THRESHOLD=0.0` **deliberately**.
Setting the producer's gate to −0.2175 would switch on the Stage-2
gating manipulation and destroy the design. The reader pins the region
constant FATALLY and pins the producer's gates to OFF, so neither error
can pass silently.

### 2.3 Arm construction (A1, verbatim in shape)

Hetero carries the mod quadruple at `scale` 1.0; flat carries an EMPTY
mod at `scale` = FLAT_SCALE. The two arms are therefore
**amplitude-matched in expectation over the pilot's visitation** and
differ only in whether that amplitude has a followable spatial
gradient. The flat arm's constant-m-via-scale is exact, so the
comparator is incentive-immune: there is no residual amplitude
advantage for the agent to chase.

### 2.4 Lane layout (registered, because it is a design property)

Declaration order is **BLOCKED** (4 hetero, then 4 flat).
`assign_lanes` is `i % nlanes`, so at 4 lanes each lane gets exactly one
run of each arm. An alternating order would produce ARM-PURE lanes and
confound arm with GPU. Replayed 30 Aug through
`wavespec.balanced_order → pair_runs → assign_lanes` on the actual runs
list: **4 lanes** → `L0{het210,flat214} L1{het211,flat215}
L2{het212,flat216} L3{het213,flat217}`; **2 lanes** → 2 hetero + 2 flat
per lane; **8 lanes** → one run per lane (A1's own layout). **3 lanes is
arm-imbalanced (2/1, 1/2, 1/1) and is not to be used.**

---

## 3. PRE-STATED PREDICTION

**H-GEN.** If A1's avoidance is a property of the objective rather than
of cheetah, then at matched delivered dose and matched delivered
contrast the finger hetero arm occupies the high-amplitude region LESS
than the amplitude-matched flat arm:
`occupancy(hetero) − occupancy(flat) < 0`.

**H-NULL / boundary.** If the effect is environment-specific — because
finger's spatial coordinate is not behaviourally costly to avoid, or
because the objective's misprice does not localize there — the
difference is zero or positive.

The direction is **registered**, exactly as in A1. It is the same
direction A1 registered and fired, so a reversal here would be a
substantive finding and is pre-declared as *not* a confirmation of
anything (§6).

---

## 4. DECISION RULE (frozen, pre-stated)

### 4.1 PRIMARY — A1's registered statistic, transplanted verbatim

**Estimand.** Per run, `occupancy` = the fraction of steps with
`position[0] > −0.21752405166625977` over the **ENTIRE synced replay
buffer** — deterministic full chunk scan, no sampling rng
(`se_m3_read.occupancy`, the amendment-2 estimand, imported not
reimplemented). Validity floor **≥ 1e5 steps**.

**Statistic.** `Δ = mean(occupancy | hetero) − mean(occupancy | flat)`
over 4 vs 4 runs.

**Test.** Exact label permutation over all **C(8,4) = 70** assignments
(`se_m3_read.exact_perm_p`, imported), **one-sided NEGATIVE**, observed
assignment included.

**FIRES iff `p ≤ .05` AND `Δ < 0`.** The minimum attainable p is
**1/70 ≈ .0143**, so the decision rule is arithmetically identical to
A1's: a fire requires the observed split to be among the **3 most
extreme of 70** — near-complete separation.

**Power disclosure (registered, inherited from A1 #27 m17).** This wave
has essentially no power against moderate effects. A non-fire does NOT
distinguish "no effect" from "moderate effect". This is stated before
the data exist and must be printed with any non-fire.

**Refusal.** The denominator is pinned at **8**. A missing, extra or
duplicated run REFUSES the whole read (there is no registered fallback
statistic for a partial panel, and dropping a run silently would change
the estimand). Refusal happens before any output is written, so the
one-execution guard survives a repair-and-rerun.

### 4.2 Secondaries

**NONE that fire.** A1's SECONDARY-1 (the ranking contrast) reads `pse2`
out of the provenance-gated `se_probe` record, which this leg does not
produce (§7). Reported unconditionally, **reporting-only, never a
fire**, mirroring A1's non-fire reporting: per-arm occupancy BCa
intervals; task return (mean `episode/score` over the final 100 entries)
as a **two-sided** exact permutation; per-run fit counters; per-run
realized dose (§5.3).

---

## 5. VALIDITY GATES

### 5.1 FATAL per-run identity pins

A defective run refuses the WHOLE read — never a silent drop (the B1
pattern). From each run's `config.yaml`:

1. **Seed ↔ arm map is the ARM AUTHORITY.** 210–213 ⇒ hetero,
   214–217 ⇒ flat. Any other seed, or a seed presented in the wrong
   arm, is FATAL. The config must AGREE with the seed's registered arm,
   never relabel it.
2. **Task** `== dmc_finger_spin`.
3. **The mod quadruple.** Hetero: `mod_key == position`,
   `mod_index == 0`, `mod_lo == -4.589493315966835`,
   `mod_hi == 7.469163326854745`, `scale == 1.0`. Flat: `mod_key == ''`,
   `mod_lo == 0.0`, `mod_hi == 1.0`,
   `scale == 0.35948428173662794`.
4. **The basesd pair.** `planted.source_key == position`,
   `planted.basesd == 0.54193702`, `distractor.dim == 8`,
   `distractor.basesd == 1.35803102`. A cheetah default (0.0976 / 1.215)
   leaking through is the single most likely producer fault and is
   FATAL on both.
5. **OU persistence.** `distractor.theta == 0.1` (φ = 0.9, the A1
   value). The φ-sweep wave moves exactly this knob on this producer
   and runs immediately before A3 on the same box.
6. **Stage-1 objective pins.** `expl.mode == p2e`,
   `disag_bootstrap is True`, `disag_bootstrap_prob == 0.8`,
   `disag_ens == 8`, `disag_scale == 1000.0`, **`disag_head == det`**
   (this key postdates A1's pin block and the alea wave ran a `gauss`
   arm on this producer — an unpinned head is a live leak path).
7. **Everything else inert.** Training-time gates OFF on both wrappers
   (`planted.gate_key == ''`, `distractor.gate_key == ''`); penalty
   machinery inert (`penalty.scale == 0.0`, `penalty.gate_key == ''`,
   `expl.penalty_mix is False`); `run.steps == 5e5`.
8. **Ingestion cross-check.** The two replay ingestion paths (the frozen
   occupancy estimand and the dose scan) must agree on the trace length
   and on the recomputed occupancy to 1e-12.

### 5.2 Adjudication order

0. per-run FATAL pins (§5.1) — refusal, not a cell;
1. panel identity: exactly 4 + 4, seeds exact, no duplicates —
   refusal;
2. **COLLAPSE CHECK** → `GLOBAL-COLLAPSE` (§5.4);
3. **occupancy validity**: 4 + 4 runs with ≥ 1e5 replay steps →
   `NOT-ADJUDICABLE`;
4. **the PRIMARY** (§4.1);
5. the realized-dose block (§5.3) — computed unconditionally, reported
   on every branch, **never a veto**.

### 5.3 REALIZED-DOSE (DOSE-DRIFTED) annotation — FLAT-ARM-KEYED

The calibration matched the dose on the **pilot's** visitation. This
gate verifies the transplant on the **adjudicating** data.

**Computation.** The solved ramp (§2.2) is applied to each run's OWN
replay visitation — `m = clip((x − mod_lo)/(mod_hi − mod_lo), 0, 1)`,
split at the registered region threshold — yielding that run's
**realized m̄** and **realized amplitude ratio** `E[m|above]/E[m|below]`.
The reader IMPORTS `ramp_stats` from `uncfield/se_a3cal_amend1_read.py`,
so the ramp algebra is literally the code that produced the registered
constants.

**Targets (registered).**

| quantity | target | band | derivation |
|---|---|---|---|
| realized m̄ | **0.35948428173662794** | ±15% | Amendment §6 `FLAT_SCALE` = the solved ramp's achieved m̄ on the pilot visitation = `0.35891393 × (1 + 0.001589104487050558)` — cheetah's registered m̄ times the amendment §4 residual (0.16%). Reproduces the artifact value EXACTLY (difference 0.0). |
| realized amplitude ratio | **1.8636677250074283** | ±25% | The solved ramp's achieved ratio on the pilot visitation (amendment §4: "1.8637× (0.33%)") = `(0.4666/0.2512) × (1 + 0.0033290452676080417)` — cheetah's registered contrast 1.8574840764331213 (amendment-3 §2 split means) times the published 0.33% residual. Reproduces `artifacts/a3_calib_amend1_20260830/constants.json` `rows[finger].solved.ratio` EXACTLY (difference 0.0). |

Both derivations are asserted at import time in the reader, so an
edited target cannot be read.

**Rule.** For the **FLAT** arm — which carries no avoidance pressure, so
its delivered amplitude is constant and its visitation is the
uncontaminated control — a run outside either band annotates the read
**DOSE-DRIFTED**. The annotation is carried into the primary's own
statement string and into `RESULTS.md`. **It NEVER vetoes and never
changes the primary verdict.**

For the **HETERO** arm the same quantities are reported
**DESCRIPTIVELY ONLY and are NEVER flagged.** Its realized dose is
endogenous to the hypothesis: Amendment 1 §2's own language is that
*"diversion itself raises visited m — that is the measured effect"*.
Flagging it would make a confirmed effect annotate itself as an
instrument fault. This asymmetry is registered before the data exist.

**BAND PROVENANCE (disclosed verbatim).** The bands were measured on
four cheetah φ-sweep control runs (distractor active, unmodulated) with
the registered cheetah ramp applied to their own replay: realized m̄
mean **0.3820**, sd **0.0178** (rel **4.7%**); systematic shift vs the
smoke target **+6.4%**; max |dev| **11.5%** (m̄) / **21.7%** (ratio).
**Two caveats, both disclosed:** (i) those runs were measured at
**88k–112k of 500k steps (~20% of training)**, so the dispersion is not
a full-training dispersion; (ii) the bands are **transplanted from
cheetah to finger**, an environment whose visitation geometry is
different by construction. The bands are set at ±15% / ±25%, i.e. above
the observed per-run maxima, and are therefore a **coarse
instrument-fault detector, not a calibration test**.

### 5.4 Collapse check (replaces A1's probe-based degeneracy gate)

A1 gated on absolute disagreement levels against **Stage-1 CHEETAH**
per-run minima. Level references are not portable across environments
and no finger Stage-1 cohort exists (§7). The registered substitute is
the weakest defensible form, using training telemetry only: the last
logged `train/expl/disag_replay_rew` must be **present, finite and
strictly positive** in every run. Any dead run ⇒ **GLOBAL-COLLAPSE**,
nothing downstream adjudicated. If the key is absent in ≥ 1 run the
primary proceeds and is annotated **COLLAPSE-CHECK-UNAVAILABLE**. No
invented absolute threshold is introduced.

### 5.5 Report-only / annotating gates

- **Fit counters** (`fit_counters`, expect 5e5): any TRUNCATION-SUSPECT
  run annotates the primary statement (7-Aug standing disclosure rule),
  never silently.
- **Instrument flag** CLEAN / VALIDITY-VIOLATIONS from the replay floor.

---

## 6. OUTCOME MAP — what each verdict licenses

| cell | licensed wording |
|---|---|
| **AVOIDANCE-GENERALIZES** (primary fires) | *"The avoidance signature transplants to a second environment at a dose- and contrast-matched manipulation"* — scoped to this substrate, this objective, this delivered contrast, and to the two environments actually run (cheetah, finger). It licenses no third environment and no mechanism (§9). Any DOSE-DRIFTED annotation must be printed beside it. |
| **AVOIDANCE-DOES-NOT-GENERALIZE** (primary does not fire, Δ ≥ 0 or p > .05) | **A GENERALITY BOUNDARY, NOT A REFUTATION OF A1.** The registered A1 cheetah result stands exactly as read on 22 Aug; what fails is the extension. The licensed wording is *"the effect did not reproduce in `dmc_finger_spin` at matched dose"*, and it MUST be printed with the §4.1 power disclosure — at n=4v4 this design cannot separate "no effect in finger" from "a moderate effect in finger". Paper 5 / the Track-A paper must then scope every avoidance sentence to cheetah and cite this leg. |
| **REVERSED** (Δ > 0, reported as a non-fire) | Reported descriptively as an unregistered direction, never as a claim. It would be the strongest available evidence that the sign is environment-dependent and would motivate a separate registration. |
| **GLOBAL-COLLAPSE** | Validity cell. No wording either way; the objective was dead. |
| **NOT-ADJUDICABLE** | Validity cell (short replay / panel below 4+4). No wording either way. |

**Asymmetric-with-duty fence, inherited from A1 #27 M8.** This leg may
never SUPPORT a Paper-5 claim in either outcome. But if it does NOT
generalize, Paper 5's descriptive avoidance paragraph and the Track-A
paper MUST cite the non-generalization as a scope caveat before any
subsequent submission. One-directional insulation from bad news is what
that clause forbids.

---

## 7. DEVIATIONS FROM THE A1 READ (each with its reason)

The primary is A1's, verbatim in estimand, statistic, sidedness, test
and n. Everything below is a deliberate divergence.

| # | deviation | reason |
|---|---|---|
| D1 | **Region threshold** −0.21752405166625977 instead of A1's −0.13009691 | The region is a property of the environment. The A3 threshold is the calibration pilot's own visitation median, i.e. the same *construction* applied to finger (Amendment §6). Using cheetah's number would split finger visitation at an arbitrary point. |
| D2 | **Ramp endpoints SOLVED, not the observed min/max** | A1's range rule is a heuristic that happened to deliver cheetah's dose on cheetah. Amendment 1 §2: the two endpoints are solved against both cheetah targets under the no-clip gate G6. Registered in the parent amendment, not here. |
| D3 | **No MISPRICE GATE** | A1's gate requires `se_probe` on the hetero arm. This leg runs no probe (a probe pass is a separate post-training job, per A1's own spec: *"the 2 se_probe passes are NOT in this wave"*). Consequence, registered: **this leg does not verify that the distractor is mispriced in finger**, and no pricing claim may be made from it (§9). |
| D4 | **No GRADIENT-DELIVERY GATE** | `uncfield/se_region_probe.py` hardcodes the CHEETAH region threshold (`REGION_THRESHOLD = se_m3_read.GATE_THRESHOLD`) and exposes no threshold flag. It is a frozen, EXECUTED instrument and may not be edited. A region-resolved delivery gate for finger needs a NEW registered instrument. The registered substitute is §5.3, which checks the delivered *amplitude field* (from replay) rather than the delivered *disagreement* (from the probe) — a weaker check, and disclosed as such. |
| D5 | **Degeneracy gate replaced** (§5.4) | A1's reference is a Stage-1 cheetah level cohort; levels are not portable across environments and no finger Stage-1 cohort exists. Tuning an absolute finger threshold now would be inventing a constant. |
| D6 | **No SECONDARY-1 (ranking)** | It reads `pse2` from the provenance-gated probe record — same cause as D3. Nothing replaces it; the leg registers no firing secondary. |
| D7 | **No mechanism decomposition** | A1's within-run `dis_high/dis_low`, `real_high/real_low` ratios come from the region probe — same cause as D4. |
| D8 | **Missing/extra runs REFUSE rather than yield NOT-ADJUDICABLE** | A1 could report a NOT-ADJUDICABLE cell for an incomplete panel; here the denominator is pinned at 8 up front so that no partial-panel statistic can be improvised after seeing which runs survived. Short-replay runs still yield A1's NOT-ADJUDICABLE cell. |
| D9 | **Added pins**: `disag_head == det`, `disag_bootstrap_prob == 0.8`, `distractor.theta == 0.1`, `run.steps == 5e5`, flat-arm inert ramp (0.0, 1.0) | These keys postdate A1's pin block, and two of them (`disag_head`, `theta`) are moved by other registered waves running on the SAME producer and the SAME box. A1's pin block cannot see them. |
| D10 | **DOSE-DRIFTED annotation added** (§5.3) | A1 needed no such check: it ran in the environment its constants were calibrated in. A3 is a transplant, so the delivered dose is a live risk and is measured on the adjudicating data. Registered as an annotation, never a veto, because a veto keyed on an endogenous quantity would let a confirmed effect disqualify itself. |

---

## 8. DISCLOSURES

### 8.1 The post-pilot amendment chain (stated first)

The environment was selected by a **post-data amendment**
(`PREREG_a3_calib_amend1_20260830.md`, written after seeing the
calibration pilot). That amendment discloses itself in its §0 and argues
— correctly, and this registration adopts the argument — that it is not
outcome-coupled: the pilot fires no hypothesis, A3 had produced no data,
and the defect corrected is a property of environment geometry and ramp
algebra, computable before any A3 run and invariant to anything A3 could
show. The parent reader `uncfield/se_a3cal_read.py` had already
executed; it is EVIDENCE, was not edited, and the amendment reader
IMPORTS it. **This reader in turn imports both**, so the ingestion path,
the ramp algebra and the config pins are literally the same code across
the pilot, the amendment and the adjudicating read.

### 8.2 SPLIT REGISTRATION

The Track-A A3 programme has two legs. **This document registers the
GENERALITY leg only** (8 fresh finger runs). The **PERSISTENCE leg**
(extending the existing A2 SCARECROW checkpoints) is a **SEPARATE
future registration**, gated on staging ~6 G of A2 checkpoint files from
RCC — a dependency that has nothing to do with this calibration.

**The two legs stand alone. Neither leg's outcome gates the other**, and
neither may be cited as support for the other. A future persistence
registration must state its own predictions and its own decision rule
without reference to what this leg returned.

### 8.3 The contrast caveat, and what remains of it

The parent prereg recorded that **neither candidate reproduced cheetah's
1.8575× contrast under the original (unsolved) range rule** — finger
delivered 5.1787×, hopper 4.7984×, against target m̄ values of 0.4375
and 0.1352 vs cheetah's 0.35891393 — and proposed carrying that as a
permanent scope caveat. Amendment 1 §5 **withdrew** the caveat: the
solved finger ramp delivers cheetah's dose to **0.16%** and cheetah's
contrast to **0.33%**, with **zero clipping** (G6).

What remains, and is registered here: that match was established on the
**pilot's** visitation — 19,232 replay steps of a 2e4-step p2e run with
`distractor.dim = 0` (the non-circularity design) — not on the 5e5-step
adjudicating runs, whose visitation is shaped by a live distractor and,
in the hetero arm, by the manipulation itself. **§5.3 is exactly the
measurement of that residual**, which is why it is registered as an
annotation on the flat (uncontaminated) arm.

### 8.4 Producer verification (no delta)

Every environment variable in the wave's `cmd` was checked line-by-line
on 30 Aug against `scripts/uncfield_se.sbatch`: all are consumed by the
launch command AND covered by the script's BIND-CHECK `want()` block —
`SOURCE_KEY`, `BASESD_PLANTED`, `BASESD_N`, `DISTRACTOR_DIM`,
`DISTRACTOR_SCALE`, `DISTRACTOR_THETA`, `DISTRACTOR_MOD_KEY/INDEX/LO/HI`,
`DISTRACTOR_GATE_KEY/INDEX/THRESHOLD`, `GATE_KEY/INDEX/THRESHOLD`,
`PENALTY_MIX`, `PENALTY_SCALE`, `PENALTY_GATE_KEY/INDEX/THRESHOLD`,
`EXPL_CONFIG`, `DISAG_BOOTSTRAP`, `DISAG_HEAD`, `SEED`, `STEPS`.
**No producer change is registered or required.**

**One gap, disclosed:** `TASK` is consumed but is **not** in the
BIND-CHECK `want()` block — the composed config's `task` key is never
asserted against the wave's `TASK`. A mis-set task would therefore not
be killed at launch. It is caught twice downstream instead: by the wave
spec's `done_when` regex on `task:` in the composed config, and FATALLY
by the reader (§5.1.2). Closing the gap would be a producer delta
touching every prior SE wave's launch path, so it is **disclosed rather
than patched** for this wave; a `want('task', env['TASK'])` line is
proposed to the ops session as a standalone, default-free change.

### 8.5 INSTRUMENT LESSON (recorded, basis of no decision here)

Two lines for the methods appendix, both established by the calibration
pilot and neither used to decide anything in this document:

> **"The registered statistic would have selected the environment that
> fails the integrity gate — a criterion can be pre-specified,
> faithfully applied, and still pick wrong when it measures one
> projection of a two-dimensional quantity."**

The parent's registered rule ranked candidates on the delivered
amplitude RATIO alone and selected `dmc_hopper_hop`; the two natural
dose statistics (m̄ and the ratio) diverge on that data, and the
amendment's response was not to adjudicate between them but to remove
the choice by solving both simultaneously.

> **"Dose defect, not vanished gradient."**

Hopper's exclusion is a statement about delivered dose, not about the
existence of a gradient. Under the A1 range rule hopper's `m` varied
from **0.047** (below the split) to **0.224** (above) — a real, followable
gradient of contrast **0.177**, against cheetah's **0.215** (0.4666 −
0.2512). What disqualified it is the LEVEL: m̄ **0.135** against
cheetah's **0.359**, and under the solved ramp it can match m̄ **or** the
ratio but never both while covering its own support (10.58% / 16.36%
residuals vs the 2% G6 tolerance), because `position[0]` is bounded
above by exactly 0 with 96% of visitation crushed into the bottom
quarter of its range. **The geometry forbids the solution — the run did
not fail.**

---

## 9. WHAT IS **NOT** CLAIMED, WHATEVER HAPPENS

Frozen so a fire cannot be over-read:

- **No misprice claim in finger.** This leg runs no probe. That the
  agent overprices the distractor in `dmc_finger_spin` is NOT measured
  here, so no sentence of the form "avoids the channel it overprices"
  is licensed by this leg alone.
- **No mechanism.** No region-resolved disagreement decomposition
  exists for finger (D4).
- **No persistence claim** (§8.2).
- **No generality beyond the two environments actually run.** Two
  environments is two, not "environments in general"; and the
  candidate set was itself restricted to tasks exposing a `position`
  key (walker exposes none, and re-keying to `height` would change the
  manipulation's meaning).
- **No claim about other objectives, other doses, or the pixel
  setting.**
- **A non-fire is a GENERALITY BOUNDARY and never a refutation of the
  registered A1 cheetah result.**

---

## 10. FROZEN ARTEFACTS

| item | path | state at registration |
|---|---|---|
| this prereg | `prereg/PREREG_a3_generality_20260830.md` | frozen |
| reader | `uncfield/se_a3gen_read.py` | written + selfchecked BEFORE compute; ONE execution; explicit `--output` |
| wave spec | `ops/waves/a3_generality/spec.yaml` | frozen at this commit |
| producer | `scripts/uncfield_se.sbatch` | UNCHANGED (§8.4) |
| calibration | `prereg/PREREG_a3_calib_20260830.md` + `amend1` | executed; `artifacts/a3_calib_read_20260830/`, `artifacts/a3_calib_amend1_20260830/` |
| A1 comparison | `prereg/PREREG_nfi_avoidance_20260821.md`, `artifacts/a1_avoidance_read_20260822/` | executed evidence |

**Selfcheck status: PASS**, 11 scenario groups — target algebra
re-derived at runtime; planted effect fires at the 1/70 floor; null and
REVERSED do not fire; wrong-arm-map and unregistered-seed are FATAL; 18
producer-default / cheetah-leak fixtures are FATAL; DOSE-DRIFTED fires
in both directions on both statistics, stays silent inside the bands
and never vetoes; the hetero arm is NOT flagged at +50% m̄ / +50% ratio
drift; missing / duplicate panels are refused; a dead objective gives
GLOBAL-COLLAPSE, a short replay gives NOT-ADJUDICABLE, absent telemetry
annotates rather than kills; end-to-end read with the one-execution
guard; and the **cheetah smoke provenance control** — the shared
ingestion path reproduces all four registered cheetah constants to
<1e-6 and `ramp_stats`, the exact function the dose block calls,
returns m̄ 0.35891393 and ratio 1.8570× on the real smoke replay.

Read invocation (ONE execution, explicit output):

```
python -m uncfield.se_a3gen_read \
    --hetero "<runroot>/se_a3gen_het_s21*" \
    --flat   "<runroot>/se_a3gen_flat_s21*" \
    --output artifacts/a3_gen_read_<date>
```

Freeze = commit of this file + the reader + the wave spec. Then, and
only then, the 8 submissions.
