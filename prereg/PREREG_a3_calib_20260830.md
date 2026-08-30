# PREREG — A3 CALIBRATION PILOT (Track-A generality leg), 30 Aug 2026

**Program:** Track A of `Plan_FollowupPrograms_20260821.md`. **Parent
registrations:** `PREREG_nfi_avoidance_20260821.md` (A1),
`PREREG_trackA_scarecrow_20260822.md` (+ amend1), and
`PREREG_nfi_scale_exhibit_amend3_20260816.md` §2, which owns the ramp
construction this pilot re-derives. User GO 24 Aug (A3 queued
GO-when-idle); promoted to build 30 Aug.

**This document registers a CALIBRATION, not a test.** It fires no
hypothesis, adjudicates no claim, and its runs never enter an
analysis. Its sole output is the constant vector that the A3
generality wave will be registered against, plus the pre-specified
rule that chooses which environment A3 runs in. It is registered
separately, and BEFORE the A3 main wave, so that the choice of
environment cannot be made after seeing an A3 outcome.

## 1. Why A3 cannot reuse the A1 constants

The A1/A2 planted quadruple is cheetah-calibrated. Four constants are
environment-specific:

| constant | cheetah value | role |
|---|---|---|
| `distractor.mod_lo` | −0.32203162 | ramp foot |
| `distractor.mod_hi` | +0.19711795 | ramp shoulder |
| `*.gate_threshold` | −0.13009691 | region split |
| flat comparator `scale` | 0.35891393 | amplitude match |

Transplanting them unchanged would put the ramp outside the second
environment's reachable range, making `m` almost constant — which
silently converts the gradient arm into a level arm and would
guarantee a null for a reason that has nothing to do with generality.

## 2. Established provenance (30 Aug, by direct reproduction)

The cheetah constants have **two different sources**, and this was
verified rather than assumed:

- `planted.basesd` = 0.0976 and `distractor.basesd` = 1.215 come from
  the **random-policy proprio inventory** already implemented as
  `SMOKE=1` in `scripts/uncfield_se.sbatch` (2000 steps, rng seed 0).
  Re-running it on the execution site returns **0.09712176** and
  **1.21378463** — within 0.5% of the registered pins, the residual
  being dm_control version drift. This is CPU-only and is **Phase A**.
- The four ramp constants do **not** come from that inventory. Under a
  random policy cheetah's `position[0]` spans only [−0.215, −0.065],
  against registered pins spanning [−0.32203162, +0.19711795]. They
  come from the **2e4-step p2e smoke run's replay** — an agent that
  actually locomotes. This needs a GPU and is **Phase B**.

Amendment-3 §2 records the cheetah split means as 0.2512 / 0.4666.
Their unweighted mean is **0.35891393**, the registered flat scale, to
all eight digits. That identity is the arithmetic signature of a median
split, and it independently confirms that this pilot's decomposition
is the registered one. The reader enforces its general
(visitation-weighted) form as gate **G5**.

**Disclosed deviation-from-unknown.** The cheetah smoke run's own
planted/distractor state is not recoverable from the record. Phase B
therefore pins its own configuration on principle (§3) rather than
claiming to match it.

## 3. Phase B design (pinned)

- **Runs:** `se_a3cal_finger_s200` (`dmc_finger_spin`, seed 200) and
  `se_a3cal_hopper_s201` (`dmc_hopper_hop`, seed 201). 2e4 steps each,
  the cheetah smoke convention. Seeds are calibration-only and disjoint
  from every registered arm seed.
- **`DISTRACTOR_DIM = 0` (NON-CIRCULARITY, registered).** The OU
  channel whose amplitude ramp is being calibrated must not shape the
  visitation used to set that ramp, or the constants are fit to their
  own manipulation. The frozen reader hard-gates `distractor.dim == 0`
  and rejects the run otherwise.
- **`planted` stays ON** at the Phase-A per-environment `basesd`
  (finger 0.54193702, hopper 0.59473863). It is Stage-1 substrate and
  carries no gate and no modulation, so it has no spatial structure and
  cannot induce the region asymmetry being measured.
- Everything else Stage-1 verbatim: `expl.mode p2e`,
  `disag_bootstrap True`, `disag_ens 8`, `disag_scale 1000.0`, penalty
  off, all gates off. The reader hard-gates each.
- **No producer change.** Every override above already exists in
  `scripts/uncfield_se.sbatch`; this wave does not touch it.

## 4. Candidate set, and why walker is excluded

`dmc_walker_*` exposes **no `position` key** — its observations are
`orientations` (14), `height` (scalar), `velocity` (9). Hosting the A1
design there would require re-keying the modulation to `height`,
which changes the manipulation from "avoid a spatial region" to "avoid
a posture". That is a change of meaning, not a re-calibration, and it
would confound any A3 outcome with the re-keying. Walker is therefore
excluded and the TODO's "walker or finger" resolves to **finger**, with
`dmc_hopper_hop` added as the second `position`-carrying candidate (a
planar locomotor, structurally the nearer analogue of cheetah).

Candidates measured 30 Aug: `finger_spin` position dim 4;
`hopper_hop` position dim 6; `cheetah_run` position dim 8 (control).

## 5. Registered read (frozen reader `uncfield/se_a3cal_read.py`,
built and selfchecked BEFORE this compute)

Per candidate, over the ENTIRE synced replay (deterministic full chunk
scan, no sampling rng): `mod_lo` = min, `mod_hi` = max,
`gate_threshold` = median, and flat `scale` = the visitation-weighted
mean of `m = clip((position[0] − lo)/(hi − lo), 0, 1)`; plus the split
means and E[m²] on each side of the threshold.

**Validity gates (all must pass):** G1 replay ≥ 90% of 2e4 steps; G2
non-degenerate spread; G3 median split within [0.40, 0.60] of
visitation (catches a mass point at the median); G4 mean `m` below the
threshold strictly positive; G5 the weighted decomposition identity to
1e-9.

**SELECTION RULE (pre-specified).** Among candidates passing every
gate, A3 runs in the one whose **delivered amplitude contrast**
E[m|above]/E[m|below] is closest to cheetah's **1.8575×**. Tie-break:
larger position dim, then task name. Rationale: a fair generality test
transplants the *same manipulation strength*; a second environment
that delivers a much weaker or much stronger contrast would confound
generality with dose. If no candidate passes, the verdict is **NO
VALID CANDIDATE** and the A3 generality leg does not launch.

**Selfcheck status:** PASS, 10 scenarios including five negative tests
— degenerate spread trips G2, a median mass point trips G3, a short
replay trips G1, `distractor.dim = 8` trips the non-circularity pin,
and a cheetah `basesd` on a finger run trips the Phase-A pin.

## 6. What this pilot does NOT license

It does not license any statement about avoidance, generality, or the
Track-A hypothesis in a second environment. The A3 main wave (8 runs,
generality leg; plus the persistence leg extending the existing A2
checkpoints) is a separate registration, written after this read and
pinned to its constants.
