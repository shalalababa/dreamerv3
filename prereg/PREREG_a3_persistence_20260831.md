# PREREG — A3 PERSISTENCE LEG: does the scarecrow fence hold at 1e6?

Registered 31 Aug 2026. Paper 5 (uncertainty field) / the Frame B
Control-Surface paper; Track A of `Plan_FollowupPrograms_20260821.md`.
Status: **FULL registration discipline** — this wave adjudicates a
paper claim, so predictions, decision rule and reader are frozen
BEFORE any compute.

**Parent registrations.** `PREREG_trackA_scarecrow_20260822.md` (A2 —
the result on trial, and the source of the primary statistic, the arm
construction and the region constant) with
`PREREG_trackA_scarecrow_amend1_20260823.md` (the full restart).
Upstream estimand: `PREREG_nfi_scale_exhibit_amend2_20260814.md` §2
(occupancy) and `amend3` §2 (the ramp construction). Executed A2
evidence: `uncfield/se_scarecrow_read.py` (frozen; **never edited by
this leg**) and `artifacts/a2_scarecrow_read_20260824/`.

**Instruments.** Reader `uncfield/se_a3per_read.py` (new, frozen,
selfchecked before this compute). Wave
`ops/waves/a3_persistence/spec.yaml`. Producer
`scripts/uncfield_se.sbatch` — **NO producer delta** (§8.4).

---

## 1. WHAT IS ON TRIAL

A2 registered, and its frozen reader returned on 24 Aug, that a
planted misprice FENCES a region: on `dmc_cheetah_run` at 5e5 steps
the scarecrow arm occupied the high-amplitude region **less** than the
amplitude-matched control —

| arm | per-run occupancy (whole replay, 5e5) | mean |
|---|---|---|
| `sc_scare` (seeds 80–83) | 0.4805 / 0.4546 / 0.4487 / 0.4736 | **0.46434** |
| `sc_ctrl` (seeds 76–79) | 0.4996 / 0.5254 / 0.5038 / 0.5813 | **0.52752** |

**Δ = −0.06319, exact C(8,4) one-sided p = .0143 = 1/70, complete
separation** (`artifacts/a2_scarecrow_read_20260824/`, cell
SCARECROW-FENCES).

That result is **scoped to 5e5 steps**. The tool claim the Control-
Surface paper wants to make — *plant a misprice, fence a region* — is
a claim about a deployed system, and a deployed system keeps training.
The single next question a reviewer asks is: *does the fence survive
continued learning, or does the agent's world model eventually learn
the planted channel is noise and walk back in?*

**Both answers are informative.** A fence that holds is a control
surface. A fence that erodes is a **shelf life**, which is a
publishable property of the tool and a mechanism claim in its own
right (the misprice is corrigible by data). Nothing about the paper's
spine depends on which way it goes.

This wave extends **only A2's registered PRIMARY pair** — `sc_scare`
and `sc_ctrl`. A2's `sc_scare2` (scale 2.0, seeds 84–87) and
`sc_pen1`/`sc_pen2` (explicit penalty, seeds 88–95) arms are **NOT
extended**: they carry A2's secondaries, not the claim on trial, and
extending them would triple the compute for contrasts nobody has asked
to be persistent. The frozen reader refuses their seeds FATALLY, so
the exclusion cannot be quietly undone at read time.

---

## 2. DESIGN

### 2.1 The eight runs

The eight A2 run dirs are **resumed IN PLACE** from staged copies and
trained from 5e5 to **1e6** total steps. Same seeds, same environment,
same manipulation, same producer: this is an EXTENSION, not a fresh
cohort, and no fresh seed is introduced.

| new run_id | A2 source run dir | seed | arm | `DISTRACTOR_SCALE` | mod quadruple |
|---|---|---|---|---|---|
| `a3per_scare_s80` | `sc_scare_s80` | 80 | scare | 1.0 | `position`, 0, −0.32203162, 0.19711795 |
| `a3per_scare_s81` | `sc_scare_s81` | 81 | scare | 1.0 | idem |
| `a3per_scare_s82` | `sc_scare_s82` | 82 | scare | 1.0 | idem |
| `a3per_scare_s83` | `sc_scare_s83` | 83 | scare | 1.0 | idem |
| `a3per_ctrl_s76` | `sc_ctrl_s76` | 76 | ctrl | 0.35891393 | `''`, 0, 0.0, 1.0 (inert) |
| `a3per_ctrl_s77` | `sc_ctrl_s77` | 77 | ctrl | 0.35891393 | idem |
| `a3per_ctrl_s78` | `sc_ctrl_s78` | 78 | ctrl | 0.35891393 | idem |
| `a3per_ctrl_s79` | `sc_ctrl_s79` | 79 | ctrl | 0.35891393 | idem |

**Why the run_ids are renamed.** The extension is a different wave
with a different bundle and a different read; A2's own run dirs still
live in the shared RCC runroot, and reusing their names is exactly the
"names are not identity" collision hazard the 28-Aug standing lesson
names. Every resume path is logdir-relative, so renaming costs
nothing. Identity is carried by the SEED in the composed config and by
the A2-PROVENANCE gate (§5.1.l), never by a directory name.

~7 h of new training each, ~56 GPU-h, **all on one instance** (the
1-Aug rule: GPU jobs are not job-to-job deterministic across hosts,
and a 4-vs-4 permutation is precisely where a device confound is
indistinguishable from the effect).

### 2.2 Pin provenance — every constant is A2's

Nothing is estimated for this leg. The reader IMPORTS rather than
restates: `MOD_KEY/MOD_INDEX/MOD_LO/MOD_HI/FLAT_SCALE/DIM/BASESD_N`
from `uncfield/se_grad_read.py` and
`GATE_KEY/GATE_INDEX/GATE_THRESHOLD/MIN_STEPS/exact_perm_p/occupancy`
from `uncfield/se_m3_read.py`, both executed instruments, and asserts
the values at import.

| constant | value | role |
|---|---|---|
| `TASK` | `dmc_cheetah_run` | environment |
| region split | `position[0] > -0.13009691` | `se_m3_read.GATE_THRESHOLD`, the CHEETAH split A2's own read used — **not** the finger constant `-0.21752405166625977` |
| `DISTRACTOR_MOD_KEY/INDEX` | `position` / `0` | A1-hetero quadruple |
| `DISTRACTOR_MOD_LO` | `-0.32203162` | A1-hetero quadruple |
| `DISTRACTOR_MOD_HI` | `0.19711795` | A1-hetero quadruple |
| scare `DISTRACTOR_SCALE` | `1.0` | A2 §1 |
| ctrl `DISTRACTOR_SCALE` | `0.35891393` | A1-flat, VERBATIM |
| `BASESD_PLANTED` / `BASESD_N` | `0.0976` / `1.215` | Stage-1 cheetah inventory |
| `DISTRACTOR_DIM` / `DISTRACTOR_THETA` | `8` / `0.1` | Stage-1 / φ = 0.9 |
| objective | `expl_p2e`, `disag_head det`, `disag_bootstrap True`, `disag_ens 8`, `disag_scale 1000.0` | Stage-1 |
| penalty block | inert (`penalty_mix False`, `scale 0.0`, gate key `''`) | A2 §1 |
| `STEPS` | `1e6` | **the only change from A2** |

The region split is a **READER constant**, not a producer knob: like
A1 and A2 this wave trains with every training-time gate OFF, and the
wave cmd passes `GATE_KEY=` / `GATE_THRESHOLD=0.0` deliberately.

### 2.3 The resume protocol

`scripts/uncfield_se.sbatch` `rm -f`s the stale `TRAINING_DONE` and
`mkdir -p`s the existing logdir; `dreamerv3/main.py` rewrites
`config.yaml` and calls `cp.load_or_save()`
(`embodied/run/train.py:86,90`), which finds the staged `ckpt`, and
loads the agent parameters, the step counter and — via
`cp.replay = replay` → `embodied/core/replay.py:Replay.load` — the
replay buffer from the run's own `replay/*.npz`. `while step <
args.steps` with `run.steps = 1e6` then continues from ~4.99e5.

**What is staged, per run:** `ckpt/` (the complete latest save —
`agent.pkl`, `step.pkl`, `replay.pkl`, `done` — plus the `latest`
pointer) and `replay/` (**all** chunks). **Nothing else.** In
particular `metrics.jsonl`, `scores.jsonl` and `ckpt_snapshots/` are
NOT staged, which is what makes `metrics.jsonl`'s first rows the
resume witnesses of §5.1.k and makes `scores.jsonl` an
extension-only record.

**"All" the replay chunks is not conservatism — it is the capacity
arithmetic.** `Replay.load` walks the directory NEWEST-FIRST and stops
once it has `capacity` items. `capacity = int(config.replay.size)` =
**5,000,000** (`dreamerv3/main.py:186`; `configs.yaml:58`; confirmed as
`replay: size: 5000000.0` in every A2 `config.yaml`). A2 produced
~4.95e5 items per run — **a tenth of capacity** — so the
newest-first cutoff never fires and a partial stage is simply a
smaller resumed buffer, silently. Measured staging cost: ~568 MB of
replay (~1 400 chunks) + ~44 MB of ckpt per run, ≈ 5 G for the eight.

**Staged copies must be MATERIALIZED BYTES.** The A2 read ran on the
full RCC bundle, which is hard-linked to the A2 runroot (the 8-Aug
"hard-linked bundles are not archives" rule). An in-place resume
MUTATES its inputs: it appends chunks to `replay/`, and
`elements.Checkpoint`'s `keep=1` cleanup DELETES the staged save
folder at the first post-resume save. A `cp -al` stage would therefore
rewrite the A2 archive. Copy content, verify, then push.

**Registered disclosure — the resume is not a bit-continuation.**
Process-level RNG streams restart at resume: the distractor and
planted wrappers are re-seeded from `SeedSequence([seed, index,
0xD0/0x5E])` at every process start (`dreamerv3/main.py:256,271`), and
the replay sampler and JAX streams likewise. Only the agent
parameters, the step counter and the replay contents carry over. The
OU distractor is stationary with φ = 0.9 (mixing time ~10 steps), so
restarting its stream is behaviourally negligible, and it happens
identically in both arms — but the extension is a fresh draw of the
noise realization, not a continuation of A2's, and no claim in this
document depends on it being one.

### 2.4 Lane layout (registered, because it is a design property)

`balanced_order` classes on the FIRST var only and `seed` is first and
unique in every entry, so the declaration order IS the lane order;
`per_gpu` is 1 and `assign_lanes` is `i % nlanes`.

The declaration order is **DOUBLE-BALANCED** — scare, ctrl, scare,
ctrl, ctrl, scare, ctrl, scare — so that at 4 lanes **every lane
carries one run of each arm** (no arm-pure lane ⇒ arm is never
confounded with GPU) **and every round carries 2 + 2** (no arm-pure
round ⇒ a round-wide disturbance, or an early instance release, cannot
load onto one arm). A blocked order balances lanes but makes round 1
arm-pure; a strictly alternating order balances rounds but makes lanes
arm-pure. **A2 Amendment 1 exists because round 1 of A2's original
layout was the entire control arm**, and this order is the one that
closes both failures at once.

Replayed 31 Aug through `wavespec.balanced_order → pair_runs →
assign_lanes` on the actual runs list:

- **4 lanes** → `L0{scare_s80, ctrl_s78} L1{ctrl_s76, scare_s82}
  L2{scare_s81, ctrl_s79} L3{ctrl_s77, scare_s83}` (1 + 1 each);
  rounds `[scare, ctrl, scare, ctrl]` and `[ctrl, scare, ctrl, scare]`.
- **2 lanes** → 2 + 2 per lane. **8 lanes** → one run per lane.
- **3 lanes is ARM-IMBALANCED** (1/2, 1/2, 2/0) and is not to be used.

---

## 3. PRE-STATED PREDICTIONS AND THE ESTIMAND

### 3.1 The hypotheses

**H-HOLD.** If the planted misprice is a durable control surface, the
displacement A2 measured is still present in behaviour generated
AFTER 5e5: `occupancy(scare) − occupancy(ctrl) < 0` on the extension
window.

**H-WASH.** If the world model eventually learns the planted channel
carries no information, disagreement over it decays, the fence stops
being paid for, and the extension-window contrast goes to zero.

No prior probability is registered on which fires; both are wanted
(§1).

### 3.2 THE WINDOWED ESTIMAND — read this before comparing anything

**The primary is computed on the EXTENSION WINDOW ONLY**: the replay
chunks produced after the resume.

A whole-1e6-buffer occupancy would be **half A2's own already-read
behaviour**, and would therefore report a fence even if the extension
contained none — mechanically, by dilution. The reader's selfcheck
demonstrates this with a positive control: on a synthetic panel whose
extension window is null (Δ = −0.0010, no fire), the same runs read
with the whole 1e6 buffer give Δ = −0.0347 at p = .0143, **a fire**.
That confound is the reason the window is registered.

> **Extension-window occupancy at 1e6 and A2's whole-replay occupancy
> at 5e5 are DIFFERENT ESTIMANDS.** Their LEVELS are not comparable.
> HOLD and WASH-OUT are defined against the **CONTRAST**, never
> against raw levels, and the two columns are never tested against
> each other.

### 3.3 Window identification is data-independent and auditable

Each run dir carries **`STAGED_MANIFEST.txt`**, the exact sorted list
of `replay/*.npz` present at staging time. The wave's own cmd writes
it on the instance **before the producer is launched**, once (a
`[ ! -f ] ||` guard, so a re-queued run cannot recapture its own
extension chunks into the staged window — verified behaviourally on
31 Aug across ten staging scenarios).

The extension window is `replay/*.npz` **minus** that list. The reader
FATALLY refuses a run whose manifest is missing, empty, duplicated,
malformed, all-swallowing, or **not a subset of what is on disk**
(superset is the normal case — the resume appends and never rewrites
or deletes a staged chunk, since `Replay.load` adds every loaded
chunk's uuid to `self.saved`; **subset is fatal**, because a
manifest-listed chunk that is gone means the staged tree was mutated
and every window below it is a guess).

---

## 4. DECISION RULE (frozen, pre-stated)

### 4.1 PRIMARY — A2's registered statistic on the extension window

**Estimand.** Per run, `occupancy` = the fraction of steps with
`position[0] > -0.13009691` over the **extension-window chunks** —
deterministic full chunk scan, no sampling rng. The scan is
`se_m3_read.occupancy`'s body with the file list supplied instead of
globbed; the selfcheck asserts it returns **bit-identical** `(frac, n)`
to `se_m3_read.occupancy` on the full listing, and that the two
windows partition that listing exactly.

**Validity floor: ≥ 4e5 extension-window steps** (nominal 5e5; the
house floor `MIN_STEPS = 1e5` is asserted to be no stronger).

**Statistic.** `Δ = mean(occupancy | sc_scare) − mean(occupancy |
sc_ctrl)` over 4 vs 4.

**Test.** Exact label permutation over all **C(8,4) = 70** assignments
(`se_m3_read.exact_perm_p`, imported), **one-sided NEGATIVE**,
observed assignment included.

**FIRES iff `p ≤ .05` AND `Δ < 0`.** Minimum attainable p is
**1/70 ≈ .0143**, identical to A2's, so a fire requires the observed
split to be among the **3 most extreme of 70** — near-complete
separation.

**Power disclosure (registered).** At n = 4 v 4 this design has
essentially no power against moderate effects. **A non-fire does NOT
license "the fence washed out" as a positive claim** and does not
distinguish "no fence" from "a moderate fence". This is stated before
the data exist and must be printed with any non-fire.

**Refusal.** The denominator is pinned at **8**. A missing, extra or
duplicated run REFUSES the whole read: the primary is an exact
permutation over C(8,4), there is no registered fallback statistic for
a partial panel, and A2 review #34 B3 demonstrated that dropping a run
can MANUFACTURE a fire. Refusal happens before any output is written,
so the one-execution guard survives a repair-and-rerun.

### 4.2 The DESCRIPTIVE wash-out comparison (registered, no test)

Emitted unconditionally, on every branch, as a side-by-side table:

- per-arm and per-run **extension-window** occupancy at 1e6, with
  per-arm BCa intervals, and the contrast Δ_ext;
- per-arm and per-run **A2 whole-replay** occupancy at 5e5, quoted
  from `artifacts/a2_scarecrow_read_20260824/se_scarecrow_read.json`
  (the reader re-verifies all eight occupancies, all eight step
  counts, Δ = −0.06318614049882765, p = 1/70 and the cell
  `SCARECROW-FENCES` **at import**, so an edited pin cannot be read);
- the **manifest-window** occupancies, which by the A2-PROVENANCE gate
  equal the A2 column exactly and exist only as that gate's witness;
- the ratio Δ_ext / Δ_A2.

Every row is labelled **CROSS-ESTIMAND and NON-INFERENTIAL**. No test
is computed between the columns and none may be reported.

### 4.3 SECONDARY-W — the registered attenuation bound (fires)

The only registered route to a **positively claimable** wash-out.

**Bound.** `Δ₀ = A2_DELTA / 2 = −0.031593070249413824` — half of A2's
measured displacement.

**Justification from A2's own dispersion.** A2's within-arm sds are
0.015140 (scare) and 0.037594 (ctrl); pooled **0.0286578**. The bound
is therefore **1.10 × A2's pooled within-arm sd** — the smallest
"the fence has more than halved" target that is still larger than the
seed-to-seed noise A2 itself measured. Below that, an "attenuation"
claim would be indistinguishable from A2's own scatter.

**Test.** Shift-model exact permutation on the same C(8,4) = 70
machinery: the scare arm's extension-window occupancies are shifted up
by `|Δ₀|` (under H₀: Δ = Δ₀ with a location shift the adjusted arms are
exchangeable) and `mean(adjusted scare) − mean(ctrl) > 0` is tested
**one-sided**.

**FIRES iff `p ≤ .05`**, i.e. `ATTENUATED-BELOW-HALF`: *the
extension-window displacement is significantly smaller in magnitude
than half of what A2 measured at 5e5.*

**Honesty about power (registered).** SECONDARY-W is an equivalence-
FLAVOURED bound, not an equivalence test at any conventional margin.
Its minimum attainable p is also 1/70, so it too requires near-complete
separation, and **a non-fire bounds nothing whatsoever**. It is
labelled SECONDARY everywhere and never rescues, vetoes or modifies
the primary cell.

The primary and SECONDARY-W can both fire (a fence that persists but
is significantly weaker than half of A2's). That combination is
registered as coherent and is reported as such.

### 4.4 Other reporting-only rows

Task return (mean `episode/score` over the final 100 entries of the
extension's own `scores.jsonl`) as a two-sided exact permutation;
per-run fit counters against 1e6; the staged-manifest sha256, chunk
counts and both resume witnesses per run.

---

## 5. VALIDITY GATES

### 5.1 FATAL per-run gates

A defective run refuses the WHOLE read — never a silent drop. Gate
ORDER is registered (cheap and decisive first, expensive scans last):

**(a) Seed ↔ arm map is the ARM AUTHORITY.** 80–83 ⇒ scare, 76–79 ⇒
ctrl. Any other seed — including A2's own 84–95, which this leg does
not extend — or a seed presented in the wrong arm, is FATAL. run_id
strings are never identity.

**(b) Task** `== dmc_cheetah_run`.

**(c) Objective pins.** `expl.mode == p2e`, `disag_bootstrap is True`,
`disag_bootstrap_prob == 0.8`, `disag_ens == 8`,
`disag_scale == 1000.0`, **`disag_head == det`** (the b4_alea wave ran
a `gauss` arm on this producer — an unpinned head is a live leak
path).

**(d) Penalty machinery inert** on both arms (`penalty_mix is False`,
`penalty.scale == 0.0`, `penalty.gate_key == ''`). A2's own penalty
arms exist on this producer and a leak would REPLACE the task reward.

**(e) The cheetah basesd pair.** `planted.source_key == position`,
`planted.basesd == 0.0976`, `distractor.dim == 8`,
`distractor.basesd == 1.215`. The A3 GENERALITY leg runs
`0.54193702 / 1.35803102` on this same producer; that leak is FATAL.

**(f) OU persistence** `distractor.theta == 0.1`. The φ-sweep wave
moves exactly this knob on this producer.

**(g) Training-time gates OFF** on both wrappers.

**(h) A2's arm construction, verbatim.** Scare: `mod_key == position`,
`mod_index == 0`, `mod_lo == -0.32203162`, `mod_hi == 0.19711795`,
`scale == 1.0`. Ctrl: `mod_key == ''`, `mod_lo == 0.0`,
`mod_hi == 1.0`, `scale == 0.35891393`. **This is not bookkeeping.**
`main.py` REWRITES `config.yaml` at resume and the env wrappers are
rebuilt from it, so an override dropped from the extension cmd would
switch the manipulation off mid-run while every other check stayed
green.

**(i) The extension target.** `run.steps == 1e6`.

**(j) The staged manifest** (§3.3): present, non-empty, no duplicates,
bare chunk basenames only, a subset of what is on disk, and leaving a
non-empty extension window.

**(k) RESUME INTEGRITY — the registered canary.** `Replay.load`
returns **silently** when the replay dir is missing or empty and
training then refills from fresh interaction: a from-scratch run
wearing an extension's name, with no error anywhere. Two FATAL
witnesses, both out of the run's own `metrics.jsonl` (which is not
staged, so its first rows are the first post-resume report by
construction):

| witness | key | threshold | failure value |
|---|---|---|---|
| buffer repopulated | first row carrying `replay/items` (logged at `train.py:109`, `logger.add(replay.stats(), prefix='replay')`) | **≥ 4.0e5** | ~3.1e3 — measured on a real SE run: 3136 items at its first log |
| resumed from 5e5 | first row carrying `step` | **≥ 4.9e5** | ~4.2e3 — the same run's first log at step 4160 |

Expected values on a correct resume are ~4.95e5 items (A2 ended at
494 624–499 648 replay steps and nothing evicts below a 5e6 capacity)
and ~5.01e5 steps. Both thresholds sit ~100× above their failure
values and ~20 % below their expected values: neither is tuned.

**(l) A2-PROVENANCE.** The **manifest window**, scanned with the same
frozen estimand, must reproduce A2's recorded numbers **for this
seed** exactly: `n_steps == A2_STEPS[seed]` and
`|occ − A2_OCC[seed]| < 1e-9`. One check proves three things at once —
the stage was **complete** (see §2.3's capacity arithmetic), the dir
staged under a seed **is** that A2 run, and no staged chunk was
corrupted or rewritten. It also catches a manifest taken LATE (after
the resume had begun), because the swallowed extension chunks make the
manifest window longer than A2's.

**(m) Extension-window validity**: ≥ 4e5 steps (→ `NOT-ADJUDICABLE`,
not a refusal).

### 5.2 Adjudication order

0. per-run FATAL gates (§5.1) — refusal, not a cell;
1. panel identity: exactly 4 + 4, seeds exact, no duplicates —
   refusal;
2. **COLLAPSE CHECK** → `GLOBAL-COLLAPSE` (§5.3);
3. **extension-window validity** → `NOT-ADJUDICABLE`;
4. the **PRIMARY** (§4.1);
5. **SECONDARY-W** (§4.3);
6. the descriptive wash-out comparison (§4.2) — computed
   unconditionally and emitted on every branch above.

### 5.3 Collapse check

**No dose gate and no drift gate are registered.** Same environment,
same manipulation, no transplant: there is no delivered-dose
transplant to verify, and the A2 arms' realized dose is what A2 itself
measured. What survives is the A3-generality collapse form, using
training telemetry only: the last logged
`train/expl/disag_replay_rew` must be **present, finite and strictly
positive** in every run. Any dead run ⇒ **GLOBAL-COLLAPSE**, nothing
downstream adjudicated. If the key is absent in ≥ 1 run the primary
proceeds and is annotated **COLLAPSE-CHECK-UNAVAILABLE**. No absolute
threshold is invented (A2's probe-referenced degeneracy gate is not
re-run: this leg produces no probe).

### 5.4 Report-only / annotating gates

- **Fit counters** against 1e6: any TRUNCATION-SUSPECT run annotates
  the primary statement (7-Aug standing disclosure rule).
- **Instrument flag** CLEAN / VALIDITY-VIOLATIONS from the extension
  floor.

---

## 6. OUTCOME MAP — what each verdict licenses

| cell | licensed wording |
|---|---|
| **FENCE-HOLDS** (primary fires) | *"The occupancy displacement produced by a planted misprice is still present in behaviour generated between 5e5 and 1e6 steps."* Scoped to this substrate, this environment, this dose, and to 1e6 steps — it licenses no statement about 2e6. If SECONDARY-W also fires, the sentence must carry *"and is significantly smaller than half its 5e5 magnitude"*. |
| **NOT-DETECTED-AT-1e6** (primary does not fire) | *"The displacement was not detected in the extension window."* **NOT "the fence washed out."** The §4.1 power disclosure MUST be printed with it: at n = 4 v 4 this design cannot separate "no fence" from "a moderate fence". The Control-Surface paper must then scope its fence claim to 5e5 and cite this leg. |
| **NOT-DETECTED-AT-1e6 + ATTENUATED-BELOW-HALF** (SECONDARY-W also fires) | The only registered wording that positively claims erosion: *"the extension-window displacement is bounded below half of A2's, i.e. the fence has more than halved."* Labelled SECONDARY, with its own 1/70 floor disclosed. A **shelf life** for the tool, which is a finding, not a failure. |
| **REVERSED** (Δ_ext > 0, reported as a non-fire) | Descriptive only, an unregistered direction, never a claim. It would motivate a separate registration. |
| **GLOBAL-COLLAPSE** | Validity cell. No wording either way; the objective was dead. |
| **NOT-ADJUDICABLE** | Validity cell (short extension window / panel below 4 + 4). No wording either way. |
| **refusal** (any §5.1 FATAL gate) | Not a cell. The staging or the continuity is broken and nothing is read. |

**Asymmetric-with-duty fence (inherited from A1 #27 M8 via A2).** This
leg may never SUPPORT a claim in only one direction. Whatever it
returns, **the Control-Surface paper and Paper 5 must cite it beside
every scarecrow-as-tool sentence**, and if the fence is not detected
at 1e6 they must scope that sentence to 5e5 before any subsequent
submission. One-directional insulation from bad news is what that
clause forbids.

---

## 7. DEVIATIONS FROM THE A2 READ (each with its reason)

The primary is A2's, verbatim in statistic, sidedness, test, region
constant and n. Everything below is a deliberate divergence.

| # | deviation | reason |
|---|---|---|
| D1 | **Windowed estimand** (extension chunks) instead of A2's whole replay | §3.2. The whole-1e6 buffer is half A2's already-read behaviour and dilutes the extension mechanically; the reader's own selfcheck exhibits a null extension that the whole-buffer read would have "fired". |
| D2 | **Only 2 of A2's 5 arms** | §1. `sc_scare2` and the two penalty arms carry A2's secondaries, not the claim on trial. Registered as an exclusion so it cannot be undone at read time. |
| D3 | **No probe leg**, hence no misprice gate, no gradient-delivery gate, no ranking secondary, no probe-referenced degeneracy gate | A probe pass is a separate post-training job; A2's probe panel is executed evidence and this leg does not re-run it. Consequence, registered: **this leg makes no mechanism claim.** The degeneracy gate is replaced by §5.3. |
| D4 | **No dose / drift gate** | Same environment, same manipulation, no transplant (contrast the A3 generality leg, where the dose is the live risk). |
| D5 | **New gates that A2 had no need of**: staged manifest, A2-PROVENANCE, and the two resume-integrity witnesses | A2 trained from scratch. This leg's entire failure surface is staging and resume, and `Replay.load`'s empty-dir path is SILENT. |
| D6 | **Added continuity pins**: `disag_head == det`, `disag_bootstrap_prob == 0.8`, `distractor.theta == 0.1`, `run.steps == 1e6`, ctrl inert ramp (0.0, 1.0) | These keys postdate A2's pin block, and three of them are moved by other registered waves on the SAME producer (φ-sweep, b4_alea, A3 generality). A2's pin block cannot see them. |
| D7 | **SECONDARY-W added** | A2 had no reason to bound its own effect from above. Without a pre-declared bound, "washed out" would be an unregistered reading of a null at n = 4 v 4. |
| D8 | **Missing / extra runs REFUSE** rather than yielding a cell | The denominator is pinned at 8 up front so no partial-panel statistic can be improvised after seeing which runs survived (A2 #34 B3). |
| D9 | **Renamed run_ids** | §2.1. Collision hazard in the shared runroot; identity is carried by seed + provenance, never by a name. |

---

## 8. DISCLOSURES

### 8.1 SPLIT REGISTRATION

The Track-A A3 programme has two legs. **This document registers the
PERSISTENCE leg only.** The **GENERALITY leg**
(`PREREG_a3_generality_20260830.md`, 8 fresh `dmc_finger_spin` runs)
is a SEPARATE registration, as that document's §8.2 already states.

**The two legs stand alone. Neither leg's outcome gates the other**,
and neither may be cited as support for the other. Nothing in this
document conditions on what the generality leg returns, and nothing
here may be cited in its read.

### 8.2 The estimand distinction, restated as a disclosure

Any table, figure or sentence that places the 1e6 numbers beside A2's
5e5 numbers is comparing **two different estimands** and must say so.
The reader emits the comparison already labelled; the paper must carry
the label. The only quantity this leg adjudicates is the
extension-window **contrast**.

### 8.3 Staging is complete, not "capacity-covering"

An earlier framing of this leg assumed the staged replay could be a
newest-first subset large enough to cover `capacity`. **The capacity
arithmetic says otherwise** (§2.3): capacity is 5,000,000 items and A2
produced ~4.95e5, so the cutoff never fires and **every chunk must be
staged**. The A2-PROVENANCE gate (§5.1.l) is what turns that
requirement into something a read can verify rather than assume.
Staged copies must be materialized bytes, never `cp -al` views,
because in-place resumes mutate their inputs.

### 8.4 Producer verification (NO DELTA)

`scripts/uncfield_se.sbatch` is **unchanged**. Verified 31 Aug by
inspection and by generating the wave's lane files:

- the resume itself needs nothing new — `rm -f TRAINING_DONE`,
  `mkdir -p` on the existing logdir, `config.save()` overwrite,
  `cp.load_or_save()`;
- `STEPS=1e6` binds and is BIND-CHECKed (`want('run.steps',
  float(env.get('STEPS','5e5')))` → `1000000.0`), and the checkpoint
  watcher's `--stop_step` is derived from it;
- all **28** environment variables the wave sets are consumed by the
  producer AND covered by its BIND-CHECK `want()` block — the
  stale-producer preflight's own greps were run against
  `scripts/uncfield_se.sbatch` on 31 Aug and all 28 pass;
- the staged-manifest step lives in the **wave cmd**, not the
  producer, and sets no ALL-CAPS variable, so it adds nothing to the
  stale-producer gate.

**One consequence, disclosed rather than patched:**
`probing/checkpoint_watcher.py`'s `--milestones` default is
`[1e5 … 5e5]` and `uncfield_se.sbatch` does not pass the flag, so
`ckpt_snapshots/nearest.json` carries **no `1000000` row at any
`STEPS`**. Adding a `MILESTONES` override would be a producer delta
touching every prior SE wave's launch path for a completion predicate
only. Instead the wave gates the realized 1e6 step on
`ckpt_snapshots/manifest.json` — the watcher's own append-only record
of every retained snapshot, which carries the exact steps — with the
pattern `"step":\s+(99\d{4}|1\d{6})`, verified 31 Aug to match a
1e6-run manifest and to REJECT both a 5e5-only manifest and one that
stopped at 989 k.

**The same `TASK` BIND-CHECK gap the generality leg disclosed applies
here** (`TASK` is consumed but not in the `want()` block). It is
caught twice downstream: by the spec's `done_when` regex on `task:` in
the composed config, and FATALLY by the reader (§5.1.b).

### 8.5 What the A2 evidence is, and that it was not touched

`uncfield/se_scarecrow_read.py` is a frozen, EXECUTED instrument. This
leg does not edit it, does not re-run it, and does not import from it.
It imports the shared constants from the same upstream executed
readers A2 imported them from (`se_m3_read`, `se_grad_read`) and reads
A2's **recorded output** from
`artifacts/a2_scarecrow_read_20260824/se_scarecrow_read.json`,
re-verifying all eight occupancies, all eight step counts, the delta,
the p-value and the outcome cell at import time.

### 8.6 Device accounting

All eight extensions on ONE instance, double-balanced across lanes and
rounds (§2.4), so no device ≡ arm term exists in this wave by
construction — the same property A2's Amendment-1 layout established
and which A2's read used to defuse A1's device caveat. **The A2
checkpoints being extended were themselves produced on one box
(vast inst 23)**, so the extension inherits a clean starting cohort;
the extension box will be a different one, but it is shared by both
arms.

---

## 9. WHAT IS **NOT** CLAIMED, WHATEVER HAPPENS

Frozen so a fire cannot be over-read and a null cannot be over-read
either:

- **A non-fire is not a wash-out.** Only SECONDARY-W can positively
  bound the effect, and only when it fires.
- **No cross-estimand inference.** The 1e6 and 5e5 columns are
  different estimands; the side-by-side table is descriptive.
- **No mechanism.** This leg runs no probe, so no sentence of the form
  "avoids what it overprices" is licensed by it alone, and no claim is
  made about *why* the fence held or eroded.
- **Nothing about A2's scale-2 or explicit-penalty arms**, which are
  not extended; in particular no claim that the ~13 % sizing against
  an explicit penalty (A2 S3) persists.
- **No generality claim** (§8.1).
- **No claim about training beyond 1e6 steps**, and none about other
  environments, objectives, doses or the pixel setting.
- **Whatever this leg returns, the registered A2 result at 5e5 stands
  exactly as read on 24 Aug.**

---

## 10. FROZEN ARTEFACTS

| item | path | state at registration |
|---|---|---|
| this prereg | `prereg/PREREG_a3_persistence_20260831.md` | frozen |
| reader | `uncfield/se_a3per_read.py` | written + selfchecked BEFORE compute; ONE execution; explicit `--output` |
| wave spec | `ops/waves/a3_persistence/spec.yaml` | frozen at this commit |
| ops handoff | `ops/waves/a3_persistence/OPS_HANDOFF.md` | frozen |
| producer | `scripts/uncfield_se.sbatch` | UNCHANGED (§8.4) |
| A2 registration | `PREREG_trackA_scarecrow_20260822.md` + `amend1` | executed |
| A2 evidence | `uncfield/se_scarecrow_read.py`, `artifacts/a2_scarecrow_read_20260824/` | executed; imported-from-record, never edited |
| A2 substrate | RCC bundle `a2_scarecrow_20260824_184000` (FULL) | the staging source |

**Selfcheck status: PASS**, 12 scenario groups —

0. the A2 artifact re-verifies at import (8 occupancies, 8 step
   counts, Δ = −0.063186, p = .0143, cell `SCARECROW-FENCES`) and the
   C(8,4) = 70 machinery reproduces its floor;
A. a persisting fence fires at 1/70 and is **not** bounded by
   SECONDARY-W;
B. a washed-out extension does not fire while its manifest window
   still carries A2's −0.0632, and SECONDARY-W fires →
   `ATTENUATED-BELOW-HALF`;
B1. **the dilution control**: the same washed-out panel read with the
   whole 1e6 buffer gives Δ = −0.0347 at p = .0143, a fire — the exact
   confound §3.2's window removes;
B2. a REVERSED extension does not fire;
C. empty-buffer resume, step reset, and each missing witness → FATAL;
D. manifest missing / empty / duplicated / malformed /
   subset-violating / all-swallowing / taken-LATE → FATAL;
E. A2-PROVENANCE: a short stage and a foreign run under a seed's name
   → FATAL;
F. wrong-arm-map, the non-extended seeds 84 / 88 / 210 / 99, and 20
   A2-quadruple and continuity leak fixtures → FATAL;
G. missing / short / duplicate panels → REFUSED;
H. dead objective → `GLOBAL-COLLAPSE`; short extension window →
   `NOT-ADJUDICABLE`; absent telemetry → annotated, not fatal — and
   both validity branches render end-to-end with the descriptive
   comparison still present;
I. `occupancy_over` on the full listing is **bit-identical** to
   `se_m3_read.occupancy`, and the two windows partition it exactly;
J. end-to-end read + `RESULTS.md` + fit counters at 1e6 + the
   ONE-execution guard.

Read invocation (ONE execution, explicit output):

```
python -m uncfield.se_a3per_read \
    --scare "<runroot>/a3per_scare_s8*" \
    --ctrl  "<runroot>/a3per_ctrl_s7*" \
    --output artifacts/a3_persistence_read_<date>
```

Freeze = commit of this file + the reader + the wave spec + the ops
handoff. Then, and only then, the staging and the 8 submissions.
