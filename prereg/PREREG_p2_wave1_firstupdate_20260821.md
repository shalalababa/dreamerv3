# PREREG — Paper-2 Wave 1: the first-update cell + C5 dual-objective oracle
# (21 Aug 2026)

**Status: FROZEN at user commit. ONE read execution** (the --pilot
dispersion look excepted, defined below). Motivations:
`Rescue_Synthesis_20260819.md` §4.1 (ranked #1; the 14-Aug compute stop
declined this cell against the WRONG power figure) and
`Rescue_Ideation_ThreeAxes_20260821.md` §4 (C5 rides these passes — the
channel-D self-test S3/S4 cannot ship without). Reviewer: ONE Opus pass,
user pre-approved, triggered at build.

## Look/power disclosure (all pre-existing numbers, sources cited)

- The 14-Aug W2 stop declined a retry against MDE80 ≈ 0.31 (cell sd ≈
  0.62, w2_mech_read_20260814/RECORD.md:63); the synthesis's recomputed
  harvest SE 0.0991 gives 0.278 — either way it is the HARVEST
  contrast's figure. The parked design is a PAIRED VARIANT contrast; the
  borrowed SE **0.0436** comes from the same-32-cell (warm−cold) paired
  contrast (P-N3) — a DIFFERENT variant pair than mode−cold1, so it is
  plausibly optimistic (disclosed; mitigation: 400 states/cell vs W2's
  200 halves the within-cluster noise component) → **MDE80 = 0.122**,
  2.3× tighter than the harvest figure. Target effect (descriptive):
  g_mode − g_plow = **+0.145** (provenance: the published
  w2_mech_read_20260814/RECORD.md:41 means 40.324 vs 40.179; used as a
  power basis only, never a decision input) → **power 0.914 at n=32**.
  Correcting a mis-attributed power figure is not re-reading data; this
  wave draws FRESH marks (env_seed 20260821 ≠ W1/W2's 20260809), and the
  W2 labels are neither re-read nor pooled anywhere in this
  registration.
- S3/S4 exposure (ThreeAxes §4): both planner facts grade a discounted
  bootstrapped short-horizon chooser against an undiscounted 100-step
  target — an unexamined channel-D. C5 is the registered test.

## Design — 32 existing TM2 LATE cells, no new training

- Cells: the registered TM2 LATE grid (cup/finger × e1/e4 × seeds
  51–58), the SAME checkpoints as PREREG_w1_wave_20260809.
- Labeler: `probing/tdmpc2_oracle_labels.py` NEW additive `--wv1_repeats`
  path (`_wave1_label_state_tm2`; no executed path modified; tag space
  21/22/30..37 disjoint from W1/W2 tags). Per state, two FIXED policy
  variants — **no selection on G anywhere** (order-statistic- and
  leak-immune by construction):
  - `mode`  = the policy-prior mode action (candidate 0; no planner call)
  - `cold1` = the planner action after exactly ONE MPPI update from a
    cold start (t0=True, iterations forced to 1 == W2_ITERS_LOW; dial
    restored in a finally block)
  Per variant: 4 raw-G rollouts under per-repeat CRN marks + ONE
  own-objective score own(a) = r + discount·V(s′) — **a discounted,
  bootstrapped ONE-step real-env score (the op_real convention), NOT the
  planner's H-step latent-model objective: C5 closes the discounting/
  horizon-length half of channel D, not the model-vs-environment half
  (review M4; this sentence travels with every C5 citation)**. 400
  states/cell; --env_seed **20260821**; output names
  `wv1tm2_{dom}_{dose}_seed{n}_late.npz`.
- **CRN null gates (review M3; the synthesis's 34 passes = 32 cells + 2
  gates)**: two `--wv1_dup` passes (act_cold1 := act_mode) on the pinned
  cells finger_e1_seed51 and cup_e4_seed58, names
  `wv1dup_{dom}_{dose}_seed{n}_late.npz`; the reader refuses unless every
  paired quantity is EXACTLY zero in both.
- **Pilot leg (registered, dispersion-only)**: the first 4 cells run
  first; `analysis/p2_wave1_read.py --pilot` prints ONLY n/cluster-SD/
  projected-SE (every mean masked — enforced in the reader and its
  selfcheck). The wave proceeds regardless unless projected SE at n=32 >
  2 × 0.0436, in which case a dated re-power FILL is recorded BEFORE the
  full read. The pilot cells' labels are wave labels (included in the
  32). Disclosed (review minor): the 2× tolerance is permissive — at the
  trigger boundary power at +0.145 is ≈0.38; the 0.914 headline is never
  quoted against a realized MDE, and MECHANISM-CLOSED-NEGATIVE is
  adjudicated on the REALIZED MDE80 the reader prints.
- **UNGATED** (the synthesis's lesson: gating a 0.12-MDE primary on a
  0.29-MDE gate is what cost W2 its wave).

## Estimands and decision rules (orientation fixed DAMAGE-POSITIVE)

- Cluster = run (n=32): d_raw = mean over states of
  [mean_r g(mode) − mean_r g(cold1)]. d_raw > 0 == the first MPPI update
  damages raw return. (The synthesis §4.1 wrote the difference in the
  opposite order with a CI>0 rule — an algebra slip against its own
  target-effect line g_mode − g_plow = +0.145; the damage-positive
  orientation registered here matches that target and is the one the
  reader implements.)
- **PRIMARY (raw)**: `one_sample_auto` (BCa + pinned-MC sign-flip) over
  the 32 cluster d_raw; DAMAGE iff perm p < .05 AND CI lower > 0.
- **C5 (own-objective)**: same form on d_own = own(mode) − own(cold1).
- Branch map (symmetric, all pre-named; reader implements verbatim):
  SEARCH-PATHOLOGY (raw fires AND own fires) / OBJECTIVE-MISMATCH (raw
  fires AND own significantly negative — S3/S4 re-scope) /
  DAMAGE-ATTRIBUTION-OPEN (raw fires, own indeterminate) /
  FIRST-UPDATE-HELPS (raw significantly negative, verbatim) /
  MECHANISM-CLOSED-NEGATIVE (raw no-fire, realized MDE80 printed —
  registered as strictly more informative than CLOSED-UNATTRIBUTED).
- Registered consequences: SEARCH-PATHOLOGY licenses the negative-VoC/
  planner-repair line; OBJECTIVE-MISMATCH forces the S3/S4 re-scope in
  every draft that cites them; no branch licenses a deployable claim.

## Gates (refusal = read not consumed; hardened to the w1_read standard, review M1)

Exactly the 32 registered cell files + the 2 registered CRN dup gates;
meta pins env_seed == 20260821, repeats == 4, horizon == 100,
label_every == 25, actions == 8, states == 400, trainer_version ==
'tm2r3_train_20260730', late_step_realized present; labeler_version
endswith '_wv1'; filename ↔ run_id identity; iters_first == 1 per state
+ iters_default recorded (dial witnesses); act_mode == cands[:, 0] (the
mode==candidate-0 witness); finite arrays; no unregistered wv1* files;
dup gates EXACTLY zero. Reader `analysis/p2_wave1_read.py` FROZEN with
this file (selfcheck PASS: 5 branches + pilot-masking + 12 refusals).

## Ops

**Checkpoint gate, concrete (review M6)**: resolve
`ls $RUNROOT/tm2r3/*/ckpt_late.pt` (32 expected). `tm2r3` is ABSENT from
`/mnt/c/dv3_archive/archive.globs` (verified in review) — the substrate
has NO archive fallback, so **archive tm2r3 BEFORE the wave, not after**
(never-delete-without-archive; the runroot_cleanup KEEP_PENDING hold's
own condition has partly lapsed).

**Amendment at freeze (21 Aug, user decision).** The checkpoint half of
this gate is RESOLVED: 32/32 `ckpt_late.pt` verified present at freeze
time under `/scratch/midway3/rickybao/dreamerv3_runs/tm2r3` (32 files,
1,004,776,992 bytes; tree 326 files / 2,090,529,123 bytes). The ARCHIVE
half is DEFERRED by explicit user decision — archive work is being
handled separately, outside this wave's ops chain — so the wave proceeds
without it. Recorded here rather than dropped silently, because the
exposure is unchanged: `tm2r3` is the sole substrate for every Wave-1
pass and still has no archive fallback, so a scratch loss before that
archive lands costs the 32 TM2 trainings (≈ Wave 2's own 101–184 GPU-h
budget) and leaves this wave's labels unverifiable against their source.

**Submission vehicle (added 21 Aug pre-freeze — ops flagged that the
wv1 path existed only in the labeler)**: `scripts/tm2r3.sbatch` stages
`wv1dup` (one job: both registered CRN gate cells, then the WV1_DUP_OK
marker) and `wv1` (one cell/job; refuses before WV1_DUP_OK; refuses
seeds outside 51–58; LATE only; the registered dials `--states 400
--horizon 100 --label_every 25 --actions 8 --seed 0 --ref_stride 5
--wv1_repeats 4 --env_seed 20260821` hardcoded; outputs to
$TM2R3_ROOT/wave1_labels/). The 4-cell pilot is procedural: submit
cup_e1_seed51, cup_e4_seed52, finger_e1_seed53, finger_e4_seed54
first, run the --pilot dispersion look, then the remaining 28.

Then: 2 `--wv1_dup` CRN gate passes (stage wv1dup) →
4-cell pilot → --pilot dispersion look → 28 remaining cells (stage
wv1) → bundle + sha manifest → **[ME] ONE read**. GPU-h note (review
M5): the 34–51 GPU-h figure is an UNVERIFIED planning estimate (no
realized per-pass wall-clock exists in any relevant artifact); ops
re-derives the budget from the first measured pass and records it in
the bundle NOTES.
