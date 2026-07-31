# PREREG: TM2-R3 — TD-MPC2 opportunity/competence cross-family replication (built 2026-07-30; freeze-commit 2026-07-31)

The R3 consumer-competence factorial landed 29 Jul with all-primary
clarity (`artifacts/r3_competence_20260729/` — opportunity +1.247
FIRES, gap +1.293 FIRES, maturity null): Paper 2's headline is the
opportunity/competence decomposition, currently evidenced in ONE model
family (DreamerV3's RSSM with a one-step plug-in consumer). This wave
asks whether the decomposition survives the family change that most
threatens it: TD-MPC2, whose action consumer is an MPPI PLANNER rather
than a one-step plug-in argmax. This file +
`probing/tdmpc2_oracle_labels.py` (the ported labeler, selfcheck PASS)
+ `probing/tdmpc2_r3_train.py` + `analysis/tm2_r3_read.py` (selfcheck
PASS) + `scripts/tm2r3.sbatch` + the `probing/tdmpc2_compat.py` dose
patch are committed BEFORE any TM2-R3 training run, smoke, or label
exists.

## Question

Does the opportunity/competence decomposition replicate in a second
model family? **Registered prediction: yes on both legs — opportunity
exists (P-TM2a) AND the consumer fails to harvest it (P-TM2b), the
cross-family competence failure.** Genuine risk, registered as a live
branch and not a failure: TD-MPC2's consumer is a PLANNER (6-iteration
MPPI over 512 samples through the same reward/Q heads that build the
candidate signal) — a categorically stronger consumer than Dreamer's
one-step plug-in, so **HARVESTED-BY-PLANNER is a real possible outcome
and is registered as the family boundary of the competence claim**
(consumer-class-scoped competence, paper-shaped either way: the
decomposition then dissociates consumer classes rather than model
families). Second genuine risk: cup/finger are sparse tasks and
TD-MPC2 trains fully online here — all-zero-G floors could dominate
(registered FLOOR gate below).

## Estimands (per labeled state, from the ported labeler's g_all)

Identical to R3 (`PREREG_r3_competence_20260725.md`):

- opportunity = max_m G[m] − G[m_now]
- achieved = G[m_real] − G[m_now]
- implementation gap = opportunity − achieved

## Port definition (frozen in `probing/tdmpc2_oracle_labels.py`)

- **Signal**: qfull [K=5, M=8] = the num_q Q-ensemble (two_hot_inv) at
  (z, a_m), z = encode(obs_t); candidates = policy-prior mode + 7
  samples at z (the dv3 mode+samples convention). m_now = argmax of
  the head-mean (the imported d0 `plugin_choice` — the SAME plug-in
  rule, deliberately, so the m_now baseline is family-comparable).
- **op_real**: per candidate, restore the env snapshot, ONE real step
  with the candidate; score = r_real + disc·Vhat(s'_real), Vhat = the
  mean-over-heads max-over-a-fresh-candidate-set Q at the observed
  next state (the dv3 `qn.mean(0).max()` mirror); disc = the agent's
  own discount (0.995 at episode length 1000).
- **Oracle**: G(a_m) = undiscounted return of a horizon-step (100)
  rollout whose FIRST step executes a_m and whose remaining 99 steps
  follow the frozen EVAL-MODE PLANNER
  (`agent.act(obs, t0, eval_mode=True)`) — 100 env steps total, the
  R3 convention, identical to the frozen code; `--oracle_all` grounds
  every candidate. g_all_boot = G +
  Vhat at the truncation state (registered companion, == G when the
  episode ends first).
- **CRN, TD-MPC2 form**: the latent is feedforward — no belief carry,
  so the d1fix branch-carry machinery collapses. The two real hazards
  are registered and handled: (1) the MPPI warm-start buffer
  `_prev_mean` is snapshotted at each label, restored before every
  branch and after the last branch, and the FIRST follower call of
  every branch is t0=True (cold start — the pre-label plan is not a
  belief and conditions on not-taking the candidate); (2) torch global
  RNG (CPU+CUDA) is marked once before op_real and reset before every
  candidate probe and every rollout — the dv3 mark/reset points.
- **Env**: branch steps drive the inner embodied node of `Dv3TaskEnv`
  ([Distractor →] DMC → ActionRepeat → FromDM — the same wrapper
  family as every study buffer); `snapshot_env`/`restore_env` are
  IMPORTED from `d0/oracle_labels.py` unchanged; the per-state
  determinism assert is kept and strengthened (reward AND full
  flattened obs, so a broken OU restore trips).
- **op_imag is NOT ported** (registered): dv3 retired imag as
  instrument-defective for its estimand (24-Jul audit) and "imag stays
  retired" is a frozen R3 consequence-map invariant; no registered
  read consumes imag columns. Schema slots are kept with sentinels
  (m_imag = −1, g_imag/delta_imag = NaN, cost_imag_* = 0). udyn has no
  TD-MPC2 analog (no disagreement ensemble): NaN, slot kept. Latent
  column = 'zlat' (512-D f16) + ref_zlat strides (ladder-ready).
- labeler_version = `tm2_oracle_20260730`; meta records dose,
  checkpoint, dials, planner dials, discount.

## Design

- **Grid**: {cup, finger} × {e1, e4} × seeds 51–58 (disjoint from
  every prior cohort: R3 31–38, D1 21–23/1–3, TM2 offline 1–16), each
  run labeled at TWO maturities — early (`ckpt_early.pt`, first
  episode boundary ≥ 2.5e4 env steps) and late (`ckpt_late.pt`, 1e5).
  32 fresh ONLINE training runs, 64 label passes, base arm only.
- **Training** (`probing/tdmpc2_r3_train.py`): the official pinned
  checkout (e9f59321933cbc8e11a002b842adc7d4ffae8ff1), model_size 5,
  state obs, mpc=True, official loss coefficients, NOTHING frozen —
  driven through the official `OnlineTrainer` (subclassed only for the
  tensordict compat shim, jsonl logging, and the early-snapshot hook
  at episode boundaries; episodes are 1000 steps so the realized early
  step is within one episode of nominal, recorded in
  `ckpt_early.STEP`). **Explicitly: NO dependency on the existing
  tm2wm_* offline-fit checkpoints — these are fresh online runs, and
  the free-arm untrained-heads caveat of `PREREG_tdmpc2_f2_20260720.md`
  does not arise because reward supervision is present (official
  coefficients).**
- **Dose**: e1 = no distractor; e4 = `d0_dose3` (OU dim 32, scale 3.0,
  theta 0.1, basesd 0, calib 1000) — composed EXACTLY as DreamerV3
  composes it (wrapper directly around the DMC env, seed =
  SeedSequence([train_seed, 0, 0xD0]); `probing/tdmpc2_compat.py`
  DOSE_TABLE, value-for-value from `dreamerv3/configs.yaml`). The
  distractor key is appended last in the obs concatenation. The
  labeler rebuilds the env with the SAME builder and the run's train
  seed: train/label env identity by construction.
- **Labeling dials**: states 200, horizon 100, label_every 25,
  actions 8 (labeler refuses any other M), labeler seed 0, ref_stride
  5, `--oracle_all` — the R3 dials.
- **Ordering**: freeze-commit → env-setup existence check (pinned
  checkout + conda env; the sbatch asserts the checkout hash) →
  training wave (32 runs) → **REGISTERED SMOKE GATE, machine-checked
  by the reader**: (i) the random-init API smoke
  (`tm2r3_apismoke_smoke.npz`: random-init model, dosed env, 2
  states, no checkpoint touched, no outcome revealed — the
  `PREREG_tdmpc2_f2` precedent) and (ii) four dosed REAL smokes of 5
  states, {cup, finger} × {e1, e4} on the seed-51 LATE checkpoints
  (`tm2r3_{dom}_{dose}_seed51_late_smoke.npz`). Smoke stdout is
  consulted only for crash/dial/determinism sanity: the labeler
  REDACTS the per-file estimand summary for `*_smoke.npz` outputs
  (frozen code), so no achieved preview exists during the open
  amendment window. Any fix required by a smoke ⇒ dated amendment
  BEFORE labels → 64 label passes → **ONE read**
  (`python -m analysis.tm2_r3_read`). Phase note: the smokes need only
  the four seed-51 runs, so label jobs of already-trained runs MAY
  interleave with training-wave stragglers once SMOKE_OK exists —
  epistemically inert (labels of run X depend only on run X; the
  reader enforces the all-or-nothing cohort), registered so the
  ordering arrow is not read as strict phases.
- **FILL slots** (values only knowable at run time; disclosed here,
  completed by a dated amendment BEFORE labels): realized early/late
  step per run [FILL: the 32 `ckpt_early.STEP`/`ckpt_late.STEP`
  values]; planner-dial audit echo [FILL: confirm each run's
  `config.json` planner block equals the pinned checkout's config.yaml
  values registered here — horizon 3, iterations 6, num_samples 512,
  num_elites 64, num_pi_trajs 24, temperature 0.5, min_std 0.05,
  max_std 2 — and record num_q 5, latent_dim 512, discount 0.995].

## Power (pre-sized from the executed R3 wave, committed sources)

Same grid geometry as R3: n = 32 run clusters, run-clustered
percentile bootstrap B=10K rng 0. From
`artifacts/r3_competence_20260729/RESULTS.md` (verbatim): "Pre-sized
power: detectable pooled gap ≈ 0.42; observed +1.29 ≈ 3× the detection
bar — this is not a marginal fire." The R3 sizing derived from per-run
gap sd ≈ 1.0 (cup) / 1.3 (finger)
(`PREREG_r3_competence_20260725.md`); if TD-MPC2's per-run dispersion
is comparable, the detectable pooled gap here is again ≈ 0.42 — a
third of the dv3 effect. Registered honestly: TM2 per-run variance is
unknown at freeze; a same-direction effect at ≥ 1/3 the dv3 size is
detectable.

## Registered decision rules (frozen in `analysis/tm2_r3_read.py`)

Cluster = training run; both maturities share the cluster. All
constants are IMPORTED from the frozen `analysis/r3_read.py` with pin
asserts (opportunity bar 0.2, B = 10K, `default_rng(0)`, floor limit
0.5) — identical thresholds by construction.

- **P-TM2a (opportunity exists)**: pooled mean opportunity CI entirely
  > 0.2.
- **P-TM2b (competence failure)**: pooled mean gap CI entirely > 0.
- **P-TM2c (maturity buys competence)**: per-run paired late−early
  achieved value, pooled CI entirely > 0.
- **FLOOR GATE** (reacher-style, registered): floor cells (all-zero G)
  are INCLUDED in all primaries (conservative); any domain with > 50%
  floor cells ⇒ **FLOOR-LIMITED** — inconclusive, fires nulled to
  None, no adjudication.
- **Machine-checked integrity** (reader trips, never silently biases):
  smoke gate (API smoke + 4 dosed smokes, labeler pin), full 64-cell
  cohort all-or-nothing, seeds 51–58 only, n_states 200, M = 8, dial
  identity from every file's meta (states/horizon/label_every/seed/
  actions/ref_stride/mass_scale/behavior_checkpoint), dose ==
  filename dose with e4 dose_config dim 32 scale 3.0, train_seed ==
  filename seed, checkpoint == the named run's `ckpt_{mat}.pt`, no
  smoke-flagged file consumed as a label.

Verdict vocabulary (frozen): **CROSS-FAMILY REPLICATION** (a+b fire; c
scopes maturity) / **OPPORTUNITY-ABSENT** (a fails) /
**HARVESTED-BY-PLANNER** (a fires, b fails) / **FLOOR-LIMITED**.

Registered secondaries (descriptive, never decisional): dose effect on
opportunity, maturity effect on opportunity, pooled achieved,
per-domain splits, floor fractions.

## Consequence map (frozen)

- **P-TM2a + P-TM2b fire (CROSS-FAMILY REPLICATION)** ⇒ the
  opportunity/competence decomposition is a two-family regularity —
  Paper 2's headline gains its strongest generality leg, and the
  planner's failure to harvest what a one-real-step probe reveals
  sharpens the claim (even a 512-sample MPPI consumer leaves the
  oracle opportunity on the table).
- **P-TM2a fails (OPPORTUNITY-ABSENT)** ⇒ information content differs
  across families at these cells; the dv3 decomposition stands
  unchanged with family scope disclosed; no TM2 competence claim
  either way.
- **P-TM2a fires, P-TM2b fails (HARVESTED-BY-PLANNER)** ⇒ the
  registered family-boundary branch: the competence failure is
  consumer-class-scoped — one-step plug-in consumers fail where an
  MPPI planner succeeds. This is a headline-grade dissociation in its
  own right (the paper reframes §competence as a consumer-class axis);
  it is NOT a replication failure of R3, whose registration was
  plug-in-consumer-scoped.
- **FLOOR-LIMITED** ⇒ inconclusive; no further TM2-R3 compute without
  a redesign (dated re-registration).
- **Regardless:** the executed R3 read stands as registered; 24+24
  stay frozen; Route A stays closed; imag stays retired (the sentinel
  columns are never consumed by any read).

## Costs

32 online training runs at ~2–4 GPU-h each (1e5 steps, 1 update/step +
5e3-update seed burst, MPPI acting) ≈ 65–130 GPU-h; 64 oracle_all
label passes at ~1–1.5 GPU-h each (≈ 200×(8 probes + 8×100 rollout
steps) ≈ 1.6e5 planner/env steps + ~5e3 base-trajectory planner steps)
≈ 65–95 GPU-h; smokes ≈ 0.5 GPU-h total. **Total ≈ 130–230 GPU-h; ~1–2
days wall-clock on 8–16 cluster lanes** (single-GPU jobs, fully
parallel across runs/passes).

## Disclosure

Known at freeze: everything through 31 Jul, including the full R3
record (all cell means, used ONLY for power sizing and threshold
inheritance), the F1/F2 TD-MPC2 offline-fit outcomes (a different
protocol — offline fits on bridged dv3 buffers; nothing here loads
them), the U2/U3/U4 unfrozen stress reads, AND the R3 reacher read
EXECUTED 31 Jul before this freeze-commit (**REPLICATES**: P-RRa
+0.828\* + P-RRb +0.813\* fire, P-RRc null,
`artifacts/r3_reacher_20260731/`) — family-relevant context for the
registered replication prediction (a third Dreamer-family domain
replicating raises the prior that the decomposition is
architecture-general, and correspondingly sharpens what a
HARVESTED-BY-PLANNER or OPPORTUNITY-ABSENT outcome would mean);
disclosed so the known-record statement is true at commit time; the
predictions and decision rules above were written 30 Jul and are
unchanged by it. Unknown: EVERY TM2-R3
quantity — no online TD-MPC2 run at these seeds/doses exists anywhere,
no TD-MPC2 oracle label of any kind has ever been computed, and the
ported labeler's torch path has never executed (it is validated only
by the registered no-outcome API smoke; the numpy core, CRN contract,
planner-state restore, schema, and guards are selfchecked on
closed-form mocks). Ordering: this file + all five instruments
committed BEFORE the first training job.
