# PREREG: Competence-repair intervention — LORO observable-features consumer (built 2026-07-30; freeze-commit 2026-07-31)

Executes the parent registration's re-entry clause: the R3 consequence
map (`PREREG_r3_competence_20260725.md`) rules that "any calibration
claim would still be a NEW registration (Route A stays closed as a
resource decision)" — this file is that NEW registration, the
interventional companion of the same-day correlational ladder
(`PREREG_r3_ladder_20260730.md`) and of the cross-checkpoint heads
amendment (`PREREG_r3_amend2_20260730.md`). This file +
`d0/train_consumer_model.py` (selfcheck PASS) + the cm-extended
`d0/oracle_labels.py` (`--consumer_model`; FULL selfcheck PASS incl.
every pre-existing leg) + `analysis/repair_read.py` (selfcheck PASS) +
`scripts/r3_rep_local.sh` are committed BEFORE the trainer is ever run
on real labels, before any repaired label exists, and before any
consumer-model quantity of any kind exists anywhere:
`--consumer_model` has never been executed on any checkpoint.

## Question

The committed R3 read (`artifacts/r3_competence_20260729/`) found
opportunity that is not harvested (P-R3a +1.247 [+0.700,+1.908];
P-R3b +1.293 [+0.723,+1.987]) and a paired maturity null on achieved
value (P-R3c +0.014 [−0.177,+0.215]). Amendment 2 asks whether the
deficit lives in the value heads; the ladder asks which observables
CORRELATE with the per-state gap. This registration asks the
interventional question: can a deliberately SIMPLE external value
model — a linear ridge probe over stored observable features (belief
latent, candidate action, qfull summaries, udyn), trained on oracle
returns from OTHER runs — make the consumer harvest?

**Registered prediction: PARTIAL** — P-REP1 fires (the oracle-return
supervision is strictly stronger than anything in the agent's own
training signal, and pooled opportunity is large), but below the
materiality bar: a linear probe on a frozen 512-d latent is expected
to leave most of the gap standing (mechanism-coherent with the
"support must be legible to the objective" account). **Genuine risk:**
the flat null (NOT-REPAIRABLE-FROM-OBSERVABLES) — per-state value
knowledge is not linearly present in the stored observables, which
strengthens the representational account; also live: REPAIRED (the
deficit is mere value-knowledge — a stronger, more surprising
headline) and HARMFUL. All four branches are written into the paper
plan below; the design cannot be outcome-steered.

## Estimands (per labeled state; states identical by construction)

- achieved_rep = `g_all[arange, m_real] − g_now` from a REPAIRED pass
- achieved_orig = the same functional from the COMMITTED R3 late label
  of the same run (the committed files are read, never recomputed
  beyond this per-state functional)
- primary target: per-state PAIRED [achieved_rep − achieved_orig] →
  per-run mean → run-clustered bootstrap

## Intervention (two frozen components)

1. **Trainer** (`d0/train_consumer_model.py`, pure numpy, no RNG —
   the output is a deterministic function of the frozen code and the
   committed labels):
   - Feature map `cmfeat1`, frozen: per candidate m at a labeled
     state — own-candidate head-mean q and head-std from qfull;
     centered head-mean and margin-to-best; candidate-set stats
     (set mean, set std, set range); udyn; the candidate action
     vector; deter (f16→f32 cast, so training and online computation
     are bitwise identical). F = 8 + A + D (= 522 on these runs).
   - Targets: `g_all[state, candidate]` (all finite under the
     committed `--oracle_all` passes).
   - **LORO scope:** the model deployed on run X is trained ONLY on
     the other 7 seeds of the same (domain, dose) group, both
     maturities — 14 source cells; the deployed chooser has never
     seen a state of its own run. Lambda by inner
     leave-one-training-run-out CV; a lambda's score for a held-out
     run is the mean over the run's two maturity files of the
     per-file mean per-state Spearman across the candidate set
     (matching the frozen code exactly; grid 1e-6…1e1; ties → largest
     lambda); standardization constants from the fitting split.
   - **Registered value-blindness:** training READS the committed
     labels — a committed, already-read artifact — as supervision;
     that is registered here as part of the intervention. The trainer
     computes NO decision-relevant estimand: the realized-choice /
     plug-in-choice / per-state delta columns are never read (field
     whitelist in code + a source-scan selfcheck leg); the output npz
     contains only weights, intercepts, standardization constants,
     lambdas, versions, and the training-data manifest; stdout
     reports only lambda choices. The internal CV rank scores are
     machinery, never printed or stored.
2. **Chooser** (labeler `--consumer_model`): at each labeled state the
   labeler computes the `cmfeat1` features from quantities `d0_eval`
   already returned and sets the realized choice to argmax ghat.
   `op_real` still runs unchanged (its own argmax is stored as
   `m_real_probe`, its scores as `real_scores`); `op_imag` still
   runs; with `--oracle_all` (required, asserted) the rollout target
   set is all-M in every pass. The chooser is pure numpy — no env
   step, no policy call, no RNG use — so the base trajectory, the
   policy-RNG schedule, and `g_all` are bit-identical to a pass
   without the flag (verified in code: the choice feeds nothing
   downstream except the saved row). **Therefore NO control passes
   are run: the committed R3 late labels are the paired control.**
   Meta version becomes exactly `d1fix_20260724_cm1` (+ model path,
   model sha256, deployed run key, feature-map version); flag empty =
   byte-identical default path, and the `rep_` filename prefix never
   matches any committed reader's filename regex (the xc isolation
   convention).

## Design

- **32 repaired passes**: the 32 core LATE cells
  (cup/finger × e1/e4 × seeds 31–38), final checkpoints, dials
  IDENTICAL to the committed R3 wave (states 200, horizon 100,
  label_every 25, actions 8, rollouts 16, **labeler seed 0**,
  ref_stride 5, `--oracle_all`) + `--consumer_model`. Files
  `rep_{dom}_{dose}_seed{s}_late.npz` in `$R3_ROOT/rep_labels`.
  Late-only is registered: late is where the committed headline
  lives, and it halves the compute.
- **DETERMINISM GATE (registered):** per cell, episode/step AND
  `m_now` must match the committed npz EXACTLY (m_now is an index
  array, as value-blind as episode/step, and enters the primary
  through `g_now = g_all[m_now]` — value-head-path drift that flips a
  plugin near-tie must gate, not contaminate) and `g_all` exactly
  (registered fallback atol 1e-5, disclosed as
  `pair_tolerance_used`); ANY violation ⇒ **QUARANTINE**, no
  adjudication — env/library drift or a chooser side-channel would
  surface here. The smoke stage checks one cell BEFORE the wave via
  the reader's `--detgate` subcommand (value-blind: meta version +
  episode/step/m_now/g_all only; no estimand computed; explicit in
  the frozen code).
- **Weights-provenance protocol (commit-ordering-sound):** the freeze
  commit precedes the trainer run, so the weights npz cannot be in
  the freeze; instead it is pinned four ways. (a) DATA: the trainer's
  manifest embeds the 64 committed file names WITH their sha256s; the
  frozen reader recomputes those shas from the committed dir AND
  asserts their sorted digest equals the reader's frozen
  `COMMITTED_LABELS_DIGEST` constant (computed at freeze from the
  byte-pinned bundle, git 4f74b187) — tying the training data to THE
  committed artifact, not to whatever dir is passed. (b) CODE: the
  trainer stamps sha256 of its own source into the manifest and the
  reader asserts it equals the checked-out
  `d0/train_consumer_model.py` — a post-freeze locally edited trainer
  cannot produce an accepted model. Registered audit path: the
  trainer is deterministic (no RNG; numpy savez with fixed zip
  epoch), so re-running the frozen trainer on the committed dir must
  reproduce sha256(model) bitwise — that retrain-and-compare is the
  full proof if ever demanded. (c) The driver's `train` stage records
  sha256(weights npz) to `rep_labels/CONSUMER_MODEL_SHA256` BEFORE
  any labeling. (d) Every repaired pass stamps the model sha in its
  meta, and the reader asserts all 32 stamps == sha256 of `--model`
  == the recorded sha, plus per-run LORO training lists == exactly
  the 14 registered sibling files. Nothing about the weights is
  chosen after freeze.
- **Existence gate (all-or-nothing):** 32 run dirs (TRAINING_DONE +
  late ckpt) + the 64 committed label files; any miss writes the
  `SUBSTRATE_GONE` marker and the frozen reader records the
  registered **SUBSTRATE-GONE** outcome. No partial substrate.
- **Ordering: freeze-commit → existence gate → trainer (local CPU;
  sha recorded before any labeling) → smoke = ONE full-dial repaired
  pass on cup e1 seed31 late (a wave pass) + `--detgate`
  (machine-checked: the reader asserts the `REP_DETGATE_OK` marker) →
  remaining 31 passes → ONE read** (`analysis/repair_read.py`).
- **Stdout disclosure:** the labeler's standard per-pass summary line
  prints that pass's mean per-state deltas (pre-existing instrument
  behavior, unchanged since Gate-D1 and shared by the executed R3
  wave); the decision rules are frozen here before any pass exists,
  so the line carries no analytic discretion.

## Power (from named committed sources; no new computation)

The parent registration measured per-run gap sd ≈ 1.0 (cup) / 1.3
(finger) and pre-sized the R3 detectable pooled effect at ≈
1.96·1.2/√32 ≈ 0.42 (`PREREG_r3_competence_20260725.md` §Power)
UNPAIRED. This wave is strictly better-powered: per-state pairing on
IDENTICAL states removes trajectory and state-sampling variance
entirely — residual variance is chooser-difference variance on shared
states, bounded by the same scale — so the detectable per-run effect
is at most 0.42 and realistically far below it, comfortably under the
materiality bar 0.6465. We register the logic, not a fake precision
number.

## Registered decision rules (frozen in `analysis/repair_read.py`)

Run-clustered percentile bootstrap (cluster = training run, 32
clusters), B = 10K, `default_rng(0)` (machinery imported from the
frozen `analysis/r3_read.py` with import-pin asserts):

- **P-REP1 (repair helps at all):** pooled per-run mean of per-state
  [achieved_rep − achieved_orig]; CI entirely > 0.
- **P-REP2 (materiality; adjudicated only when P-REP1 fires):** point
  estimate ≥ **0.6465** = 50% of the committed R3 pooled gap point
  1.29304688 (`artifacts/r3_competence_20260729/r3.json`
  `p_r3b_gap.point`, verified at freeze) — the SAME registered
  constant as `PREREG_r3_doubling_20260729.md`. **Honest
  truncation-direction note:** the doubling registration truncated
  0.64652344 → 0.6465 because that direction was conservative there
  (a smaller bar makes IMMATERIAL harder); here the same truncation
  is anti-conservative by 2.3e-5. It is kept identical anyway for
  cross-registration consistency: the discrepancy is four orders of
  magnitude below the committed CI width, and P-REP2 is a labeling
  threshold on a point estimate, not a CI fire.
- **Verdict map:** **REPAIRED** (P-REP1 + P-REP2) / **PARTIAL**
  (P-REP1 only) / **NOT-REPAIRABLE-FROM-OBSERVABLES** (CI straddles
  0; worded scope-limited in the frozen reader: "a LINEAR probe on
  stored observables", explicitly NOT "no repair possible") /
  **HARMFUL** (CI entirely < 0) / **QUARANTINE** (determinism gate) /
  **SUBSTRATE-GONE** (existence gate; no adjudication).
- **Guards (machine-checked per file):** exact `d1fix_20260724_cm1`
  pin on the repaired side and exact `d1fix_20260724` on the
  committed side (a base, xc, or blended file trips); full dial
  identity incl. labeler seed 0 and feature-map `cmfeat1`; late-ckpt
  path suffix; train_seed vs filename; `consumer_model_run` == the
  file's own run (LORO deployment); chooser identity
  `m_real == argmax ghat` per state (the pass provably deployed the
  chooser it stamps); parent Amendment-1 array guards; `g_now =
  g_all[m_now]` oracle_all identity; all-or-nothing 32-cell grid both
  sides; unregistered seeds trip; the full weights-provenance
  protocol above.
- **Secondaries (descriptive, never decisional):** per-domain and
  per-dose splits of the paired diff; pooled achieved_rep /
  achieved_orig levels; chooser-vs-oracle agreement rates (repaired
  and committed); chooser change rate; probe agreement (the repaired
  pass's own op_real argmax vs the committed realized choice —
  expected 1.0 on identical trajectories; redundant instrument
  sanity); residual gap fraction; pooled late opportunity recomputed
  ONLY as a pairing sanity against the committed per-state values
  (max cell absdiff reported; disclosed as such in the output).

## Consequence map (frozen)

- **REPAIRED** ⇒ registered result: the implementation gap is
  value-knowledge harvestable from stored observables by a linear
  probe — the competence failure is chooser calibration, not
  representation; Paper 2 gains an interventional repair arm and the
  representational reading of P-R3c is weakened accordingly.
- **PARTIAL** ⇒ registered result: a quantified slice of the gap is
  linear value-knowledge; the remainder is not linearly
  observable-legible — both numbers go into the decomposition
  section.
- **NOT-REPAIRABLE-FROM-OBSERVABLES** ⇒ registered scope-limited
  null: a linear probe on stored observables cannot harvest — the
  representational account is strengthened, coherent with a
  heads-irrelevant Amendment-2 outcome and a flat ladder. Any
  nonlinear or in-agent repair attempt is a NEW registration.
- **HARMFUL** ⇒ disclosed result, no headline: the external linear
  chooser is worse than the probe-informed realized choice it
  replaces (the real probe carries one-step information the ridge
  lacks); reported with both CIs.
- **QUARANTINE** ⇒ instrument audit before ANY use of repaired
  labels; the wave is not interpretable evidence in any direction.
- **SUBSTRATE-GONE** ⇒ recorded as the registered outcome; no re-run
  on a partial substrate.
- **Regardless:** the committed R3 read and every prior registration
  (reacher, doubling, ladder, Amendment 2) stand as registered and
  are never recomputed; the committed labels are never modified;
  24+24 stay frozen; imag stays retired; Route A stays closed.

## Costs

Trainer: < 1 CPU-hour, local, once. Labeling: 32 full `--oracle_all`
passes at the R3 per-pass rate (≈ 1–2 h each on one GPU,
`PREREG_r3_competence_20260725.md` §Costs) ≈ 32–64 GPU·h ≈ half the
R3 label wave; sequential on one box ≈ 1.5–3 days, cluster-lane
parallel < 1 day wall-clock. Read: local CPU minutes. No training
compute on the RL side (substrate reuse is the point).

## Disclosure

Known at freeze: everything through the 31-Jul record — the committed
R3 read and its labels (bundle
`local_results/r3_competence_20260729_105146/labels/`, byte-pinned in
git at the read commit 4f74b187 and by this reader's frozen
`COMMITTED_LABELS_DIGEST`; the RCC `r3_labels` dir is
bitwise-identical per the results-sync policy), Amendment 1, the
doubling/ladder/xc registrations (none of those waves executed), the
U2/U3/U4 reads, the parent power constants used above, AND the R3
reacher read EXECUTED 31 Jul before this freeze-commit
(**REPLICATES**: P-RRa +0.828\* + P-RRb +0.813\* fire, P-RRc null,
`artifacts/r3_reacher_20260731/`) — family-relevant context (a third
domain with achieved ≈ 0), disclosed so the known-record statement is
true at commit time; it supplies no per-state quantity and every
decision rule above was written 30 Jul and is unchanged by it. The committed label files' schema was
inspected at freeze (fields/shapes/meta only; no estimand statistic
was computed). Unknown: every repair quantity — no consumer model has
ever been trained on real labels, `--consumer_model` has never run,
no feature→G or feature→gap statistic on real data exists anywhere at
freeze (the ladder is registered but unexecuted), the CV lambda
choices are unknown, and whether the 32 run dirs survive on scratch
is itself unverified (the existence gate adjudicates it). Ordering:
this file + the four code deliverables are committed BEFORE the
trainer's first real run.
