# PREREG R3 Amendment 2: cross-checkpoint consumer wave (built 2026-07-30; freeze-commit 2026-07-31)

Amends `PREREG_r3_competence_20260725.md` (freeze commit
`8b3380ce548dec00c7319237f04989f943f64afa`, verified against `git log`
30 Jul). The parent registered the exact re-entry clause this
amendment executes: "The cross-checkpoint consumer (early WM × late
critic and vice versa) from plan v4 needs a labeler extension and its
own smoke: it is NOT part of this registration and would be a dated
amendment before any such label." This file + the xc-extended
`d0/oracle_labels.py` (`--consumer_checkpoint`; selfcheck PASS incl.
the new xc legs) + `analysis/r3_xconsumer_read.py` (selfcheck PASS) +
`scripts/r3_xc_local.sh` are committed BEFORE any cross-checkpoint
label exists: `--consumer_checkpoint` has never been run anywhere, and
no cross-checkpoint statistic of any kind has been computed by anyone
or anything.

## Question

The committed R3 read (`artifacts/r3_competence_20260729/`) found
opportunity that is not harvested (P-R3a +1.247 [+0.700,+1.908];
P-R3b +1.293 [+0.723,+1.987]) and a paired maturity null on achieved
value (P-R3c +0.014 [−0.177,+0.215]). Does the competence deficit live
in the VALUE HEADS (rew/con/valens — the chooser), or elsewhere
(features/policy/interface)? The cross-checkpoint consumer separates
these: swap only the heads between the early and late checkpoints of
the SAME run and measure the per-state paired change in achieved
value.

**Registered prediction:** both primaries straddle 0 —
**HEADS-IRRELEVANT**. The committed P-R3c null (maturity buys no
achieved value at all) is most parsimonious if the deficit is not
head-resident; a heads-irrelevant outcome strengthens the
representational account (cross-link to the Paper-1/3 "support must be
legible to the objective" framing) and is the paper-shaped null.
**Genuine risk:** P-XC2 fires (mature heads rescue the early side) —
then part of the deficit IS head knowledge, the representational
reading of P-R3c weakens, and the decomposition section must be
rewritten around a head-learning bottleneck. The design cannot be
outcome-steered: both branches are written into the paper plan.

## Mechanism (overlay route — labeler-side parameter surgery)

- `--consumer_checkpoint <path>`: AFTER the main full load of the eval
  checkpoint, a SECOND load with regex `^(rew|con|valens\d+)/` from
  the consumer checkpoint (the `embodied/run/train.py`
  `bind(agent.load, regex=...)` pattern). Effect: `qfull` — hence
  `m_now`, `op_imag`, and the `op_real` chooser — reads the EVAL
  agent's features through the CONSUMER's heads; the follower policy,
  the WM (enc/dyn/dec), and disag stay the eval agent's. Both loads
  happen before any labeling; every RNG mark is taken after.
- **Counter restore (the pairing mechanism, verified in code):**
  `JAXAgent.load` overwrites `n_updates`/`n_batches`/`n_actions`
  UNCONDITIONALLY from whichever checkpoint it loads
  (`embodied/jax/agent.py:364-371`), and every policy call's seed
  derives from `(config.seed, n_actions)` (`agent.py:231-233,
  405-408`). `overlay_consumer` therefore snapshots the counters after
  the main load and restores them after the overlay. Base trajectories
  and oracle rollouts depend only on pol + counters + env RNG — the
  overlaid heads never influence them (verified in
  `d0/oracle_labels.py`: head outputs enter only chooser indices;
  policy-call COUNTS are head-independent; `--oracle_all` makes the
  rollout target set all-M in every pass). Consequence, registered as
  a machine-checked guard: a CONTROL pass (consumer = the eval
  checkpoint itself, still through the overlay code path) and a
  swapped pass label IDENTICAL states with IDENTICAL `g_all` —
  per-state PAIRED contrasts. The reader QUARANTINES on any violation
  (episode/step exact; `g_all` exact with registered fallback atol
  1e-5, disclosed as `pair_tolerance_used`).
- **No-op-overlay guard (frozen in the labeler)**: the regex load path
  silently updates nothing when zero param keys match; a silent no-op
  would make control and swapped passes byte-identical and FABRICATE
  the registered HEADS-IRRELEVANT null with every guard green.
  `overlay_consumer` therefore asserts, on the live param set after
  the load, that the regex matched at least one key AND that the
  matched set covers each consulted head family (`rew/`, `con/`,
  `valens0`); param-layout drift refuses instead of fabricating.
- **Config-safety asserts** (the regex load path skips shape asserts):
  the consumer run's `config.yaml` must exist and match the eval run
  on `task` + the full `agent.*` subtree, and both must have
  `agent.valnorm.impl == 'none'` (heads move without normalizer
  state); any mismatch errors loudly before labeling.
- **Version stamping:** flag empty ⇒ byte-identical `d1fix_20260724`
  path (no meta keys added; existing frozen readers unaffected).
  Overlay active ⇒ `labeler_version = d1fix_20260724_xc1` +
  `consumer_checkpoint` + `consumer_regex` in meta.
  Cross-contamination note, disclosed honestly: the r3 readers pin
  their version by SUBSTRING, which the xc string would also satisfy —
  isolation is therefore enforced by (a) file naming (`xc_` prefix
  never matches the r3 readers' filename regexes, and xc labels live
  in their own `xc_labels` dir) and (b) the xc reader pinning the
  version by EXACT equality (a plain `d1fix_20260724` file trips; leg
  in its selfcheck).

## Estimands (per labeled state, from `g_all` under `--oracle_all`)

- achieved = `g_all[arange, m_real] − g_now` (identical to the parent;
  `g_now = g_all[m_now]` is asserted per file as the oracle_all schema
  identity)
- per (run, eval side) paired per-state difference:
  achieved(swapped heads) − achieved(control heads)

## Design

- **Substrate = the 32 executed R3 training runs** (RCC scratch
  `r3_local` runroot: `r3_{cup,finger}_{e1,e4}_seed{31..38}`,
  `ckpt_early` + `ckpt`). **REGISTERED EXISTENCE GATE,
  all-or-nothing:** the driver's `gate` stage requires all 32 runs
  with both checkpoints; any miss writes the `SUBSTRATE_GONE` marker,
  the wave never starts, and the frozen reader records the registered
  **SUBSTRATE-GONE** outcome. No partial substrate, ever.
- Per run, 4 passes: eval ∈ {early, late} × consumer heads ∈
  {early, late}. heads = eval is the control (through the overlay code
  path). 128 passes + smokes. Naming:
  `xc_{dom}_{dose}_seed{s}_{evalmat}_h{headsmat}.npz` in
  `$R3_ROOT/xc_labels`.
- Dials: identical to R3 (states 200, horizon 100, label_every 25,
  actions 8, rollouts 16, ref_stride 5, max_steps 1000 — stamped into
  overlay meta and pinned by the reader, since it gates the
  labeled-state window — and `--oracle_all`, REQUIRED and asserted by
  the labeler for every overlay pass) EXCEPT
  **labeler seed 1** (registered fresh state draw: the labeled states
  differ from the committed R3 states — disclosed; every contrast is
  within-pass-pair, never against the committed labels, which are not
  re-read here).
- **Ordering: freeze-commit → existence gate → smoke (REGISTERED GATE,
  machine-checked by the reader: the 4 dosed cup-e4-seed31 5-state
  smokes `xc_cup_e4_seed31_{early,late}_h{early,late}_smoke.npz`,
  covering both swap directions and both controls — the overlay has
  never labeled a real dosed agent) → pairing gate (REGISTERED GATE:
  ONE full-dial pair, cup e1 seed31 eval-late control + swapped — two
  of the 128 wave passes — checked by the reader's `--pairgate`
  subcommand BEFORE the wave; this is an integrity gate, not an
  outcome read: achieved values exist inside those files, so the gate
  touches ONLY the meta version and the episode/step/g_all arrays and
  computes NO estimand — value-blindness is explicit in the frozen
  code) → labels (remaining 126) → ONE read**
  (`analysis/r3_xconsumer_read.py`).
- **Stdout value-blindness (frozen in the labeler)**: `save_rows`
  REDACTS the per-pass estimand summary (mean delta_real ≡ mean
  achieved) whenever a consumer overlay is active (version `_xc1`; also
  `_cm1`), printing state/episode counts only — without this, the
  paired stdout means of a control+swapped pair would reveal that
  run's primary contribution, and the full wave's logs would contain
  both primaries by subtraction before the ONE read. Smoke/pairgate
  stdout is therefore consultable for crash/dial/determinism sanity
  with no estimand exposure. (The default no-overlay path prints
  exactly as before — byte-identical for every existing registered
  caller.)

## Power (pre-sized from named committed sources; no new computation)

The parent registration measured per-run gap sd ≈ 1.0 (cup) / 1.3
(finger) and pre-sized the R3 detectable pooled effect at ≈
1.96·1.2/√32 ≈ 0.42 (`PREREG_r3_competence_20260725.md` §Power, from
the 25-Jul corrected relabel). This wave is strictly better-powered
per cluster: the primaries are PER-STATE PAIRED within a pass pair —
identical states, identical `g_all`, so all trajectory and
state-sampling variance cancels and residual variance comes only from
chooser disagreement on shared states. The detectable per-run effect
is therefore far below 0.42; we register the logic, not a fake
precision number, and note the committed achieved levels (≈ 0
everywhere) mean even small absolute rescues would be material.

## Registered decision rules (frozen in `analysis/r3_xconsumer_read.py`)

Run-clustered percentile bootstrap (cluster = training run, 32
clusters), B = 10K, `default_rng(0)` (machinery imported from the
frozen `analysis/r3_read.py` with import-pin asserts):

- **P-XC1 (late eval):** per-run mean of paired per-state
  [achieved(heads=early) − achieved(heads=late)]; CI entirely < 0 ⇒
  immature heads HURT (head knowledge matters on the late side).
- **P-XC2 (early eval):** per-run mean of paired per-state
  [achieved(heads=late) − achieved(heads=early)]; CI entirely > 0 ⇒
  mature heads RESCUE (head-knowledge deficit on the early side).
- **Verdict map:** HEADS-IRRELEVANT (both straddle 0) / HEAD-DEFICIT
  (P-XC2 fires +) / HEAD-HARM (P-XC1 fires −) / MIXED (both fire, or
  any CI excursion in an unregistered direction — disclosed, no
  headline claim) / QUARANTINE (pairing guard violated anywhere: no
  adjudication, audit first) / SUBSTRATE-GONE (existence gate).
- **Guards (machine-checked per file):** exact `d1fix_20260724_xc1`
  version pin; full dial identity incl. labeler seed 1 and
  `consumer_regex`; checkpoint AND consumer_checkpoint path suffixes
  vs the filename cell; train_seed vs filename; finiteness/m_real
  bounds/shapes (parent Amendment-1 guards); `g_now = g_all[m_now]`
  identity; all-or-nothing 128-file grid; unregistered seeds trip.
- Secondaries (descriptive, never decisional): per-domain XC1/XC2
  splits; chooser disagreement rates (`m_real`, `m_now`) per side;
  opportunity level per (eval, heads) cell; per-pair mean max-G
  identity (redundant guard sanity — the exact pair-invariant is
  `max_m g_all`, NOT opportunity, which shifts with `m_now` when
  choosers disagree).

## Registered interpretive limit

The overlaid heads read the OTHER checkpoint's features while trained
on their own. With same-run early/late checkpoints (phase B resumes
from the early snapshot) the feature mismatch is bounded but
irreducible in this route: mismatch degrades the transplanted heads
and biases BOTH contrasts toward null. A HEAD-DEFICIT fire is
therefore a LOWER bound on head-knowledge transfer; a HEADS-IRRELEVANT
null is correspondingly weaker than a same-feature null would be and
is claimed only as "not detectably head-resident under transplant".

## Consequence map (frozen)

- **HEADS-IRRELEVANT** ⇒ registered result: the competence deficit is
  not head-resident — the opportunity/competence decomposition gains
  its mechanism-side scoping and the representational account is
  strengthened; Paper 2 §decomposition cites both straddling CIs.
- **HEAD-DEFICIT** ⇒ registered result: part of the deficit is head
  knowledge (lower bound, per the interpretive limit); the P-R3c
  representational reading is weakened and the paper reports the
  split; any follow-up (e.g. head-only fine-tuning) is a NEW
  registration.
- **HEAD-HARM** ⇒ late-side competence depends on head knowledge;
  reported alongside whichever early-side branch obtains.
- **MIXED** ⇒ both CIs reported, no headline claim; interpretation
  deferred to a new registration if pursued.
- **QUARANTINE** ⇒ instrument audit before ANY use of xc labels; the
  wave is not interpretable evidence in any direction.
- **SUBSTRATE-GONE** ⇒ recorded as the registered outcome; no re-run
  on a partial substrate.
- **Regardless:** the committed R3/reacher/doubling reads stand as
  registered and are never recomputed; the committed R3 labels are
  never re-read by this wave; 24+24 stay frozen; imag stays retired;
  Route A stays closed.

## Costs

≈ 2× the R3 label wave: 128 full passes (the pairgate pair included)
+ 4 five-state smokes at the R3 per-pass rate (≈ 1–2 h each on one
GPU, `PREREG_r3_competence_20260725.md` §Costs) ≈ 130–260 GPU·h;
sequential on one box ≈ 6–11 days, cluster-lane parallel ≈ 2–4 days
wall-clock. No training compute (substrate reuse is the point).

## Disclosure

Known at freeze: everything through the 31-Jul record — the full R3
read (`artifacts/r3_competence_20260729/`), Amendment 1, the doubling
registration (unexecuted), the U2/U3/U4 reads, the parent's power
constants used above, AND the R3 reacher read EXECUTED 31 Jul before
this freeze-commit (**REPLICATES**: P-RRa opp +0.828\* + P-RRb gap
+0.813\* fire, P-RRc null, `artifacts/r3_reacher_20260731/`). The
reacher result is family-relevant context — a third domain where
achieved ≈ 0 and maturity does not close the gap is consistent with
(and mildly strengthens the prior for) the registered HEADS-IRRELEVANT
prediction — and is disclosed here precisely so the known-record
statement is true at commit time; the predictions and decision rules
above were written 30 Jul and are UNCHANGED by it. The `runroot_cleanup.sh`
classifier was updated in the same freeze to move the R3 run dirs'
label dir out of DELETE-AFTER and name this amendment (+
`PREREG_competence_repair_20260730`) as dependents of the R3 substrate
(donor-dependency lesson). Unknown: every cross-checkpoint quantity —
no xc label, no overlay pass, no consumer-swapped statistic exists
anywhere; `--consumer_checkpoint` has never been executed on any
checkpoint; whether the 32 run dirs still exist on RCC scratch is
itself unverified (that is what the existence gate adjudicates).
