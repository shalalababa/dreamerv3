# PREREG: W1 adjudication wave — decision rules — 2026-08-09

Implements `PREREG_w1_adjudication_constraints_20260809.md` (committed
first; constraints 1–7 inherited unweakened). Frozen BEFORE any W1
label exists. Instruments (built + selfchecked 2026-08-09, this
session; ONE batch reviewer over {both prereg files, both labelers,
this reader} runs before submission — user-approved):

- `d0/oracle_labels.py` — `--w1_repeats/--w1_dup_cand/--env_seed`,
  `_w1_label_state` (repeat marks base+(r+1)·1e5, probe mark base+5e4,
  deployed-actor branch, r_real stored, dup_cand, seeded env via
  `seed_env_task`, `_w1` stdout redaction). Selfcheck PASS incl. the
  stochastic-follower fixture (within-repeat max bias VISIBLE,
  split-selection kills it) and the exact duplicate-null leg.
- `probing/tdmpc2_oracle_labels.py` — same flags; `_w1_label_state_tm2`
  (seeded torch marks via `w1_mark`, MPPI-planner branch = the actual
  `follower_act` at the decision state, r_real, dup_cand). Selfcheck
  PASS (same fixture classes).
- `analysis/w1_read.py` — frozen reader, subcommands dv3/tm2; selfcheck
  PASS (fires/no-fires on planted signal/noise/leak/value fixtures;
  duplicate gate trips; count guard). ONE execution per family.

## Cohorts (existing checkpoints; NO retraining)

- dv3: the 32 R3 LATE cells (cup/finger × e1/e4 × seeds 31–38),
  `--w1_repeats 8 --oracle_all --env_seed <registered below>`,
  standard dials (200 states, horizon 100, label_every 25, M=8).
  Output names `w1_{dom}_{dose}_seed{n}_late.npz`.
- tm2: the 32 TM2 LATE cells (cup/finger × e1/e4 × seeds 51–58), same
  dials, names `w1tm2_{dom}_{dose}_seed{n}_late.npz`.
- Duplicate-null passes (GATE, constraint 3): 2 per family —
  cup_e1_seed31 + finger_e1_seed31 (dv3), cup_e1_seed51 +
  finger_e1_seed51 (tm2), `--w1_dup_cand`, names `w1dup_*` /
  `w1tm2dup_*`.
- `--env_seed` = 20260809 for every pass (also the TM2 mark seed base).
- Budget: 68 passes × ~(8×9+8+9) rollouts/state × 200 states ≈ within
  the ~1–2 GPU-day envelope on existing checkpoints.

## Frozen decision rules (implemented verbatim in `analysis/w1_read.py`)

Statistics: per-cell means of per-state values; cluster = run;
run-clustered BCa 95% (B=10K, rng 0) + sign-flip permutation p
(standing rule 08-08: permutation-primary + BCa).

- **P-W1a (estimand rescue)** per family: split-selected opportunity
  `mean_s [Ḡ_odd(sel_even) − Ḡ_odd(m_now)]`. FIRES iff BCa CI entirely
  > 0 AND perm p < .05. Fire ⇒ an identified opportunity exists (its
  magnitude replaces every retired headline number; rescue framing (c)
  re-opens at honest scale). No-fire ⇒ the winner's-curse account is
  complete for that family; the construct-invalid re-grade of
  P-R3a/b (P-TM2a/b) becomes final.
- **P-W1b (actual-consumer harvest)** per family: consumer branch
  (dv3 = the action d0_eval sampled from the DEPLOYED belief and the
  default path discards — batch-review B1: never a re-sample from the
  post-assimilation carry; tm2 = the WARM-STARTED planner action,
  `follower_act(obs, t0=False)` with the label-site `_prev_mean`
  restored — batch-review M1: the deployed convention; cold start
  remains registered for candidate branches only) vs the held-out
  candidate mean, same fire rule. Fire ⇒ "the agent
  was never tested" becomes "the actual agent harvests above a random
  candidate" — the competence claim inverts. No-fire ⇒ the original
  competence-failure claim revives in identified, consumer-correct
  form (scoped to whatever P-W1a left alive).
- **P-W1c (leak-free dissociation)** per family, three-way verdict on
  held-out ranks (non-degenerate states):
  `DISSOCIATION-SURVIVES` iff Spearman(real_scores − r_real, Ḡ_odd)
  BCa CI > 0 ∧ perm p < .05 (the learned-value component carries
  signal — §6's dissociation confirmed leak-free);
  else `GROUND-TRUTH-REWARD-ONLY` iff Spearman(r_real, Ḡ_odd − r_real)
  CI > 0 ∧ perm p < .05 — **the TAIL form (batch-review M2): r_real is
  bitwise the first-step component of every repeat's G (same restored
  transition), so the naive Spearman(r_real, Ḡ_odd) fires
  tautologically whenever r_real varies; the registered test is
  ground-truth first-step reward predicting the REST of the return**
  (sharpened headline: one step of ground-truth reward outranks the
  entire learned value stack);
  else `LEAK-WAS-ALL` (the probe-skill numbers were wholly the shared
  mark; the ensemble-flatness result stands alone).
- **DUPLICATE-NULL GATE** (constraint 3): both dup passes per family
  must give bit-identical branch returns within every repeat,
  split-opportunity EXACTLY 0, AND constant per-state `r_real` /
  `real_scores` (batch-review M3 — the historical leak channel was
  probe-side); any violation ⇒ that family's read REFUSES
  (INSTRUMENT-INVALID), no estimand is reported.
  **Constraint-3 scope (batch-review M5, registered clarification, no
  weakening):** the estimands with a 0-by-construction duplicate
  expectation are P-W1a (split opportunity) and the P-W1c inputs
  (branch/probe constancy) — these ARE the gate. P-W1b's duplicate
  expectation (consumer branch vs a duplicated candidate) is NOT zero
  by construction (the consumer action legitimately differs from the
  duplicated candidate) and is therefore excluded from the gate and
  never reported on dup passes.
- Descriptives (no decision weight): per-(state, action) sd(G) across
  repeats; argmax stability across repeats; within-pass vs split
  opportunity side by side; per-domain splits.
- **Estimand scope disclosure (batch-review m1):** the env snapshot
  (incl. the OU distractor RNG) is shared across all repeats and both
  halves, so W1 opportunity is REALIZATION-CONDITIONAL — identified
  relative to the frozen state and wrapper-noise realization; follower
  sampling is the only randomized dimension. Cross-realization
  generality is carried by the 32-cell / 2-family breadth, not within
  a state.
- **Reader pins (batch-review M4):** `analysis/w1_read.py` refuses any
  label file whose meta lacks env_seed=20260809, repeats=8, states=200,
  horizon=100, or whose filename (dom, dose, seed) is absent from its
  `run_id` (fail-closed against submission-script mislabeling).
- **Spearman convention (batch-review B2):** tie-averaged midranks;
  (near-)constant raw inputs return NaN and drop out — index order can
  never masquerade as ranking signal on sparse-reward states.

Registered expectations (against-interest bookkeeping, not rules):
P-W1a no-fire in dv3 (cross-fit −0.105 predicts it); P-W1a TM2 open
(bias-corrected +0.34 predicts a small fire); P-W1b open both families
(never measured before); P-W1c open (duplicate-leak witness ρ=0.47
makes LEAK-WAS-ALL live in TM2).

Known at freeze: everything in `artifacts/review_response_p2_20260809/`
(committed pre-wave), incl. the cross-fit, the duplicate-leak witness,
and both reviews. Unknown: every W1 number (all marks fresh; env seeded
20260809 — no prior pass ever ran seeded, so no W1 quantity exists
anywhere).

Labels dir convention (registered): `$RUNROOT/w1_labels/` — protected
in `runroot_cleanup.sh` KEEP_PENDING until both reads verify.

Ordering: [YOU] freeze-commit {constraints file (if not yet), this
file, both labelers, the reader} AFTER the batch review lands →
smoke (1 cell per family, ≤5 states, `_smoke.npz` suffix, **plus one
dup_cand smoke cell per family** — the reviewer's cheap-insurance
recommendation: the gate must be seen to hold on real hardware before
the 68-pass spend) → 68 passes → ONE read per family via
`analysis/w1_read.py` → bundle + ledger row.

Batch review (ONE reviewer, 2026-08-09, user-approved): 2 BLOCKING /
5 MAJOR / 5 MINOR + NITs — ALL B+M applied same-day (B1 actor=act0
with carry-sensitive fixture; B2 midrank Spearman + constant-input
NaN + tie fixtures + tautological-form mutant caught; M1 warm-started
planner; M2 tail-form reward test; M3 probe-side dup gate; M4 reader
pins; M5 scope note; m1/m3/m4/m5 applied). Reviewer verified: mark
discipline collision-free at registered dials; within-invocation
same-mark CUDA determinism holds on this stack (761/761 duplicate
groups bit-identical in existing TM2 labels — DUP_TOL=0.0 defensible);
split estimator sound; BCa/perm correct; redaction complete; regexes
exact; no cleanup glob eats W1 outputs; dv3 constraint-6 genuinely
closed (distractor already SeedSequence-seeded; the dm task RNG was
the only hole).
