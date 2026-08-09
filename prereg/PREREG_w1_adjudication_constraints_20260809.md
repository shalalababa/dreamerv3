# PREREG: W1 adjudication-wave design constraints — 2026-08-09

Frozen BEFORE any W1 instrument code or label exists (user GO 2026-08-09:
"all three" — framings (a)+(b) adopted AND the rescue adjudication run).
The wave's own primaries/power freeze in a separate `PREREG_w1_<date>.md`
once the instruments are built and selfchecked; this file registers the
design constraints the review resolution showed are load-bearing. The
wave prereg may not weaken constraints 1–7.

## What W1 must decide (three registered questions)

1. **Is any of the measured "opportunity" real?** (estimand rescue,
   review §4.2 item 8) — repeated-rollout split-selection.
2. **Do the actual consumers (Dreamer actor; TD-MPC2 MPPI planner)
   harvest?** (the D1/D3 fix — the claim the paper made and never
   tested.)
3. **Is the one-real-step ranking signal real once the shared-mark leak
   is removed?** (the dissociation's decisive test — NEW, motivated by
   the 2026-08-09 duplicate-leak witness: inside TM2 duplicate-action
   groups, where true signal is 0 by construction, Spearman(real_scores,
   g_all) = **+0.47 [+0.39, +0.57]** — `artifacts/
   review_response_p2_20260809/p2_mech_D_leak.json`. The committed
   probe-skill numbers (dv3 +0.023*, TM2 distinct-only +0.117*) are
   leak-contaminated to an unknown degree; only a de-shared measurement
   settles them.)

## Registered design constraints (violations invalidate the wave)

1. **Repeated evaluation with independent follower marks.** Every
   (state, candidate) is evaluated R ≥ 8 times from the SAME restored
   snapshot, varying ONLY the follower rng_mark across repeats (marks
   deterministic from a per-state base + repeat index; CRN across
   candidates WITHIN a repeat, independent ACROSS repeats). Stored as
   `g_all_rep (S, R, M)`.
2. **Split-selection estimator.** Selection statistics (argmax, chooser
   scores) computed on one half of the repeats; evaluation on the held-
   out half. Within-half quantities may be reported descriptively but
   NEVER as primaries.
3. **Duplicate-candidate null pass.** At least one registered pass per
   family in which all M candidates are the SAME action. Every estimand
   the wave reports must return 0 on it within CI — this is a GATE, not
   a finding; a nonzero duplicate-null invalidates the affected
   estimand's pass.
4. **Actual-consumer branches.** The labeler additionally evaluates, as
   extra branches under the same repeat structure: (a) DreamerV3 — the
   deployed actor's sampled action (the action currently computed and
   discarded at the label site); (b) TD-MPC2 — the MPPI planner's
   action (`agent.act(eval_mode=True)` at the decision state). Consumer
   primaries compare these, not the plug-in rule. The plug-in rule is
   retained as a comparison arm.
5. **Leak-free probe decomposition.** `op_real` runs under a mark
   INDEPENDENT of every G-evaluation mark, and per-candidate `r_real`
   is STORED (the `return_rewards` path built for xc Amendment 1).
   Registered decomposition: rank signal of `real_scores`, of `r_real`
   alone, and of `real_scores − r_real` (the learned-value component),
   each against held-out-repeat G. The dissociation claim is
   adjudicated on the held-out, de-shared version only.
6. **Environment seeding.** The labeler env is constructed with an
   explicit seed (dmc `use_seed`-equivalent wiring; seed recorded in
   meta). The unseeded path is retired for all new labeling.
7. **Global stdout redaction.** No estimand mean is printed by any W1
   pass (extend the `_xc1/_xc2/_cm1` redaction to the W1 labeler
   version unconditionally).

## Registered outcome map (adjudication only; exact rules in the wave prereg)

- Split-selected opportunity ≈ 0 in both families ⇒ the winner's-curse
  account is COMPLETE; framings (a)/(b) proceed with the estimand
  retired; a nonzero duplicate-null anywhere instead invalidates that
  pass's instrument (gate, constraint 3).
- Split-selected opportunity > 0 with the TM2 point near its
  bias-corrected +0.34 ⇒ a smaller identified opportunity exists;
  rescue framing (c) re-opens with honest magnitudes.
- Actual-consumer branches: if the planner/actor track the split-
  selected oracle better than the plug-in rule, D1/D3's "the agent was
  never tested" becomes "the agent partially harvests" — a NEW result
  either way; if they match the plug-in's chance level, the original
  competence claim revives in identified form.
- Leak decomposition: if `real_scores − r_real` retains CI>0 rank
  signal on held-out repeats ⇒ the dissociation survives leak-free
  (framing (a) headline confirmed, now with mechanism: the signal's
  carrier identified as learned-value vs ground-truth-reward). If only
  `r_real` carries signal ⇒ the honest statement becomes "one step of
  GROUND-TRUTH REWARD outranks the entire learned value stack" — a
  sharper, still-publishable dissociation. If neither ⇒ the probe skill
  was wholly leak; the ensemble-flatness result (leak-free, already
  established: candidate spread ≈ 5% of own head noise, both families)
  stands alone.

## Budget + process

~1–2 GPU-days total on EXISTING checkpoints (no retraining): R3 late
cells (32) + TM2 late cells (32) at R=8, plus duplicate-null passes and
consumer branches. Instruments built inline (additive, flag-gated,
default paths byte-identical); labeler selfcheck EXTENDED with a
stochastic-follower fixture and a duplicate-candidate fixture whose
required output is zero (the 13.4 lesson: deterministic fixtures are
blind to realization noise). ONE reviewer over {this file, the wave
prereg, the modified labelers, the frozen reader} before submission
(user-approved in the GO). Reader frozen + selfchecked BEFORE the read;
ONE read per registration.

Freeze ordering: commit this file BEFORE any W1 instrument code lands.
