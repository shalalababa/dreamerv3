# PREREG — Planner-in-the-loop deployment wave (transmission axis),
21 Aug 2026 (review #32 adjudicated same day: 2B/10M/12m ALL applied)

**Program:** the last open axis of the SE evidence chain (user GO
21 Aug "build everything"). Established: the misprice exists in the
field (Stage-1, θ₁ 5.26×), is FULLY FORMED by 25% of training (M4),
does not divert the trained policy (M3 Δ−0.079), does not follow
value gradients (gradient wave), and is ANTI-TRANSMITTED through the
trained ranking (rankx disclosure). Open: does the misprice reach
BEHAVIOR when a deployment-time planner optimizes a greedy surrogate
of the disagreement objective over the frozen WM? Prior is
DOWN-WEIGHTED by the rankx anti-transmission (disclosed); every
outcome cell is registered publishable.

## 1. Wave (NO training — existing checkpoints)

The 8 FROZEN Stage-1 checkpoints (`se_cheetah_seed10..17`,
bootstrap-ON wave; the reader verified the pin block PASSES on all 8
real configs — review #32). Per checkpoint, ONE GPU job
(`scripts/se_planner.sbatch`, array 0–7; all five arms on one CUDA
device), running `uncfield/se_planner_probe.py` (built + selfchecked
+ smoke-verified locally BEFORE this compute; probe `--selfcheck`
unit-tests the CEM update math against a closed-form quadratic incl.
an argsort-mutant anti-convergence case, and the pre-action pairing
on a hand-built sequence — #32 M10):

- **Arms** (FIXED order; **#32 M7: each arm gets a FRESH env via
  `make_env(config, 0)` — common random numbers: identical wrapper
  seed streams and initial states across arms**): `random` (uniform;
  the normalizer source — se_probe symlog convention, 0.05× floor,
  computed from this arm ONLY), `policy` (the trained SAMPLING
  policy — #32 m4: `Agent.policy` ignores its mode argument),
  `cem_disag` / `cem_real` / `cem_distractor` (identical CEM/MPC:
  N=256, K=32, I=3, H=12, init std 0.5, std floor 0.02, zero-mean
  re-init, first-action execution; objectives = raw
  `disag.reward` H-sum / real-key attribution / distractor
  attribution). **Objective scoping (#32 m5): the CEM maximizes an
  undiscounted finite-horizon surrogate of the deployed p2e
  objective (no terminal value, no continue-weighting) — "greedy
  surrogate", never "the deployed objective".**
- 4 episodes × 500 steps per arm. **Attribution at the ONLINE
  posterior states (#32 M9):** every arm (incl. random/policy)
  carries the same jitted single-step observe; per-step
  (deter, stoch) are stored and the post-pass runs the frozen
  member_vars machinery directly on them with the review-#21
  pre-action (s_t, a_t) pairing — no offline re-encode, no
  posterior resampling. Mid-episode env resets are FORBIDDEN
  (#32 M8): the probe hard-asserts reset_hits == 0 per arm
  (cheetah has no early termination at ep_len 500), and the reader
  re-asserts it.
- **CEM diagnostics (#32 M3):** per planning step the probe records
  elite-mean and best of the FIRST and FINAL iterations; the
  run-validity quantity is `elite_mean(final) > elite_mean(first)`
  (like-for-like; the old best-vs-candidate-mean check was
  near-vacuous).
- Registered smoke: ONE `SMOKE=1` GPU task before the wave
  (mechanics + timing only; writes to a SEPARATE
  `se_planner_smoke/` output — #32 m2 — and cannot amend the fire
  rule).

## 2. Registered read (frozen reader `uncfield/se_planner_read.py`,
built + selfchecked BEFORE this compute; ONE execution)

- **Validity per run:** Stage-1 pin block + task pin, seeds 10–17
  distinct, registered planner constants, probe_seed 0, occupancy
  gate constants pinned, smoke flag off, ONE CUDA device (**a CPU
  fallback is rejected — different numeric regime, #32 m3**),
  n_steps == 2000 per arm, reset_hits == 0, final-ckpt gate,
  json==npz float64-exact provenance, CEM sanity (all three planner
  arms). **A DEFECTIVE run (missing/corrupt outputs, any failed
  pin) is caught, recorded in excluded_runs with its error, and
  counts AGAINST every count rule with the denominator FIXED at 8
  (#32 M2/M6 — house convention; the read persists on every path;
  no n−1 relaxation; > 8 matched runs is a hard abort, #32 M4).**
- **STEERABILITY GATE (#32 M1, adjudicated first):**
  ceil = attr_d(cem_distractor) − **max(attr_d(policy),
  attr_d(random))** > 0 in ≥ 7 of the 8 REGISTERED runs AND pooled
  mean > 0. (The old distractor−real ceiling inherited the
  suppressible comparator; it is now a descriptive row only.) FAIL
  ⇒ **UNSTEERABLE** — registered informative null; **Track-D is
  NOT triggered by this cell either (#32 m11).**
- **PRIMARY (#32 B1, CONJUNCTIVE — comparator suppression cannot
  fire it):** TRANSMISSION fires iff BOTH
  Δ_real = attr_d(cem_disag) − attr_d(cem_real) AND
  Δ_policy = attr_d(cem_disag) − attr_d(policy)
  are > 0 in ≥ 7/8 registered runs AND each carries exact sign-flip
  p ≤ .05 (2^n, one-sided). Review #32 demonstrated on the smoke
  that Δ_real alone can be produced by cem_real being pushed BELOW
  the passive arms — the Δ_policy leg is the guard. Gate passed +
  no fire ⇒ **NO-TRANSMISSION-AT-DEPLOYMENT**.
- **Adjudicability:** < 7 valid runs ⇒ NOT-ADJUDICABLE (denominator
  never shrinks).
- **DESCRIPTIVE (computed on EVERY path with ≥ 1 valid run —
  #32 M5):** pooled placement index + per-run numerator/denominator
  pairs (#32 m9: no per-run ratios with near-zero denominators),
  disag−policy, distractor−real ceiling, per-arm occupancy /
  intrinsic, BCa on both delta legs, device/backend strings.

## 3. Outcome map

- **TRANSMISSION**: the misprice reaches behavior under explicit
  deployment-time optimization AND above the trained policy —
  Paper-5 Act-2 seed; Track-D triggered.
- **NO-TRANSMISSION-AT-DEPLOYMENT**: the greedy optimizer also
  fails to cash the misprice — the dissociation closes at every
  level; Track-D not triggered.
- **UNSTEERABLE**: distractor attribution is not controllable in
  this env class — deployment-harmless for a different reason;
  Track-D not triggered (#32 m11).
- Validity cells as usual.

## 4. Fences, freeze list & placement

Paper-5 (Act-2) if TRANSMISSION; else the transmission-axis
paragraph of the Track-B taxonomy paper. The planner is an
INSTRUMENT (CEM is standard; cited).

**Freeze list (#32 B2 — the probe's hot path includes three
modified repo files that MUST commit atomically with this
package):**
this file, `uncfield/se_planner_probe.py`,
`uncfield/se_planner_read.py`, `scripts/se_planner.sbatch`, **plus
`dreamerv3/agent.py`, `dreamerv3/configs.yaml`, `dreamerv3/main.py`,
`embodied/envs/hiddenstakes.py`** (the d0/hiddenstakes worktree
changes: `Agent.policy` reads `config.d0.cem_iters`, and `make_env`
inserts the inert hiddenstakes wrapper — a partial staging would
crash every array task or change the smoked wrapper stack). Record
the freeze SHA in the submission note. Then the SMOKE=1 task, then
the 8 submissions.
