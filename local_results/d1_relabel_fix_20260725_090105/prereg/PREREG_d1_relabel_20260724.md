# PREREG: D1 corrected-instrument relabel campaign (frozen 2026-07-24)

Registered instrument correction + relabel of the shift-consequence
probe. Trigger: external review
(`research_notes/other research/D1_GPT_Analysis_20260724.md`) claimed,
and code inspection CONFIRMED (see analysis/DEVIATIONS.md 24-Jul
deviation entry), two defects in the shared D1 real-operation labeler
that affect EVERY D1 label ever produced (Gate-D1, R2, ladder, shift).
This file + the repaired labeler + the updated read + driver are
committed BEFORE any corrected label exists.

## The two confirmed defects (old labeler, all versions through
## shift_ext_20260723)

1. **Stale-carry candidate branches.** The labeling loop discarded the
   posterior carry returned by `d0_eval` at the labeled state; both
   `op_real` and `rollout_return` evaluated obs_{t+1} from the
   PRE-obs_t carry, with prevact still a_{t-1} rather than the
   candidate. Candidate-conditioned values were therefore computed
   from a belief that (a) never assimilated obs_t and (b)
   mis-attributed the transition action. All branches shared the same
   corrupted convention, so the labels are a well-defined but
   UNINTENDED estimand (value differences under a belief-corrupted
   follower); nulls on those labels cannot be read as absence of value
   (attenuation), and positive constants cannot anchor a paper
   (validity).
2. **Policy-RNG not common across branches.** The eval-mode policy
   SAMPLES (dreamerv3/agent.py `sample(policy)`; `mode` unused) with a
   seed derived from the global `n_actions` counter
   (embodied/jax/agent.py), which was never restored between branches
   — follower sampling noise was NOT common across candidates, so the
   "exact CRN" claim covered env state and belief only.

## The registered fix (labeler_version `d1fix_20260724`)

`d0/oracle_labels.py`: `label_run` keeps the posterior carry
(`carry_post`) from the labeled-state eval; every branch (each op_real
probe, each rollout) starts from
`oracle.branch_carry(carry_post, cand)` = posterior through obs_t with
prevact = the candidate; a policy-RNG mark is taken once per labeled
state and restored before every branch (`rng_mark`/`rng_reset` on the
outer agent's n_actions counter), making follower sampling noise
common across candidates. op_imag intentionally keeps the pre-obs_t
carry (it re-assimilates obs_t each sample) and gets no RNG reset
(fresh samples are its estimand); the imag op remains retired for
decisions regardless. NEW row fields `g_all`/`g_all_boot` (per-
candidate realized returns, NaN where not computed) support the
opportunity/achieved-value decomposition; label passes run with
`--oracle_all` (every candidate grounded).

Validation: selfcheck extended with (a) a carry-sensitive recurrent
mock whose closed-form branch scores REQUIRE obs_t assimilation and
prevact = candidate (the stale-carry convention provably fails it),
(b) branch-start carry recording (all 8 branch entries per 2-state run
carry (pos_t, cand)), (c) a policy-RNG contract check (one reset per
branch, all to the per-state mark), (d) oracle_all fill checks, (e)
structural checks of the real-adapter branch_carry/rng plumbing.
SELFCHECK PASS. Read-side: `analysis/d1_shift_read.py` version assert
updated to `d1fix_20260724` in this same freeze (it no longer accepts
defective labels); its own selfcheck PASS.

### Audit additions (same day, pre-freeze — full-code audit findings)

A same-day audit of the labeling stack found three further issues, all
fixed IN THIS SAME FREEZE (still zero corrected labels in existence;
DEVIATIONS 24-Jul audit entry is the tracked record):

3. **xpol trajectory prevact mis-attribution** (labels-affecting, xpol
   arm only): on behavior-driven trajectories the eval carry's prevact
   was the eval agent's OWN sampled action while the env executed the
   BEHAVIOR action — the eval belief mis-conditioned every trajectory
   step. FIX: `with_prevact` substitutes the executed action into the
   eval carry after each behavior step. Selfcheck: behavior-driven
   recurrent mock asserts trajectory calls arrive with prevact = the
   executed behavior action. (Retroactive: the 23/24-Jul xpol cells
   carried this additional defect.)
4. **Distractor OU state not snapshotted** (no effect on THIS wave —
   e1 only, wrapper not applied; affects any dosed labeling, e.g. R2's
   old e4 cells and the future R3): snapshot captured the RNG but not
   the OU value `_state` or the Welford calibration triple — branches
   replayed identical noise increments from branch-shifted starting
   values. FIX: `Distractor.oracle_get_state/oracle_set_state`
   (exact round-trip verified incl. live calibration; rng-only restore
   demonstrated inexact); `snapshot_env`'s custom slot is now a LIST
   (was last-writer-wins). Any future dosed labeling wave must smoke
   ON a dosed env before labels.
5. **`FromDM._done` not restored** (latent — verified NEVER fired:
   episodes run 1000 wrapper steps, labels stop at 875, branch
   rollouts reach ≤ 975): a branch ending an episode would have left
   the done-latch set, silently auto-resetting the next branch and the
   resumed base trajectory. FIX: done-latches are snapshotted and
   restored; selfcheck covers the latch and multi-node custom state.

## Campaign design (existing checkpoints; user's 5090)

- Checkpoints: the 6 existing shift pilots `d1s_{cup,finger}_seed{21,
  22,23}` (final ckpts; TRAINING_DONE — the pilots stage skips).
- Arms and dials: IDENTICAL to PREREG_d1_shift_20260723 ({base, xpol
  sibling-rotation 21→22→23→21, phys mass ×1.3}; states 200, horizon
  100, label_every 25, actions 8, rollouts 16, labeler seed 0,
  ref_stride 5) + `--oracle_all`. 18 relabel passes into
  `$D1S_ROOT/d1_labels_fix` (the defective `d1_labels/` is preserved
  untouched as provenance).
- Ordering: freeze-commit → `./scripts/d1shift_local.sh smoke` (both
  shift arms, real MuJoCo path — the repaired branch plumbing has not
  yet touched a real agent; any fix ⇒ dated amendment BEFORE labels)
  → `labels` → frozen read.
- Cost [prov.]: oracle_all grounds 8 candidates/state vs ≤3 ⇒ ~2–2.5×
  per pass; 18 passes ≈ 1–2 days sequential.

## Registered decision rule (PRIMARY, unchanged from the shift prereg)

Per shift arm (2 looks): D_r = mean(Δ_real | arm) − mean(Δ_real |
base) paired per run, 6 runs pooled; arm FIRES iff the run-clustered
bootstrap 95% CI (B = 10K, rng 0) of mean(D_r) is entirely > 0
(`analysis/d1_shift_read.py`, frozen). Sensitivity disclosure: the
small-cluster t-interval will be REPORTED alongside (registered
descriptive, not decisional; n = 6 clusters).

## Consequence map (frozen)

- **≥ 1 arm fires** ⇒ the shift-consequence leg is REVIVED on the
  corrected instrument; the 24-Jul defective-wave null is recorded as
  an instrument artifact. Which arm scopes the claim.
- **Neither fires** ⇒ the consequence leg fails ON THE CORRECTED
  INSTRUMENT — the registered negative now carries interpretive
  weight the defective wave could not (the attenuation objection is
  discharged up to the design's power, which the t-interval
  disclosure bounds honestly).
- Either way: 24+24 stay frozen; Route A stays closed; imag stays
  retired. Gate-D1/R2/ladder conclusions REMAIN
  instrument-invalidated regardless of this outcome (they are not
  re-adjudicated by this wave; see plan v4 for what would be).

## Registered secondaries (descriptive, never decisional)

1. **Defect-impact panel**: per-run mean Δ_real and change_rate, old
   (shift_ext) vs new (d1fix) labels on the SAME checkpoints —
   distributional A/B (trajectories re-sample; not state-matched).
2. **Base-cell late×e1 constant** with cluster CI on corrected labels
   (the only cell of the old 2×2 these checkpoints instantiate).
3. **Opportunity/achieved decomposition** (from g_all): oracle
   opportunity = max_m G[m] − G[m_now]; achieved = G[m_real] −
   G[m_now]; implementation gap = difference; per arm × domain.
   These PARAMETERIZE the prospective R3 design (plan v4 §"R3"); they
   adjudicate nothing here.
4. C1-style calibration per arm (frozen machinery), as before.

## Disclosure

Known at freeze: everything through the 24-Jul defective-wave read
(`artifacts/d1_shift_20260724/`), including its per-cell means; the
reward-floor fact (finger 22/23 base cells all-zero G at 1e5 pilots)
is PRE-EXISTING and will not be "fixed" by relabeling — finger
sensitivity stays limited at these checkpoints (design limit carried
forward, disclosed). Unknown: every corrected-label quantity. No
corrected label of any kind exists at freeze. The old wave's states
were labeled under a different follower convention; nothing here
constitutes a peek at corrected outcomes.
