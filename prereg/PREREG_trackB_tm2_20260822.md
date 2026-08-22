# PREREG — Track-B5: the TD-MPC2 column (decoder-free attribution), 22 Aug 2026 — rev 2

**Program:** Track B (user GO 22 Aug "do all the builds"; rev 1
same-day per review R4 — anchor/action pairing corrected,
specificity + inflation cells added, pins extended; **rev 2 same
day per R-A1-M12**: the registered fire channel — distractor batch
permutation — was STRUCTURALLY UNABLE TO FIRE: with gates/mod off
the SE distractor is a pure exogenous AR(1), so permutation is
law-preserving and its population delta is zero by exchangeability
for ANY statistic (`artifacts/det_semask_baseline_20260822/NOTE.md`
+ correction). The fire channel is re-specified LAW-CHANGING; all
revisions pre-outcome, no B5 data exist). The matrix's second architecture family: does
TD-MPC2's uncertainty proxy — its Q-ENSEMBLE std (K=5; TM2 has NO
dynamics-disagreement ensemble, `tdmpc2_oracle_labels.py:45`) —
price the SE planted channels? Filled cells for contrast:
dreamer-disagreement (farms stochasticity 5.26×, duplicate-immune),
APT (immune to both; its velocity control ran −0.028 in 8/8 — the
fact that motivates the specificity cell below). The B5 instrument
is itself the registered contribution the plan gated this column
on: a DECODER-FREE attribution (TM2 has no decoder, so the dreamer
decoder-projection line cannot port).

## 1. Machinery (built + core-selfchecked BEFORE compute; the torch
path is gated by the registered cluster smokes)

- `probing/tdmpc2_compat.py` extensions: `dmc_cheetah_run` in
  OBS_ORDER; dose `'se'` (the Stage-1 distractor pins: dim 8, scale
  1.0, theta 0.1, basesd 1.215); `planted=True` support in
  Dv3TaskEnv (Planted wraps AFTER Distractor — the main.py order;
  seed `SeedSequence([seed, 0, 0x5E])`); **extra-key order PINNED**
  `('distractor','planted_dup0','planted_dup1','planted_dup2',
  'planted_const')` (it defines the flat layout);
  `channel_slices()` helper.
- `probing/tdmpc2_r3_train.py`: cheetah + `--dose se --planted`;
  the audit json records planted + extra_keys + num_q + realized
  steps.
- **`probing/tm2_qmask.py`** (the decoder-free instrument): S=512
  single-state anchors collected under the trained policy (TM2's
  encoder is feedforward — no windows). **Anchor pairing (R4-B1):
  each anchor is (s_t, a_t) with a_t = π(s_t) captured BEFORE the
  env step — the state acted FROM; the t=0 post-reset state is
  excluded (its t0=True planner state differs qualitatively).**
  **Baseline action FROZEN per anchor across every variant** (the
  estimand is the accounting, not the policy). Anchors are
  stride-subsampled (stride 7) from ~4 episodes of one trajectory
  and are NOT independent — anchor-level p/BCa are DIAGNOSTIC for
  that reason (R4-M4). The statistic is RNG-free (encode/Q are
  sampling-free, model asserted in eval mode — dropout off); the
  anchor collection is ONE-SHOT (the instrument refuses to
  overwrite a recorded pass) and its sha256 is recorded, so a
  silent resample is detectable (R4-M2). `mpc=True` + seed passed
  as explicit cfg overrides (the house precedent; recorded and
  reader-pinned, R4-M1). **Variant suite (rev 2, R-A1-M12):**
  **distractor MEAN-SUBSTITUTION = the FIRE channel** (every
  anchor's distractor replaced by the anchor-population mean
  vector — LAW-CHANGING, level-sensitive: it tests whether the
  Q-heads' disagreement depends on the channel carrying its
  realized values at all); **velocity MEAN-SUBSTITUTION = the
  form-matched specificity comparator** (upgrades R4-M5);
  **distractor batch permutation = the built-in
  EXCHANGEABILITY-NULL calibration row** (population delta 0 by
  construction for the exogenous channel; a BCa-separated,
  fire-comparable value flags an instrument defect, report-only);
  velocity batch permutation = the coupling teeth (velocity is
  state-coupled — informative); dup0 bitwise no-op hard-assert;
  dup substitution + fresh-resample. House RNG offsets (R4-m17).
  Statistic = std over the 5 Q heads of
  `two_hot_inv(Q(encode(flat), a_frozen, return_type='all'))`.
  Torch-free core selfcheck PASS incl. the rev-2 exogenous case
  (permutation EXACTLY zero-mean on a channel-only statistic while
  mean-substitution fires).

## 2. Wave (8 TM2 trainings, seeds 132–139 + registered smokes)

`tm2_qmask`-gated chain on the tdmpc2 conda env (torch;
`scripts/tm2_b5.sbatch`): (i) **apismoke** — random-init agent on
the cheetah+se+planted env: env construction, the 53-dim flat
layout, planner act, env stepping, and the tm2_qmask import chain
(rc-gated, no outcome); (ii) **real smoke** — one 5k-step training
+ a REDUCED-S tm2_qmask mechanics pass (`--n_eval 128`; the S=512
pin is the READER's gate, house form — R4-B2) on its ckpt (dup0
zero, finite); (iii) the 8 trainings (`tdmpc2_r3_train --task
dmc_cheetah_run --dose se --planted --seed 132..139`, 100k steps,
walltime 24 h, ckpt_late) + one `tm2_qmask` pass per run (`--ckpt
ckpt_late.pt`; skip-if-recorded). The torch-side smokes are the
registered cluster gates (house rule: torch selfchecks are cluster
smokes). Submit recipe (R4-m14): export `TM2_CONDA_ENV`,
`TDMPC2_CHECKOUT`, `RUNROOT`; then `TM2B5_STAGE=apismoke sbatch
scripts/tm2_b5.sbatch` → `TM2B5_STAGE=smoke …` → per seed
`TM2B5_STAGE=train TM2B5_SEED=$s …` → `TM2B5_STAGE=qmask
TM2B5_SEED=$s …`.

## 3. Registered read (frozen `uncfield/tm2_qmask_read.py`,
selfcheck PASS; ONE execution; registered invocation `--runs
"<runroot>/tm2se_s13[2-9]" --output <artifacts dir>` — the glob
must NOT sweep the smoke dir, R4-M8)

- Validity: audit pins (task/dose/planted/extra-key order/num_q/
  realized steps ≥ 100k — the 7-Aug fit-counter rule), instrument
  pins (ckpt == ckpt_late.pt, S=512, num_q=5, mpc=True,
  instrument-side extra-key order, per-channel FORM pins), seeds
  132–139, done-markers, dup0 vector-zero, float64 provenance;
  defective → excluded, missing runs count AGAINST (denominator 8).
- **PRIMARY (rev 2 — single fire channel, no BH; the APT-read
  form):** `channels["distractor_meansub"]["delta_qstd_mean"] < 0`
  in ≥ 7/8 registered runs (binomial 9/256).
- **Outcome cells (registered BEFORE any number exists):**
  - **i-TM2-PRICES-DISTRACTOR** — fire AND channel-specific: the
    Q-value channel's disagreement depends on the distractor's
    realized values.
  - **iii-TM2-NONSPECIFIC** — fire, but in a majority of loaded
    runs the FORM-MATCHED velocity mean-substitution is also
    negative with |Δ_vel| > 0.5·|Δ_dist|: the evidence is
    explained by "any off-manifold substitution lowers this
    statistic"; no pricing claim licensed.
  - **iv-TM2-PRICES-INFLATING** — Δ > 0 in ≥ 7/8 (same 9/256):
    the sign under the intervention is not theoretically pinned;
    systematic inflation is pricing, NOT immunity.
  - **ii-TM2-IMMUNE** — no fire in either direction with ≥ 6
    loaded runs. Worded: the Q-std does not respond to the
    LAW-CHANGING removal of the distractor's realized values —
    a genuine level-immunity statement this time (rev 2 closed
    the R-A1-M12 hole: the rev-1 permutation form could not have
    fired on any pricing).
  - **NO-EVIDENCE** (R4-m15) — < 6 loaded runs and no fire: an
    immunity claim needs data.
- Descriptive: the EXCHANGEABILITY-NULL calibration row (the
  distractor permutation — inert by construction; BCa-separated,
  fire-comparable values are flagged as instrument defects),
  velocity permutation (the coupling teeth), velocity meansub row,
  dup substitution + fresh-resample nulls, base Q-std levels,
  p_inflate per channel. Registered caveat: Q-std measures VALUE
  disagreement — a null does not license "TM2's exploration is
  safe" (TM2 as configured has no intrinsic bonus); the cell is
  about the uncertainty PROXY's accounting.

## 4. Fences

Track-B taxonomy paper. The instrument (decoder-free attribution)
is claimable as a contribution; the TM2 architecture is not. TM2
pin: checkout e9f59321 (the sbatch asserts it).

Freeze = review R4 adjudicated (done 22 Aug — 2 BLOCKING + 9 MAJOR
+ 11 minor) + the rev-2 fire-channel re-spec (R-A1-M12, done 22
Aug; both selfchecks PASS post-rev-2) + commit of this file + the
tdmpc2_compat/tdmpc2_r3_train deltas + `probing/tm2_qmask.py` +
`uncfield/tm2_qmask_read.py` + `scripts/tm2_b5.sbatch` — then
apismoke → smoke → the 8 trainings.
