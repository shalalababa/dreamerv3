# PREREG — Track-B4: aleatoric-normalized disagreement (the repair column), 22 Aug 2026

**Program:** Track B (user GO 22 Aug "do all the builds"; revised
same-day per review R3 BEFORE any compute — the first-draft repair
ratio was demonstrated to fire on three non-repairs; estimand,
gates, head wiring, and pins all revised pre-outcome). The
constructive column: does an aleatoric-aware intrinsic remove the
5.26× misprice without killing exploration?

**DESIGN NOTE (pre-registration toy, 22 Aug — a falsified first
design, disclosed):** the naive repair (members trained by
heteroscedastic NLL, reward = var-of-μ) is BACKWARDS — 1/σ²
gradient scaling FREEZES μ on noisy dims at init dispersion (toy:
noise/signal dispersion 0.53 → 6.32, WORSE). The registered design
trains **μ by the SAME MSE as Stage-1**, **σ by NLL against
stop-gradiented residuals**, and moves the repair into the DEPLOYED
REWARD: `var(μ) / (ensemble-mean predicted aleatoric variance)` per
dim. **R3-M1 revision:** the logvar head reads a STOP-GRADIENTED
copy of the member trunk — without that, the σ-NLL gradient reaches
the shared trunk and (via per-tensor AGC) rescales μ's updates, so
"μ training is det-identical" would hold in loss but not in
gradient path; with it, it holds by construction. Post-revision toy
(`b4_toy3`): σ learns the true noise floor (1.02 vs truth 1.0;
0.0025 on clean dims), normalized noise/clean contribution 0.00085
(repair ≈ 448×), μ dispersion gauss/det = 0.87. **Scope honesty
(R3-M2):** identical μ LOSS does not imply identical DATA — the
actor follows the normalized reward, so visitation/replay/WM can
drift from Stage-1; the RAW-MISPRICE-ABSENT cell covers exactly
that, and `expl/intr_rew` over training is a registered diagnostic
on both arms. `disag_scale` stays 1000 (checked: retnorm perc-limit
normalizes the advantage; Stage-1 telemetry shows the normalizer
active, so the ~O(1)-ratio-valued reward is absorbed). **Registered
constants (R3-M3):** `agent.expl.disag_logvar_min = -8.0`,
`disag_logvar_max = 6.0` — config-registered, BIND-CHECKed,
instrument-recorded, reader-pinned; the min floors alea at 3.35e-4
and therefore caps the normalization gain (~3000×). Known
limitation (R3-N2, disclosed): the hard clip has no restoring
gradient (one-way ratchet); the instrument records bound fractions
so it is observable. The instrument evaluates `_raw` twice
(predict + predict_logvar); verified deterministic (dup0 exact-zero
on both statistics in the smoke) — construction-level pairing
deferred (R3-N4, disclosed). **Consumer disclosure (R3-D8):**
`disag.reward` is now HEAD-DEPENDENT — any future gauss run
consumed by the planner/refine/region instruments optimizes or
measures the normalized objective, not the Stage-1 one; `predict()`
(μ block) is shape- and semantics-compatible everywhere (verified
against all five probe consumers).

## 0. FREEZE GATE — det se_mask baseline (R3-B5, zero GPU)

The manipulation check below rests on a quantity (`se_mask`-form
delta of the intrinsic under distractor permutation) that has NEVER
been measured on a full Stage-1 det run (the only measurements are
a 2e4-step smoke and the APT arm — both with POSITIVE deltas).
**Before this wave's freeze:** run `uncfield.se_mask` on the 8
existing Stage-1 det checkpoints (one CPU job on RCC, existing
data; descriptive instrument-calibration pass, not a registered
read; value-aware: it informs the manipulation check's FORM, and is
executed before any B4 data exist). If the det baseline does not
show a reliably negative distractor delta, the manipulation check
is re-specified (relative-share form) by dated amendment BEFORE
submission.

## 1. Machinery (built + selfchecked BEFORE compute)

- `dreamerv3/explore.py` Disag head='gauss' (separate sg-trunk
  logvar head), `agent.py`/`configs.yaml` plumbing (`disag_head`,
  `disag_logvar_min/max`), BIND-CHECK extended (R3-D4: gate
  indices/thresholds both wrappers, source_key, both basesds, seed,
  steps, disag_scale, the logvar constants — whose PRESENCE also
  proves the checkout carries the B4 config keys).
- `uncfield/se_alea_mask.py`: BOTH statistics (raw var-of-μ and
  deployed normalized reward) on identical masked variants, house
  suite + house RNG offsets; gauss diagnostics recorded (logvar
  bound fractions, alea level/spread, deter/stoch block
  decomposition of the reward — the mean-of-ratios statistic is
  dominated by low-alea dims, a re-weighting distinct from
  de-pricing, R3-N3). Selfcheck = mechanics gate on the local gauss
  smoke ckpt with a fabricated replay chunk (exercises the real
  model end-to-end; does NOT exercise multi-chunk replay, S=512, or
  realistic observation scales — stated per R3-N6).
- `scripts/se_b4_probe.sbatch` (R3-M4): ONE CPU job, all 4 runs,
  se_probe + se_alea_mask, ckpt-identity skip rule (never
  "json parses").

## 2. Wave (4 runs, seeds 120–123, one instance, one round)

Stage-1 config + `DISAG_HEAD=gauss`; the spec passes the FULL
variable set incl. inert values (R3-D5). 5e5 steps. `replay/`
required in the bundle. Submit: wave spec `ops/waves/b4_alea` after
freeze-commit; then `RUNS_ROOT=<runroot> sbatch
scripts/se_b4_probe.sbatch` post-training.

## 3. Registered read (frozen `uncfield/se_b4_read.py`, selfcheck
PASS incl. the three R3 adversarial cases; ONE execution)

- **Validity:** pins (gauss head, logvar bounds, Stage-1 rest,
  penalty inert), seeds 120–123, json==npz float64 provenance,
  dup0 exact-zero both statistics, S=512, BOTH instruments on the
  SAME checkpoint + newest-ckpt-dir cross-check (R3-M5), mask-json
  run-dir identity (R3-N5). Defective → excluded; **any outcome
  cell requires the full 4/4 loaded+valid (R3-D2)**. Fit counters
  reported; non-OK ANNOTATES the primary (disclosure-carried,
  R3-D3).
- **DEGENERACY** vs --stage1_runs: real leg = se_probe real level;
  intrinsic leg = the mask's `base_raw` (var-of-μ — the ONLY
  det-commensurable intrinsic quantity; the pse2 intrinsic under
  gauss is the normalized ratio, ~1000× off det scale — R3-B4).
  Collapse → **REPAIR-BY-DEATH**.
- **MANIPULATION CHECK:** delta_raw(distractor) < 0 AND its BCa
  upper bound < 0 (R3-D1) in 4/4, else **RAW-MISPRICE-ABSENT**
  (form contingent on §0's det baseline).
- **ALIVENESS GATE (R3-B3):** per run, the normalized statistic
  must have teeth — velocity-control delta_norm BCa excludes 0 AND
  sign-matches its raw delta (when the raw delta is itself
  distinguishable; direction two-sided per R3-D7 — the control's
  sign under permutation is not theoretically pinned), AND anchor
  dispersion cv(base_norm) ≥ 0.25 × cv(base_raw). Any failure →
  **REPAIR-BY-FLATTENING** (a flat reward zeroes deltas without
  repairing).
- **PRIMARY (scale-free, sign-constrained — R3-B1/B2):**
  r_raw = delta_raw/base_raw, r_norm = delta_norm/base_norm,
  **R_rel = r_norm/r_raw; REPAIRED iff 0 ≤ R_rel < 0.5 in 4/4.**
  A constant rescaling gives R_rel ≡ 1 (correctly no fire); any
  delta_norm > 0 → **OVER-CORRECTED** (sign inversion is not
  repair); otherwise **PARTIAL-REPAIR**. Per-run paired anchor
  bootstrap of R_rel reported. **Power note (R3-N8):** 4/4
  unanimity ≈ 1/16 null rate, one discordant run demotes with no
  gradation; all comparisons are within-run on identical variants,
  so the 1-Aug cross-invocation rule does not bind (why one
  instance/one round suffices).
- Secondaries (no fire): velocity rows, dup + resample rows, θ₁
  rows (Stage-1-band consistency, descriptive), gauss diagnostics,
  coverage + score tails (exploration cost), base levels.

## 4. Fences

Track-B taxonomy paper (the constructive column). The claim ceiling
is "this aleatoric normalization removes the misprice on this
substrate" — never "aleatoric heads fix disagreement exploration".

Freeze = review R3 adjudicated (done 22 Aug — 5 BLOCKING + 5 MAJOR
+ 8 moderate + 8 minor, all resolved or registered pre-freeze) +
**the §0 det baseline executed** + commit of this file + the
explore/agent/configs deltas + `uncfield/se_alea_mask.py` +
`uncfield/se_b4_read.py` + `scripts/se_b4_probe.sbatch` +
`scripts/uncfield_se.sbatch` + `ops/waves/b4_alea/spec.yaml` — then
the 4 submissions.
