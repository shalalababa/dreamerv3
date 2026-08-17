# PREREG AMENDMENT 3 — SE gradient arm (level-vs-gradient), 16 Aug 2026

**Parent:** `PREREG_nfi_scale_exhibit_20260812.md` v2.2 + Amendments
A1/2. **Context:** M3 (Amendment 2 §2) returned NO-DIVERSION with a
complete post-hoc reversal (`artifacts/se_m3_read_20260816/`). The
leading candidate mechanism — registered there as a hypothesis — is
that the OU channel's disagreement is a state-independent PEDESTAL
inside the gate-open region: level, not gradient. This arm is the
direct test, targeted for the ICLR version of Paper A (internal
cutoff: read by ~5 Sep). User GO 16 Aug.

**Timing disclosure:** written after the Stage-1/M4/D-only/M3
outcomes. Pins are (a) parent-forced, (b) pre-Stage-1 smoke-derived,
or (c) indexing conventions; additionally this arm's DESIGN is
motivated by the M3 outcome (that is its registered purpose) but no
Stage-1/arm attribution or occupancy NUMBER enters any pin below
except the reuse of the already-registered threshold −0.13009691 and
the smoke range [−0.32203162, +0.19711795].

## 1. The registered question

Does the mispriced stochastic channel steer behavior when its
emission amplitude has a CONTINUOUS, followable spatial gradient —
or is the accounting/behavior dissociation robust even then?
- If diversion appears: the hazard propagates given spatial
  structure; M3's null is channel-geometry, and the boundary of the
  dissociation is mapped.
- If not: the dissociation deepens into an immunity result.
Both outcomes feed the Paper-A ICLR revision; neither re-adjudicates
any prior read.

## 2. Manipulation (pinned)

`embodied/envs/distractor.py` amplitude modulation (built 16 Aug,
strictly additive, default `mod_key=''` BITWISE-INERT — selfcheck:
modulation tracks m exactly over 3000 steps incl. both clip ends;
OU recursion and rng consumption mod-independent):
emission = OU_state × scale × refsd × m, with
**m = clip((position[0] − mod_lo)/(mod_hi − mod_lo), 0, 1)**,
`mod_lo = −0.32203162`, `mod_hi = +0.19711795` (the pre-Stage-1 smoke
replay's observed position[0] range). The modulation coordinate is
itself observed ⇒ the channel remains zero-marginal-information
(parent §3 argument). Under the linear ramp, the already-registered
threshold −0.13009691 (the smoke median) splits the space into a
low-amplitude region and a high-amplitude region. **Registered
delivered contrast, SMOKE-VISITATION-WEIGHTED (recomputed exactly,
review #24 B1 — the review corrected an earlier miscalculation):
mean m = 0.2512 below vs 0.4666 above (amplitude ratio 1.857×),
E[m²] = 0.0707 vs 0.2259 (disagreement-variance contrast 3.195×);
under the UNIFORM measure over the pinned interval the split means
are 0.185/0.685 (3.70× amplitude, 11.0× variance).** The realized
contrast under trained behavior is endogenous (diversion itself
raises visited m — that is the measured effect); the smoke-weighted
figures are the honest H0 anchor. A followable gradient exists
everywhere, with no boundary discontinuity (removing M3's
snap-transition confound). Overall smoke-weighted mean
**m̄ = 0.35891393** — the flat-comparator pin below.

## 3. Arms & seeds

- **hetero**: Stage-1 config + the mod quadruple
  (`DISTRACTOR_MOD_KEY=position DISTRACTOR_MOD_INDEX=0
  DISTRACTOR_MOD_LO=-0.32203162 DISTRACTOR_MOD_HI=0.19711795`),
  seeds **28–31**, RUN_IDs `se_het_s28..31`.
- **flat (amplitude-matched comparator — review #24 M2: without it,
  gradient and dose are confounded, since hetero also reduces mean
  amplitude ~2.8× vs Stage-1):** Stage-1 config +
  `--distractor.scale 0.35891393` (the smoke-weighted mean m — zero
  new code: a constant m is exactly a global scale). Same mean
  emitted amplitude as hetero under the H0 visitation measure, NO
  spatial gradient, and immune to the incentive (the agent cannot
  change its amplitude by moving). Seeds **36–39**, RUN_IDs
  `se_flat_s36..39`.
- **ungated2** (dose reference): Stage-1 config VERBATIM, seeds
  **32–35**, RUN_IDs `se_ungate2_s32..35`. Gives the pure-dose
  secondary (flat vs ungated2) and the cross-wave descriptive anchor
  vs the prior ungated arm (0.560; seeds 24–27 never enter any test).
- All three arms in the SAME wave on the SAME site (the Amendment-2
  arms ran matched on Midway3; cross-site arm comparisons barred by
  the 1-Aug nondeterminism rule). 12 runs × 5e5 steps; arms never
  seed-paired; seeds disjoint from {0, 10–17, 20–27}.
- Producer: the Stage-1 template with the four new
  `DISTRACTOR_MOD_*` env overrides (defaults = config defaults =
  m≡1, Stage-1-inert; same pattern as the Amendment-2 addendum) plus
  `DISTRACTOR_SCALE` (verification pass item 8: the launcher
  hardcoded scale 1.0, so the flat arm was inexpressible; override
  default 1.0 = the Stage-1 hardcoded value, inert).

## 4. Registered read (frozen reader `uncfield/se_grad_read.py`,
built + selfchecked BEFORE this arm's compute)

- **MANIPULATION-CHECK GATE (validity, adjudicated FIRST):** the
  misprice must survive the modulation for the behavioral read to be
  interpretable. Per hetero run, the frozen per-run instrument
  (`se_probe`, final ckpt, frozen defaults, all probe passes on one
  device class — the 16-Aug single-GPU rule) must show the distractor
  above its permutation null: **p_perm < .05 in ≥ 3 of 4 hetero
  runs**, with the primary reader's provenance/validity gates
  (S ≥ 256, calibration < 0.5, p-equality recompute, final-ckpt
  status surfaced in the aggregate). Registered honesty (review #24
  M4/m12): the "modulation removed the misprice" wording requires
  **≥ 3 VALID probes with the gate failing on their p-values**; if
  fewer than 3 probes are valid, the outcome is "manipulation check
  NOT-ADJUDICABLE (probe instrument)" with NO removal claim. The
  per-run dim-permutation p is the parent-§5-registered
  anti-conservative per-run evidence; it serves here as a VALIDITY
  gate on a channel that fired at the p=.001 floor in 8/8 Stage-1
  runs, not as a confirmatory test.
- **PRIMARY (directional; comparator = the amplitude-matched flat
  arm, review #24 M2):** occupancy(hetero) − occupancy(flat) in the
  high-amplitude region (position[0] > −0.13009691, the registered
  threshold; full-replay deterministic scan, ≥ 1e5-step floor — the
  Amendment-2 §2 estimand verbatim), exact one-sided label
  permutation over C(8,4) = 70, **fires iff p ≤ .05** (top 3 of 70;
  min attainable 1/70 ≈ .0143). The flat arm carries the same mean
  amplitude with no gradient, so a fire cannot be a pure dose
  effect.
- **SECONDARY (reporting):** hetero − ungated2 and flat − ungated2
  occupancy deltas (the dose axis), same permutation form,
  DESCRIPTIVE; task-return deltas (final-100 scores.jsonl mean),
  two-sided, reporting; per-arm occupancy BCa; hetero per-run
  distractor shares + θ₁ and (when probes exist) flat-arm shares —
  the amplitude-dose descriptive for the taxonomy seed; ungated2
  occupancy vs the prior ungated arm's 0.560 (cross-wave DESCRIPTIVE
  replication check only).
- **Outcome map:** (i) gate PASSES ∧ primary FIRES →
  "gradient-following: the misprice propagates to behavior when the
  channel has a followable spatial gradient; the M3 null is
  channel-geometry" (boundary-of-dissociation wording licensed;
  amplitude-matched comparator discharges the dose alternative).
  (ii) gate PASSES ∧ primary does not fire → "the dissociation is
  robust to a followable gradient at the registered delivered
  contrast (1.86× amplitude / 3.2× variance, smoke-weighted)" —
  immunity wording licensed AT THAT DOSE, not beyond; reversed-
  direction delta reported descriptively, post-hoc-labeled, exactly
  as in M3. (iii) gate FAILS with ≥ 3 valid probes →
  amplitude-dose finding (modulation removed the misprice), no
  diversion wording. (iv, validity cells — review #24 m14):
  fewer than 3 valid hetero probes → manipulation check
  NOT-ADJUDICABLE; any occupancy-floor failure or wrong arm counts →
  primary NOT-ADJUDICABLE (4+4 valid required). No other cells.
- Identity gates as in prior readers: arm identity from CONFIG —
  **all three arms**: `distractor.dim == 8`, `basesd == 1.215`, gate
  keys EMPTY on both wrappers (review #24 M3); hetero:
  `mod_key == 'position'` + pinned index/lo/hi to 1e-9,
  `scale == 1.0`; flat: mod empty, `scale == 0.35891393` to 1e-9;
  ungated2: mod empty, `scale == 1.0`. Train seeds exactly
  {28..31}/{36..39}/{32..35}, duplicates fatal; fit counters; ONE
  execution, this chat.

## 5. Bundle spec

hetero: `replay/`, `scores.jsonl`, `config.yaml`, `metrics.jsonl`,
`ckpt/latest` (+ listing), **`se_probe/`** (the manipulation-check
input; run AFTER training, final ckpt, frozen defaults, default
output, one device class across all probe passes). flat: same as
hetero INCLUDING `se_probe/` (amplitude-dose descriptive). ungated2:
same minus `se_probe/`. Manifest per results-sync v2.

## 6. What does NOT change

No frozen reader, executed read, or prior verdict is touched. The
Distractor modulation is verified default-inert for every existing
consumer (same argument chain as the Amendment-2 gate review #23:
merge-inherited defaults; wrapper skipped entirely at dim 0).
Freeze = review #24 adjudicated (1B/4M/9m ALL adopted, incl. the
flat-arm redesign for M2 and the corrected contrast arithmetic for
B1) + a delta verification pass + commit of: this file,
`embodied/envs/distractor.py`, `dreamerv3/configs.yaml`,
`scripts/uncfield_se.sbatch`, `uncfield/se_grad_read.py` — then the
12 submissions.
