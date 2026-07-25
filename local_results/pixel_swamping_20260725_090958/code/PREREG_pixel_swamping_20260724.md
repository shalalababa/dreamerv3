# PREREG: pixel reconstruction-swamping diagnostic (frozen 2026-07-24)

The registered next step of the X2 decision tree
(`PREREG_pixel_repl_20260719.md` §5: "does not fire ⇒ ... the
reconstruction-swamping diagnostic ..., which gets its own
registration before it is read"). X2 read landed 24 Jul
(`artifacts/pixel_x2_20260724/`): primary +3.6 [−29.9,+40.9], both
simples null, all four cells on a common AUC floor. This file is
committed BEFORE any pixel E4 pass or pixel probe set exists.

## Question

Did the pixel task arm's reward gradients LOSE the subspace
competition (swamping — the bar sits above G2, no cell includes θ;
theory-continuous), or did the task trunk include θ without
behavioral transfer (membership-without-transfer — the
membership→transfer link itself fails in pixels; a genuine theory
problem)? The task arm's own reward head discriminates: it trained
on TRUE reward with trunk gradients (`reward_grad: true`, audited),
so its achievable in-regime reward-NLL is bounded by what the trunk
retained.

Disclosure on the apt arm: its reward head trains on the APT
INTRINSIC reward (expl.mode apt), so its true-reward NLL is NOT a
membership readout — reported descriptively only, never decisional.

## Instrument

1. **Pixel probe set** `fingerpx_v1`: `probing.stratified_error
   build-probeset` over the image-bearing X1 index sources
   (`$RUNROOT/pixel_x0/episodes_x1.json` universe), same regime
   thresholds as the frozen proprio `finger_v1` set (proprio keys
   retained in these episodes label regimes; images feed the
   encoder). FROZEN marker + sha manifest before any scoring. Exact
   build flags = the finger_v1 invocation with the X1 sources
   [FILL: recorded in the probe-set manifest at build time, before
   any fit is scored].
2. **Pass**: `PROBESET=$RUNROOT/e4_probesets/fingerpx_v1
   GLOB='ax1wm_finger_*pxpxq1m*' sbatch scripts/e4_measure.sbatch`
   (existing frozen E4 machinery; horizons 1 5 20 default; scores
   all 32 pixel fits at final checkpoints, inference-only).

## Registered read (decision rule frozen now)

Statistic: task-arm in-regime reward-NLL LEVEL at h0, pooled over
sides, seed-clustered (n=8 per side; B=10K bootstrap rng 0). Proprio
anchors on the same task + same reward labels + same head class
(frozen record, `artifacts/e4_amend1_optionc_20260723/` and the
18-Jul panels): membership band full 1.02 / rgo 1.21; non-membership
band sh 2.03 / sgb 2.30 / vgo 2.31.

- **P-SW1 SWAMPING-CONSISTENT** iff the task-arm pixel level's 95%
  CI lies entirely ≥ 2.0 nats (at/above the non-membership band's
  lower edge): reward information is absent from the pixel trunk —
  the all-cells-floor X2 pattern is the g > G2 regime; the theory's
  account of the pixel boundary is mechanism-complete.
- **MEMBERSHIP-form** iff the CI lies entirely ≤ 1.5 nats: the task
  trunk carries reward information while behavioral transfer is
  null — registered as a membership→transfer boundary (theory
  problem, reported as such).
- Otherwise **UNRESOLVED** (reported; no further registered claim
  from this pass).

Guard (uninformative outcome): the read also computes the probe
set's trivial-predictor floor (per-side-mean-reward NLL). If that
floor itself is ≥ 2.0 nats, the swamping threshold is unreachable by
construction and the read is declared UNINFORMATIVE (reported, no
fire) — the thresholds are cross-modality conventions and this guard
is the honesty valve. Registered secondaries (descriptive):
d_errin (in-vs-out dynamics error) per arm — expected arm-invariant
as in proprio (coverage learned, membership contested); apt-arm
true-reward NLL (disclosed non-readout); per-side splits.

## Consequences (frozen)

- SWAMPING-CONSISTENT ⇒ Paper 1's pixel boundary paragraph is
  written as predicted structure (the bar framework's high-g regime;
  cites P-E1's ordering law extended upward); no further pixel arms
  pre-deadline (G-X3 stays NO-GO).
- MEMBERSHIP-form ⇒ the pixel section reports a genuine boundary of
  the membership→transfer link; the theory's honesty box gains this
  as an open failure; any follow-up (readout-budget ladder on pixel
  fits) would need its own registration and is NOT authorized here.
- UNRESOLVED/UNINFORMATIVE ⇒ boundary reported without mechanism
  attribution; no further pixel compute pre-deadline.

## Disclosure

Known at freeze: everything through the X2 read (all cell AUCs, the
null, audit). NOT known: any pixel reward-NLL, any pixel probe-set
statistic, any pixel E4 number. The 2.0/1.5 thresholds are
registered conventions anchored to frozen proprio levels on the same
task and labels; head/probe-set differences across modalities are
acknowledged and partially covered by the trivial-floor guard.
