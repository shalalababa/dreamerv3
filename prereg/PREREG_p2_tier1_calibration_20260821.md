# PREREG — Paper-2 Tier-1 CALIBRATION BUNDLE (21 Aug 2026)

**Status: FROZEN at user commit. ONE read execution of
`--component main` + ONE of `--component ab`** (two pre-named
invocations of one frozen reader, reported together). Motivation:
`Rescue_Ideation_ThreeAxes_20260821.md` — the three-lens convergence
("the program has no positive control", §0), items 2–5 of the Tier-1
slate plus the item-9 admissibility gate, registered as ONE bundle with
ONE reviewer per the board's explicit instruction (§3, §1.2:
register-before-compute — these are new estimands over the archived
`g_all_rep` arrays, and computing first would burn them the way the
substrate screen burned itself; none of the quantities below has been
computed). Reviewer: ONE Opus pass, user pre-approved, triggered at
build.

## Substrate (archived, committed, no new compute)

- `local_results/w1_labels_20260811_090727/w1_labels/` — the 64 W1
  cells (dv3 seeds 31–38, tm2 seeds 51–58; cup/finger × e1/e4) + 4 dup
  gates, loaded through the FROZEN `analysis/w1_read.py` machinery
  (`load_family` + `dup_gate` imported, not copied — every inherited
  gate applies byte-identically; the frozen reader itself is NOT edited
  and NOT re-executed).
- Published anchors (value-aware, DISCLOSED — these numbers are already
  read and are used only as identity gates and placement targets, never
  as new evidence): `artifacts/w1_read_20260811/{dv3,tm2}/w1_*.json` —
  P-W1a dv3 +0.0048 (p=.898) / tm2 +0.0617 (p=.016); P-W1b dv3 +0.0238
  / tm2 −0.301.
- **Identity gates (refusal = read not consumed)**: recomputed pooled
  P-W1a per family must equal the published point to ANCHOR_ATOL=1e-9
  (bit-exact in practice — the review verified the JSON round-trip);
  the published point must ALSO equal the disclosed constants
  (dv3 +0.004805 / tm2 +0.061719, |Δ| ≤ 5e-5 — review M11: relative
  self-consistency alone would admit a doctored labels+published
  pair); the labels dir basename must be the committed `w1_labels`;
  Stage A at S=0 must reproduce the archived estimator bit-exactly.
  Any mismatch refuses the whole bundle.
- **Governance hardening (review B3/B4/M13)**: the reader refuses
  under `python -O` (the inherited w1_read gates are asserts and would
  be silently stripped); the ONE-read guard is PATH-PINNED to
  `artifacts/p2_tier1_20260821` (a fresh --output cannot restart it);
  the output records full provenance (per-npz sha256, reader sha,
  w1_read sha, dirs, counts).

## Components and decision rules

Reader `analysis/p2_tier1_bundle_read.py` FROZEN with this file
(selfcheck PASS). Inference: cluster = cell (n=32/family),
permutation-primary + BCa (the 08-08 standing rule) at α=.05 via the
frozen `w1_read` stat machinery. All RNG pinned (CCD 20260821,
Stage A 20260822, CE 20260824/25).

### 1. CCD — consumer-competence dial (positive control at zero compute)

Synthetic choosers on the frozen arrays: sel_σ(s) = argmax_m
[even-half mean(s,m) + σ·ε], ε ~ N(0,1) pinned, evaluated on the
odd half against the mean-candidate baseline
h_σ(s) = ho(s, sel_σ) − mean_m ho(s,m); 64 draws/state; σ grid
{0, .25, .5, 1, 2, 4, 8} (raw G units) + uniform-random endpoint
(truth 0 by exchangeability). **σ=0 uses P-W1a's SELECTOR against
P-W1b's mean baseline; it is NOT P-W1a** (review M5 corrected the
ThreeAxes identity claim — the selector is identical, the baseline is
swapped, deliberately: the mean baseline makes the curve commensurable
with P-W1b harvest; any apparent divergence from the published P-W1a
null is baseline arithmetic, not selector competence).

- **CCD-PC-FIRES** iff h_{σ=0} fires (perm < .05 AND CI lower > 0), per
  family. Registered meaning: the chain CAN register harvest when a
  competent selector exists — the missing positive control is bought.
- **CCD-PC-FAILS**: even a perfect selector cannot harvest this
  substrate at n=32 — the substrate-empty reading is licensed for that
  family **only if Stage A's measured floor could have seen
  ceiling-level signal (mds80 in opportunity units ≤ 0.2026 — review
  M7)**; otherwise the verdict is **CCD-PC-UNDERPOWERED** (a power
  fact, licensing nothing about the substrate).
- Placement (descriptive): the published consumer harvest is placed on
  the curve — σ_equiv by linear interpolation across the grid AND the
  rand endpoint; labels BELOW-RANDOM / ON-CURVE / **ON-CURVE-TAIL**
  (between the σ=8 point and rand, where σ is unidentifiable — the
  upper grid collapses toward rand at the realized noise scale;
  review M6) / ABOVE-PERFECT; a curve-monotonicity flag is emitted
  (review m25). Expected from the published values: tm2 −0.301 lands
  BELOW-RANDOM (consistent with S3 anti-harvest; no re-adjudication of
  S3 happens here and none is licensed).

### 2. C7 — identification-gap reader

evsi(s) = ho(s, m_real) − ho(s, m_now) on the odd half (m_real is
selected under the probe mark, disjoint from every evaluation mark —
channel-A/B immunity verified at ThreeAxes §2.3; odd-half evaluation is
conservative and symmetric with P-W1a, disclosed). Measures the term
`FullRecord:1963` names as unmeasured.

- **PROBE-IDENTIFIES** (fires positive) / **PROBE-MISLEADS** (CI upper
  < 0 and perm < .05) / **PROBE-BLIND** (otherwise), per family.
- Identification gap = opp_split − evsi per cell, reported as a
  descriptive stat block (opp_split is published; disclosed).

### 3. Stage A — certified injection on REAL noise

Planted per-candidate values μ(s,m) ~ N(0, S²) added to the archived
arrays (constant across repeats = true candidate value), S grid
{0, .05, .1, .2, .5, 1, 2}, 50 plants/point (S=0: single identity
plant). RNG pins: plants [20260822, family], permutation
[20260822, 777] (review m24).

- **No recovery-bias estimand exists (review B1, adopted)**: the plant
  is additive and the estimator linear, so (opp′ − truth) is an
  algebraic IDENTITY equal to the archived noise-selection term — it
  measures nothing about plant recovery, and the originally drafted
  CALIBRATED/BIASED branch is WITHDRAWN pre-freeze (it would have been
  mechanically forced to BIASED by a healthy positive control).
  Registered Stage A outputs are: the zero-plant identity gate; the
  detection curve; and the descriptive selector-efficiency curve
  (recovered − base)/planted-ceiling.
- Detection curve: fraction of plants firing under the pinned
  vectorized sign-flip test (B=4096; perm-only fire criterion for the
  MC — DISCLOSED approximation; the read-grade conjunction rule is not
  MC-affordable and the perm leg is the binding leg in this program's
  history). **MDS80** = smallest grid S with rate ≥ 0.8, reported in
  BOTH sd units and **opportunity units** (mean[μ.max − μ(m_now)]
  ≈ 1.427·S at M=8 — review B2: the substrate ceiling +0.2026 is an
  opportunity, so the comparison must be in those units; the sd-grid
  point nearest the bar is S≈0.14, between grid points — resolution
  disclosed). **dv3-primary** (tm2's curve starts from an
  already-firing base — P-W1a tm2 fired — so its detection curve is
  descriptive only; disclosed). Size at S=0 is not separately
  estimable on the fixed realization (the S=0 detection indicator IS
  the archived fire status); the CE fixture's power at shift 0 serves
  as the machinery size witness (review m18).
- **Registered consequence (opportunity units)**: if dv3
  mds80_ceiling_units > 0.2026 (the corrected substrate ceiling from
  the archived screen — non-registered archived compute, cited as
  context not evidence), the sentence "the calibrated instrument could
  not have detected this substrate's own ceiling-level signal at n=32"
  is licensed in drafts; if ≤ 0.2026 the flatness verdict gains teeth.
  If detection never reaches 0.8 on the grid the verdict is
  DETECTION-FLOOR-ABOVE-GRID and the ceiling sentence is NOT licensed
  (review m17).

### 4. CE — harvest-side power calibration

Primary: realized-dispersion resampling — per-family per-cell harvest
values (their mean is published; using their spread is dispersion
around a known point, disclosed), centered residuals + shift grid
{0, .05, .1, .15, .2, .3, .5, .8}, 500 replicates/point (flip and
bootstrap matrices drawn fresh per 64-replicate chunk — review m20),
fire = vectorized perm < .05 AND percentile-bootstrap CI lower > 0
(B=2048; percentile-not-BCa DISCLOSED — power-curve resolution, not
read-grade inference). Output: **MDE80_harvest** per family, reported
WITH the realized cell-sd's bootstrap band (review M10: the curve
conditions on one realized sd whose relative s.e. is ~13% at n=32 —
no draft may quote MDE80 as a point fact without the band). Registered
consequence: every "power event" sentence about W2/P-W1b in any draft
must quote this measured number with its band (this can hurt the
draft; that is why it is credible). Secondary: the `cons_edge` sweep
through the frozen reader's own fixture generator (S=200, R=8, M=8;
fixture grid {0, .1, .2, .4, .8} × 200 MC — review m22 registered
here); fixture power at shift 0 is the machinery size witness.

### 5. AB — run-level purchase A/B admissibility gate (dispersion-only)

**PER DOMAIN** (review M12: a mixed cup+finger glob would make run_sd
a between-task sd and silently guarantee DEAD-ON-ARRIVAL, confirming
the board's prior on no evidence — the reader refuses any glob
spanning domains; ops supplies ONE glob per domain, comma-separated,
in the single registered invocation). Per domain, from ≥8 per-run
eval-score files: run-level mean over the last 20 episodes, run-level
sd (its ~27% relative s.e. at n=8 disclosed in the output),
MDE80(n/arm) = t_factor(n)·sd·√(2/n) for n ∈ {8, 16, 32} with t-based
factors {3.013, 2.896, 2.848} (review m21 — the normal 2.8 understates
at n=8).

- **AB-ADMISSIBLE** iff ANY domain has MDE80(16) ≤ 0.05 × |mean| (a
  purchase-policy effect below 5% of return is not worth a deployment
  wave — pinned here, before the dispersion is seen).
- **AB-DEAD-ON-ARRIVAL** otherwise ⇒ item 9 is retired without
  compute and recorded as such (the board's own suspicion, ThreeAxes
  §3: "with 84–95% floors this may be dead on arrival").

No estimand mean is computed by this leg (dispersion + a known-scale
mean of published-style eval scores only).

## What this bundle does NOT license

No re-adjudication of P-W1a/P-W1b/S3/S4; no pooling with any past or
future read; no new behavioral claim. It calibrates the instrument and
prices two future waves. The substrate screen stays non-registered
archived compute (ThreeAxes §1.1) and is cited only as context.

## Ops

Zero GPU. `--component main` runs locally on the committed label
bundle (CPU minutes); `--component ab` needs the dv3 R3 per-run
eval-score files (RCC; cup+finger confirmed present 21 Aug) — ops
resolves ONE glob PER DOMAIN and runs the leg there or ships the score
files back. Output path is CODE-PINNED to
`artifacts/p2_tier1_20260821/` (review B4); the reader writes
`p2t1_main.json` (with full substrate provenance — per-npz shas,
reader/module shas, review M13) + `p2t1_ab.json`; RECORD.md is
authored at read (review m23).
