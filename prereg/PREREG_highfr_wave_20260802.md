# PREREG: high-f_R falling-limb wave (frozen 2026-08-02)

Registers the wave that adjudicates the frozen theory predictions
`PREREG_highfr_theory_20260730.md` (P-HF1 falling limb, P-HF2
support-starvation dissociation, P-HF3 weak interior maximum, +
registered failure modes), per that file's registered ordering (theory
note → freeze → THIS wave registration → curated buffers). This file +
`probing/curate_frew.py` (selfcheck PASS) + `analysis/highfr_read.py`
(selfcheck PASS) are committed BEFORE any curated high-f_R buffer,
fit, or adaptation run exists.

## Design: curated quad `q1f`, task arm, seeds 1–8

Two curated 200-episode buffers from the FROZEN Axis-1 finger episode
index (`$RUNROOT/axis1_finger/episodes.json` — the same collector pool
every q1/q1v200/q1v400 buffer came from), materialized by the frozen
builder as quad `axis1_finger/q1f`:

- **side0**: target mean rewarded-regime occupancy **0.60**;
- **side1**: target **0.80** (the deep end of the freeze's
  f_R ∈ [0.60, 0.80] manipulation range; PRIMARY cell).

Selection rule (deterministic, `curate_frew search-frew`): episodes
sorted by occupancy descending; targets processed descending; each cell
= the contiguous K=200 window of the remaining pool whose mean is
closest to its target (ties → earlier window); selected episodes are
removed before the next target (sides disjoint by construction).
Registered tolerance ±0.05. Registered fallback: if no window reaches
0.80±0.05, side1 = the top-200 window iff its mean ≥ 0.60. Any other
infeasibility (either cell SHORT) ⇒ the wave HALTS before any fit and
a dated amendment is required — no outcome exists at that point.
Deliberately NO coverage matching: f_R itself is the manipulation;
support breadth is MEASURED afterward (P-HF2), not controlled.
Episode volume and episode-length distribution are matched to the
v200 comparator by construction (200 episodes at the index's modal
length; the frozen builder's chunk contract and confound checklist
apply unchanged).

Fits + adapts: task arm only, seeds 1–8 per side, IDENTICAL protocol
and config to the v200/v400 volume cells (default Axis-1 offline_fit
config, 500k updates, frozen-readout adapt 1.25e5 steps, AUC100k) —
the matched-fit-budget comparison the freeze requires. Apt-arm cells
are NOT run (the freeze names them optional controls only; three
reward-free nulls already anchor objective-specificity; task-only
keeps the wave at 16 fits + 16 adapts).

Commands (in order; curation + instrument on the login node):

```
test -f $RUNROOT/axis1_finger/episodes.json
python -m probing.curate_frew search-frew \
  --index $RUNROOT/axis1_finger/episodes.json \
  --n_episodes 200 --targets 0.60 0.80 --tol 0.05 \
  --output $RUNROOT/axis1_finger/frew_pairs.json
python -m probing.build_controlled_replay build \
  --index $RUNROOT/axis1_finger/episodes.json \
  --pairs $RUNROOT/axis1_finger/frew_pairs.json \
  --which q1 --output_root $RUNROOT/axis1_finger/q1f
python -m probing.spectral_measure measure \
  --replay $RUNROOT/axis1_finger/q1f/side0 --domain finger --side lo \
  --tag q1f_side0 --output $RUNROOT/axis1_finger/q1f/spectral_side0.json
python -m probing.spectral_measure measure \
  --replay $RUNROOT/axis1_finger/q1f/side1 --domain finger --side hi \
  --tag q1f_side1 --output $RUNROOT/axis1_finger/q1f/spectral_side1.json
# HALT RULE (binding): if side1's instrument f_rewarded falls outside
# [0.60, 0.85], DO NOT SUBMIT ANY FIT — the wave halts here and a
# dated amendment is required. side0 out of its range only excludes
# side0 (side1 + the primary proceed). Then:
AXIS1_ID_PREFIX=ax1hf AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger \
  AXIS1_QUADS=q1f AXIS1_SEEDS="1 2 3 4 5 6 7 8" \
  ./scripts/submit_all.sh axis1-bundles
# after fits complete (E4 membership rows, frozen machinery; HORIZONS
# deliberately left at its default "1 5 20" — horizon 0 is emitted
# unconditionally by probing/stratified_error.py, and passing 0 in
# HORIZONS is invalid there, reviewer-verified):
PROBESET=$RUNROOT/e4_probesets/finger_v1 GLOB='ax1wm_finger_hfq1f*' \
  COLLATE=$RUNROOT/e4_finger_v1_hf.csv \
  sbatch $REPO/scripts/e4_measure.sbatch
# collation + the ONE read (executed once, wherever all inputs exist):
python -m analysis.adaptation_auc --runroot $RUNROOT --output <outdir>
python -m analysis.highfr_read --auc <outdir>/auc.csv \
  --auc_frozen local_results/volume_repl_20260729_204854/analysis/auc/auc.csv \
  --spectral_lo <q1f>/spectral_side0.json \
  --spectral_hi <q1f>/spectral_side1.json \
  --spectral_ref artifacts/volume_repl_20260729/diversity_q1v200_s1.json \
  --e4 e4_finger_v1_hf.csv --output artifacts/highfr_<date>/
```

Run ids: `adapt_ax1hfq1fs{0,1}_finger_seed{1..8}_ckpt500000`; WM runs
`ax1wm_finger_hfq1fs{0,1}_seed{1..8}`; AUC csv modes `ax1hfq1fs0` /
`ax1hfq1fs1`.

## Validity gates (frozen; `highfr_read` enforces)

- Both curated-side spectral jsons: version `spectral_v1_1_20260724`,
  200 episodes.
- **side1 instrument `f_rewarded` ∈ [0.60, 0.85]** — outside ⇒ the
  read REFUSES (invalid wave; halt happened pre-fit or a dated
  amendment is required). side1 is the primary cell precisely so the
  fallback branch cannot silently weaken the manipulation below the
  freeze's range.
- side0 `f_rewarded` ∈ [0.55, 0.65] AND side1−side0 ≥ 0.08 — else
  side0 is EXCLUDED (disclosed; never fatal; primary unaffected).
  **The same never-fatal rule covers side0's data legs**: an
  incomplete side0 8-seed cohort or a qc-failed side0 row also
  EXCLUDES side0 (with the reason recorded) rather than aborting the
  read — only side1 and the comparators are load-bearing. Stated
  plainly: a side0 realizing f ∈ [0.55, 0.60) sits below the freeze's
  manipulation range yet still enters the monotone chain and the
  P-HF3 grid (it is a grid point, not a manipulation-range claim).
- All-or-nothing 8-seed cohorts + qc_pass on side1 and both
  comparators (fatal if violated); rows filtered to
  domain=finger, milestone=500000.
- Comparator rows from the ARCHIVED volume bundle
  (`local_results/volume_repl_20260729_204854/analysis/auc/auc.csv`,
  read 2026-07-29): `ax1v2q1v200s1` exactly seeds 1–12,
  `ax1v2q1v200s0` exactly seeds 7–12. Frozen v200s1 instrument record
  pinned bit-exact from
  `artifacts/volume_repl_20260729/diversity_q1v200_s1.json`
  (f_R 0.32317182817182816, diversity_pr 9.230033291492585).

## Registered read (frozen `analysis/highfr_read.py`)

- **PRIMARY (sole confirmatory, P-HF1): mean[`ax1hfq1fs1`] −
  mean[`ax1v2q1v200s1`], AUC100k, two-sample seed-cluster bootstrap
  (B=10K, `default_rng(0)`); FIRES iff CI entirely < 0 ⇒ the falling
  limb is real.** The freeze's "per-seed [B_hi-f − B_q1s1]" is
  operationalized as this cell-mean contrast with seed-cluster
  bootstrap — the `volume_repl` operationalization on the same
  substrate; n differs (8 vs 12) and independent jobs are not
  job-to-job deterministic (1-Aug finding), so no per-seed pairing is
  licensed.
- **Monotone-rise refutation branch (frozen failure mode)**: cell
  means nondecreasing in f across {v200s1, side0 (if included),
  side1} AND the primary CI entirely > 0 ⇒ the breadth-patch
  trade-off form (`Theory_CompCapacity_Addendum_20260724.tex` §4) is
  REFUTED — revision required, not annotation.
- **Saturation branch (frozen)**: primary CI straddles 0 AND
  diversity_pr(side1) ≥ 9.230033291492585 ⇒ saturation reading —
  compatible, uninformative, reported as such.
- **P-HF2 (registered secondary)**: side1 task fits' in-regime
  reward-NLL at horizon 0 (frozen E4 machinery, `finger_v1` probe
  set), 8-seed cluster bootstrap CI (`default_rng(2)`): membership
  intact iff CI entirely ≤ 1.5 nats; inclusion-effect alternative iff
  CI entirely ≥ 2.0 nats (the registered pixel-swamping constants —
  same task, same labels, same head class, proprio anchors 1.02/1.21
  member vs 2.03–2.31 non-member); between ⇒ indeterminate,
  dissociation not adjudicated. Dissociation HOLDS iff membership
  intact AND diversity_pr(side1) < 9.230033291492585 (bare direction,
  `volume_repl` P-E4b precedent). Scope notes: the constants are
  imported WITHOUT pixel-swamping's trivial-predictor-floor valve —
  unnecessary here because the cells share the anchors' modality,
  task, labels, and head class (the floor guard existed for the pixel
  modality change); and `spectrum_pr` (whole-buffer, off-regime
  inclusive — the index already known to move OPPOSITE to
  diversity_pr on the volume pair) is recorded from the same jsons as
  a registered DESCRIPTIVE, so the mechanism claim has its natural
  counterpart on the record; diversity_pr remains the sole decisional
  index (the freeze names it).
- **P-HF3 (registered secondary, weak form)**: over the
  matched-200-episode grid {v200s0 (f=.054, n=6 fresh), v200s1
  (f=.3232, n=12), side0 (if included), side1}, the maximum cell-mean
  AUC100k is NOT at the highest-f cell. Disclosed: with the frozen
  cells' means known (v200s0 161.95, v200s1 187.72), this weak form
  carries almost no information beyond the sign of the new cells'
  contrasts — any negative primary point estimate already implies it;
  it is reported as the freeze's registered form, not as independent
  evidence. Also disclosed: a monotone-DECREASING curve satisfies the
  weak form (the freeze's own gloss is "not at the highest-f cell",
  not a two-sided interior test), and when side0 is excluded the
  monotone chain degenerates to two points — a 2-point nondecreasing
  chain still counts toward the refutation branch.
- Registered descriptives (never decisional): side0−v200s1 contrast
  (`default_rng(1)`); side0 E4 rows (`default_rng(3)`); per-cell seed
  maps; `spectrum_pr` from the curated-side jsons.

## Power (disclosed; stated against the registered pooled-12 cohort)

All three candidate comparator cohorts' statistics are archived and
known at registration: orig 1–6 mean 204.28 sd 90.61; fresh 7–12 mean
171.15; pooled-12 mean 187.72 sd 77.18. The registered comparator is
POOLED-12 — the against-interest choice (its mean is lower than the
orig batch's, so the predicted negative contrast is SMALLER and
harder to fire). MDE for the 8-vs-12 contrast ≈ 69 at sd 77.18 (≈ 81
at the conservative sd 91). The predicted falling limb (side1
dropping toward reward-free levels ~80–100 from 187.72) is a −88 to
−108 effect — adequately powered. Smaller drops read as a straddle;
the saturation branch then adjudicates informativeness.

## Consequence map (frozen; inherits the theory freeze)

- P-HF1 fires + P-HF2 dissociation ⇒ the breadth-patch trade-off is
  supported end-to-end: high f_R starves support while inclusion
  stays intact; the flagship's support-breadth axis gains its falling
  side; the interior-maximum claim (weak) stands per P-HF3.
- P-HF1 fires + inclusion reading (trunk leaves the band) ⇒ the
  deficit is an inclusion effect; the breadth trade-off account gains
  NO support from this wave (freeze's registered alternative).
- Monotone rise ⇒ §4 trade-off form REFUTED; revision required.
- Straddle + no diversity separation ⇒ saturation; uninformative.
- **Source-composition rider (registered reporting rule):** the
  occupancy-sorted window is expected to be collector-skewed —
  possibly maximally (source_l1 ≈ 1 vs the comparator), the known
  E3v2-style entanglement of pooled searches. f_R and
  collector-composition-as-realized are therefore INSEPARABLE in this
  design by construction: a fired P-HF1 is reported as the effect of
  the curated high-f_R draw (composition included), never as a pure
  f_R coefficient; mechanism attribution rests on P-HF2, and any
  composition-controlled follow-up (e.g. a `search-matched`-style
  per-collector-count-matched curation) is a NEW registration. The
  builder's `q1f/manifest.json` `mixture` + `source_l1` (and the
  comparator's from `q1v200/manifest.json`) are DISCLOSED pre-fit
  quantities, to be copied into the wave artifact record before any
  fit is submitted.

## Disclosure and ordering

- Known at the theory freeze (30 Jul): all reads through 2026-07-30.
  Known additionally at THIS registration (2 Aug): U1
  (protocol-robust), the 25m confirmatory read + E4 counterpart, the
  doubling read, the second-k retry. None of these measures or
  manipulates f_R. NO curated high-f_R buffer, fit, or adaptation
  exists; no transfer outcome of any curated cell exists.
- The comparator cell means are PUBLISHED (volume_repl read,
  2026-07-29) and therefore known: pooled-12 187.72 (orig 204.28,
  fresh 171.15), v200s0 fresh 161.95. This is disclosed; it cannot
  bias the wave because the manipulanda (window rule, targets, gates,
  seeds, config) are frozen here before any hi-f outcome exists, and
  the primary's unknown side is entirely the new cells.
- **The P-HF2 comparator level is likewise already KNOWN**, not
  contingent: the v200s1 task fits' E4 h0 in-regime rew-NLL rows are
  archived (`artifacts/e4_amend1_optionc_20260723/e4_finger_v1_optionc_volume.csv`,
  seeds 1–6: 0.746/0.813/0.790/1.139/0.729/0.746, mean 0.827) — deep
  inside the 1.5-nat membership band, which is what makes the P-HF2
  threshold a real bar for side1 rather than a formality.
- Grid-coordinate correction: the session TODO paraphrase listed
  f=.094 (v400s1) as the P-HF3 low point; the freeze's own "matched
  volume" clause excludes 400-episode cells — the registered low grid
  point is v200s0 (f=.054, 200 episodes). The freeze governs.
- Curation + spectral measurement run on the login node any time
  after commit (buffer-only; they reveal f_rewarded and diversity_pr
  — registered gate/mechanism quantities — and no transfer outcome;
  same post-commit allowance volume_repl registered).
- Ordering: commit this file + `probing/curate_frew.py` +
  `analysis/highfr_read.py` → curation + build + spectral measure →
  gates → fits + adapts → E4 pass → ONE read
  (`analysis.highfr_read`, executed once, wherever both AUC csvs +
  jsons are present). No unregistered statistic on real hi-f data
  before the read; the frozen reader is the single consumer.

## Costs

16 offline fits + 16 adapts at the default (small) Axis-1 config +
one E4 sweep job; curation/instrument are login-node CPU minutes; all
comparator rows are archived — no re-runs. Days, not weeks; far under
the 25m cap regime.
