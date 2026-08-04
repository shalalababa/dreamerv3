# PREREG: breadth-causal wave — matched-f_R diversity pair (frozen 2026-08-03, post-review)

Registers the wave that adjudicates the frozen theory predictions
`PREREG_breadth_theory_20260803.md` (commit `481aced9`; P-BD1 causal
breadth effect, P-BD2 anomaly-attribution ordering, P-BD3 membership
dissociation, + registered failure modes), per that file's registered
ordering (theory freeze → feasibility → THIS wave registration →
build → gates → fits → ONE read). This file + `analysis/breadth_read.py`
(selfcheck PASS) + the amended `probing/curate_frew.py` (block-pr +
check-pairs additions, selfcheck PASS) are committed BEFORE any q1d
buffer, fit, or adaptation exists — only the value-blind feasibility
outputs exist. ONE adversarial reviewer ran pre-freeze (user
pre-approved); findings and fixes in the Review-provenance section.

## Feasibility outputs (pre-fit disclosables, known at this freeze)

Run on the RCC login node after the theory-freeze commit, exactly as
that file's ordering prescribes (both value-blind — pool spectra and
occupancies only). The commands, verbatim (constants registered here;
review MAJOR-3 fix):

```
python -m probing.curate_frew scan-moments \
  --index $RUNROOT/axis1_finger/episodes.json \
  --output $RUNROOT/axis1_finger/div_moments.npz
python -m probing.curate_frew search-div \
  --index $RUNROOT/axis1_finger/episodes.json \
  --moments $RUNROOT/axis1_finger/div_moments.npz \
  --n_episodes 200 --div_target 0.3232 \
  --output $RUNROOT/axis1_finger/div_pairs.json
# defaults registered: --div_tol 0.02, --div_corridor 0.10,
# --min_rew 10, --seed 0, --boot 200
```

- scan: 7424 episodes scanned, **288 unavailable** (3264 missing chunk
  files — the availability rule excluded them, so every candidate was
  buildable at scan time), feature dim 524.
- search (`div_pairs.json`): side0 (lo-div) occ **0.3038**, search
  diversity_pr **7.495**; side1 (hi-div) occ **0.3037**, search
  diversity_pr **12.336**; separation **4.841**; eligible-pool PR
  9.844; random-K baseline 9.689. The pair STRADDLES the frozen
  natural reference (7.495 < 9.2300 < 12.336); the cells' occupancy
  means match each other to 1e-4 and sit 0.0194 below the comparator's
  0.3232 — at the edge of the search tolerance (the repair pass stops
  at the first in-tolerance exchange, not at the optimum; disclosed,
  review m8), inside the registered f-band below.
- **The emitted `div_pairs.json` IS the wave** (review MAJOR-3): its
  copy in `artifacts/breadth_reg_20260803/` (with
  `div_moments.meta.json`) is committed WITH this registration, its
  sha256 recorded in `freeze_shas.txt` and enforced by the registered
  `check-pairs` preflight below. Never re-run `search-div` for this
  wave; a re-search after further purges selects a DIFFERENT wave and
  requires a dated amendment.

## Instrument-space decomposition (review BLOCKING-1 — load-bearing)

The frozen spectral instrument's obs_keys are [dist_to_target,
dyn/deter, position, target_position, touch, velocity] — 524 dims at
H=1, of which **512 (97.7%) are `dyn/deter`: the COLLECTOR's stored
recurrent latents**. These are not part of obs_space; the fitted WM
never encodes them (they enter training only as a 1-frame replay-carry
that `Replay.update()` overwrites); the repo's own probe-set builder
excludes `/`-keys as collection artifacts. A diversity_pr manipulation
could therefore in principle be pure collector-latent variety — a
near-relabeling of collector identity with little change to the
OBSERVABLE support the WM actually trains on. Consequences, registered:

- `curate_frew block-pr` decomposes any cell's PR into the full space,
  the **proprio block** (the 12 observable dims), and the latent block,
  from the existing moments cache (moments mode, pre-build) or from a
  built buffer's chunks (replay mode, instrument-grade).
- **Pre-freeze disclosure (ordering step 0 — EXECUTED, numbers frozen
  here):** moments-mode block-pr on the emitted pair and replay-mode on
  the frozen reference buffer, archived in the registration record:
  - pair: side0 pr_proprio **3.0359** / pr_latent 93.41 / latent trace
    share 0.380 / n_rew 60,776; side1 pr_proprio **3.7718** /
    pr_latent 110.25 / latent trace share 0.467 / n_rew 60,760.
    **Proprio-block separation 0.736 ≥ 0.5 — the observable-support
    pre-check PASSES**, and the n_rew match (Δ 0.03%) kills the
    PR-sample-size confound (review m7).
  - reference (q1v200/side1, replay mode): pr_full
    **9.230033291492585 — BIT-EXACT equal to the frozen instrument
    pin**, live confirmation on real data of the block-pr ≡
    spectral_measure equivalence the selfcheck proved synthetically;
    pr_proprio **3.2500**, pr_latent 64.85, latent trace share 0.430,
    n_rew 64,699.
  - Disclosed structure: the proprio ordering STRADDLES the reference
    exactly as the full index does (3.036 < 3.250 < 3.772) — the
    observable manipulation is directionally coherent at roughly half
    the full index's relative swing (±11% vs ±26% around the
    reference); the full-space separation is latent-majority by trace
    (latent shares 0.38–0.47), which is exactly why the gate exists
    and why the licensed vocabulary is "observable support" with the
    latent block reported alongside.
- **Observable-support gate (frozen; enforced at HALT and by the
  reader): proprio-block PR separation (hi − lo, replay-mode on the
  BUILT buffers) ≥ 0.5.** Below it, no "support breadth" statement
  about the WM's training distribution is licensed in either direction
  — fire or null — and the wave halts / the read refuses.
- The full-space index remains the registered manipulated quantity
  (comparability with the frozen reference and every prior curated
  cell); the proprio block is the licence for the vocabulary. Latent
  block PRs and trace shares are registered descriptives.

## Design: quad `q1d`, task arm, seeds 1–8 per side

Both cells from the FROZEN Axis-1 finger episode index (the same
collector pool as q1/q1v200/q1v400/q1f), materialized by the frozen
builder as quad `axis1_finger/q1d` from the COMMITTED `div_pairs.json`
(sha-enforced). Fits + adapts: task arm only, seeds 1–8 per side,
IDENTICAL protocol and config to the v200/v400/q1f cells (default
Axis-1 offline_fit config, 500k updates, frozen-readout adapt,
AUC100k). Run ids: WM `ax1wm_finger_bdq1ds{0,1}_seed{1..8}`; adapts
`adapt_ax1bdq1ds{0,1}_finger_seed{1..8}_ckpt500000`; AUC csv modes
`ax1bdq1ds0` / `ax1bdq1ds1`. Name derivation verified against
`submit_all.sh` line-by-line by the reviewer (no quad special-casing);
the family was glob-checked CLEAN against `runroot_cleanup.sh`
DELETE-SAFE and DELETE-AFTER (reviewer machine-verified).

Commands (in order; build + instrument on the login node, submitted
from a clean shell — the --export=ALL module-leakage gotcha):

```
# 0) preflight (review BLOCKING-2 / MAJOR-3): identity + availability
test ! -e $RUNROOT/axis1_finger/q1d   # refuse to build onto an existing quad
python -m probing.curate_frew check-pairs \
  --index $RUNROOT/axis1_finger/episodes.json \
  --pairs $RUNROOT/axis1_finger/div_pairs.json \
  --expect_sha 48032a7598ac36c7651db060c34cf5bd68ffe6b442848824aa32791f15184ba0
# check-pairs verifies: byte-identity with the committed search output,
# decision == OK, and that every member's covering chunks still exist
# (the purge re-check scan-time availability cannot give). Any failure
# => HALT, dated amendment. RECOVERY RULE (registered): if a build ever
# fails partway, delete $RUNROOT/axis1_finger/q1d ENTIRELY before the
# rebuild — the builder appends, never overwrites, and a doubled side
# passes every f/diversity gate undetected (duplication-invariant).
# 1) build + instrument
python -m probing.build_controlled_replay build \
  --index $RUNROOT/axis1_finger/episodes.json \
  --pairs $RUNROOT/axis1_finger/div_pairs.json \
  --which q1 --output_root $RUNROOT/axis1_finger/q1d
python -m probing.spectral_measure measure \
  --replay $RUNROOT/axis1_finger/q1d/side0 --domain finger --side lo \
  --tag q1d_side0 --output $RUNROOT/axis1_finger/q1d/spectral_side0.json
python -m probing.spectral_measure measure \
  --replay $RUNROOT/axis1_finger/q1d/side1 --domain finger --side hi \
  --tag q1d_side1 --output $RUNROOT/axis1_finger/q1d/spectral_side1.json
python -m probing.curate_frew block-pr \
  --index $RUNROOT/axis1_finger/episodes.json \
  --replay $RUNROOT/axis1_finger/q1d/side0 --side lo --tag q1d_side0 \
  --output $RUNROOT/axis1_finger/q1d/block_side0.json
python -m probing.curate_frew block-pr \
  --index $RUNROOT/axis1_finger/episodes.json \
  --replay $RUNROOT/axis1_finger/q1d/side1 --side hi --tag q1d_side1 \
  --output $RUNROOT/axis1_finger/q1d/block_side1.json
# 2) HALT RULE (binding, pre-fit): the wave halts and a dated amendment
# is required if ANY of the following fails on the BUILT buffers:
#   - both spectral jsons: version spectral_v1_1_20260724 AND
#     n_episodes == 200 (duplication witness, review B2)
#   - side0 AND side1 f_rewarded in [0.27, 0.34]
#   - |f_side1 - f_side0| <= 0.02
#   - diversity_pr(side1) - diversity_pr(side0) >= 3.0
#   - block jsons: |pr_full - spectral diversity_pr| <= 1e-6 per side
#     AND pr_proprio(side1) - pr_proprio(side0) >= 0.5
# (identical constants enforced again by the frozen reader). Then:
AXIS1_ID_PREFIX=ax1bd AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger \
  AXIS1_QUADS=q1d AXIS1_SEEDS="1 2 3 4 5 6 7 8" \
  ./scripts/submit_all.sh axis1-bundles
# 3) after fits complete (E4 membership rows, frozen machinery;
# HORIZONS left at its default "1 5 20" — h0 is emitted
# unconditionally by probing/stratified_error.py, passing 0 is invalid)
PROBESET=$RUNROOT/e4_probesets/finger_v1 GLOB='ax1wm_finger_bdq1d*' \
  COLLATE=$RUNROOT/e4_finger_v1_bd.csv \
  sbatch $REPO/scripts/e4_measure.sbatch
# 4) collation + bundle; the ONE read (executed once, on the bundle —
# all paths bundle-relative, review m4):
python -m analysis.adaptation_auc --runroot $RUNROOT --output <outdir>
python -m analysis.breadth_read \
  --auc <bundle>/analysis/auc/auc.csv \
  --auc_frozen local_results/volume_repl_20260729_204854/analysis/auc/auc.csv \
  --spectral_lo <bundle>/q1d/spectral_side0.json \
  --spectral_hi <bundle>/q1d/spectral_side1.json \
  --spectral_ref artifacts/volume_repl_20260729/diversity_q1v200_s1.json \
  --block_lo <bundle>/q1d/block_side0.json \
  --block_hi <bundle>/q1d/block_side1.json \
  --e4 <bundle>/e4/e4_finger_v1_bd.csv \
  --output artifacts/breadth_read_<date>/
```

## Validity gates (frozen; `breadth_read` enforces, all FATAL)

Unlike the f_R wave's side0, BOTH cells here carry the primary — there
is no never-fatal side:

- Both curated-side spectral jsons: version `spectral_v1_1_20260724`,
  **n_episodes == 200**, domain finger, sides lo/hi (swap-pinned).
- **f-band (legibility control): side0 AND side1 instrument
  `f_rewarded` ∈ [0.27, 0.34] AND |Δf| ≤ 0.02** — outside ⇒ the read
  REFUSES. Band note: the realized cells sit at ≈0.304, 0.0194 below
  the comparator's 0.3232; the primary (within-wave hi−lo) is exactly
  matched by construction; the P-BD2 anchor comparison carries the
  offset — from the archived grid's local slope (≈96 AUC per unit f
  between 0.054→161.95 and 0.3232→187.72) it is worth ≈ **1.9 AUC
  units downward on both cells** (review m6): it slightly eases the
  P-BD2 left leg and hardens the right leg. Occupancy drift precedent
  between search and instrument on this regime: ~1e-3.
- **Manipulation-validity: instrument diversity_pr(side1) −
  diversity_pr(side0) ≥ 3.0** (directional — a swapped pair reads
  negative and refuses). Search value 4.841; the moment↔instrument
  equivalence is proven at 1e-6 on a small synthetic and 1e-7 at
  dim-64/offset-50 scale (review m5 — the margin is expected headroom;
  the gate, not the proof, is what binds at full scale).
- **Observable-support: proprio-block PR separation ≥ 0.5** on the
  built buffers' block jsons, which must bind to the same buffers
  (|pr_full − diversity_pr| ≤ 1e-6 per side).
- All-or-nothing qc-passing 8-seed cohorts on BOTH cells; comparator
  rows exactly `ax1v2q1v200s1` seeds 1–12 from the ARCHIVED volume
  bundle; frozen v200s1 instrument record pinned bit-exact
  (f 0.32317182817182816, diversity_pr 9.230033291492585 — both pins
  tamper-tested).
- Rows filtered to domain=finger, milestone=500000.

## Registered read (frozen `analysis/breadth_read.py`)

- **PRIMARY (sole confirmatory, P-BD1): mean[`ax1bdq1ds1`] −
  mean[`ax1bdq1ds0`], AUC100k, two-sample seed-cluster bootstrap
  (B=10K, `default_rng(0)`, 2.5/97.5 percentiles — machinery pinned by
  golden values in the selfcheck); FIRES iff CI entirely > 0; REVERSED
  iff CI entirely < 0; else the registered INFORMATIVE NULL.**
- **P-BD2 (secondary, weak point-ordering, non-confirmatory):**
  mean[lo] < mean[v200s1] ≤ mean[hi] HOLDS/FAILS (strict left, ≤
  right; descriptive CIs rng 1/2, never decisional). **Registered
  reading rule (review MAJOR-6):** the anchor is a NATURAL pooled draw
  while both cells are CURATED draws; the f_R wave measured the
  curated-draw offset at +49.7 and +56.9 AUC (source_l1 = 1.0, roughly
  flat across very different f) — the left leg is therefore biased
  toward FAIL by roughly that offset. A FAILED ordering is
  UNINFORMATIVE about breadth (confounded with curation lift); a HELD
  ordering is stronger than its weak form suggests (the lo cell
  overcame the lift). The theory freeze's "converts direction into
  account" role is carried by a HOLD only; it cannot be carried by the
  legs' defaults.
- **P-BD3 (secondary, mechanism — SCOPE DISCLOSURE, review MAJOR-5):**
  BOTH cells' E4 h0 in-regime rew-NLL 8-seed cluster CIs (rng 3 = hi,
  rng 4 = lo); member iff CI entirely ≤ 1.5, nonmember iff entirely ≥
  2.0, else indeterminate. On this regime the finger reward IS the
  regime indicator (in-regime ⊂ rewarded; reference json:
  mean_reward = f_rewarded, s2_rewarded = 0), so the in-regime target
  is the constant 1 and a degenerate always-predict-reward model also
  reads member; empirically every finger task fit ever measured sits
  at 0.75–1.18, far from the 2.0 bar. **P-BD3 is therefore a
  GROSS-COLLAPSE SCREEN, not a discriminating instrument: its
  realistic role is to catch outright legibility collapse, and a
  "member" reading contributes little beyond that. Mediation
  attribution rests on the matched-f design (the manipulation never
  touched legibility inputs), not on P-BD3's member label.**
  `rew_nll_out` cell means are recorded as registered descriptives so
  a degenerate-low `rew_nll_in` is auditable.
- **Verdict map (frozen):**
  - FIRES + both cells member ⇒ **BREADTH-CAUSAL CONFIRMED** (support
    effect licensed — with the observable-support gate passed, "broader
    observable support transfers better"; P-BD3's role per the scope
    disclosure).
  - FIRES + lo-div nonmember ⇒ **INCLUSION-LEAK** (registered
    alternative; inclusion reading; pure-breadth mediation NOT
    licensed).
  - FIRES + HI-div nonmember ⇒ **HI-CELL LEGIBILITY ANOMALY**
    (registered disclosure clause, unpredicted direction; mediation
    not licensed; reported for follow-up) (review m2).
  - FIRES + any indeterminate ⇒ CONFIRMED, mediation unadjudicated.
  - REVERSED ⇒ λ-monotonicity form REFUTED (revision required).
  - Straddle ⇒ **INFORMATIVE NULL**: no breadth effect at an
    instrument-verified ≥3.0-PR (and ≥0.5 proprio-PR) separation ⇒
    the λ-input reading of the volume anomaly loses its principal
    support; the anomaly requires a non-breadth account (frozen
    consequence).
  - P-BD2 clause appended to every branch under its reading rule.
- Registered descriptives (never decisional): both cells vs v200s1
  CIs; `spectrum_pr` both sides; latent-block PRs + trace shares;
  `rew_nll_out` means; straddles_ref; per-cell seed maps.

## Power (disclosed)

8-vs-8 primary at the comparator's archived sd 77.18 → SE ≈ 38.6, MDE
(α=.05, 80% power) ≈ 108 AUC units (reviewer-verified arithmetic). The
wave is powered for effects of the volume anomaly's scale (+331 with
Δdiversity ≈ +1.4 at doubled volume; this pair realizes Δdiversity
4.84 at MATCHED volume), not for ~50-unit effects — a mid-size true
effect reads as the straddle branch, whose registered consequence is
stated above and accepted. Against-interest notes: the archived sd is
the pooled-12 comparator's; the curated cells' variances are unknown
and may be larger; no magnitude prediction is frozen anywhere.

## Source-composition rider (inherited, binding)

The diversity-sorted selection is expected to be collector/tier-skewed
(possibly source_l1 → 1, as every curated wave so far). Breadth and
composition-as-realized are INSEPARABLE in this design by
construction: a fired P-BD1 is reported as the effect of the curated
breadth draw (composition included), never as a pure diversity_pr
coefficient; mechanism attribution rests on the matched-f_R control
(and the P-BD3 screen). The emitted `div_pairs.json` mixture +
`source_l1` + per-side `n_rew` (the PR sample sizes — matched by the
occupancy match, ≈61k rewarded frames per side; review m7) and the
builder manifest's `mixture` + `source_l1` + confound deltas are
disclosed pre-fit quantities, copied into the wave artifact record
before any fit is submitted. Any composition-matched follow-up is a
NEW registration. Search asymmetry disclosed: the lo cell is grown
first from the full eligible pool, the hi cell from the remainder
(deterministic; re-running with other data selects a different wave —
see the sha-pinned identity rule).

## Registration record

`artifacts/breadth_reg_20260803/` holds RECORD.md + freeze_shas.txt
(this file, the theory freeze, `analysis/breadth_read.py`,
`probing/curate_frew.py`, **`div_pairs.json` + `div_moments.meta.json`
— copied from scratch BEFORE the freeze-commit**, review MAJOR-3) +
the pre-freeze block-pr disclosure outputs (moments-mode on the pair;
replay-mode on the reference buffer). The q1d builder manifest joins
the wave bundle as usual.

## Review-provenance (2026-08-03, ONE adversarial reviewer, pre-approved)

**2 BLOCKING / 4 MAJOR / 8 MINOR / 5 NIT — all B+M fixed and
machine-verified before this freeze.** Reviewer verified independently:
submit-script name derivation, cleanup-glob safety, builder/schema
compatibility, comparator csv integrity (12 rows, no duplicates, no
foreign modes), power arithmetic, no-subsample condition, rng
independence. Findings and dispositions:

- **B1 (dyn/deter dominance)**: diversity_pr's space is 97.7%
  collector latents the WM never models ⇒ block-pr decomposition
  built; observable-support gate (proprio-block sep ≥ 0.5) added to
  HALT + reader; pre-freeze block disclosure registered; latent
  descriptives recorded. Verdict vocabulary now says "observable
  support".
- **B2 (duplication-blind gates)**: builder appends on rebuild; all
  registered gates provably duplication-invariant ⇒ `test ! -e q1d` +
  `check-pairs` chunk preflight + `n_episodes == 200` HALT witness +
  registered recovery rule (delete the quad entirely; never
  re-search).
- **M3 (unpinned wave identity)**: `div_pairs.json` committed with the
  registration, sha in freeze_shas.txt, enforced by `check-pairs
  --expect_sha`; feasibility commands + constants registered verbatim.
- **M4 (vacuous f-band fixtures)**: both-sides-moved band fixtures +
  message-text assertions distinguishing every gate; mutation-verified
  (band-widening/bound-deletion mutants now CAUGHT).
- **M5 (P-BD3 non-discrimination)**: scope disclosure above;
  `rew_nll_out` descriptives added.
- **M6 (P-BD2 curation offset)**: reading rule above (+49.7/+56.9
  prior, leg biases, FAIL-uninformative/HOLD-strong).
- Minors: reversed-mirror + directional-separation + swapped-block +
  percentile-golden + cohort-cardinality + domain-filter + ref-f-pin
  fixtures added (12/12 mutants CAUGHT on mirror copies, clean-copy
  control PASS); hi-cell nonmember = registered disclosure clause;
  `cmd_build` OFF-TARGET input refused by check-pairs; read command
  bundle-relative; medium-scale exactness leg (dim-64, offset-50,
  1e-7); f-offset arithmetic (≈1.9 AUC) disclosed; n_rew disclosed;
  repair-edge behavior disclosed. NITs: drift wording reconciled to
  ~1e-3; straddles_ref moved to descriptives; growth-order asymmetry
  disclosed; clean-shell sbatch reminder; load_e4 skips non-integer
  suffixes (smoke ids) instead of crashing.

## Disclosure and ordering

- Known at this freeze: everything through 2026-08-03 including the
  high-f_R saturation read, the volume read, all published cell means
  and instrument values, the feasibility outputs quoted above, and the
  pre-freeze block-pr disclosure (recorded in the registration record
  when run). NO q1d buffer, fit, adaptation, E4 row, or transfer
  outcome exists.
- Ordering: pre-freeze block-pr disclosure + archive pairs/meta into
  the registration record → commit this file + reader + curate_frew +
  record (+ ledger) → preflight → build → spectral + block instruments
  → HALT gates → fits + adapts → E4 pass → ONE read
  (`analysis.breadth_read`, executed once, on the bundle). No
  unregistered statistic on real q1d data before the read; the frozen
  reader is the single consumer.

## Costs

16 offline fits + 16 adapts at the default Axis-1 config + one E4
sweep job; build/instrument are login-node CPU minutes; comparator
rows archived — no re-runs. Same envelope as the f_R wave; far under
the 25m cap regime.
