# AMENDMENT 1 — PREREG_mindose_reacher_20260820 (20 Aug 2026, pre-outcome)

**Trigger**: registration-integrity review post-freeze (d2247e96); applied
pre-outcome, nothing submitted, no curation run. Frozen body unedited;
superseded clause-by-clause.

## B10 — the curation draw is pinned (the walker lesson, completed;
## command corrected 20 Aug pre-commit after the ops preflight — the
## earlier draft's `--holdout 16` misread the flag's type)
The registered draw is the FIRST invocation of exactly:

```
python -m probing.build_controlled_replay search-dose \
  --index $RUNROOT/axis1_reacher/episodes.json \
  --ref_replay <REF> --levels 4 --beam 40 \
  --n_episodes 200 --n_candidates 400 --dirichlet 0.3 --seed 0 \
  --output $RUNROOT/axis1_reacher/dose.json
```

where `<REF>` is the SAME `--ref_replay` the registered axis1_reacher q1
search used (copied verbatim from that search's recorded provenance;
the literal path is recorded in the occupancy FILL below). Every other
knob runs at the subparser defaults (max_overlap 0.2, max_frames 3000,
knn 12, logc 1.0, cov_match_frac 0.05, occ_sep_mult 3.0, boot 100) —
pinned AS those defaults; any deviation is a refused wave. Then
`build-dose` materializes **level 1 of that 4-level search** (the
~0.13-occupancy analog rung).

**--holdout disposition (supersedes the earlier draft's `--holdout 16`)**:
the flag takes E4 probeset manifest PATHS, and **no reacher E4 probeset
exists** (verified 20 Aug: `$RUNROOT/e4_probesets/` holds cup/finger/synth
only) — there is nothing to hold out, so the flag is omitted, WITH the
obligation inverted and registered here: any FUTURE reacher probeset that
will be measured on this wave's fits must exclude the dose buffer's
episodes at probeset-build time (the D1-leak guard applied at the other
end). The buffer manifest records
`holdout_disposition: "no-probeset-exists"`; the reader accepts exactly
that value or `holdout_applied: true`, nothing else.

The first invocation's stdout and the resulting buffer manifest sha are
recorded here by dated edit BEFORE any fit. Re-invocation with different
sampling knobs is gate-shopping and is forbidden; if level 1 of this draw
misses the [0.08, 0.20] occupancy window, the wave REFUSES pre-outcome
(walker precedent) — full stop.

## B11 — decision rule made permutation-primary
PROCEED iff permutation p < .05 AND BCa CI lower bound > 0 (two_sample,
capdescent lineage). A significant negative (CI upper < 0) is
DROP-WITH-REVERSAL-RECORDED. Everything else is DROP. The frozen body's
CI-only clause is superseded.

## M12 — secondary labeled cross-protocol
The "dose arm vs W0-wave reacher task-arm cells" secondary compares
UNFROZEN dose adapts against FROZEN-readout W0 cells (verified:
adapt_ax1q1s1_reacher config has frozen_wm: true). It is cross-protocol
context, not a like-for-like level comparison, and is labeled so in the read.

## M13 — matched-budget scope on PROCEED
The dose arm receives 500k pretraining updates that scratch does not; the
comparison is unmatched-budget by design (conservative for DROP, liberal
for PROCEED). A PROCEED therefore licenses only a matched-budget follow-up
registration for the algorithm paper — never a recipe claim by itself. The
budget asymmetry is printed in the PROCEED branch string.

## m3–m6 — nits
(m3) "cleared Pocock by .0001" is correct (.0294 − .029297 = .000103); the
dose_ext RECORD's "by 0.0007" is the artifact-side error, noted there-not-
here. (m4) Interpretive reference for the occupancy window: reacher
full-occupancy side1 = 0.2795 (08-16 domains read), so [0.08, 0.20] sits
below the built side at roughly the finger level-1 fraction. (m5) The
materializing step is `build-dose`, level 1, as pinned above. (m6) The
frozen body's "within-invocation comparisons" gate has no referent for an
unpaired two-sample and is superseded by: both arms' cells must come from
THIS wave's invocations; no pre-existing rows enter the estimand.

**Ride-along commit**: this file + analysis/mindose_reacher_read.py +
STUDY_LEDGER.md.
