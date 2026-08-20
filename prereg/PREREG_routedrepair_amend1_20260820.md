# AMENDMENT 1 — PREREG_routedrepair_20260820 (20 Aug 2026, pre-outcome)

**Trigger**: registration-integrity review post-freeze (d2247e96); applied
pre-outcome, nothing submitted. Frozen body unedited; superseded
clause-by-clause.

## B13 — in-wave baseline control added (the review's central finding)
The pinned apt-unfrozen baseline (89.8984625) was computed on the LOST July
donors; the surviving fq1s1 donors are the 08-10 REGENERATION, and Midway3
is not job-to-job deterministic — so the pin and this wave's donors are
different objects. Fix: the wave adds an **8-cell in-wave apt-unfrozen
control** — plain full fine-tune adapts of the SAME regenerated fq1s1
seeds 1–8 donors (no refit, no perturbation). Wave becomes **24 cells**:
8 rgo-refits (+ their adapts), 8 shrink-and-perturb adapts, 8 apt-unfrozen
control adapts (~12 extra GPU-h).

## B12 + B13 — D1 redefined (paired, decidable)
**D1**: paired per-donor differences [repair_i − aptft_i], i = seeds 1–8,
`domains_read.one_sample` (exact 2^8 sign-flip + BCa). FIRE iff perm
p < .05 AND BCa CI lower bound > 0, two-sided reporting. The July pin
89.8984625 is DEMOTED to descriptive context (cross-era, disclosed); it is
no longer a decision input. The frozen body's "two_sample … one_sample"
contradiction is superseded by this paragraph.

## D2 clarified (unchanged bar, exact literal)
**D2**: one_sample of the repair adapts against the fixed constant
147.6823; kill bar = repair mean ≥ `SCRATCH − 25.0` **computed in the
reader** (= 122.6823; the frozen body's 122.68 literal was 0.0023 loose —
m1). Neighborhood criterion, not equivalence; the pin's own sampling error
(n=8, sd 33.22) is not propagated — anti-conservative, disclosed.
PROCEED iff D1 fires AND D2's bar is met (unchanged).

## B14 — the within-invocation gate scoped
"Within-invocation comparisons" applies to the in-wave panel (repair, S&P,
apt-control — all from this wave's invocations). D2 and the demoted July
pin are cross-era comparisons against committed constants, disclosed as
such. The frozen body's unscoped gate (which the wave's own rules violated)
is superseded.

## M14 — repair-vs-overwrite limitation registered
A 500k rgo re-fit on the same q1 side1 buffer approaches a fresh rgo fit;
no unfrozen rgo-from-random-init cell exists to separate "repaired the
sunk trunk" from "overwrote it". This wave cannot make that separation and
its branch strings say so: a PROCEED licenses the follow-up registrations
(short-refit dose curve; rgo-from-random-init control; matched-budget
comparison) — never the deployable salvage claim by itself. Also disclosed:
the ideation brief specified a SHORT refit on TARGET data; this wave
changes the length (to house 500k, for comparability — was disclosed) and
runs on the pretraining-side buffer (q1 side1), which the frozen body left
implicit.

## Donor generation declaration (added 20 Aug pre-commit, after the ops
## preflight found the fq1s1 dirs hold TWO checkpoint generations)

Ops finding: each fq1s1 donor dir contains both the July original apt fits
and the August REFIT_20260808 regeneration in the same ckpt tree, and
NEITHER generation is complete 8/8 at 500000 — under `latest_ckpt()`
(what axis1.sbatch actually resolves for AXIS1_INIT_WM) the August set has
seed5 truncated at 200000; under the `ckpt/latest` file the July set has
seed6 truncated at 200000. Left alone, one repair cell would have started
from a half-trained trunk.

**Decision (registered here, pre-outcome):**
1. **The donor generation is the August REFIT_20260808 regeneration** —
   the same declaration B13 already made (the July originals belong to the
   lost era the pin was demoted over) and the generation `latest_ckpt()`
   resolves natively. The July checkpoints are used NOWHERE in this wave.
2. **Option (b): complete the generation** — re-run the registered
   finger_refit recipe (PREREG_finger_refit_20260808 form) for fq1s1
   seed5 to a full 500000 updates (~5 h, one lane) BEFORE any wave job.
   No n=7 drop: the completion costs almost nothing, and the estimand is
   the within-donor paired contrast, which absorbs donor heterogeneity
   across seeds (the completed seed5 donor is a fresh invocation of the
   same recipe — same era-class as its 7 siblings, not bitwise their
   batch; disclosed, and immaterial to the within-donor pairing).
3. **Post-completion gate (ops attests in the witness `_meta`)**:
   `latest_ckpt()` must resolve to a 500000-update August-generation
   checkpoint for 8/8 donors (`_meta.donor_generation =
   "REFIT_20260808+seed5_completion_20260820"`,
   `_meta.latest_ckpt_500k_8of8 = true`); the reader's donor_sha keys are
   taken from exactly those resolved checkpoints, so any mixed-generation
   init is caught by the existing sha-pairing gate.
4. The prereg's "donors 8/8 present" inventory line is corrected to
   "8/8 present, 7/8 complete at freeze; seed5 completed pre-wave by the
   registered recipe."

**Ride-along commit**: this file + analysis/routedrepair_read.py +
STUDY_LEDGER.md.
