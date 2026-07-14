# E4 registration — regime- and reward-stratified WM prediction error (frozen 2026-07-14)

Registers the E4 estimator, probe-set construction, and descriptive
signatures **before any E4 measurement exists** (no `e4_*` output directory
exists under any runroot or snapshot at freeze). Supersedes the E4 sketch
in the post-5a addendum (2026-07-07) where they differ; plan-v4 rules
stand (fixed common probe sets; "capacity-allocation signature", never
mediation).

## Status and role

E4 is **descriptive mechanism evidence only**. No confirmatory or
secondary contrast is registered here; no decision rule in any frozen
protocol consumes an E4 number. It exists to characterize *how*
reward-capable objectives spend model capacity differently on the same
data — the mechanism behind the confirmed occupancy × reward-supervision
interaction (W0 read, 2026-07-13).

## Probe sets (one per domain, frozen before first measurement)

- Built by `probing/stratified_error.py build-probeset`; whole episodes of
  the modal length; deterministic sampling (`np.random.default_rng(seed)`,
  seed 0); sha256-frozen manifest (marker = file digest, loader verifies).
- **Sources: the Gate-0 pilot replays** (`pilot_goal_<dom>`,
  `pilot_random_<dom>`, `pilot_p2e_<dom>`), 20 episodes each (60 per
  domain) — held out by construction from every Axis-1/E3v2 buffer (those
  are composed exclusively from Phase-4 pretrain replays). Goal supplies
  in-regime/reward-dense frames; random/p2e supply the out-of-regime bulk.
- Regime thresholds: the frozen `probing/regimes.py` spec values.
- Canonical IDs: `$RUNROOT/e4_probesets/finger_v1`, `.../cup_v1`.

## Estimator (frozen)

Per WM checkpoint (final fit checkpoint of each `ax1wm_*` run; inference
only, no training):

- **Error:** per-frame decoder negative log-likelihood summed over proprio
  observation keys, target-aligned.
  - Horizon 0: filtering-posterior reconstruction NLL at t.
  - Horizon k ∈ {1, 5, 20}: NLL of the k-step open-loop prior anchored at
    t−k (rolled with logged actions, model-sampled stochastic states)
    against obs at t; first k frames of each episode undefined (NaN,
    excluded from means and counts).
- **Strata of the target frame:** in/out-of-regime; reward-bearing
  (reward > 0) or not; their 2×2 cross; per probe source. Per-key NLL by
  regime/reward strata as a supplementary panel.
- **Reward-head NLL** per horizon for reward-aware fits only (the head is
  untrained under `reward_free`; recorded null there).
- Implementation: `probing/stratified_error.py` (`selfcheck` covers
  alignment and stratification exactly; measure path validated end-to-end
  on a real apt WM checkpoint, 2026-07-14). Sweep:
  `submit_all.sh e4-measure` → `scripts/e4_measure.sbatch` (resumable,
  per-run skip on existing summary).

## Descriptive signatures stated in advance

On the Axis-1 2×2 checkpoint population (task `ax1wm_<dom>_q*` vs apt
`ax1wm_<dom>_fq*`, both sides, seeds 1–8/9–16 where fitted):

1. Fit-buffer occupancy ↑ ⇒ (err_out − err_in) ↑ within an arm (post-5a
   addendum signature, unchanged).
2. Finger Q1: the high-occ side shows lower err_in / higher err_out than
   the low-occ side (capacity re-allocation with buffer composition).
3. **Interaction mechanism (new, from the W0/battery read):** on the SAME
   buffer, task-arm fits show lower error on reward-bearing frames
   relative to apt-arm fits (and a larger reward-head advantage on the
   high-occ side); the effect tracks the reward stratum, not the regime
   stratum, where the two dissociate (cup — battery occ↔reward corr
   −0.5/−0.7) and coincide in finger (corr +1.0).
4. No signature is expected to reverse across horizons; horizon panels are
   robustness views.

Failure of any signature is reportable as-is (descriptive); it cannot
gate or un-gate any registered decision.

## Ordering statement

At freeze: no E4 probe set has been built on the cluster; no
`e4_*/summary.json` exists in any snapshot; the only E4 execution to date
is the local validation run on `ax1wm_finger_fq1s0_seed9` against a
2-episode throwaway probe set built from that run's *adapt* replay
(scratchpad only, not a registered measurement, discarded).
