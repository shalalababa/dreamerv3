# PREREG AMENDMENT 1: pixel swamping — E4 measure repair + read freeze (2026-07-25)

Amends `PREREG_pixel_swamping_20260724.md`. Registered BEFORE any pixel
E4 number exists: the first E4 job crashed before writing any output.

## What happened

Job 52626590 (`PROBESET=fingerpx_v1 GLOB='ax1wm_finger_*pxpxq1m*'`)
crashed on the FIRST run (`ax1wm_finger_fpxpxq1ms0_seed1`) with
`KeyError: 'image'` inside the encoder (`dreamerv3/rssm.py:228`), before
any `summary.json`/`errors.npz` was written; `set -e` aborted the sweep
and the collate never ran (log in the 25-Jul bundle). Zero pixel E4
quantities exist — the amendment ordering guard holds by construction.

## Root cause (instrument defect, not a probe-set defect)

`probing/stratified_error.py cmd_measure` selected observation keys with
`len(space.shape) <= 1` — a proprio-era assumption baked into the E4
machinery. The obs dict fed to the encoder therefore lacked `image`,
which is the ONLY key a pixel WM's encoder consumes (`model_obs:
image`). The frozen `fingerpx_v1` probe set is NOT defective: its npz
carries `image` (120×1001×64×64×3 uint8) alongside the proprio regime
keys, exactly as registered. The probe set stays FROZEN and unchanged
(sha in its manifest).

Secondary latent defect fixed in the same pass: the decode-NLL loop
iterated over the same vector-key list, but a pixel WM's decoder
reconstructs `image` only — it would have KeyError'd next.

## The fix (probing/stratified_error.py, this freeze)

The model's own enc/dec spaces (already filtered by `model_obs` in
`dreamerv3/agent.py`) now decide everything:

- obs fed to the encoder = `enc.obs_space ∪ dec.obs_space` keys;
  image-shaped keys stay uint8 (the encoder asserts uint8 and applies
  its own /255 − 0.5), vector keys cast float32 as before.
- decode-NLL targets = `dec.obs_space` keys; image targets scored
  against f32/255, matching the agent's own training loss convention
  (`dreamerv3/agent.py` `isimage` branch).

Backward identity: for every proprio WM, `dec.obs_space` equals the old
vector-key list exactly (proprio envs have no 3-dim obs; the excluded
specials are already dropped in `agent.py` enc/dec space construction),
so all EXISTING proprio E4 numbers are unchanged — no re-measurement of
any proprio panel is triggered. `python -m probing.stratified_error
selfcheck` PASS after the change.

## Read instrument frozen in this same amendment

`analysis/pixel_swamping_read.py` (selfcheck PASS) — committed BEFORE
any pixel E4 number exists. It implements the registered rule exactly:
task-arm (`ax1wm_finger_pxpxq1ms*`) in-regime reward-NLL LEVEL at h0,
sides pooled, seed-clustered percentile bootstrap (B=10K, rng 0),
verdicts ≥2.0 / ≤1.5 / else UNRESOLVED. Two conventions the base
registration left open are pinned here, pre-outcome:

1. **Trivial-predictor floor** = cross-entropy of the best CONSTANT
   reward-head prediction: the marginal two-hot distribution of the
   probe set's own rewards on in-regime frames, under the agent's
   `symexp_twohot` 255-bin convention (`embodied/jax/heads.py`). The
   probe set is common to both sides, so the registered "per-side"
   floor is one number (disclosed). Floor ≥ 2.0 ⇒ UNINFORMATIVE.
2. **Apt-arm secondary is absent-by-design**: the frozen E4 machinery
   scores the reward head only when `expl.mode == task`
   (`reward_aware` gate), so the apt arm's (intrinsic-reward-trained,
   disclosed non-readout) head produces no true-reward NLL at all. The
   read asserts this gate still holds and records the secondary as
   absent-by-design rather than as a number.

Selfcheck coverage: planted swamping/membership/unresolved levels give
the right verdicts; a high-entropy reward set trips the floor guard; a
missing side row and an apt row carrying a reward NLL both trip.

## Execution addendum

Re-run the SAME registered command from a clean shell (the sweep is
resumable; no pixel run has a summary, so all 32 rescore under the
fixed code):

```
PROBESET=$RUNROOT/e4_probesets/fingerpx_v1 GLOB='ax1wm_finger_*pxpxq1m*' \
  sbatch scripts/e4_measure.sbatch
```

then ONE read execution:

```
python -m analysis.pixel_swamping_read \
  --e4_csv $RUNROOT/e4_fingerpx_v1.csv \
  --probeset $RUNROOT/e4_probesets/fingerpx_v1 --output <dir>
```

## Disclosure

Known at freeze: everything in the base registration's disclosure, plus
the crash traceback, the probe-set manifest (occ_frame 0.0726,
reward_frame_frac 0.0725), and the per-run E4 banner lines for the
first run (expl.mode/horizons — no scores). NOT known: any pixel
reward-NLL, any pixel decode-NLL, any strata statistic. The decision
thresholds, statistic, and consequence map are UNCHANGED from the base
registration.
