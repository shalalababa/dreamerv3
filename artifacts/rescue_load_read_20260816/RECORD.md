# Rescue-under-load — registered ONE read (16 Aug 2026)

**Registration:** `prereg/PREREG_rescue_load_20260815.md` (freeze commit
9d8f50ec, 2026-08-15 15:28 — before the wave ran 16 Aug). Reader
`analysis/rescue_load_read.py` sha256 47ac18bd… (committed same freeze,
working tree clean at read time); prereg sha256 6bdb9e8d…. Selfcheck
PASS immediately before execution. ONE execution, this artifact.

**Bundle:** `local_results/ruw_20260816_142429` — 3650 files, sha256
manifest 0 failures. Fits+adapts on Vast (dv3ops wave rescue_load),
COMPLETION 32/32 DONE, every adapt scores=112.

## VERDICT — RESCUE-CONFIRMED

w_r = 100 restores task reward-legibility under σ=2 load. Registered
paired primary (per-(side,seed) deltas AUROC(w100) − AUROC(w1), n=8,
exact sign-flip + BCa):

- **mean delta +0.2144 [BCa +0.1389, +0.2639], perm p = .0078**
  (8/8 pairs positive — the exact-test floor at n=8), d_z = 2.27.
- Panel means: w1 0.7542 (s0 0.7091 / s1 0.7992) → w100 **0.9686**
  (s0 0.9786 / s1 0.9586).
- Realized paired sd 0.0943 → MDE80 0.113; the observed effect exceeds
  the registered FULL-rescue reference (+0.16): w100 lands at/above the
  unloaded σ=1 anchor level (ceiling censoring note in prereg — 0.954
  anchor; w100 exceeds it).

**Registered consequence:** the s×w_r inertness is reconciled as
saturation-off-boundary — the loss-weight lever DOES reach the
spectrum at the inclusion boundary, and β (the a²-term weight) is an
experimentally manipulable parameter. Combined with the σ-ladder
(g-side dose response, EXCESS-TASK-LOSS) the competition mechanism now
has causal confirmation on BOTH sides of the inclusion inequality.

## Gates (all passed)

nzs2 manifest (σ=2.0, dims, source, B2-era holdout sha pin — manifest
sourced from the sha-verified σ-ladder bundle copy
`sgl_20260815_123342/replay/q1_nzs2/manifest.json`, the identical file
the pin was minted from); **buffer linkage 16/16** (every fit's REPLAY
endswith q1_nzs2/side<side>); fit counters 16/16 update==total; ckpt
steps 16/16 at 500000; config gates (loss_scales.rew == {1,100} both
directions, dzs2 distractor block, task expl) — reader passed all.

## Input-production conformance note (disclosed)

The bundle shipped raw materials; the registered collate/witness inputs
were produced locally from the sha-verified bundle (precedent: goodhart
partial, σ-ladder): ridge/<run_id>.json renamed copies of per-run
ridge_probe.json (run_logdir field verified pointing at the ruw run
dirs); fit_counters/ckpt_steps dumped from the bundle runroot's own
OFFLINE_FIT_PROGRESS + done-ckpt names; **linkage json extracted from
the bundle's runs.json per-task cmd env (REPLAY=…)** — the prereg's
literal producer targeted the queue-template runs.csv; this wave ran on
the dv3ops route, whose runs.json is the same generator's task manifest
(AXIS1_WR present in every cmd env as the prereg's ops-route note
requires). All inputs archived in `inputs/`.

## Provenance discovery (instrument-grade, disclosed)

The registered w1-replication descriptive returned **diff exactly 0.0**:
all 8 fresh w1 per-fit AUROCs are **bitwise identical** to the pinned
SGL_NZS2_TASK values — yet the w1 WMs were freshly trained in-wave
(ckpt stamps 2026-08-16, ridge run_logdir = the ruw dirs, ridge json
bytes differ only in metadata). Conclusion: the static-buffer offline
fit + same-GPU ridge pipeline is **end-to-end bitwise reproducible** on
the Vast stack (fixed seed, same buffer, same GPU model) — training
included, not just evaluation. This (a) is the strongest possible
same-hardware replication of the σ-ladder task panel (8/8 exact), (b)
confirms the paired design's cells were genuinely fresh and same-stack,
(c) bounds the instrument noise floor at exactly zero for this
configuration. (The 1-Aug Midway3 non-determinism finding concerned
ONLINE training; offline static-replay fits are the deterministic case.)

## Honest notes

- w100 exceeding the unloaded anchor (0.969 vs 0.954) is reported as
  observed; the registered verdict does not depend on it, and
  interpretation (w_r=100 helping beyond load-compensation) is
  post-read theory work, not a read output.
- The seed2/side1 w1 value 0.954 (high outlier, reproduced exactly from
  the σ-ladder panel) leaves its pair's delta positive (+0.038);
  the primary is insensitive to it (sign-flip exact).
