# R3 consumer-competence factorial — READ EXECUTED 2026-07-29

Registered: `prereg/PREREG_r3_competence_20260725.md` (freeze 8b3380ce)
+ Amendment 1 `prereg/PREREG_r3_amend1_20260729.md` (reader seed-grid
repair + value-blind integrity guards; decision rules unchanged;
commit bbe86983 at 10:40:20 −0500, BEFORE the read). ONE read executed
on RCC midway3 (slurm job 52773785, ~10:53) on
`/scratch/.../r3_local/r3_labels`; output committed 4f74b187. Bundle:
`local_results/r3_competence_20260729_105146/` (labels + code
snapshots + `artifact/r3.json` + read log). Reacher cohort: absent
(registered conditional-GO, absent-but-never-partial — valid state).

## Verdict (frozen consequence map, first branch)

**CONSUMER-COMPETENCE FAILURE: opportunity exists (P-R3a) and is not
harvested (P-R3b); maturity does not detectably buy competence
(P-R3c null).**

| primary | estimate [95% CI] | rule | fires |
|---|---|---|---|
| P-R3a pooled opportunity | **+1.247 [+0.700, +1.908]** | CI > 0.2 | **YES** |
| P-R3b pooled gap | **+1.293 [+0.723, +1.987]** | CI > 0 | **YES** |
| P-R3c paired late−early achieved | +0.014 [−0.177, +0.215] | CI > 0 | no |

n = 32 run clusters (both maturities per cluster), run-clustered
percentile bootstrap B=10K rng 0. Pre-sized power: detectable pooled
gap ≈ 0.42; observed +1.29 ≈ 3× the detection bar — this is not a
marginal fire. Per the frozen map: the 25-Jul shift-consequence
negative is now interpretable as a **competence ceiling, not an
information ceiling**; Paper 2's headline becomes the
opportunity/competence decomposition, with P-R3c scoping that
maturity (2.5e4 → 1e5, the run's own continuation) does NOT close the
gap in this range.

## Registered secondaries (descriptive, never decisional)

- **Per-domain** (the gap is domain-robust, not one domain's artifact):
  cup opp +0.982 [+0.384, +1.710], ach +0.065 [−0.060, +0.202], gap
  **+0.917 [+0.373, +1.583]**; finger opp +1.512 [+0.664, +2.610],
  ach **−0.157 [−0.289, −0.050]**, gap **+1.669 [+0.735, +2.880]**.
  Finger's achieved value is NEGATIVE with CI < 0: the trained
  consumer's realized choice does marginally worse than the incumbent
  action — harvest is not merely absent, it is slightly
  counterproductive there.
- **Dose effect on opportunity**: e1 +1.510 [+0.707, +2.606], e4
  +0.984 [+0.335, +1.756] — opportunity exists at BOTH dose levels
  (distraction does not eliminate information content; point estimate
  lower under dose, CIs overlap broadly).
- **Maturity effect on opportunity**: early +1.193 [+0.554, +2.008],
  late +1.301 [+0.626, +2.149] — opportunity is stable across
  maturity; the gap's persistence is not because opportunity vanished.
- **Floors**: cup 8/32 cells (25.0%), finger 6/32 (18.75%) — neither
  domain floor-limited (bar 50%); floor cells included in all
  primaries per the frozen conservative policy.

## Execution verification (this session)

- Amendment-before-read ordering held: bbe86983 (10:40:20) precedes
  the read (~10:53) and the results commit 4f74b187 (10:53:56).
- Labels bitwise identical to the audited 27-Jul bundle (all 68 npz
  md5-identical; that bundle passed a 3-agent value-blind integrity
  audit: 64/64 exact registered dials + doses, all-finite G, smoke
  gate 4/4 on real dosed envs, 32/32 two-phase training with the
  continuation property verified).
- Bundle code snapshots (reader, driver, both preregs) byte-identical
  to repo HEAD; committed `r3.json` ≡ bundle copy.
- Local verification re-execution of the amended reader on the bundle
  labels: **bit-identical** to the cluster read on every primary,
  secondary, fire flag, and verdict string (only the recorded
  `labels_dir` path differs). No float noise at all this time.
- Selfcheck PASS before the read (amendment session), including the
  Amendment-1 guard-trip cases.

## Provenance notes (carried from Amendment 1)

Training + smoke on the Vast box; labels + read on RCC after rsync;
labels-stage stdout absent from the 27-Jul bundle; bundle assembly
reset mtimes (smoke-before-labels ordering rests on the worker log +
`SMOKE_OK`). The read log `logs/r3_read_52773785.out` is in this
bundle.

## Standing consequences

- 24+24 stay frozen; Route A stays closed; imag stays retired
  (resource decisions, reaffirmed by the frozen map "Regardless"
  clause).
- Any calibration/repair claim (making the consumer harvest) is a NEW
  registration, not an extension of this one.
- Cross-checkpoint consumer (early WM × late critic) remains OUTSIDE
  this registration (labeler extension + own smoke + dated amendment
  required).
