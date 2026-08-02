# pe pixel arm READ — pretrained-frozen-encoder lever test (2026-08-02)

ONE registered execution of `analysis.pe_pixel_read` (never re-run).
Registration: `prereg/PREREG_pe_pixel_20260730.md` (committed with the
frozen reader + code additions BEFORE any pe run existed). Outputs
here: `pe_pixel.json` (full registered output) + `read_stdout.txt`.

## Provenance (verified pre-read)

- Bundle `local_results/pe_pixel_20260802_165601/`: `sha256sum -c
  MANIFEST.sha256` → 0 mismatches (152 files); repo copy
  `manifests/pe_pixel_20260802_165601.sha256`.
- Bundle git head `bcc73387` IS a local commit (the 2 Aug reads
  commit) and all 7 registered code shas are byte-identical to the
  committed finals with zero drift since (`git diff bcc73387..HEAD`
  empty on all 7): reader `34380e0f…`, check_frozen_enc `00acbdff…`,
  axis1.sbatch `8c222d13…`, offline_fit `214324c3…`,
  stratified_error `0379a35d…`, agent.py `1966a6d7…`,
  configs.yaml `ca1e6216…`.
- **Registered integrity sweep** (`pe_integrity_sweep.log`, ships with
  the bundle per prereg): **16 PASS, 0 QUARANTINE** — every stage-2
  fit has its 12 enc params BYTE-IDENTICAL to its fpxpx donor, the
  TRAINING WITNESS at 500000/500000 with counters reset, and 5 rew
  params present. This kills the one instrumentation trap the prereg
  named as able to masquerade as NO-RELIEF (counter-carryover
  training skip).
- ADAPT_DONE 16/16; auc.csv complete 2×8 grid, qc_pass=1 on all 16;
  E4 csv 64 rows (16 fits × 4 horizons), reward_aware=1 throughout;
  exclusions.csv contains only stale non-pe runs.
- Reader selfcheck PASS immediately before the read.
- Independent anchor: hand-computed mean of the 16 h=0 `rew_nll_in`
  values = 23.0324 ≡ the reader's P-PE1 point.

## Result

- **P-PE1 (representation, primary): NO-RELIEF.** In-regime rew-NLL
  23.03 [20.14, 25.68] — statistically indistinguishable from the
  committed swamping baseline 23.34 [19.41, 27.29]; not below the 2.0
  swamping bar (INCLUSION-RESTORED), not even below the baseline's
  lower bound 19.41 (PARTIAL-RELIEF). Membership band: no.
- **P-PE2 (behavior): no lift.** Paired [pe − X2-task] −8.5
  [−26.0, +9.6], n=8, 3/8 positive — the pe arm sits ON the X2 pixel
  floor (large per-seed spread −47 to +40, centered slightly below 0).
- **P-PE3: premise-idle** (P-PE2 did not fire); descriptive s1−s0
  −7.9 [−27.4, +10.1].
- **VERDICT (registered map, verbatim): NO-RELIEF — the
  encoder-competition lever fails its first intervention; registered
  honest negative; the swamping account needs revision before further
  pixel arms.** P-SW1 (the 23.34 measurement) is NOT re-adjudicated.

## What this means (interpretation, non-verdict)

The swamping account's causal claim was that reconstruction's
fit-time gradient bid for encoder capacity is what excludes reward
structure at pixel. The lever test removed the encoder from the
gradient competition entirely (frozen pretrained encoder; dyn+heads
fit the identical FULL-arm objective on top) — and the reward head's
in-regime NLL did not move AT ALL (23.03 vs 23.34). Two readings,
both for §revision (neither is licensed as a headline by this read):

1. The competition story mislocates the bottleneck: the exclusion is
   not (only) at the encoder's gradient allocation, since fixing the
   features doesn't help. The competition could live in dyn, or the
   pixel reward signal is too sparse/hard for heads regardless.
2. Mechanism-coherent alternative: the donor encoder is REWARD-FREE
   trained (reconstruction+dynamics), so its features were selected
   with no reward legibility pressure — under the paper's own
   headline ("support must be legible to the objective") a
   reconstruction-trained encoder failing to make reward fittable is
   the EXPECTED outcome, i.e. this null lands as a fourth
   reward-free-transfer null rather than evidence against
   spectral competition per se. Separating (1) from (2) needs a
   task-trained-donor or trainable-encoder-anchored arm — NEW
   registration if ever pursued; the frozen-RANDOM-encoder control
   named in the prereg was conditional on PE1 firing and is NOT
   queued.

## Consequences

- U4's live "pretrained-encoder decision" resolves: the pixel floor
  is protocol-independent (U4) AND survives encoder pretraining —
  the pixel boundary stands as a hard negative-result boundary for
  the paper, now with its leading mechanistic account demoted to
  "measured fact (P-SW1) + falsified first lever".
- fpxpx donor fits: the registered KEEP-PENDING condition ("until
  this wave's read verifies") is now met — release is a user
  cleanup decision; releasing forecloses donor-dependent follow-ups
  (any future pe variant would need the donors or fresh fits).
- No further pixel arms are queued by default (registered map:
  account revision comes first).
