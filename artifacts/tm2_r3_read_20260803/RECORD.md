# TM2-R3 — TD-MPC2 opportunity/competence cross-family replication: ONE read record (2026-08-03)

Registered ONE read of the TM2-R3 wave via the frozen reader
`analysis/tm2_r3_read.py`, under `PREREG_tm2_competence_20260730.md`
(freeze-committed 2026-07-31 with the five Paper-2 bundles).

## VERDICT: **CROSS-FAMILY REPLICATION** (strongest registered branch)

- **P-TM2a (opportunity exists) FIRES**: pooled mean opportunity
  **+0.546 [+0.325, +0.808]**, CI entirely above the 0.2 bar
  (n = 32 run clusters, run-clustered bootstrap B=10K, rng 0).
- **P-TM2b (competence failure) FIRES**: pooled implementation gap
  **+0.530 [+0.315, +0.787]**, CI entirely > 0. The 6-iteration,
  512-sample MPPI PLANNER leaves the one-real-step oracle opportunity
  on the table — pooled achieved is a tight null
  (+0.016 [−0.036, +0.074]): the planner harvests essentially NONE of
  what the probe reveals.
- **P-TM2c (maturity buys competence) does NOT fire**:
  −0.034 [−0.158, +0.098] — the same maturity null as DreamerV3-R3.

The decomposition is now a TWO-FAMILY regularity with the same shape
in both families: opportunity real, gap ≈ opportunity, achieved ≈ 0,
maturity buys nothing. Effect size ≈ 42% of the dv3 pooled gap
(+0.53 vs +1.29), above the ≈0.42 pre-sized detection bar.

Registered secondaries (descriptive): opportunity GROWS with dose
(e1 +0.304 → e4 +0.789 — distractors create more unharvested
opportunity, the dv3 pattern); both domains same direction (cup gap
+0.777 [+0.399, +1.202], finger +0.283 [+0.140, +0.453]); maturity
does not move opportunity (early +0.572 / late +0.521). FLOOR gate
passes both domains (cup 0%, finger 15.6% — both < 50%; the
sparse-task floor risk did not materialize, mirroring reacher).

## Provenance (verified before the read)

- Bundle `local_results/tm2_r3_20260803_175551/`; committed manifest
  byte-identical to the bundle's; **sha256 sweep 370/370 OK, 0
  FAILED**.
- RCC head at run time `481aced9` (committed ancestor, contains the
  31-Jul instrument freeze); RCC tree clean except untracked slurm
  logs. All FIVE frozen instruments byte-identical between the
  bundle's `code/code_sha256.txt` and the local committed files
  (tm2_r3_read 00e83fc4…, tdmpc2_oracle_labels 8eeb5949…,
  tdmpc2_r3_train 422b1d60…, tdmpc2_compat 09172a74…, tm2r3.sbatch
  14004651…).
- 32/32 runs `TM2R3_TRAIN_DONE`; SMOKE_OK marker present; the API
  smoke + 4 dosed seed-51-late smokes present as `*_smoke.npz`
  (reader-enforced smoke gate; smoke files never consumed as labels).
- 64/64 label passes present; reader enforced the all-or-nothing
  cohort, seeds 51–58, n_states 200, M=8, dial identity, dose/seed/
  checkpoint identity per file.

## Registered FILL slots — executed at read time (deviation disclosed)

The prereg registered two FILL slots "completed by a dated amendment
BEFORE labels". **No such amendment file exists — a process deviation,
recorded here honestly.** The fill DATA is in the bundle and the audit
was executed machine-checked at read time (value-blind quantities:
step counters + config echoes):

- Realized snapshot steps: `ckpt_early.STEP` = **25000 exactly for
  all 32 runs** (first episode boundary ≥ 2.5e4 with 1000-step
  episodes ⇒ exactly 25000 — coherent); `ckpt_late.STEP` = **100001
  for all 32**.
- Planner-dial audit: the slim per-run `config.json` has no planner
  block (the prereg's assumed location), but every LABEL's meta
  records the full planner block; **64/64 labels match the registered
  values exactly** (mpc true, horizon 3, iterations 6, num_samples
  512, num_elites 64, num_pi_trajs 24, temperature 0.5, min_std 0.05,
  max_std 2.0; discount 0.995), and all 32 run configs record
  num_q = 5, latent_dim = 512. The audit's intent is fully
  discharged; only its registered timing (pre-labels amendment) was
  missed. No decision quantity is touched by the deviation.

## Consequences (frozen map)

- Paper 2's headline gains its **strongest generality leg**: the
  opportunity/competence decomposition is architecture-general
  (DreamerV3 finger/cup/reacher + TD-MPC2 cup/finger), and the
  competence failure now covers a categorically stronger consumer
  class — a 512-sample MPPI planner fails to harvest what a single
  real step reveals, sharpening the claim beyond the plug-in-consumer
  scope.
- The HARVESTED-BY-PLANNER family boundary did NOT materialize; the
  consumer-class-scoped fallback framing is unnecessary.
- Venue decision input (user's 29-Jul rule: "ICML if TD-MPC2 repl
  fires"): **the ICML branch condition is now MET.** Venue call
  remains ~Dec as decided.
- Standing invariants unchanged: R3 read stands; 24+24 frozen; Route
  A closed; imag retired (sentinel columns never consumed).

## Read execution

```
python -m analysis.tm2_r3_read \
  --labels local_results/tm2_r3_20260803_175551/tm2r3/labels \
  --output artifacts/tm2_r3_read_20260803/
```

Outputs: `tm2_r3.json` (full, incl. per-run values), `read_stdout.txt`.
Executed once, 2026-08-03.
