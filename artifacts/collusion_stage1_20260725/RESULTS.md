# Collusion pilot stage 1 — baseline reproduction: GO (2026-07-25)

Snapshot: `local_results/collusion_pilot_baseline_20260725_090946/`
(100 sessions, seeds 0–99, tabular Q-learning pair on the Calvano
duopoly, 2M-iteration cap). Read per the pilot design of record
`research_notes/Design_Collusion_Pilot_20260724.md` (stage-1 rule:
GO iff mean Δ_profit in the published range ≈0.7–0.9 AND fingerprint
rate is majority).

## Stage-1 verdict: **GO** (both legs)

| quantity | value | rule |
|---|---|---|
| mean Δ_profit | **0.831** (sd 0.12; med 0.828; IQR 0.752–0.933) | in published ≈0.7–0.9 ✓ |
| fingerprint rate | **68/100** | majority ✓ |
| converged sessions | 75/100 (25 hit the 2M cap) | descriptive |
| fingerprint \| converged | 0.67 | descriptive |
| punishment recovery rate | 0.71 | descriptive |
| mean Δ_price | 0.688 | descriptive |

The phenomenon reproduces: supra-competitive profits at the published
level, sustained in the majority of sessions by genuine
punishment-and-recovery strategies (mean punish depth 0.30 grid units
among fingerprinted sessions, return within the 15-period window).

A minority pattern worth carrying forward: high-Δ sessions WITHOUT the
fingerprint exist (e.g. seeds 98/99 at Δ≈0.95) — the Abada–Lambin
failure-to-learn-to-deviate signature the design anticipated. The
confirmatory prereg's measurement section must treat "collusion" as
fingerprint-conditional, not Δ-conditional.

## Stage-2 calibration findings (from the same logs)

- **Coverage is saturated: 1.000 in ALL 100 sessions** (minimum
  1.000). At 2M iterations with ε-greedy decay, every one of the 225
  joint states is visited. Consequence: RQ1's matched (Cov, Occ) pairs
  are NOT constructible from these logs — coverage has zero variance.
  The confirmatory design needs a coverage knob (shorter sessions /
  faster ε decay / larger grid m / restricted exploration), to be fixed
  in the confirmatory prereg before any RQ1 claim.
- **Punishment occupancy has healthy spread**: 0.46–0.99 (mean 0.73,
  sd 0.17), and correlates weakly with Δ_profit (r ≈ −0.10) — the
  occupancy axis is usable as-is.
- undercut_rate mean 0.33 (med 0.30); iters mean 1.81M.

## Run provenance note

Two slurm jobs appear in the bundle: 52624560 was CANCELLED mid-sweep
(around seed 36); 52625412 completed all 100 sessions and wrote the
csv. Sessions are seed-deterministic end-to-end (selfcheck-validated),
so the partial first job contaminates nothing.

## Next (per the design of record)

- Stage-2: pick the punishment-phase regime definition on these logs
  (undercut+W=5 vs distance-from-collusive-path) — analysis on this
  artifact, no new compute.
- Stage-3: Design-B viability probe (batch Q-iteration on logged
  transitions vs frozen opponent checkpoints) — build next; Design A
  runs regardless (staged-both default).
- Coverage-knob pilot (small): vary session length / ε schedule to
  de-saturate coverage BEFORE the confirmatory prereg fixes the RQ1
  design.

## Provenance

- `baseline_sessions100.csv` — one row per session (Δ_profit, Δ_price,
  fingerprint, punish_depth, recovered, coverage, punish_occupancy,
  undercut_rate, iters, converged).
- Command: `python -m collusion.pilot run --sessions 100
  --max_iters 2000000 --output ...` (cluster CPU; ~35 core-min).
