# PREREG: volume-anomaly replication — P-E4a (frozen 2026-07-26)

Registers the P-E4a replication authorized "post-B" by
`PREREG_compcapacity_theory_20260724.md` §P-E4: does the 18-Jul
v400s1 anomaly REPLICATE on fresh seeds? The anomaly (Opt-C rider,
`artifacts/scaling_optc_20260718/`): the task arm on the LOW-occupancy
400-episode buffer (v400s1, occ .094) beat the HIGH-occupancy
200-episode buffer (v200s1, occ .323) 432.9 vs 204.3 at matched fit
budget — with the WORST rep-level reward NLL (rep-level excluded then;
the behavioral gap is what replicates or dies). This file +
`analysis/volume_repl_read.py` (selfcheck PASS) are committed BEFORE
any fresh-seed volume run exists.

## Design: 24 jobs on the EXISTING frozen buffers, fresh seeds 7–12

The volume buffers are frozen and shared per cell (seeds vary fit/
adapt RNG only), so the replication is fit+adapt-only. The standard
submit loop covers both sides, giving the registered 24-job form
(task arm only — the anomaly is a task-arm fact; apt cells are not
re-run):

```
AXIS1_ID_PREFIX=ax1v2 AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger \
  AXIS1_QUADS=q1v200 AXIS1_SEEDS="7 8 9 10 11 12" ./scripts/submit_all.sh axis1-bundles
AXIS1_ID_PREFIX=ax1v4 AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger \
  AXIS1_QUADS=q1v400 AXIS1_SEEDS="7 8 9 10 11 12" ./scripts/submit_all.sh axis1-bundles
```

- s1 cells (primary): `ax1v2q1v200s1` / `ax1v4q1v400s1`, +6 seeds
  each → n=12 total per cell.
- s0 cells (secondary): `ax1v4q1v400s0` extends to 12; `ax1v2q1v200s0`
  (occ .054, never run) gets its first 6 seeds — the
  occupancy-conditional cell P-E4b wants. All-or-nothing on each
  fresh cohort (the frozen read enforces).

## Registered read (frozen `analysis/volume_repl_read.py`)

- **PRIMARY (sole confirmatory): fresh-batch (seeds 7–12)
  mean[v400s1] − mean[v200s1], AUC100k, two-sample seed-cluster
  bootstrap (B=10K, rng 0); FIRES iff CI > 0 ⇒ P-E4a REPLICATES.**
- Power (disclosed): observed gap +228.6, pooled sd 183.1 ⇒ the 6+6
  fresh contrast detects ≈207 — powered for effects near the observed
  size, which is the right bar for a winner's-curse check (the claim
  under test IS the observed magnitude).
- Registered descriptives (never decisional): pooled-12; the original
  batch restated; the s0 fresh contrast; per-cell seed maps.

## Diversity secondary (P-E4b operational form, registered now)

Pinned index: `diversity_pr` from the frozen spectral instrument
(`probing/spectral_measure.py`, `spectral_v1_1_20260724`) run on the
two s1 volume buffers (same MEASURE_VERSION gate and conventions as
the domains pass). Registered directional prediction: diversity_pr
(v400s1) > diversity_pr(v200s1) — breadth, not occupancy, is what the
400-episode buffer buys. This is descriptive here; the full
conditional-on-occupancy regression form of P-E4b needs more buffers
and is NOT registered by this file.

## Consequence map (frozen — from P-E4a's registered consequences)

- **FIRES** ⇒ the anomaly is real: support breadth (episode
  diversity) enters the theory as a λ-structure input distinct from
  occupancy fraction; the P-E4b diversity leg engages; the flagship
  gains its support-breadth axis.
- **Does not fire** ⇒ registered demotion to winner's curse: no
  theory term is added; the 18-Jul cell is reported with the
  replication failure; rep-level exclusion stands.
- Diversity direction failing while the primary fires ⇒ a
  non-spectral account of the volume effect is forced (P-E4's own
  registered consequence).

## Disclosure and ordering

Known at freeze: the full Opt-C rider + corrective (all seeds-1–6
volume cells on both arms and both sides where run, incl. the
anomaly), occupancy calibrations, the 26-Jul Scaling B read. Unknown:
every seed-7–12 volume quantity and both buffers' diversity_pr (the
spectral instrument has never been run on the volume buffers).
Ordering: this file + `analysis/volume_repl_read.py` committed BEFORE
submission; the diversity measurements may run on the login node any
time after commit (buffer-only, no fit dependence).
