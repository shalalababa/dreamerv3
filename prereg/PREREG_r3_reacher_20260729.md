# PREREG: R3 reacher third-domain replication (frozen 2026-07-29)

Paper-2 conference-band leg (task-domain axis of "replicated across
tasks/model families"). The original R3 registration
(`PREREG_r3_competence_20260725.md`) carried reacher as a
conditional-GO cohort inside its ONE read; that read executed 29 Jul
with reacher ABSENT (a registered-valid state,
`artifacts/r3_competence_20260729/`). This file supersedes that clause:
reacher is adjudicated as a FRESH single-domain replication with its
own frozen reader — the cup/finger primaries are never recomputed.
This file + `analysis/r3_reacher_read.py` (selfcheck PASS) + the
`SMOKE_DOM` driver parameterization in `scripts/r3_local.sh` are
committed BEFORE any reacher R3 run exists.

## Question and registered prediction

Does the opportunity/competence decomposition replicate in a third
domain? Registered prediction: YES on both legs (opportunity exists;
gap > 0) — the mechanism story (oracle-visible value the trained
consumer cannot harvest) is not domain-tuned. Genuine risk: reacher
_hard is sparse; heavy floors are possible at 1e5-step pilots, which is
why the floor gate below is registered as INCONCLUSIVE, not as failure.

## Design: 16 fresh runs + 32 label passes (idle-instance workload)

- Grid: reacher × {e1, e4} × seeds 31–38, two-phase training (early
  2.5e4 snapshot → resume to 1e5), base arm, task `dmc_reacher_hard`
  (gate0-verified regime record `artifacts/gate0_20260702/gate0_reacher/`:
  threshold 0.025, windows_ok). Everything (dials, labeler
  `d1fix_20260724`, `--oracle_all`, seeds, maturities, dose e4 =
  d0_dose3) IDENTICAL to the executed R3 wave — the driver's existing
  reacher mapping is used unmodified.
- **SMOKE GATE (registered, machine-checked at BOTH ends):** reacher
  has never been labeled dosed — the four reacher smokes (e1/e4 ×
  early/late, 5 states, seed-31 checkpoints) must exist and carry the
  registered labeler version; the reader ASSERTS both and refuses to
  read otherwise. The driver's smoke stage is parameterized
  (`SMOKE_DOM=reacher`, committed with this file; default stays `cup`
  — the executed waves are untouched), and the driver's labels stage
  now ALSO refuses to label any non-cup/finger domain whose four smoke
  files are absent — a stale `SMOKE_OK` from the executed cup wave
  cannot authorize reacher labels (pre-freeze review finding, fixed
  before this freeze).
- Submit sequence (idle instance, activated dv3 env, R3_ROOT set):

```
DOMS="reacher" ./scripts/r3_local.sh pilots      # 16 two-phase runs
SMOKE_DOM=reacher ./scripts/r3_local.sh smoke    # 4 reacher smokes
DOMS="reacher" ./scripts/r3_local.sh labels      # 32 oracle_all passes
```

- Existence caveat: if the instance's `$R3_ROOT/r3_labels` is fresh,
  the smoke stage creates `SMOKE_OK` from the reacher smokes alone —
  acceptable (the reader's gate is reacher-specific regardless).

## Registered read (frozen `analysis/r3_reacher_read.py`, ONE execution)

Estimands, floor definition, bootstrap (B=10K, rng 0, cluster = run,
both maturities share the cluster) imported unchanged from the frozen
`analysis/r3_read.py` (Amendment-1 version). n = 16 run clusters.

- **FLOOR GATE**: floor fraction > 50% ⇒ FLOOR-LIMITED — inconclusive;
  primaries reported, NOT adjudicated; no further reacher compute
  without a redesign. (Sparse-task floors are an information-free
  outcome; registering this now prevents a post-hoc argument later.)
- **P-RRa** pooled opportunity CI entirely > 0.2 (same existence bar).
- **P-RRb** pooled gap CI entirely > 0.
- **P-RRc** per-run paired late−early achieved, pooled CI entirely > 0.

Power: R3 per-cell gap sd ≈ 1.0–1.3 ⇒ at n=16 the detectable pooled
gap ≈ 1.96·1.2/√16 ≈ 0.59, well under the observed cup/finger gap
+1.29 — powered for a same-sized effect, honest about smaller ones.

## Consequence map (frozen)

- **REPLICATES (a+b fire)** ⇒ the decomposition is a three-domain
  fact; the conference-band task axis lands; P-RRc scopes maturity
  per domain.
- **NO-OPPORTUNITY (a fails)** ⇒ registered domain-scope statement:
  reacher offers no oracle-visible opportunity at these cells; the
  cup/finger decomposition stands unchanged; the paper reports the
  boundary.
- **HARVESTED (a fires, b fails)** ⇒ a registered competence-SUCCESS
  domain: the cup/finger competence failure becomes domain-scoped and
  the paper's headline gains a contrast domain (a stronger paper,
  differently shaped — the outline's §7 gains a panel either way).
- **FLOOR-LIMITED** ⇒ inconclusive; reported; reacher dropped from the
  band argument without prejudice.
- Regardless: cup/finger R3 results untouched; 24+24 frozen; Route A
  closed; imag retired.

## Costs and disclosure

16 two-phase trainings (~2–3 h each) + 32 oracle_all passes (~1.5–2 h
each) ≈ 3–4 days on one GPU. Known at freeze: the full cup/finger R3
record (all primaries/secondaries), gate0 reacher regime facts, the
sparse-task floor risk. Unknown: every reacher R3 quantity — no
reacher run, label, or smoke exists in the R3 context. Ordering: this
file + reader + driver edit committed BEFORE submission.
