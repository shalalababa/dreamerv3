# R3 reacher third-domain replication — READ: REPLICATES (2026-07-31)

Registered read (`PREREG_r3_reacher_20260729`, freeze commit 596a7f93;
frozen reader `analysis/r3_reacher_read.py`, ONE execution 31 Jul,
local). Bundle `local_results/r3_reacher_20260731_100509/` under
results-sync policy v2.

## Primaries

| primary | estimate [95% CI] | rule | fires |
|---|---|---|---|
| P-RRa pooled opportunity | **+0.828 [+0.565, +1.114]** | CI > 0.2 | **YES** |
| P-RRb pooled gap | **+0.813 [+0.554, +1.097]** | CI > 0 | **YES** |
| P-RRc paired late−early achieved | +0.047 [−0.023, +0.121] | CI > 0 | no |

**Verdict (frozen reader, verbatim): REPLICATES: the
opportunity/competence decomposition holds in a third domain —
opportunity exists (P-RRa) and is not harvested (P-RRb). P-RRc does
not fire — maturity does not close the gap in reacher either.**

n = 16 run clusters ({e1,e4} × seeds 31–38; both maturities share the
cluster), run-clustered percentile bootstrap B=10K rng 0 — machinery
imported unchanged from the frozen `analysis/r3_read.py`
(Amendment-1 version). Pre-sized detectable pooled gap ≈ 0.59;
observed +0.813 clears it with margin.

## Registered consequence engaged

**REPLICATES ⇒ the decomposition is a three-domain fact (cup, finger,
reacher); the conference-band task axis LANDS.** P-RRc scopes
maturity: the null replicates per domain — "maturity does not buy
competence" now holds in all three domains. Regardless-invariants
reaffirmed: cup/finger R3 results untouched; 24+24 frozen; Route A
closed; imag retired.

## Secondaries (descriptive, never decisional)

- Opportunity by dose: e1 +0.995 [+0.523, +1.486], e4 +0.661
  [+0.455, +0.874] — survives the e4 distractor dose.
- Opportunity by maturity: early +0.683 [+0.260, +1.261], late
  +0.973 [+0.660, +1.317] — present at both snapshots.
- Pooled achieved: +0.015 [−0.038, +0.071] — tight null; the
  consumer harvests essentially none of it.
- **Floors: 1/32 cells (3.1%), `floor_limited: false`** — the
  registered sparse-task floor risk (the reason the FLOOR GATE
  existed) did NOT materialize; the gate is nowhere near tripping
  (bar 50%).

## Chain of custody / verification (pre-read, in order)

1. Manifest: `scripts/bundle_manifest.sh verify` → **OK, 14,554
   files**, bundle MANIFEST.sha256 == committed pin
   `manifests/r3_reacher_20260731_100509.sha256` (staged at read
   time).
2. Code snapshots in `code/` byte-identical (`cmp`) to the frozen
   repo files: `r3_reacher_read.py`, `PREREG_r3_reacher_20260729.md`,
   `scripts/r3_local.sh`. Bundle `code/COMMIT` = 9e9402dd (cluster
   checkout postdates the 596a7f93 freeze; executed files verified
   byte-identical to the frozen versions, so the registered
   instrument ran).
3. Grid complete before read: 32 full passes ({e1,e4} × seeds 31–38 ×
   {early,late}) + the 4 registered dosed smokes
   (`r3_reacher_{e1,e4}_seed31_{early,late}_smoke.npz`) + `SMOKE_OK`;
   smoke gate + labeler-version pin (`d1fix_20260724`) machine-checked
   by the frozen reader at load.
4. Reader `--selfcheck` PASS immediately before the read (verbatim:
   "selfcheck PASS: four verdict branches incl. floor-limited gate,
   cohort/stray-seed/n_states/duplicate trips, smoke-gate missing +
   wrong-labeler trips").
5. ONE execution of the registered read; output archived unmodified as
   `r3_reacher.json` (sha256 a2538b766b2116cf04df25bb198733ef072c9e
   cbc038e66c5984cc254b4cb487). File-location note: the reader treats
   `--output` as a directory, so the json landed one level deeper and
   was MOVED (not modified) to the artifact root.

One-read discipline: no prior read output existed anywhere (first and
only execution); the cup/finger primaries were not recomputed
(reacher-only reader by construction).
