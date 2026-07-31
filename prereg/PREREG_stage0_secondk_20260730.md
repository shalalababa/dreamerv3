# PREREG: Stage-0 second-k density-proxy robustness — k=10 → {5, 20} (built 2026-07-30; freeze-commit 2026-07-31)

Executes the Stage-0 memos' own registered-but-unexecuted caveat before
Paper 2 externalizes the density-meter headline. The committed Stage-0
record (`artifacts/latent_uq_stage0_20260710_221232/DECISION.md`) reads
10/10 held-out p2e seed-cells TRACKS_DENSITY at the final checkpoint
(anchor, h=5) — "disagreement is a *density* meter, not an error meter" —
and every memo carries, verbatim: "kNN density in high-dim latent spaces
is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict
should be sanity-checked against a second k before externalizing"
(`cup_p2e/memo.md`). That sanity check has never run: the committed claim
is a single-k claim. This registration is the second-k check, run as a
controlled single-variable change with a machine-checked integrity guard.
This file + `analysis/stage0_secondk_read.py` (selfcheck PASS) are
committed BEFORE any k ≠ 10 dump exists anywhere.

## Question

Is the 10/10 TRACKS_DENSITY verdict a property of the latent geometry or
an artifact of the k=10 dial in the kNN density proxy? **Registered
prediction**: robust at both flanks — mean-of-k-smallest-distances is
rank-stable across overlapping neighbor sets, and the committed verdicts
were unanimous across 10 independent runs and strengthened with training.
**Genuine risk**: at k=5 the proxy is a high-variance boundary-dominated
estimate in the >1000-dim standardized anchor space, and at k=20 it
smooths over exactly the local structure the attractor-bias story needs;
either flank can reshuffle density ranks enough to push Δ inside the
2σ band — and under the strict rule below, AMBIGUOUS counts as a flip.

## Estimand

Per cell × k2: the final-checkpoint anchor-h5 verdict of the SAME frozen
machinery that produced the committed claim —
Δ = pcorr(D,E|ρ) − pcorr(D,ρ|E) (rank-based partials), stream-clustered
bootstrap SE (n_boot=300), verdict TRACKS_DENSITY iff Δ < −2·SE
(`probing/latent_uq_analysis.py`: `within_signals`/`row_stats`/
`verdict_of`/`analyze_cell`, VERDICT_SIGMA=2.0, imported with pin
asserts) — where ρ is `density_knn` recomputed at k2. Nothing else about
the estimator changes; k is the single manipulated variable.

## Design

- **Substrate fact (verified)**: a second k CANNOT be computed from disk.
  `knn_distance` keeps only the mean of the k smallest distances
  (`probing/latent_uq_analysis.py:89–111`) and the dumper saves neither
  anchors nor reference encodings (`probing/latent_uq.py` — `uq.npz`
  holds only derived arrays). A second-k pass = re-running
  `python -m probing.latent_uq --knn K2` per run, which needs (a) the
  run's 5 retained checkpoint snapshots, (b) its training replay buffer
  (density reference), (c) the frozen probesets. Probesets are LOCAL
  (`local_results/evpi_stage0_20260710_221924/probesets/{cup_v1,finger_v1}`,
  sha256 pinned in the reader). Checkpoints/replay lived at
  `/workspace/dreamerv3_runs` on a Vast box per the committed
  `index.json`; current location UNKNOWN at freeze (candidate: RCC
  scratch home_offload archive).
- **REGISTERED EXISTENCE GATE**: run the existence-check command below
  first. ALL-OR-NOTHING: if ANY of the 10 runs lacks its 5 committed
  snapshots or its replay, record which, run NO passes at all, and take
  the **SUBSTRATE-GONE** branch (a valid registered outcome, not a
  failure): every external TRACKS_DENSITY statement permanently carries
  the single-k qualifier.
- **Cohort**: the 10 held-out p2e cells,
  `pretrain_p2e_{cup,finger}_seed{1..5}` — all-or-nothing, because the
  committed claim is 10/10 and a partial cohort cannot adjudicate its
  robustness. The apt/random cross-policy cells are an OPTIONAL
  secondary cohort (all-or-nothing if run, descriptive only, no decision
  role — their committed verdicts are TRACKS_ERROR/AMBIGUOUS and are not
  the claim being externalized).
- **k2 = both flanks, 5 and 20** (registering both costs the same order
  as one; a one-sided check would leave the other flank as a fresh
  caveat).
- **Dials**: every non-knn dial pinned to the committed run's values,
  machine-checked by the reader from each dump's `index.json`/`meta.json`
  — probeset (sha-pinned), burn_in 32, eval_steps 8, ref_windows 2048,
  dumper `--seed 0`, `--ep_batch 64` (batching sets the rng marks, so it
  is a value-relevant dial; the dumper is patched IN THIS FREEZE to
  record `ep_batch` in `meta.json`, and the reader REQUIRES the key on
  every new dump — the committed k=10 dumps predate the key and are
  exempt from its presence, never its value), and the EXACT committed
  snapshot dirs passed via `--checkpoints` (names pinned by equality
  with the committed dump's `index.json`). The smoke dumps carry the
  FULL dial pins too (probeset id+sha, burn_in/eval_steps/ref_windows,
  own-run ref_replay, dumper seed, ep_batch) — a mis-dialed calibration
  must not select the guard mode on a false premise. Output dirs are
  suffixed `_k5`/`_k20` and NEVER overwrite the committed k=10 dumps.
- **Platform**: the re-dumps pin `--platform cuda`, matching the
  committed dumps. Rationale (this REPLACES any weaker claim): the
  stack computes in bfloat16 and the RSSM SAMPLES the categorical stoch
  at every observe/imagine step, so cross-backend drift can flip
  discrete draws and diverge whole windows O(1) — "drift is numeric
  only" is NOT a safe premise, which is exactly why the guard mode is
  selected by an explicit calibration (below), never assumed.
- **REGISTERED INTEGRITY GUARD (machine-checked, the crux)**: for each
  re-dump, every k-INdependent array (`disag_steps`, `err_steps`,
  `err_*`, `pred_*`, `true_*`, `source_label`, `stream_start`; key sets
  must coincide) must match the committed k=10 dump for the same run
  and checkpoint. TWO registered modes, frozen in the reader:
  - **tolerance** — exact equality for non-float arrays; float arrays
    within rtol 1e-4 / atol 1e-6;
  - **fallback** — exact non-floats + identical key sets and shapes +
    `true_*` within the numeric tolerance (numpy-computed from the
    sha-pinned probeset, backend-independent) + per-array Spearman rank
    correlation ≥ **FALLBACK_RANK_FLOOR = 0.9** for the remaining float
    arrays (sample-flip drift re-draws the same windows and preserves
    cross-window order in aggregate; a changed anchor/probe-window/
    reference set decorrelates ranks toward 0).
  ANY violation under the SELECTED mode ⇒ **QUARANTINE**: no verdict is
  computed — the pass was not a controlled single-variable change.
- **SMOKE + GUARD-CALIBRATION GATE (registered, machine-checked;
  `check_smoke`, also standalone as `--smoke_check` run between the
  smoke dumps and the full passes; value-blind — only k-independent
  arrays and the k=10 calibration are touched, no k2 estimand)**: TWO
  single-checkpoint dumps on the earliest committed cup-seed1 snapshot
  must precede the full passes:
  - `cup_v1_k10cal_smoke` (`--knn 10`): compared to the committed dump
    with `density_knn` INCLUDED (same k — this also exercises the
    density REFERENCE side, replay chunks → ref windows, end to end).
    Within tolerance ⇒ guard mode **tolerance**; fails tolerance but
    passes the fallback guard ⇒ guard mode **fallback** (recorded in
    the output); fails BOTH ⇒ **invalid** ⇒ registered QUARANTINE (the
    backend cannot reproduce the committed substrate at the SAME k, so
    no k2 pass is interpretable as a single-variable change).
  - `cup_v1_k5_smoke` (`--knn 5`): checked under the selected mode
    (`density_knn` exempt — k differs). Any failure ⇒ QUARANTINE.
- Ordering: freeze-commit → existence gate (writes
  `secondk_substrate_gate.json`) → rsync probesets + committed index
  files → the two smoke dumps → `--smoke_check` → 20 full passes →
  rsync back → ONE read (`analysis/stage0_secondk_read.py`, consuming
  the gate record via `--gate_json`).

## Power (pre-sized from named committed sources)

This is a verdict-robustness check, not a new effect estimate, and the
instrument's power is inherited unchanged: same 2560 probe windows per
checkpoint, same stream-clustered n_boot=300, same 2σ rule that produced
10/10 unanimous TRACKS_DENSITY with the phase-change signature
(`artifacts/latent_uq_stage0_20260710_221232/{cup,finger}_p2e/analysis.json`).
No new estimator or sample size is introduced, so a flip is attributable
to k, not to power; conversely the strict rule (AMBIGUOUS = flip) means
low-power outcomes register as K-SENSITIVE, never silently as robust.

## Registered decision rules (frozen in `analysis/stage0_secondk_read.py`)

The reader pins the 10 cell names and their committed TRACKS_DENSITY
verdicts as constants and asserts at read time that the committed
artifact still says so; it NEVER recomputes anything about k=10 beyond
the integrity comparison.

- **P-K1 (SECOND-K ROBUST)**: fires iff ALL 10 cells' final-checkpoint
  anchor-h5 verdicts are TRACKS_DENSITY at BOTH k=5 and k=20 (20/20).
- Any other verdict (TRACKS_ERROR or AMBIGUOUS) anywhere ⇒
  **K-SENSITIVE**: flipped cells enumerated in the output; the
  externalization qualifier becomes mandatory.
- Any integrity-guard violation under the calibrated mode, any smoke
  failure, or a k10-calibration verdict of 'invalid' ⇒ **QUARANTINE**:
  no verdict, audit first. If the read adjudicates under guard mode
  'fallback', the mode is recorded in the output and reported alongside
  any externalization (a weaker guard than 'tolerance', disclosed).
- Registered secondaries (descriptive, never decisional): per-checkpoint
  verdict trajectories at each k2 (does the committed phase-change shape
  — earliest checkpoint error-tracking, then flip to density-tracking —
  reappear?), and the h=1 addendum read per cell.

## Consequence map (frozen)

- **SECOND-K ROBUST** ⇒ the memos' registered caveat is discharged by
  execution: Paper 2 §4 may externalize "on-distribution, ensemble
  disagreement is a density meter, not an error meter" citing this read,
  without the single-k qualifier.
- **K-SENSITIVE** ⇒ the qualifier becomes mandatory wording: every
  external TRACKS_DENSITY statement names k=10 and enumerates the
  flipped cells; the §4 headline weakens to "at the registered proxy
  dial". Any k-sweep or alternative density estimator would be a NEW
  registration.
- **QUARANTINE** ⇒ instrument audit before any use; a mismatch means the
  single-variable property failed (substrate changed under the pass),
  not that the claim is fragile.
- **SUBSTRATE-GONE** ⇒ no passes run; the recorded missing runs and the
  permanent single-k qualifier are the registered outcome.
- Regardless: the committed Stage-0 record and DECISION.md stand as
  written (this read re-adjudicates nothing at k=10); D0 stays frozen;
  Gate-D1 criteria and the 24+24 resource decisions are untouched.

## Commands

```bash
# 0) Existence gate (SRC = candidate archive root; resolve first).
#    Writes the registered gate record the read consumes (--gate_json):
#    verdict + per-run replay chunk counts + missing list.
export SRC=<archive root holding pretrain_p2e_* run dirs>
export OUT=$SRC/secondk_dumps; mkdir -p $OUT
python - <<'EOF'
import json, os
root = os.environ['SRC']
base = 'local_results/evpi_stage0_20260710_221924/raw'
missing, runs = [], {}
for dom in ('cup', 'finger'):
  for s in range(1, 6):
    run = f'pretrain_p2e_{dom}_seed{s}'
    idx = json.load(open(f'{base}/{run}/latent_uq/{dom}_v1/index.json'))
    for c in idx['checkpoints']:
      p = os.path.join(root, run, 'ckpt_snapshots', c['name'])
      os.path.isdir(p) or missing.append(p)
    rp = os.path.join(root, run, 'replay')
    chunks = (len([f for f in os.listdir(rp) if f.endswith('.npz')])
              if os.path.isdir(rp) else 0)
    runs[run] = dict(replay_chunks=chunks)
    chunks or missing.append(rp + ' (no replay chunks)')
rec = dict(verdict=('SUBSTRATE OK' if not missing else 'SUBSTRATE-GONE'),
           missing=missing, runs=runs)
out = os.path.join(os.environ['OUT'], 'secondk_substrate_gate.json')
json.dump(rec, open(out, 'w'), indent=2)
print(rec['verdict'], '->', out)
EOF
# SUBSTRATE-GONE => STOP here; the gate record + this prereg are the
# registered outcome (no passes, no read).

# 1) Frozen inputs to the box (probesets ~7.4 MB + the 10 index files):
rsync -av local_results/evpi_stage0_20260710_221924/probesets/ \
    $SRC/probesets_stage0/
rsync -av --include='*/' --include='index.json' --exclude='*' \
    local_results/evpi_stage0_20260710_221924/raw/ $SRC/committed_index/

# 2) The TWO smoke dumps (cup seed1, its earliest committed snapshot
#    only): first the k=10 guard CALIBRATION, then the k=5 smoke.
FIRST=$(python -c "import json;print(json.load(open('$SRC/committed_index/pretrain_p2e_cup_seed1/latent_uq/cup_v1/index.json'))['checkpoints'][0]['name'])")
for KV in 10:k10cal 5:k5; do
  K=${KV%%:*}; TAG=${KV##*:}
  python -m probing.latent_uq --probeset $SRC/probesets_stage0/cup_v1 \
      --run_logdir $SRC/pretrain_p2e_cup_seed1 \
      --checkpoints $SRC/pretrain_p2e_cup_seed1/ckpt_snapshots/$FIRST \
      --knn $K --seed 0 --ep_batch 64 --platform cuda \
      --output $OUT/pretrain_p2e_cup_seed1/latent_uq/cup_v1_${TAG}_smoke
done

# 2b) Guard calibration (machine-checked, value-blind; selects the
#     integrity-guard mode and refuses the wave if 'invalid'):
python -m analysis.stage0_secondk_read --smoke_check \
    --dumps_root $OUT \
    --committed_root local_results/evpi_stage0_20260710_221924/raw

# 3) Full passes (20 jobs, embarrassingly parallel per run x k):
for dom in cup finger; do for s in 1 2 3 4 5; do for K in 5 20; do
  RUN=pretrain_p2e_${dom}_seed${s}
  CKPTS=$(python -c "import json,os;print(' '.join(
      os.path.join('$SRC','$RUN','ckpt_snapshots',c['name']) for c in
      json.load(open('$SRC/committed_index/$RUN/latent_uq/${dom}_v1/index.json'))['checkpoints']))")
  python -m probing.latent_uq --probeset $SRC/probesets_stage0/${dom}_v1 \
      --run_logdir $SRC/$RUN --checkpoints $CKPTS \
      --knn $K --seed 0 --ep_batch 64 --platform cuda \
      --output $OUT/$RUN/latent_uq/${dom}_v1_k${K}
done; done; done

# 4) Sync $OUT back (incl. secondk_substrate_gate.json), then the ONE
#    read. A read invoked without --gate_json records substrate_gate:
#    null in the output json — an auditable omission; the registered
#    command passes it.
python -m analysis.stage0_secondk_read \
    --dumps_root <synced $OUT> \
    --committed_root local_results/evpi_stage0_20260710_221924/raw \
    --gate_json <synced $OUT>/secondk_substrate_gate.json \
    --output artifacts/stage0_secondk_<date>
```

## Costs

Existence gate + rsync: minutes. Dumper: pure inference; per checkpoint
≈ 2560 probe windows (40 steps) + 2048 reference windows (32 steps)
through the encoder + an 8-step open-loop roll ⇒ ~2–6 min per
checkpoint, **102 checkpoint-passes** (2 smokes + 20 runs × 5) ≈
**3–10 compute-hours total**, embarrassingly parallel per run × k
(20 jobs ⇒ well under an hour wall-clock on one GPU box; `--platform
cuda` pinned — a CPU run is possible but would predictably land in
guard mode 'fallback' or QUARANTINE at the k10 calibration, by design
not by accident). Reader: ~15–30 CPU-min (400 clustered-bootstrap rows
at n_boot=300). No training.

## Disclosure

Known at freeze: the complete committed Stage-0 record — all k=10
verdicts, Δ values, memos, DECISION.md, the phase-change shape — plus
the substrate facts above (anchors/reference encodings discarded at dump
time; probesets local; checkpoint/replay location unknown). Also known
at the freeze-commit date (31 Jul): the executed R3 reacher read
(REPLICATES, `artifacts/r3_reacher_20260731/`) — it bears on no
quantity in this registration (different instrument, different claim)
and is named here only so the known-record statement is true at commit
time. Unknown: every k ≠ 10 quantity — no k=5 or k=20 density value has
ever been computed on these runs (or anywhere on this stack); whether
the substrate still exists. Residual risks, registered honestly:
(1) the integrity guard covers the probe side via the array comparison
and the density REFERENCE side via the `ref_windows == 2048` pin PLUS
the k=10 calibration dump (which recomputes density against the live
replay and compares it to the committed values — a truncated-but-
2048-window replay shifts the k10 calibration densities and lands the
ladder in 'fallback' or 'invalid'); the existence-gate record
additionally stores per-run replay chunk counts, reported alongside the
read. (2) The committed local raw bundle predates the results-sync-v2
manifest policy; its integrity rests on the DECISION.md verification
(cup_p2e re-run from these raw dumps was bit-identical to the
cluster-side analysis). (3) The dumper prints a per-checkpoint
`density_knn mean` to stdout, so k2 mean densities appear in job logs
before the ONE read — the registered estimand is RANK-based (Δ of
partial Spearmans) and a mean density magnitude carries no information
about it; this stdout is registered as disclosed-informational, and
smoke stdout is consulted only for crash/dial sanity, never as
evidence. Ordering: this file + reader (+ the dumper's one-line
`ep_batch` meta patch) are committed before the existence gate runs;
the read executes exactly once.
