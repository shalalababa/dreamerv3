# AMENDMENT 1 to PREREG_waveb_advd_20260822 — uniform convergence
extension of ALL 16 actors + full fresh probe panel, 23 Aug 2026
(rev 2: one reviewer, 3 BLOCKING / 7 MAJOR-MINOR findings applied
pre-freeze; rev 1 was never frozen or executed)

## What happened (first invocation, 23 Aug)

The ONE registered execution of `uncfield/se_advd_read.py` on the
verified `wb_advd_20260823_083129` bundle returned
NOT-ADJUDICABLE-INSTRUMENT-WEAK on the AMBIENT (carrier) wing: the
in-model dominance gate passed 6/8 vs the registered α-bar 7/8
(WB-B2). Per the carrier-keyed R-A1-M13 rule the execution was
PRESERVED (side file `se_advd_read_NOTADJ_20260823T091032.json`;
`se_advd_read.json` does not exist). The gate-failing runs are
se_advd_s10 (margin −11.9 %) and se_advd_s13 (−2.2 %). The plateau
descriptive ranks convergence heterogeneity across the panel
(s10 1.259 > s17 1.247 > s13 1.243 — i.e. the failers are NOT
cleanly "the two least-converged"; s17 passes with the largest
margin). The prereg named the in-model gate as the
under-trained-actor catch; this amendment is that catch firing.

## Amendment (registered BEFORE any new compute)

1. **Extension — ALL 16 adaptations, uniformly, to `STEPS=2e5`.**
   Every actor (8 ambient + 8 localized) continues from its OWN
   run dir for one additional 1e5-step leg (~1 GPU-h/leg realized;
   ~16 GPU-h total). UNIFORM DOSE is the anti-selection property
   (reviewer finding 2): extending only the gate-failers would
   selectively re-roll the wing's only negative delta — the single
   run that decides the primary — after its sign was seen. A
   uniform, outcome-independent dose to every actor cannot steer
   the gate or the primary in a preferred direction, and keeps the
   two wings dose-commensurable.
   Mechanics (verified in `embodied/run/train.py`): the producer
   re-runs with the staged-back FULL run dir (ckpt/ AND replay/
   required) — `from_checkpoint` regex-loads the frozen WM from the
   source first, then `cp.load_or_save()` restores the run's own
   full state (actor, critic, step, replay) over it, so the net
   effect is continued actor training on the bitwise-identical
   frozen WM. The read's bitwise WM-IDENTITY gate re-verifies this
   independently per pair. Producer change (registered here): a
   resume-guard block in `scripts/uncfield_advd.sbatch` that (i)
   refuses resume without a resolvable `ckpt/latest`, ≥150 replay
   chunks, or with a leftover `error` file (Replay.load otherwise
   falls back SILENTLY to an empty buffer; portal aborts opaquely
   on `error`), and (ii) removes the stale `config.yaml` so the
   BIND-CHECK reads the relaunch's config, not leg-1's (it would
   otherwise deterministically kill the run on
   `run.steps 1e5 != 2e5`). BIND-CHECK pins `run.steps == 2e5` via
   the existing STEPS var. Wave spec:
   `ops/waves/wb_advd_ext/spec.yaml` (16 runs, original wb_advd
   vars per run + STEPS=2e5; milestone gate at 200000 with
   tolerance ≥190000 — snapshot spacing is ~7.8k steps with no
   final save, so a 5k tolerance would false-fail ~35 % of
   completed runs; scores.jsonl floor 185 = 96 lines/1e5 measured,
   appended across resume).
2. **Fresh probe panel — ALL 16 pairs, ONE GPU job, ON RCC, pinned
   to panel-1's GPU model.** Panel 1 ran on RCC/Midway3 (probe
   jsons record /scratch/midway3 paths). Panel 2 therefore also
   runs on RCC, in ONE job on ONE device of the SAME GPU model as
   panel 1 (ops recovers panel-1's node/GPU from the
   `se_advd_probe_<jobid>.out` log / `sacct` and pins via the
   scheduler; the probe json cannot certify the device, so
   `scripts/se_advd_probe.sbatch` now logs
   `nvidia-smi name,driver_version` as the panel's device
   provenance). Every existing `se_advd/` probe output is
   QUARANTINED first (renamed `_SUPERSEDED_WBA1_se_advd_*`) — RCC-
   side by explicit rename in all 16 run-dir copies BEFORE any
   pull-back (this also closes the rsync --append-verify same-size
   shadowing hole), and job-side by `FRESH_PANEL=1`, which (a)
   quarantines any pre-existing output before the skip rule can
   see it, (b) refuses to run over a bundle/archive tree
   (COMPLETION.txt / manifest.sha256 guard), and (c) succeeds ONLY
   if all 16 pairs were freshly probed in THAT job (n_skip must be
   0 — a resubmission after a partial job must set FRESH_PANEL=1
   again; splicing jobs/devices is the failure the flag exists to
   prevent). Probe code, `--seed 0`, N_IMAG, and all registered
   probe params are UNCHANGED — changing the probe seed would be
   re-rolling for a better draw and is expressly NOT done.
3. **Re-read — the ONE preserved execution, bars UNCHANGED.**
   `se_advd_read.py` untouched: AMB_GATE_NEED stays 7/8, all
   cells, estimands, and gates unchanged. Consumption stays
   carrier-keyed. The EXACT registered invocation (pinned;
   reviewer finding 14):
   `python -m uncfield.se_advd_read
   --runs "<rcc_runroot>/se_advd_*"
   --output artifacts/wb_advd_read_20260823 --expect_steps 2e5`
   run on the verified `wb_advd_r2` bundle tree. `--expect_steps
   2e5` makes the standing fit-counter check discriminative
   natively: every run must show `config_steps == 2e5` and
   `ratio ≥ 0.95` (flag `OK`); any `TRUNCATION-SUSPECT` or
   `CONFIG-STEPS-MISMATCH` is a REAL fit failure (no pre-declared
   mismatch signature needed — rev-1's `SUSPECT (2 runs)`
   signature was not discriminative against a truncated
   extension and is superseded).
4. **What counts as a look (single-retry finality made exact).**
   A "look" = an execution in which the full panel loads:
   `excluded_runs == []`, 8 ambient + 8 localized records present.
   An execution that aborts on panel completeness/validity (a
   missing pair, a failed probe, a provenance gate) is an OPS
   FAILURE: it is repaired and re-run WITHOUT consuming the retry,
   and the repair may not touch bars, seeds, probe params, the
   dose, or which runs were extended. This amendment authorizes
   exactly ONE look beyond the preserved first invocation. If that
   look returns carrier <7/8, the ambient wing is
   **INSTRUMENT-WEAK-FINAL**: fork (b) takes effect automatically —
   Tier-2 unreachable under this prereg, no further Wave-B
   compute, the localized side-file result and the closed-gap
   mechanism finding kept as descriptive. No bar change, no second
   extension, no probe re-roll will be proposed.
5. **The LOCALIZED wing's second roll is registered, not free**
   (reviewer finding 10). Panel 2 recomputes the localized cell
   after its panel-1 value (LOCALIZED-CONTAINED-MATCHED,
   interaction +3.4e-5, p=.50) was seen. Registered handling: if
   the carrier adjudicates, the PANEL-2 localized cell is the
   consumed one; the panel-1 value is reported alongside it as an
   instrument-sensitivity row; any panel-1→panel-2 change is
   disclosed as a two-look quantity, never presented as a fresh
   finding.

## Value-aware disclosure

The side-file descriptive rows WERE seen before this amendment
(ambient Δ positive 7/8 with s10 the lone negative; closed
imag/real gap 16/16; localized interaction ≈0), and the reviewer
computed that the ambient primary currently sits at sign-flip
p=.328 and would move to p=.0039 if s10's delta alone crossed
zero — which is precisely why the extension dose is UNIFORM over
all 16 actors rather than targeted at the failers. Mitigations,
all registered here: uniform dose; whole-panel re-roll in one job
on the panel-1 device model with no selective retention; probe
params and bars unchanged; consumption gate-keyed; the exact-one-
look rule of §4. This disclosure is carried into the read artifact
and any downstream writing.

## Ops chain (execution order; the registered authority)

1. RCC: `mv se_advd _SUPERSEDED_WBA1_se_advd_<ts>` in ALL 16
   adapted run-dir copies. The verified bundle
   `wb_advd_20260823_083129` + its committed manifest are the
   panel-1 archive — NEVER touched (and FRESH_PANEL refuses
   bundle trees).
2. Stage to ONE instance: all 16 adapted run dirs FULL (ckpt/
   incl. the latest pointer + replay/ ≥150 chunks each) +
   `_wb_srcs` with all 16 source dirs.
3. `python scripts/wb_advd_preflight.py --srcroot <staged>` must
   return 16/16 OK (zero-GPU); the producer's resume guard then
   re-checks each run dir at launch.
4. Submit `wb_advd_ext` (16 runs, per_gpu 1; prune `generated/`
   first; read lane files back — assert non-empty cmd bodies).
   Canary: first resume must LOG PAST its restored step (~96k)
   before the rest are fed.
5. Pull continuously to RCC (extended ckpt/ incl. the latest
   pointer — the append-verify lesson — plus scores/metrics/
   config/TRAINING_DONE).
6. Panel on RCC, pinned to panel-1's GPU model: `SMOKE=1` first,
   then `FRESH_PANEL=1` full 16-pair job (RUNS_ROOT = the RCC
   runroot, SRCS_ROOT = the RCC source dirs, panel-1 paths).
7. Re-bundle ALL 16 run dirs from RCC as `wb_advd_r2` (the ext
   wave spec's own instance-side bundle is named `wb_advd_extlegs`
   and is NOT the read bundle), commit the sha256 manifest, hand
   the bundle name over for verify-then-read.

Compute: ~16 GPU-h training + ~1 GPU-h panel. Panel = RCC light
GPU (substrate lives there — a registered RCC reason).

## Rev 2.1 (23 Aug, pre-read; ops finding): leg-1 training-device
layout + single-lane extension

Ops verified (lane = i % nlanes, empirically checked against
generated lane files) that leg 1 ran the wb_advd declaration order
on 2 lanes of instance 16, so ALL 4 het actors trained on one card
and ALL 4 flat actors on the other; the 8 ambient actors split
4/4. Registered consequences, recorded BEFORE the re-read:

- **AMBIENT wing: unaffected.** Training cards are balanced 4/4
  across the ambient panel; the primary is a within-run contrast
  and the gate is per-run — no cross-run arm contrast exists for a
  card effect to align with.
- **LOCALIZED wing: training-device ≡ arm, PERMANENTLY.** The
  het-vs-flat interaction is exactly the contrast the card split
  aligns with, and the extension RESUMES the leg-1 actors, so the
  confound is carried in the actors themselves — it applies to the
  panel-1 sensitivity row AND the panel-2 consumed cell alike, and
  no panel or extension can remove it (only a full retrain would,
  not warranted for a side-filed descriptive cell). The magnitude
  of same-model/same-box/cross-card training drift has never been
  measured here (the 0.113-AUROC figure is cross-model and
  probe-side, not training-side). The localized cell therefore
  carries a standing device-confound caveat in the read artifact
  and all downstream writing; its per-run dominance gates (within-
  run) are unaffected.
- **Extension runs SINGLE-LANE** (user decision): all 16 legs on
  one GPU, ~16 h. This adds zero new device-arm alignment in leg 2
  and needs no reorder of the frozen runs list. (For the record:
  the runs list's period-4 interleave IS lane-pure-by-arm on a
  4-lane box under lane = i % nlanes — the list must not be run
  4-lane as declared.)

No estimand, bar, cell, gate, or reader changes; disclosure only.
