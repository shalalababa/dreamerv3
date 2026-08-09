# PREREG: cup WM regeneration wave (P-C3 adjudication substrate) — 2026-08-08

Registered after the confirmed loss of ALL fitted cup world models
(`ax1wm_cup_*` removed without archive; user + storage-agent verdict
2026-08-08). What SURVIVES, archive-verified: the `axis1_cup` buffer tree
(q1/side{0,1}, q2, dose, r1; 5.2 G), the cup collectors
(pretrain_{p2e,apt,random}_cup, 5 each), and `d1pilot_cup_e1`. **The
training data therefore restores byte-identical — this regeneration is
stronger than the rl/sh replication (`PREREG_rlsh_ownlabel_refit_20260808`):
identical buffers, identical objective, fresh network draws (Midway3 GPU
non-determinism, 1 Aug detprobe).** All archived cup verdicts (W2, E4
panels) are untouched; this wave exists to give review item #12 / R4's
restored **P-C3** (cup probe parity: cup apt features support reward
prediction ≈ cup task features, while finger maximally violates —
`PREREG_theory_predictions_20260717.md`) a checkpoint substrate, since
"adjudicate on existing cup checkpoints" is no longer executable.

## Design (fits + measure passes only — NO adapts)

- **32 fits** on the RESTORED `axis1_cup/q1` pair (sha-verified restore
  per the archive policy before any job): arms task
  (`AXIS1_EXPL_MODE=task` → `ax1wm_cup_q1s<side>_seed<k>`) and apt
  (reward-free, → `ax1wm_cup_fq1s<side>_seed<k>`), sides {0,1}, seeds
  1–8, 500K updates, `dmc_proprio`, protocol byte-identical to
  `PREREG_axis1_corrective_20260711.md` (same sbatch path, same config
  guard; `expl.mode` in saved config.yaml = audit anchor).
- **Naming**: original reserved names reused (cluster copies destroyed ⇒
  no collision); marker file `REFIT_20260808` in each run dir;
  `ax1wm_cup_*_seed*` sits in the cleanup DELETE tier — re-fit names are
  added to KEEP_PENDING in the same freeze-commit (KEEP wins, B4
  precedent).
- **Measure passes per fit, NEW output dirs, archived summaries never
  touched**: (1) `ridge_probe measure` on the `cup_v1` probeset (the
  P-C3 instrument, resolution-doc item #12); (2) E4 `stratified_error`
  h=0 pass to `e4_cup_v1refit` (anchors the ridge numbers to the NLL
  panel). Any collate goes to a NEW csv, never merged with archived rows.
- **Preflights (halt + dated amendment on failure)**: (a) archive restore
  sha + srcstat verification of `axis1_cup/q1` both sides; (b)
  `e4_probesets/cup_v1` exists on cluster — if gone, rebuild from the
  restored collector replays and verify the (replay_dir, picked-ordinal)
  manifest identity against the archived manifest before any measure
  (probeset construction is deterministic given the same replays; a
  mismatch is a HALT, not a silent re-pick); (c) glob check vs
  `runroot_cleanup.sh` with the KEEP_PENDING addition; (d) fit-counter
  verification at read time (realized-training standing rule).

## Registered decision rule (P-C3 adjudication; permutation-primary + BCa)

Let `R(fit)` be the ridge-probe OOF reward-prediction score (AUROC
primary, R² descriptive) on `cup_v1`, and define the domain contrast
`Δ_dom = mean R(task fits) − mean R(apt fits)` per domain (cup from this
wave; finger from the REGENERATED finger q1 fits — the originals were
destroyed in the same incident, full inventory 2026-08-08 late:
`PREREG_finger_refit_20260808`, same instrument, same probe protocol).

- **P-C3 FIRES (parity)**: cup Δ_dom CI (seed-cluster BCa, B=10K, rng 0)
  includes 0 with |Δ_dom| ≤ 0.10 AUROC, WHILE finger Δ_dom CI is
  entirely > 0 — the registered parity-with-finger-violation pattern.
- **P-C3 REFUTED**: cup Δ_dom CI entirely > 0 at a magnitude
  overlapping the finger contrast (no domain dissociation).
- **AMBIGUOUS (registered)**: any other pattern (e.g. finger contrast
  fails to materialize under the ridge instrument) ⇒ no P-C3 wording
  licensed either way; disclosed as instrument-limited.
- Replication rider (descriptive, no verdict): the re-fit cup E4 NLL
  panel vs the archived 14-Jul cup panel values — a population-level
  consistency check of the regenerated fits; divergence is disclosed,
  never silently absorbed.

Known at freeze: all archived cup E4/W2 values (fits destroyed); the
finger-side ridge passes are queued but UNEXECUTED (no ridge number in
either domain exists yet — the parity contrast is fully prospective).
Unknown: every ridge value and every re-fit value.

Reader: rules above are binding; implemented either in
`analysis/rlsh_ownlabel_read.py`'s sibling or a dedicated
`analysis/cup_refit_read.py`, frozen + selfchecked BEFORE the read
(two-stage pattern). ONE read execution. No reviewer (rides the
just-reviewed ridge_probe + cleanup-precedence instruments; single
two-CI decision rule) — skip disclosed per standing policy.
