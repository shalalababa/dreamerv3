# PREREG: finger q1 main-arm regeneration wave (#9 substrate) — 2026-08-08

Registered after the full loss inventory (user + storage agent,
2026-08-08 late): the finger q1 MAIN ARMS — `ax1wm_finger_q1s{0,1}`
(task) and `ax1wm_finger_fq1s{0,1}` (apt), ALL seeds — are destroyed
with no archive copy. The `axis1_finger` buffer tree was NEVER lost
(still on RCC), so regeneration runs on the ORIGINAL buffers —
byte-identical data, fresh network draws (Midway3 GPU non-determinism).
All archived verdicts (P0/W0/W1/E4 panels, every bundle and manifest)
are intact and untouched; per the inventory, the losses cost only future
work on old substrate. This wave regenerates the substrate that two
queued review-response measure passes need:

- **#9 apt-trunk ridge probe** (review D-series centerpiece:
  present-but-unusable vs absent on the apt trunk) — needs apt fits;
- **the finger side of P-C3** (`PREREG_cup_refit_20260808` — cup parity
  vs finger violation) — needs the task-vs-apt finger ridge contrast.

## Design (fits + measure passes only — NO adapts)

- **32 fits** on the on-RCC `axis1_finger/q1` pair: arms task
  (`AXIS1_EXPL_MODE=task` → `ax1wm_finger_q1s<side>_seed<k>`) and apt
  (reward-free → `ax1wm_finger_fq1s<side>_seed<k>`), sides {0,1}, seeds
  1–8, 500K updates, `dmc_proprio`, protocol byte-identical to
  `PREREG_axis1_corrective_20260711.md` (same sbatch path, same
  `AXIS1_EXPL_MODE` config guard; saved config.yaml = audit anchor).
- **Naming**: original names reused (cluster copies destroyed ⇒ no
  collision); marker file `REFIT_20260808` per run dir; the names appear
  in a delete-eligible cleanup list ("U1 read verified") — re-fit globs
  are added to KEEP_PENDING in the same freeze-commit (KEEP wins, B4
  precedent).
- **Measure passes per fit, NEW output dirs, archived summaries never
  touched**: (1) `ridge_probe measure` on `finger_v1` (#9 + P-C3 finger
  side; `--holdout`-era probeset unchanged — the D1 leak facts and their
  sensitivity analyses carry to regenerated fits only via the registered
  leak-exclusion rule, disclosed in any output); (2) E4
  `stratified_error` h=0 pass to `e4_finger_v1refit`. Collates go to a
  NEW csv, never merged with archived rows. rgo/sgb ridge anchors come
  from the ARCHIVE-RESTORED seeds 1–8 fits (genuine recovery,
  byte-identical originals — no re-fit needed for those arms).
- **Preflights (halt + dated amendment on failure)**: (a) q1 chunk
  availability (chunkgap rule); (b) glob check vs `runroot_cleanup.sh`
  with the KEEP_PENDING addition; (c) rgo/sgb seeds 1–8 archive restore
  sha+srcstat verify (anchors); (d) fit-counter verification at read
  time (realized-training standing rule).

## Registered decision rules

This wave is SUBSTRATE regeneration: its own outputs are consumed by
already-registered instruments, and it introduces exactly one new
decisional quantity:

- **#9 apt-trunk probe (primary, permutation + BCa, B=10K, rng 0)**: the
  apt fits' OOF ridge AUROC for reward prediction on `finger_v1`,
  in-regime vs out-of-regime frames. Graded (frozen here):
  - AUROC CI entirely > 0.5 both regimes at ≥ 0.70 point ⇒
    **PRESENT-BUT-UNUSABLE established** (linear reward information in
    the apt trunk despite the behavioral null + high rew-NLL — the
    strongest form of inclusion ≠ usefulness, feeds the dual-lead
    mechanism section).
  - AUROC CI including 0.5 (either regime) ⇒ **ABSENT-OR-NONLINEAR**:
    the apt trunk carries no linearly-decodable reward signal — the
    legibility account shifts from "present but illegible to the
    objective" toward "not linearly present at all"; wording change
    registered, no verdict flip (behavioral nulls already archived).
  - Task fits are the calibration arm (expected high AUROC; if TASK
    fails the probe, the instrument is declared uninformative — no #9
    wording licensed either way).
- **P-C3 finger side**: consumed by the frozen rule in
  `PREREG_cup_refit_20260808` (finger Δ_dom = task − apt mean AUROC,
  seed-cluster BCa). No number exists in either domain — fully
  prospective.
- **Replication rider (descriptive, no verdict)**: regenerated-fit E4
  NLL panel vs the archived q1 panel values (population consistency of
  regeneration; divergence disclosed, never absorbed).

Known at freeze: all archived q1 behavioral/E4 values (fits destroyed);
no ridge value exists for any arm in any domain. Unknown: every
regenerated-fit and ridge number.

Reader: frozen + selfchecked BEFORE the read (two-stage pattern; may
share `analysis/cup_refit_read.py`'s implementation with a domain flag —
the cup prereg's rule is unchanged either way). ONE read execution. No
reviewer (rides the just-reviewed ridge_probe instrument + the
twice-used regeneration pattern; skip disclosed per standing policy).
