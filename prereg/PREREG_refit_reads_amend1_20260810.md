# AMENDMENT 1 to the three refit registrations — 2026-08-10

Amends: `PREREG_finger_refit_20260808.md`, `PREREG_cup_refit_20260808.md`,
`PREREG_rlsh_ownlabel_refit_20260808.md`. Written AFTER bundle delivery and
provenance/validity scans, BEFORE any outcome value (AUROC / NLL / R²) was
read — the scans below consumed only metadata: fit-counter files, checkpoint
directory names and timestamps, `checkpoint`/`witness_match`/`probeset_sha256`/
`expl_mode`/override-sidecar fields, and stratum **counts** (probeset label
properties). No decision rule is changed; this amendment records a registered
preflight FAILURE and the substituted witness, plus two structural-instrument
facts the readers must handle.

## 1. Preflight (d) fit-counter verification FAILED — witness substituted

Registered: "fit-counter verification at read time (realized-training
standing rule: OFFLINE_FIT_PROGRESS update == total)". Found:

- **finger**: all 32 `OFFLINE_FIT_PROGRESS` files carry PRE-LOSS dates
  (Jul 10 ×16, Jul 13 ×8, Jul 29–Aug 1 ×8) — an **archive-restore overlay**
  clobbered the fresh counters (old files rsync'd over the run dirs with
  preserved mtimes; `ckpt/latest` similarly points at old checkpoints in
  overlaid dirs). Counters are stale witnesses, not evidence of missing
  training.
- **cup**: 10/32 counters show update < total (137k–460k of 500k) while full
  fresh 500k checkpoint chains exist — a **duplicate truncated job** re-fired
  into the same run dir after the completing job (per-GPU-lane queue
  double-fire), rewriting early checkpoints, moving `latest` backwards, and
  leaving its own truncated counter as the last write.
- **rl/sh**: counters clean (32/32 update == total, fresh dates).

**Substituted witness (all three reads)**: the measure passes' recorded
`checkpoint` field. A consumed fit is VALID iff the checkpoint actually
loaded by BOTH its ridge and E4 passes is at step **500000**. Verified
before freeze of this amendment: 96/96 measure targets at step 500000.
Timing scan (config-write → final fresh ckpt: 7.5–17.7 h; no run shows the
~15-min resume-from-restored-450k signature) rules out silent partial
training behind the fresh 500k checkpoints.

## 2. Finger substrate is MIXED — 30 fresh refits + 2 pre-loss originals

`ax1wm_finger_fq1s0_seed6` and `ax1wm_finger_fq1s1_seed5` were measured on
their restored JULY original checkpoints (step 500000) — the axis1
idempotent-skip hole fired on the restored dirs, so no fresh fit was
trained there. These are genuine pre-loss original fits (better provenance
than a refit, but a different draw population). Handling, frozen here:
**primary analysis = all 32** (substrate column disclosed per run);
**registered sensitivity = fresh-only (n=30, apt arm 14 instead of 16)**.
A verdict is reported as substrate-robust only if primary and fresh-only
grades agree; disagreement is disclosed, primary stands, and the mixed
substrate is named as the reason.

## 3. Structural instrument facts (probeset geometry, not outcomes)

- `finger_v1`: out-of-regime rewarded frames = **0 exactly** (leak-free
  probeset) ⇒ `auroc_out_regime` is structurally undefined for every run.
  The #9 "both regimes" clause is evaluated on the two DEFINED legs —
  pooled AUROC and in-regime AUROC — at the registered bars (CI > 0.5 and
  point ≥ 0.70 on BOTH legs for the strong verdict); the out-regime leg is
  reported as structurally unevaluable. Disclosure: the in-regime leg has
  only 6 negative frames (low-powered; 5440 positives).
- `cup_v1`: in-regime rewarded frames = 0 (regime mask anti-aligned with
  reward) ⇒ in-regime AUROC undefined for cup. P-C3's registered quantity
  is the unqualified "OOF ridge reward-prediction score (AUROC primary)" =
  **pooled AUROC**, defined in both domains; the domain contrast Δ_dom uses
  it. No rule change.
- Headline ridge score = alpha 1e-3 (the instrument's registered headline;
  collate default). Other alphas reported descriptively.

## 4. Unconsumed files disclosed

The rl/sh bundle also contains `ridge_probe_finger_v1.json` per run (a
ridge pass not referenced by any registered rl/sh rule). It is NOT read by
the frozen reader and remains unconsumed; any later use requires its own
registration.

ONE read execution per registration still applies. Reader:
`analysis/cup_refit_read.py` (shared finger/cup, `--domain`) and
`analysis/rlsh_ownlabel_read.py`, selfchecked before execution.
