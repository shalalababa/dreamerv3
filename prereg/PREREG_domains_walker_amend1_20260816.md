# Amendment 1 to PREREG_domains_20260812 — walker leg fit-witness substitution (16 Aug 2026)

**Scope: two named cells only.** `ax1wm_walker_fq1s0_seed6` and
`ax1wm_walker_fq1s0_seed8`. Nothing else in the walker leg or the
registered analysis changes. Registered pre-outcome: written and dated
before the walker read executes; at freeze time no walker outcome
quantity (auc row, contrast, verdict) has been observed by the deciding
session — the collate was produced mechanically and not inspected.

## What happened

The registered read gates every fit on OFFLINE_FIT_PROGRESS
`update == total`. For the two named cells the bundle
(`domains_walker_20260816_152906`, sha-verified) carries:

- progress witnesses BOTH frozen at `updated_at=2026-08-14T20:47:40+0000`
  — the identical second, a sync-event signature, with update 176000 /
  171875 < 500000;
- `done`-marked checkpoints at exactly 500000, stamped 2026-08-15T14:06
  (seed6) and 2026-08-16T16:54 (seed8) — in-wave, POSTDATING the frozen
  witness by 17–44 h.

This is the documented pull-clobber artifact
(`artifacts/stale_fit_witness_20260816/RECORD.md`, ops commit 1816a7fd:
pass-2 pulls checksum-forced over witness filenames clobbered RCC
progress files with older instance copies; 63 dirs affected including
`walker_fq1s0`; 0 of 63 genuinely short). A fit that stopped at 176000
cannot write a done checkpoint at 500000; a checkpoint postdating its
witness is evidence about the sync, not the training.

## The substitution (10-Aug precedent)

Exactly as in PREREG_refit_reads_amend1/2_20260810 (fit-counter
preflight failed → checkpoint-step witness substituted): for the two
named cells the fit-completion witness is the **done-marked checkpoint
at step ≥ 500000** (already independently gated by the reader's
ckpt_steps check, which both cells PASS with authentic bundle bytes).
Implementation: the reader is frozen and is not edited; the
fit_counters input json carries `update = total = 500000` for the two
named cells, derived from the checkpoint-step fact and disclosed as
SUBSTITUTED-WITNESS entries here and in the read RECORD — they are not
authentic progress-file bytes (the authentic stale bytes are preserved
in the bundle). The other 62 walker-leg witness entries are authentic
and unmodified.

## What this does NOT do

- No walker outcome was observed before this freeze.
- No fabricated witness enters the record as if authentic — the
  substitution is labeled everywhere it appears.
- The 16-Aug refined standing rule applies: checkpoint-step witnesses
  are the authoritative completion fact; progress counters are
  sync-fragile in both directions.
