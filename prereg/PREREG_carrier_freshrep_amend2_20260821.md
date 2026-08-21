# Amendment 2 — carrier fresh replication: schema-drift normalization
# of the Amendment-1 config gate (21 Aug 2026)

**Applies to**: PREREG_carrier_freshrep_20260820.md (+ Amendment 1).
**Status**: dated pre-read amendment. Adjudicated BEFORE the ONE read,
from configs only — no estimand value has been seen (the builder exited
3 and the reader never ran; value-blindness intact).

## The finding (ops, 21 Aug — diagnosis verified)

All 160 fresh fit configs vs the four July Amendment-1 references:
**0 shared-key value differences, 0 removed keys, 23 added keys —
uniformly, in all 160, with 1 distinct key set across the wave**. The
added keys are codebase additions made after the July fits
(agent.frozen_enc, agent.model_obs, agent.recon_grad,
distractor.gate_*/mod_*, env.synth.*, planted.const_dim/source_key,
orthreward.task). The byte-compare therefore fails on SCHEMA drift,
not on any arm or dial difference. (Ops disclosure carried: two of
their three check scripts were buggy and are disregarded; the table
above comes from the explicit pair check + per-fit sweep only.)

## Why the additions are science-inert at their defaults

Every one of the 23 keys was introduced under this repo's additive-only
discipline with a reviewed "default path byte-identical /
default-inert" claim, and the repo defaults are exactly the
reproduce-old-behavior values: `agent.frozen_enc: False`,
`agent.recon_grad: True`, `agent.model_obs: '.*'` (documented as
reproducing the pre-key include-everything behavior),
`orthreward.task: ''` and `planted.source_key: ''` (wrappers not
applied), distractor gate/mod defaults bitwise-inert (their own
selfchecks), `env.synth.*` (a different env's declaration). A fresh fit
carrying these keys AT DEFAULT runs the identical science the July code
ran. The wave-internal identity gate (one config sha per (arm, side))
is unaffected by this amendment and already passes.

## The registered normalization (the d1_relabel_amend2 precedent)

The labeler chain already has the registered pattern for exactly this
situation: `d0/sweep._backfill_defaults` (PREREG_d1_relabel_amend2 —
"schema-drift compatibility: only ABSENT keys are added, saved values
always win"). Amendment 2 adopts it for the config gate:

**amend1_config_match := True iff, for every fresh fit vs its
(arm, side) July reference, after excluding the Amendment-1 M5
exception keys (seed, logdir):**
1. every SHARED key is value-identical;
2. NO reference key is absent from the fresh config;
3. every fresh-only (added) key's value equals the current repo
   default for that key (`dreamerv3/configs.yaml` defaults, flattened)
   — i.e. the addition is the schema's own backfill value, never a set
   dial.

The builder records, in the witness `_meta`: the full added-key list
with values, the defaults they were checked against, and the
comparison mode (`schema_normalized_v2`). Any added key whose value
differs from the repo default, any shared-key difference, or any
removal ⇒ False, exactly as before. The flag still cannot silently
pass: absent references, unparsable configs, or any non-default
addition all yield False.

## What this amendment does NOT do

No estimand, threshold, branch, look, or seed changes. The reader is
NOT edited (it gates on the boolean, as frozen). `--allow_problems` is
NOT the mechanism and would not work (the reader refuses on the flag
independently of the builder's exit code) — recorded to close that
path explicitly.

## Ops

Re-run `ops/waves/carrier_freshrep/build_read_bundle.sh` with the
updated builder (same references). Expected: match True iff the 23
additions are all at repo defaults; the added-key table lands in
`_meta` for the read RECORD.
