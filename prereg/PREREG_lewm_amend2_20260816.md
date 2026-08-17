# Amendment 2 to PREREG_lewm_20260812 — same-device pe anchor panel (16 Aug 2026)

**Scope: the pe-side E4 anchor csv pin, nothing else.** Registered
pre-outcome: at freeze time NO jp outcome quantity has been observed —
the LeWM read was deliberately HELD (ledger 08-16) when the bundle
landed carrying the historical pe csv; the reader has never been
executed on real inputs.

## Why

The frozen jp↔pe primary (P-J1-rep) is a per-seed paired contrast
[pe − jp] on rew_nll_in, with pe from the historical csv
(sha 9907d733…, device unrecoverable) and jp freshly measured on
RCC rtx6000. The 16-Aug 1-cell device diff priced the cross-measurement
term at −0.099 nats; the FULL 16-cell re-measure on rtx6000
(`pe_e4_remeasure_20260816_174813`, sha-verified, all cells on
midway3-0282 Quadro RTX 6000 per per-cell device sidecars, job
53421265) prices it at **mean −0.172 nats, per-cell sd 0.32, range
[−0.82, +0.26]** — not a clean offset but cell-level scatter, which
would enter the paired deltas as both bias and noise. A same-device
anchor removes the entire term by construction.

## The change

1. New pe anchor csv: `e4_fingerpx_v1_pe_rtx6000.csv`, produced by the
   REGISTERED collate (`python -m probing.stratified_error collate
   --probeset_id fingerpx_v1`) over the 16 re-measured summaries;
   sha256 `8bd7a68389a1f05cc78fc4b6f3b4be41fb1cdbfa348f305c8e8e1a86d7f8c112`.
2. `analysis/lewm_read.py` (frozen, never executed): `PE_E4_SHA256`
   constant swapped to the new sha; the selfcheck's own pin assertion
   (`startswith('9907d733')` → `startswith('8bd7a683')`) updated to
   match. No other reader line changes; selfcheck re-run required PASS.
3. The historical csv and pin remain archived (bundle
   `lewm_20260816_121536/collates/` + pe read artifacts) — the executed
   pe read's verdict was within-panel on its own device and is
   untouched.

## Direction disclosure (value-blindness)

The known shift is pe-side DOWNWARD (−0.172 mean), which makes
pe − jp deltas SMALLER, i.e. *disfavors* the JEPA-MORE-LEGIBLE fire
branch — the amendment is conservative with respect to the headline
prediction. jp values remain unobserved at freeze.
