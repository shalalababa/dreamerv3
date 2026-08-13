# AMENDMENT 1 to PREREG_tm2_bridge_20260812 — selfcheck-only repair — 2026-08-12

The registered cluster smoke
(`TDMPC2_ROOT=... python -m probing.tdmpc2_recon_fit --selfcheck`)
crashed before any leg ran: the two selfcheck-internal `build_cfg`
calls (the B7 reference-init replay and the official-class strict-load
leg) passed `tdmpc2_root=None`, and `build_cfg` has no env fallback
(`os.path.join(None, ...)` TypeError). The FIT path was never affected
— `fit()` passes `args.tdmpc2_root or os.environ.get('TDMPC2_ROOT')`.

Repair: both selfcheck call sites now pass
`os.environ.get('TDMPC2_ROOT')` (the env the registered command
already exports). Diff scope: two lines inside `selfcheck()` only; the
training/checkpoint code path is byte-identical to the freeze-commit
(7b51ad58). Zero bridge runs exist; the smoke gate has not yet passed;
no estimand, gate, or decision rule changes. Committed before any
bridge fit is submitted.
