# Anti-harvest mechanism wave (TM2 W2) — ONE registered read — 2026-08-14

- Registration: `prereg/PREREG_antiharvest_mech_20260811.md` (sha256
  5cb4b0050fbffbbbdd55ddada9b1d97f8c6000f7a96325d3d915829342a56787,
  freeze-committed e1ffe5ab on 2026-08-11, before any W2 pass ran;
  2026-08-11 batch reviewer applied pre-freeze).
- Bundle: `local_results/w2_mech_20260812_213627` — MANIFEST.sha256
  verified (78 entries, all OK; byte-identical to committed
  `manifests/w2_mech_20260812_213627.sha256`) before any file was
  opened. 32 cell files + 2 duplicate-null gate files, exactly the
  registered names.
- Reader: `analysis/w2_mech_read.py` (sha256 d8b66461…196d68),
  selfcheck PASS re-run immediately before execution. THIS is the
  wave's single read. All load gates passed: filename↔run_id,
  horizon/label_every/actions pins, per-state `iters_realized`
  witness, planner meta pins (mpc:true, iterations:6), dup-file meta
  + one dup per domain, duplicate-null exact-zero panels.

## Verdict (registered rules)

**REPLICATION-FAILED: in-wave anti-harvest absent at this power;
mechanism legs reported with ZERO decision weight.**

- **G0 (in-wave replication gate)**: harv_warm = g(warm) − gbar =
  **−0.128** [−0.354, +0.034], cluster sign-flip p = .217 (32 cells).
  Direction matches W1; the gate's CI∌0 ∧ p<.05 conjunction fails.
  Not sign-reversed. **Honesty note: the CI CONTAINS W1's executed
  −0.301** — this wave is compatible with both zero and the W1
  magnitude; the gate failure is an in-wave power event, not a
  contradiction. The W1 anti-harvest fact is untouched (it has its
  own read); this wave loses its premise, per the frozen rule.
- Mechanism legs (ZERO decision weight, reported as required):
  - P-N1 curse: **−0.794** [−1.175, −0.435], p = .0005; coverage
    27.1% retained (above the 25% floor; 29/32 cells contribute —
    finger is heavily G-degenerate, its per-domain curse panel is
    NaN). The sign is OPPOSITE the curse prediction: the deployed
    planner's action is model-UNDER-valued relative to ground truth
    on the retained states.
  - P-N2 pressure: slope over ln(iters) ≈ 0 (+0.003, p = .93);
    variant means g_plow 40.179 / g_cold 40.177 / g_phigh 40.190 —
    flat; prior-mode reference 40.324 sits ABOVE all planner
    variants (descriptive).
  - P-N3 inertia: +0.034 [−0.048, +0.123], p = .46 — null.
- Descriptives: pooled Spearman(q, g) over 9-action sets = **+0.020**
  (the ZC P-Z2 "Q carries no ranking signal" fact reproduced in-wave);
  per-domain harvest cup −0.036 / finger −0.219.
- Cross-stage coherence (registered sentence, no verdict): the ZC
  read's P-Z1 (planner-below-MODE, −0.261*) is descriptively echoed
  here — the un-optimized prior mode again outscores every planner
  variant — while the candidate-mean-relative harvest does not reach
  significance in-wave.

## What this licenses (per the frozen texts)

- **No mechanism wording.** The anti-harvest remains REAL (W1's own
  read) but UNATTRIBUTED: curse, pressure, and inertia all carry zero
  decision weight from this wave. Paper-2 text may cite the W2
  descriptives only as labeled, zero-weight panels.
- The mechanism program for the anti-harvest is now empirically
  closed on registered compute: ZC (planner-search-specific, Q-side
  null) + W2 (premise lost in-wave). Any further mechanism attempt is
  a new registration with a power analysis that the realized G0
  spread (cell sd ≈ 0.62 ⇒ MDE80 ≈ 0.31 at n=32 — right at the W1
  magnitude) would have to beat, e.g. by pooling more cells or more
  states per cell.

## Post-read notes (labeled, non-registered)

- The zero-weight P-N1 panel is the most interesting number: strongly
  model-UNDER-valued planner actions coexisting with (W1's)
  ground-truth-worse outcomes is the signature of a planner that
  wanders off the value landscape entirely (search pathology), not
  one that climbs a wrong value peak (curse). That is exactly the ZC
  read's "anti-harvest is PLANNER-SEARCH-SPECIFIC" reading, now with
  a value-side witness — but it carries zero registered weight and
  would need its own registration to be more than a footnote.
- Flat pressure (iters 1→12 changes nothing) plus mode-above-planner
  suggests the damage is done by the FIRST MPPI update, not
  accumulated optimization — a design hint for any future wave
  (iters=0 vs iters=1 contrast, i.e. mode-vs-single-update, would be
  the sharpest cell).
