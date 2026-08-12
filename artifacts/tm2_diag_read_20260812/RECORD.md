# TM2 legibility diagnostics — ONE registered read — 2026-08-12

- Registration: `prereg/PREREG_tm2_diag_20260811.md`
  (sha256 e3f58a0ea86419411456460335904dead4825fa10ef6bbcd8cdc6b1d13e078bc,
  freeze-committed 8cd62845 on 2026-08-11, before any probe pass ran)
  + `prereg/PREREG_tm2_diag_amend1_20260812.md`
  (sha256 ad7a41ef90e8079e25d3883bd73ae54bbe9cc8e513d910ae30b235ac9ba9a6e1,
  committed b93c1e25 — the mid-wave inner-split redraw repair, zero
  estimands observed pre-repair).
- Bundle: `local_results/tm2_diag_20260812_162518` — MANIFEST.sha256
  verified (68 entries, all OK) before any json was opened. Exactly 32
  probe jsons (aware/free × s0/s1 × seeds 1–8), no duplicates.
- Reader: `analysis/tm2_diag_read.py`, selfcheck PASS re-run
  immediately before execution. THIS is the wave's single read.

## Amendment-1 provenance (disclosed)

- 8 jsons are PRE-amendment (`aware_s0_*`, no `inner_redraws` field);
  24 are POST-amendment. Validity of the mixed bundle rests on the
  amendment's RNG-stream identity property, WITNESSED LIVE here two
  ways: (a) all 8 post-amendment `free_s0` runs — the same side1 probe
  data the pre-amendment aware_s0 runs consumed — realized
  `inner_redraws == [0,0,0,0]` (the amended code path is byte-identical
  where no redraw fires); (b) all 16 s1-fit runs realized the identical
  redraw pattern `[1,0,0,3]` (redraws depend only on the shared side0
  labels — cross-seed identity is the expected signature, and its
  uniformity is an instrument-sanity witness).

## Verdict (registered rule, thresholds on gap_TM2 ratios)

**PREMISE-CONSISTENT** — ratio = gap_TM2 / gap_dv3 =
0.3365 / 0.53370 = **0.6305**, in the registered [0.5, 1.0) tier.

- gap_TM2 (side-balanced, the registered numerator) = **+0.3365**;
  pooled house CI [0.244, 0.427] (context, not decisional; point rule
  governs) — the gap is real AND its CI sits entirely below the dv3
  anchor 0.5337.
- Arm means (fold-centered pooled-OOF AUROC, cross-side episodes):
  aware **0.9993** (16/16 ≥ 0.997 — TD-MPC2's reward/TD-carried
  encoder is near-perfectly reward-legible, vs dv3 task 0.93477);
  free **0.6629**.
- Per-fit exclusions: none (no degenerate-label AUROCs; ≥6-per-arm
  gate trivially satisfied at 16/16).
- Side structure (registered side-balance motive, realized): side-0
  gap 0.152 vs side-1 gap 0.521. The free arm is strongly
  probe-side-dependent: free fits probed on the hi-occupancy side1
  episodes score 0.78–0.88; probed on the sparse side0 episodes they
  score 0.43–0.54 (≈ chance). The cross-side design confounds fit-side
  with probe-data-side within each cell — exactly why the registered
  scalar balances sides.

## What this licenses (per the frozen texts)

- **P-F1 premise check (R1 addendum, "runs first"): PASSES.** The TM2
  free arm's reward-legibility sits closer to its aware arm than
  Dreamer apt (0.401, below chance) sits to Dreamer task (0.935) —
  and, more sharply, the free arm is ABOVE chance on balanced probing
  while dv3 apt is below it. The mechanism premise of R1's attenuation
  prediction holds; the powered P-F1 contrast (now carried by
  `PREREG_tm2_bridge_20260812`, frozen 7b51ad58 BEFORE this read —
  timing consistent with that prereg's disclosure) proceeds as
  registered, with no re-registration (that path is reserved for
  PREMISE-VIOLATED / PREMISE-INVERTED).
- No behavioral claim, no family-scope wording change: this is a
  representation-level premise fact.

## Post-read notes (labeled, non-registered)

- The aware ceiling (0.999 vs dv3 task 0.935) and the free arm's
  above-chance floor are BOTH mechanism-coherent with R1's account of
  the family boundary: TD-MPC2's objective keeps reward-relevant
  support legible even in its weakest arm, compressing the behavioral
  contrast the bridge wave now tests causally (P-F2: does ADDING
  reconstruction re-open it?).
- The free-arm side asymmetry (chance-level on sparse side0 probing)
  is a descriptive lead only — n_data = 2 sides; any use would need
  its own registration.
