# σ-ladder (B2 crowding-account discriminator) — ONE registered read — 2026-08-15

- Registration: `prereg/PREREG_sigma_ladder_20260814.md` (sha256
  fe2af9a383b7a6f3c02db47b881f9575b032f6df3522813c9c7880f5305324eb,
  freeze-committed d5390c4a 2026-08-14 13:32 — before any σ∈{2,8}
  fit existed; ONE-Opus sequential review applied pre-freeze,
  2 BLOCKING + 3 MAJOR + 8 minor). Account-(b) prediction FROZEN in
  the prereg before outcome: normalized task−apt drop contrast ≈0
  below the knee, POSITIVE past it.
- Bundle: `local_results/sgl_20260815_123342` — MANIFEST.sha256
  verified (72 entries, all OK; identical to committed manifest)
  before any file was opened. Provenance conformance (disclosed in
  bundle NOTES, per the prereg's conformance clause): wave ran on
  three Vast instances, not Slurm; `sgl_runs_manifest.csv` =
  concatenation of the instances' runs.csv (schema unchanged, REPLAY
  at index 6), dup-checked, seed99 smoke excluded by the
  seed-anchored regex. `sgl_replay_linkage.json` present, 32/32
  entries, every path verified `q1_nzs<σ>/side<side>`-consistent
  with its run name (M5 gate).
- Reader: `analysis/sigma_ladder_read.py` (sha256 727ed311…d9eaeb,
  byte-identical to the freeze-commit), selfcheck PASS re-run
  immediately before execution. THIS is the wave's single read. All
  gates passed: 32-name dual witness, config gates (dim/basesd/theta,
  expl mode), replay manifests (σ, dims, 16/side, tool, source
  equality, holdout-set equality, **B2-era holdout sha MATCHES the
  pin** — the pinned baselines' probe distribution is intact),
  ridge-json gates, exact-32 inventory.

## Verdict (registered rules)

**EXCESS-TASK-LOSS at BOTH new loads — the
competition-with-swamped-rescue account is supported;
proportional-compression is rejected at σ = 2 and σ = 8.**

- **σ=2 (nzs2, INFORMATIVE)**: task 0.754 / apt 0.708; normalized
  drops **0.385 vs 0.046**; contrast **+0.339** [+0.134, +0.457],
  perm p = .0016 (BH thr .05).
- **σ=8 (nzs8, INFORMATIVE)**: task 0.679 / apt 0.675; normalized
  drops **0.568 vs 0.197**; contrast **+0.371** [+0.239, +0.513],
  perm p = .0006 (BH thr .025).
- Both loads fired via the statistical route (no collapse branches
  involved; every raw mean comfortably above the 0.55 floor).
  Realized per-load MDE80 ≈ 0.24/0.21 — both observed contrasts
  exceed their MDE; this is a powered, double-confirmed fire.
- **P-S2 curve (descriptive)**: S(σ) over {1, 2, 4, 8} =
  {0, +0.140, +0.163, +0.153} (raw-drop units). The excess loss is
  ALREADY near-fully expressed at σ=2 — **the knee sits below σ=2**
  — and the task arm's raw legibility then flattens (0.754 → 0.679 →
  0.679 across σ 2→4→8) while apt declines gently (0.708 → 0.681 →
  0.675). Sharpest single fact: at σ=2 the task arm has surrendered
  38.5% of its reward-legibility headroom while the reward-free arm
  has lost 4.6%.

## What this licenses (per the frozen consequence mapping)

- **The B2 NEGATIVE-DID anomaly now has a registered
  competition-mechanism account**: the spectral model in the
  g-swamped regime — reward-relevant support is low-variance, so a
  rising noise floor excludes it FIRST despite the a² rescue —
  **predicted in the frozen prereg, then confirmed at both fresh
  loads**. Paper wording may claim the mechanism, citing this wave
  as its predicted test; R1's scope note is written from this
  account (the load×arm interaction R1-as-written missed is the
  spectral competition term acting on low-variance reward features).
- This wave does NOT reopen the Part-B λ adjudication (complete and
  split, as frozen). What it adjudicates is the ACCOUNT of B2's
  anomaly — and it lands on the competition side, which means the
  theory ledger's λ-side now carries: two registered λ-negatives in
  the ORIGINAL direction (breadth, B2-as-registered) AND one
  registered competition-mechanism confirmation in the REVERSED
  regime (this wave). The honest synthesis: competition exists and
  binds, but against the low-variance reward-relevant features — the
  opposite victim from the original λ prediction.
- The parked conditional (14 Aug) now TRIGGERS: B1′ capacity-descent
  at ~600k becomes worth registering as the capacity-side convergent
  test. GO decision is the user's.

## Post-read notes (labeled, non-registered)

- The knee-below-σ2 localization plus B2's dose (σ 1→4) means the
  original B2 wave sampled almost entirely PAST the knee — its
  "NEGATIVE-DID" was the saturated form of this curve. The two waves
  compose into one clean story: crowding excludes reward-legible
  support early (by σ≈2 in obs units, ≈1.6× in symlog loss-sd
  units), then both arms drift toward a shared floor.
- The apt arm's small-but-real normalized loss at σ=8 (0.197)
  says the reward-free arm's support is not immune, just
  higher-variance — consistent with the spectral account's ordering,
  not with a task-specific artifact.
