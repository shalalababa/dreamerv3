# Routed-repair PAIRED certification (rrp2) — ONE read (24 Aug 2026)

Registration: PREREG_rrp2_20260823.md (frozen 2775e027 pre-run).
Bundle: rrp2_20260824_133000 (manifest OK; witness problems NONE;
path A — surviving carrier rgo fits, source FILL honored; invocation
pairing 8/8 seeds, all three arms per seed in one job; init-sha
equality + recipe pins + realized replay all clean; repo_commit
recorded).

## Verdict: **NO-CALL-UNDERPOWERED at E1** (registered branch) —
## with the wave's S1 instrument reinterpreting WHY

| quantity | value |
|---|---|
| repair′ / rca′ / rcb′ means | 295.4 / 303.1 / 323.5 — all within ~28 |
| E1′ paired [repair′ − rca′] | **−7.75 [−106.9, +122.2], p = .96** — point ≈ ZERO |
| E2′ (descriptive) | −28.2 [−122.8, +75.6], p = .63 |
| E3′ (2nd rgo phase) | +20.4 ns — worthless, third time |
| realized paired MDE80 | 175.1 (planning figure was 67) |
| **S1: repair′ − pinned original, SAME refits, same seeds** | per-seed −209…+193, **mean +3.9, sd 124.1** |

## What this settles

- **The invocation "shift" was never a batch offset — it is per-run
  noise.** S1 re-adapted the identical refits at the identical seeds
  and got shifts spanning ±200 with mean ≈ 0. The earlier ~+99
  cross-batch difference (rca 394.7 vs q1s1 295.5) was two draws from
  a per-run distribution with sd ≈ 124 — same GPU, same job, same
  seed does NOT cancel it. Seed-level pairing buys nothing here;
  that is why both control waves NO-CALLed. **Scope**: measured on
  high-performing (~300 AUC) unfrozen finger adapts; low-mean arms
  (crowding at ~85, paired sd 25) are far quieter — the noise scales
  with level.
- **The salvage question is now closed by measurement arithmetic,
  permanently.** All rgo-family pipelines (repair, fresh rgo,
  rgo→rgo) land at ~300 with per-run sd ~124; their pairwise
  differences (≲30) would need n ≈ 130+ seeds to resolve. No further
  wave will be proposed (resolving a sub-noise difference is not
  worth 10× the program's seed budget). The Paper-1 practice section
  ships: **"re-fit through the reward head rather than fine-tuning a
  sunk trunk (+201, paired, far above the noise floor); the init
  source among rgo-family pipelines makes no measurable difference
  (bounded only by the ~124-AUC per-run floor)."**
- **The original +201 refit-vs-finetune result is untouched and is
  the ONLY difference in this family that clears the measurement
  floor** — aptctl (~90) vs the rgo family (~300) is a 200-point gap
  against 124-point noise; everything else was never resolvable.
- Memory rule REVISED (supersedes the 23-Aug "cross-invocation
  shift" reading): unfrozen adapt AUCs at these budgets carry
  per-RUN noise sd ≈ 124 at high performance levels, irreducible by
  venue, batch, or seed pairing; only effects ≳ the floor are
  measurable at n=8.
