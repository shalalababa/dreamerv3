# B1 dose-ladder registered read — 22 Aug 2026

**Prereg:** `PREREG_nfi_dose_20260821.md` (review #28). ONE
executed read of the frozen `se_dose_read` ON RCC. (A first
invocation REFUSED at the final-checkpoint gate — se_dose10_s53's
probe was a stale-ckpt output the resumable driver had skip-kept;
quarantined, re-probed on caslake (job 54274457, frozen defaults,
probe==latest verified), then this single completed execution. No
outcome was produced or seen by the refused attempt.)
Bundle `b1_dose_20260822_101552` verified (32,966 files); 16/16
valid, floored_runs=[], collapsed_runs=[], instrument CLEAN,
degeneracy gate clean vs the Stage-1 reference.

## Outcome: **DOSE-LAW-NOT-CONFIRMED** — and the registered
prediction is not just unconfirmed, it is directionally reversed
with structure

- **Primary:** Spearman ρ(θ₁, scale) = **+0.728**, one-sided
  NEGATIVE label permutation p = 0.9992 (100,000 draws, seed
  20260821), fires=False, n=16.
- **The registered descriptive curve (per-dose θ₁):**

  | scale | per-run θ₁ | mean |
  |---|---|---|
  | 0.10 | 0.150, 0.113, 0.097, 0.091 | **0.11** |
  | 0.25 | 1.52, 0.130, 0.093, 1.55 | **0.83 (bimodal)** |
  | 0.50 | 9.24, 8.44, 7.47, 8.49 | **8.41** |
  | 0.75 | 6.76, 6.54, 7.18, 5.75 | **6.56** |
  | anchors (cross-wave, descriptive) | 0.359 → ~7.1 · 1.0 → 5.26 | |

  Below scale ~0.25 the channel is UNDER-priced (θ₁ ≈ 0.1); at 0.25
  the panel is bimodal (two runs off, two near parity — a stochastic
  onset); by 0.5 the misprice PEAKS above the full-scale anchor;
  above 0.5 it declines gently toward 5.26. **The shape is a
  threshold-onset hump, not a monotone law.**
- **Cost/coverage curve (registered descriptive, no fire):**
  Spearman(coverage proxy, scale) = −0.728.

## Registered consequences

1. The 17-Aug post-hoc observation ("relative misprice grows as
   amplitude falls", 12 runs, review-#25 rec-2 provenance) is
   **demoted to wave-specific** per the outcome map — it sampled
   only the DESCENDING limb (0.359 vs 1.0) of a non-monotone curve;
   extrapolating it downward was exactly the unregistered step this
   wave existed to test, and it fails.
2. **[writing chat] Paper 5's drafted one-sentence dose observation
   must be reworded**: as written it is contradicted below scale
   ~0.36. The licensed replacement is the onset-hump description
   with this read's registered-descriptive curve (or dropping the
   sentence).
3. The taxonomy keeps the curve descriptively; the DOSE AXIS
   becomes an ONSET AXIS. Follow-up seed (unregistered, noted): an
   onset ladder over 0.25–0.50 with more seeds — the bimodality at
   0.25 (2-on/2-off) is a seed-level phase transition, and the
   candidate mechanism race (decoded disagreement vs per-dim
   normalizer) now has a sharp target.

Stale-ckpt panel disclosure as in the A1 record (6 B1 cells
repointed pre-read; probed checkpoints 496k–499k; table:
`$RUNROOT/CKPT_LATEST_REPOINT_20260822.md`).
