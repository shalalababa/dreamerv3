# E3 pre-submit check — 2026-07-11

Inputs: `local_results/e3_search_outputs_20260711_095108/` (searches + built
buffers, both domains). All builds verify: `occ_recomputed == occ_search`
exactly at every level/side, `n_transitions` = 200,200 everywhere,
`terminal_fraction` Δ = 0, action deltas within the family of the original
Axis-1 builds (mean ≤ 0.073, std ≤ 0.147).

## Verdicts

| build | verdict | detail |
|---|---|---|
| finger dose (d0–d3) | **GO as built** | occ 0.054/0.131/0.249/0.323, tight to targets (total dev 0.110), pairwise dcov 0.0078 ≤ tol 0.0108, overlap ≤ 0.115 |
| finger r1 | **GO as built** | docc 0.207 (target 0.270), dcov 0.0013, hi = random1/random5-mix (random2 excluded), hi-side overlap with original q1 hi: 31/200 |
| finger r2 | **GO as built** | docc 0.171, dcov 0.0001, hi = **pure random3** (200 eps), source_l1 = 1.0 vs its lo — a genuinely third composition |
| cup r1 | **GO as built — submit side1 only** | docc 0.139 (target 0.161), dcov 0.0140; **lo side = original q1 side0 exactly (200/200 members)** ⇒ reuse `adapt_ax1q1s0_cup_seed{1..5}` as the paired control; submit only `ax1r1s1` (5 jobs) |
| cup r2 | **REDO search** | selected the *identical* pair to r1 (hi contains neither random4 nor apt2, so both exclusion sets share one lexicographic optimum) — as built it adds no second composition |
| cup dose | **REDO search (cheap, deterministic)** | levels 0.060/0.121/0.191/0.199: top two nearly duplicate and level0 missed the available 0.0126 candidate. Diagnosis: the beam-40 shortlist at target 0.229 holds only low-coverage candidates, so every DFS completion from the true low-occ candidate (0.0126@0.8855) dies and the search falls back to a worse clique. A wider beam admits e.g. {0.013, 0.121, 0.173, ~0.20} (total dev ≈0.19 < 0.226 found). |

## Registered fixes (logged in DEVIATIONS.md before any E3 run executes)

1. **Cup dose re-search**: same frozen criteria, `--n_candidates 1200
   --beam 300` (seed 0, deterministic). Adopt the re-search iff
   `total_abs_dev` improves; rebuild. Finger may be re-run with the same
   parameters for uniformity — adopt per-domain by the same lower-dev rule.
2. **Cup r2 re-search**: exclusion widened until the pair differs from r1 —
   primary attempt `--exclude_sources random1 random2 random3 random4
   random5` (high side with *zero* random-policy episodes = strongest
   composition test); fallback if SHORT: `--exclude_sources random4 apt2
   random2 apt5 random3` (original + r1-hi dominants). Same
   `--target_docc 0.161`.
3. **Control-reuse rule**: where an r-pair side's member set is *identical*
   to an already-run buffer (cup r1 lo ≡ q1 s0), the existing adapt runs are
   the paired control (`fit_mixed_effects paired --cond_a ax1r1s1 --cond_b
   ax1q1s0`); the duplicate side is not re-run.

## Job count after fixes

finger dose 20 + finger r1 10 + finger r2 10 + cup dose 20 + cup r1 5
+ cup r2 5–10 (5 if its lo again ≡ q1 s0) = **70–75 jobs**.
