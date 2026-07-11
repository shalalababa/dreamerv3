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

---

## Redo check — 2026-07-11 (inputs: `local_results/e3_cup_redo_search_20260711_111744/`)

**Cup dose: ADOPTED (rule satisfied).** Re-search (beam 300) improved
total_abs_dev 0.2258 → 0.1567; rebuilt buffers verify (occ_recomputed ≡
occ_search, 200,200 transitions). Levels **0.061 / 0.117 / 0.230 / 0.232**
at cov 0.709–0.728 (pairwise dcov 0.0198 ≤ tol). The top two levels
duplicate at occ ≈0.23 — that is the cup frontier: a coverage-matched
4-clique cannot reach the 0.336 target (high-occ cup episodes are all
low-coverage). Kept as built: the replicate at the top dose serves as a
pure-error estimate; the registered analysis regresses on *measured*
occupancy, so unequal spacing is fine. GO — 20 jobs.

**Cup r2 (all-random exclusion): built pair is too weak — do not submit;
run the fallback search.** The primary attempt returned OK (9 valid pairs,
lo = pure p2e4 0.020@0.994, hi = pure apt5 0.068@0.978, exclusion honored,
sides distinct from r1) but with **docc 0.048 = 0.30× the 0.161 target and
only 1.18× the separation floor** (occ_sep_min 0.0405). At cup's outcome
noise (Q1 CI half-width ≈65 AUC at n=8, docc 0.161) a 5-seed contrast at
30% dose is uninformative — it would enter the paper as a dead cell.
**Amendment (registered here and in DEVIATIONS before any E3 outcome
exists): minimum-dose criterion for r-pairs — adopt a searched pair only if
docc ≥ 0.5 × target_docc.** The all-random pair fails it ⇒ run the
registered fallback exclusion on the cluster and rebuild r2:

    python -m probing.build_controlled_replay search-rpair \
      --index $RUNROOT/axis1_cup/episodes.json --ref_replay <cup REF> \
      --exclude_sources random4 apt2 random2 apt5 random3 \
      --target_docc 0.161 --n_candidates 1200 \
      --output $RUNROOT/axis1_cup/rpairs_r2.json

If the fallback also lands below 0.5×target, cup keeps r1 as its single
composition-robustness pair and the all-random pair is reported as a
descriptive frontier note (no adapt runs). The all-random search output is
retained at `rpairs_r2_redo.json` either way.
