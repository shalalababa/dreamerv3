# Axis-1 pair-search evaluation + Gate-0 caveat recheck — 2026-07-06

Inputs (produced on Midway3, copied to
`local_results/runroot_snapshot_20260705_101044/runroot_light/axis1_{cup,finger}/`):
`episodes.json` (index over 15 Phase-4 pretrain replays + 4 Gate-0 pilot
replays per domain, modal episode length 1001) and `pairs.json`
(`build_controlled_replay search`, n_episodes=200/side, frozen criteria:
cov_match_frac=0.05, occ_sep_mult=3.0, max_overlap=0.2, knn=12, seed=0).

## Verdict: GO for both domains, all four builds

| domain | quadrant | Δcov (tol) | Δocc (sep. threshold) | overlap | feasible pairs |
|---|---|---|---|---|---|
| cup | Q1 cov-matched / occ-split | 0.013 (≤ 0.021) | **0.161** (≥ 0.041; ≈12× median boot noise) | 5% | 2,037 |
| cup | Q2 occ-matched / cov-split | **0.325** | 0.038 (< 0.041) | 0.5% | 15,476 |
| finger | Q1 | 0.008 (≤ 0.011) | **0.270** (≥ 0.061; ≈11× noise) | 7.5% | 7,297 |
| finger | Q2 | **0.164** | 0.011 (< 0.061) | 0% | 20,398 |

Both `pairs.json` decisions: OK. Selected pair = the search's argmax under the
frozen criteria (no post-hoc pair swapping).

## Gate-0 decision caveats (GATE0_DECISION.md) — both closed

1. **"Re-verify occ-matching at Phase-6 buffer size."** Closed: this search
   ran at the real Phase-6 granularity — whole 1001-frame episodes, 200 per
   side ≈ 200K transitions per buffer — not the Gate-0 window-compose scale.
   Occupancy separation (Q1) and matching (Q2) both hold with margin.
2. **"High-occ strata thin in pilots; Phase 4 scales them ~25×."** Closed:
   high-occ episodes went 3 → 59 (cup, ~20×) and 20 → 203 (finger, ~10×)
   once Phase-4 replays entered the index. Occ-bin strata counts
   (zero / mid / high, occ_bin thresholds 1e-6 / 0.5):
   - cup: 310 / 7,327 / 59 of 7,696 episodes (high mostly from random seeds
     11–12 each; apt3 has 2; p2e none — cup explorers rarely dwell in-regime).
   - finger: 3,712 / 3,797 / 203 of 7,712 (high from random 21–28 each and
     pilot_goal 17; every source has some).

## Residual imbalance to report at build time (runbook Phase 6 item 2)

- **Collection-policy composition differs strongly between sides**
  (source_l1 0.90–1.00): e.g. cup Q1 side0 is pure p2e2 episodes while side1
  is a broad mixture (random4/apt2/p2e1/pilots). This is inherent — moving
  occupancy at fixed coverage *is* a composition change — and is the
  "where possible" item of the checklist, not a violation. Action-stat and
  terminal-fraction deltas land in each build's `manifest.json`
  (`confound_deltas`); inspect them before submitting the adapt grid.
- Q1 sides share a few episodes (cup 10/200, finger 15/200; ≤ max_overlap).
  They are kept on both sides; shared content only dilutes the contrast
  (conservative).
- Q1 Δocc (0.16 / 0.27) is smaller than the Gate-0 compose values
  (0.575 / 0.668) because whole-episode selection averages occupancy within
  episodes, where Gate-0 composed cherry-picked windows. The achieved dose is
  still ≈ half the natural explorer↔goal occupancy range and ≥11× the
  bootstrap noise floor.

## Critics (VSA prerequisite, PREREG §4) — both usable

- cup: held-out R² = 0.841 (gate 0.2) → VSA primary.
- finger: held-out R² = 0.243 (gate 0.2) → VSA primary, marginal — report
  retdec alongside as pre-registered.
- walker: no critic by design (retdec substitution, PREREG §4 domain notes).

## Next actions

1. Build the four buffer pairs (login node, CPU, minutes each):
   `python -m probing.build_controlled_replay build --index
   $RUNROOT/axis1_<dom>/episodes.json --pairs $RUNROOT/axis1_<dom>/pairs.json
   --which <q> --output_root $RUNROOT/axis1_<dom>/<q>` for
   (dom, q) ∈ {cup, finger} × {q1, q2}; then eyeball each
   `manifest.json` `confound_deltas`.
2. `./scripts/submit_all.sh axis1` — 64 jobs (2 domains × 2 quadrants ×
   2 sides × 8 paired seeds), each = offline WM fit (500K updates,
   gradient-count equalized to the 500K-step pretrains) + frozen-readout
   adapt (1.25e5 steps). Run ids `adapt_ax1<q>s<side>_<dom>_seed<k>_ckpt500000`
   feed `analysis/adaptation_auc.py` unchanged and the Phase-6 paired
   contrasts (`fit_mixed_effects paired --domain <dom> --cond_a ax1q1s1
   --cond_b ax1q1s0`); they are excluded from the dose-response models by
   the `--modes` filter (see analysis/DEVIATIONS.md code notes).
