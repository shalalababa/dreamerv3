# P0 Axis-1 reward-free corrective refit — registered read (2026-07-13)

**Protocol:** `prereg/PREREG_axis1_corrective_20260711.md` +
`prereg/PREREG_axis1_corrective_amendment1_20260711.md` (both frozen before
any corrective outcome existed).
**Data:** `local_results/p0_axis1_corrective_20260713_151650/runroot_light`
— 64 WM fits (`ax1wm_<dom>_f<q>s<side>_seed<k>`, apt, 500K updates, same
four materialized buffer pairs and seeds 1–8 as the 10-Jul reward-aware arm)
+ 64 frozen-readout adapt runs (`adapt_ax1f*`, 125K steps, task mode by
design). Analysis: frozen `analysis/adaptation_auc.py` →
`analysis/fit_mixed_effects paired` (cluster bootstrap B=10,000, seed 0).

## Audit (registered per-run rule) — PASS 64/64

- Saved `config.yaml: agent.expl.mode == apt` in all 64 WM fits.
- Offline-fit loss lines in all 64 job logs contain only observation-decoder
  + dynamics losses (finger: `con,dyn,position,rep,velocity`; cup:
  `con,dyn,position,velocity,dist_to_target,target_position,touch,rep`),
  **no `rew`/`value`/`repval` term**.
  [Correction 2026-07-13, W0 read: the finger/cup key lists above are
  swapped — finger turn_hard logs carry the 8-component set including
  `touch/target_position/dist_to_target`. The audited component *sets* and
  the no-`rew`/`value` verification are unaffected.] (Config-file `reward_grad`/
  `loss_scales.rew` defaults are present but inert on the `reward_free=True`
  code path — established by the 11-Jul smoke test,
  `artifacts/smoke_axis1_expl_20260711/`.)
- QC: 64/64 adapt runs pass the ≥20-episodes-in-100K rule; no exclusions.

## Registered decision (Amendment 1 §A: CI-only, AUC₁₀₀ₖ)

| Contrast (s1−s0) | Role | mean Δ | 95% CI (B=10K) | pos | read |
|---|---|---|---|---|---|
| **finger Q1** (occ) | **headline confirmatory** | **+2.04** | **[−22.04, +24.16]** | 5/8 | **NULL** |
| cup Q2 (cov) | secondary | −70.23 | [−343.85, +212.74] | 3/8 | null (direction: low-cov hurts) |
| cup Q1 (occ) | specificity | +151.04 | [−17.75, +327.88] | 7/8 | null, directionally + |
| finger Q2 (cov) | specificity | +17.85 | [−3.29, +40.42] | 6/8 | null |

**⇒ The frozen decision tree's second branch fires:** the occupancy-rescue
claim is **unsupported as reward-free**. Per the registered branch, the
paper pivots to (i) the observational sign reversal under intervention and
(ii) the occupancy × reward-supervision interaction; E3 as designed
(reward-free occupancy dose) is deprioritized.

Robustness windows (AUC₅₀ₖ/AUC₁₂₅ₖ; final10 descriptive): finger Q1 null at
every window (final10 +31.6 [−45.6, +122.4]). Cup Q1 grows with window and
final10 excludes zero (+322 [+51, +675]) — descriptive only, noted for
follow-up. Full JSONs in `paired/`.

Cell means (AUC₁₀₀ₖ): finger f-q1s0 78.9, f-q1s1 80.9 — **both reward-free
finger sides sit on the online-pretrain floor** (Gate-F population μ=83.4);
the reward-aware arm had 108.6 vs 267.9. Cup levels healthy in both arms
(450–625 free vs 559–695 aware).

## Registered 2×2 read: occupancy × reward-supervision interaction

Interaction = within-seed (Δ_task − Δ_apt), same seeds (1–8), same buffers;
bootstrap over seeds (B=10K, seed 0). From `sensitivity_and_2x2.json`:

| domain-quadrant | Δ aware | Δ free | interaction | 95% CI | pos |
|---|---|---|---|---|---|
| **finger Q1** | +159.32 | +2.04 | **+157.28** | **[+76.51, +240.68]\*** | **8/8** |
| finger Q2 | −1.59 | +17.85 | −19.43 | [−70.82, +17.84] | 3/8 |
| cup Q1 | +46.10 | +151.04 | −104.93 | [−296.43, +72.04] | 3/8 |
| cup Q2 | −120.67 | −70.23 | −50.44 | [−294.80, +176.20] | 4/8 |

Sensitivity suite on the finger-Q1 interaction (robustness only, per
Amendment 1): t CI [+50.9, +263.6], exact sign-flip permutation p=0.0078,
Wilcoxon p=0.0078, d_z=+1.24, LOO range [+127.7, +179.4]. The reward-free
simple effect is a tight null (t CI [−27.6, +31.7], d_z=+0.06).

**The interaction is significant, seed-consistent, and specific to the one
cell where the physical regime is definitionally the reward condition.**
High-occupancy replay helps transfer in finger *only when the WM fitting
objective carries reward/value gradients into the representation*
(reward-bearing frame fraction ≈0.32 vs ≈0.05 by side). Regime-relevant
data is not sufficient; the objective must be able to exploit it.

## Supplementary (NOT registered for this read; labeled per DEVIATIONS)

The reward-aware seed-9–16 extension (launched before the 11-Jul pause,
completed for finger Q1 + all cup; snapshot
`local_results/axis1_seed_extension_20260711_201510/`) provides a fresh-seed
replication check of the reward-aware simple effects:

- finger Q1 aware, seeds 9–16: **+30.71 [−15.51, +75.59]** (5/8) — strong
  attenuation vs seeds 1–8 (+159.32); both cells converge (s1 268→163,
  s0 109→132). Pooled seeds 1–16: **+95.02 [+44.20, +152.91]\*** (12/16) —
  still excludes zero.
- cup Q1 aware 9–16: +51.00 [−121, +186] (consistent with 1–8's +46).
- cup Q2 aware 9–16: −20.71 [−147, +99] (attenuated vs −120.67).

**Caveat carried forward:** the registered interaction estimate (+157, from
seeds 1–8) is likely inflated by a favorable seed batch (winner's curse);
the honest magnitude range given pooled-aware is roughly +40..+150. The
qualitative claim (aware ≫ free ≈ 0) is unchanged — the reward-free null is
tight and the pooled aware effect excludes zero. **Fix registered as the
next step:** re-point the seed extension per the protocol — reward-free
(apt) finger Q1 fits for seeds 9–16 (16 jobs) to give a fully-paired n=16
interaction estimate.

## Consequences (per the frozen tree)

1. Headline structure: (i) observational diagnostics get the causal sign
   wrong (5a occ− → intervention null/positive), (ii) coverage replicates
   observationally in 3 domains but its causal contrast is directional-only,
   (iii) occupancy × reward-supervision interaction +157* [+77,+241] —
   regime data pays off only through reward-capable objectives.
2. E3v2 (reward-free within-collector occupancy dose) **deprioritized** —
   its target simple effect is now known-null in finger. Re-purposing E3v2
   under task-mode fits would require a new amendment (decision: RB).
3. Seed extension re-pointed: apt finger-Q1 seeds 9–16 (16 jobs; §above).
4. E4 (mechanism) is now central and its spec should stratify
   representation error by reward-bearing frames, not only by regime.
5. Files: `auc_p0.csv` (frozen extractor output), `paired/` (16 registered
   contrast JSONs + 3 labeled supplementary `ext_aware_*`),
   `sensitivity_and_2x2.json`, `adaptation_curves_p0.png` (descriptive),
   `auc_aware_1_16.csv` (pooled robustness table).
