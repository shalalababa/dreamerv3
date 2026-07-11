# E1 reacher replication — outcome-side read (2026-07-11)

Inputs: `local_results/e1_reacher_pretrain_20260710_230406/` (15/15 pretrains
`TRAINING_DONE`; probeset `reacher_v1` frozen; `critics/critic_reacher_v1.npz`
present — **held-out R² not stored in the npz; recover it from the fit-critic
log before the measure grid decides VSA vs retdec, gate 0.2**) and
`local_results/e1_reacher_adapt_20260711_095948/adapt_scores/` (75/75 adapts,
75/75 QC pass, 0 exclusions). AUC re-derived locally with the frozen pipeline
— bit-identical to cluster-side `auc_reacher.csv`.

## Descriptives (replication population; drivers pending measure grid)

Mean AUC over 25 runs/mode (5 seeds × 5 milestones) ± SD:

| mode | AUC₅₀ₖ | AUC₁₀₀ₖ | AUC₁₂₅ₖ | final10 |
|---|---|---|---|---|
| p2e | 40.3 ± 38.7 | **94.6 ± 77.7** | 110.8 ± 86.2 | 193.0 ± 161.8 |
| apt | 35.5 ± 23.7 | **98.1 ± 67.0** | 118.4 ± 73.1 | 249.7 ± 169.9 |
| random | 21.9 ± 12.0 | **42.4 ± 28.0** | 50.7 ± 35.4 | 95.6 ± 86.6 |

- **The 5a direction replicates at the mode level:** exploration pretraining
  (p2e/apt ≈ 95–98 AUC₁₀₀ₖ) transfers ~2.3× better than random (42), in a
  third decoupled domain, seeds/milestones fully populated.
- **No clean milestone dose trend** (per-milestone means non-monotone),
  consistent with the 5a primary-population M0 null.
- Absolute levels are modest (reacher_hard); spread is large — the
  registered M1 driver fits (cov/occ_phys) are the informative read.

## Pending (this is NOT the registered E1 analysis yet)

The registered analysis is the frozen 5a pipeline with reacher as its own
population (`--primary_domains reacher --control_domain none`), which needs
the driver rows: **15 reacher measure jobs + latent dumps** (after the
critic-gate R² is confirmed). Key question given Axis-1: does the
observational occ_phys association come out negative here too (composition
confound signature) while cov stays positive?
