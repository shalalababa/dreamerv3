# s×w_r wave read — 2026-08-10

Registration: `prereg/PREREG_swave_wave_20260809.md` +
`PREREG_swave_theory_correction_20260808.md` (both frozen 00d6ce15,
08-09, pre-outcome). Reader: `analysis/swave_read.py` (frozen in the same
commit; selfcheck re-PASS immediately before execution). ONE execution.

Inputs: `swave_rgo_20260810_202214/auc_rgo.csv` + `swave_vgo…/auc_vgo.csv`
(mechanically concatenated, 88 rows, zero exclusions both files),
3 E4 collates from `swave_measure_20260810_202458/collates/`,
fit counters + registered `loss_scales.rew` config-audit sweep from
`paper1_aux_20260810_215605/swave/` (audit: checked 88 bad: [] — PASS).
All bundles sha-manifest-verified. Counters 88/88 update==total.
modal_n_ep = 96, no qc or sub-window exclusions.

## Verdict (P-SW4): **BOTH-AXES-INERT — corrected a²-term refuted in the tested range**

The registered honest theory negative. With the twohot-floor correction
applied (registered constants 0.6735/0.3981/0.5286):

- **P-SW1** (s membership, own-label, floor-corrected, swr31−swr11):
  −0.470 [−1.237, +0.139], perm p=.229 — NO FIRE.
- **P-SW2** (w_r membership, true-label, swr13−swr11):
  −0.648 [−1.445, +0.146], perm p=.158 — NO FIRE.
- **P-SW3** (behavioral auc100k, 8 contrasts vs swr11, BH q=.05):
  **no survivors** (extremes: r22 +99.1 p=.148, r33 +83.9 p=.306,
  r13 +82.7 p=.085 — all CIs straddle 0).
- **P-SW5** (vgo repair leg, swv31−swv11): +0.127 [+0.018, +0.247],
  p=.067 — NO FIRE (and the point sign is opposite the predicted
  direction).

## Theory-ledger consequence

Third registered λ-side/energy-side negative in a row (after P-HF1
saturation and P-BD1 breadth null): amplifying reward ENERGY (scale s)
or reward WEIGHT in the objective (w_r) does not move feature membership
in the tested range — the reward-gradient path's causal story stays
g-side (which arm carries gradients), not magnitude-side. The
descriptive shape stats (ln(1+s)² R²=0.66 vs raw-s² R²=0.47) are
reported for the theory note but license nothing.

Output: `swave.json` (unedited). Bundles referenced remain under their
committed manifests.
