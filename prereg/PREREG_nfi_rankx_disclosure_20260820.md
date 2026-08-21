# PREREG — SE ranking-probe DESCRIPTIVE DISCLOSURE (rankx), 20 Aug 2026

**Status:** registered descriptive disclosure of UNREGISTERED data.
**Context:** review #25 (`Review_NFIv3_20260820.md` addendum A3) and the
algorithm-ideation panel discovered that the gradient-wave se_probe
passes (hetero + flat, bundle `uncfield_se_grad_20260817_101139`)
carry populated P-SE2 (`pse2`) fields that NO frozen reader ever
consumed — the probe ran as part of the frozen per-run instrument; the
gradient read (`se_grad_read.py`) never references `pse2`. The panel
derived the numbers by hand; per the standing discipline they must
enter the record through a frozen reader with ONE registered
execution, as a DISCLOSURE, before any writing or algorithm leans on
them. User GO 20 Aug.

## Binding caveats (registered up front)

- The data is UNREGISTERED for these arms: the gradient wave's
  registered estimands were occupancy + the manipulation check;
  nothing in Amendment 3 pre-specified a ranking read.
- n = 4 per arm; direction not pre-specified; the like-for-like anchor
  (Stage-1's P-SE2, −0.0170 [−0.0277, −0.0064]) IS registered but its
  fire rule was 8-run, not 4.
- **Nothing here is a claim.** Output is labeled DESCRIPTIVE
  throughout; no fire rules, no outcome map, no BH, no verdict beyond
  faithful numbers. Any paper use is as a disclosed exploratory
  observation with this provenance attached.
- Reviewer economy: no fresh instrument review — review #25's addendum
  A3 independently derived the headline values from the same artifacts
  (hetero overall ≈ −0.082 with per-run −0.0840/−0.1073/−0.0562/
  −0.0812; flat ≈ −0.009; hetero noise-family ≈ −0.138 4/4; hetero
  dup-family ≈ +0.055 4/4); the reader's ONE execution must REPRODUCE
  those independently-derived values, which serves as the
  cross-implementation check. Any mismatch = stop and investigate,
  not adjudicate.

## Registered reader

`uncfield/se_rankx_read.py`, built + selfchecked BEFORE its single
execution. Inputs: the 8 gradient-wave probe jsons (hetero 28–31,
flat 36–39) + the 8 Stage-1 probe jsons (seeds 10–17) as the
registered-baseline context row. Per run: Δ_overall = pse2.top_share −
pse2.base_share; Δ_dfam and Δ_noise from pse2.families; p_rank
carried; train seed from config.yaml; arm identity from config
(hetero: mod quadruple; flat: scale pin; Stage-1: neither) — the
se_grad_read/se_read pin values reused. Aggregates: per-arm means +
seed-level BCa (reporting only). Output json + a RESULTS.md in
`artifacts/se_rankx_disclosure_20260820/`.

## What this may and may not feed

MAY: the NFI paper's §8 as a disclosed exploratory paragraph (with
the unregistered-direction clause, same treatment as the occupancy
reversal); the structured-noise-avoidance follow-up seed; the
transmission-axis design discussion. MAY NOT: any confirmatory
sentence, any abstract/contribution claim, any fire wording.
