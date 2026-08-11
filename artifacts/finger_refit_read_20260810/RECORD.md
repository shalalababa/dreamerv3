# Finger q1 regeneration read — 2026-08-10

Registration: `prereg/PREREG_finger_refit_20260808.md` (frozen 825de241,
08-08) + `PREREG_refit_reads_amend1_20260810.md` (checkpoint-step witness,
regime-leg structure) + `PREREG_refit_reads_amend2_20260810.md` (substrate
correction). Reader: `analysis/cup_refit_read.py` (selfcheck PASS before
execution). Bundle: `regen_ridge_e4_20260810_220849` (sha-verified, 394
files). ONE execution (`--domain finger`); output `read.json` (not edited
post-execution).

## Gates

32/32 runs loaded (16 task q1, 16 apt fq1); witness_match 32/32 True
(post-amendment ep_batch=4 ridge panel); probeset `finger_v1`
sha a78fe58ca526… uniform incl. E4; expl_mode/reward_aware wiring correct
both arms; measured checkpoints 64/64 at step 500000 (ridge + E4).
Fit-counter preflight (d) FAILED (restore-overlay stale counters) —
substituted checkpoint-step witness per Amendment 1. Substrate per
Amendment 2: 26 original-July-weight + 6 fresh apt.

## Primary — #9 apt-trunk ridge probe: **ABSENT-OR-NONLINEAR**

- Calibration (task arm): pooled AUROC **0.935** [0.908, 0.952]; in-regime
  **0.911** [0.855, 0.946] — passes the strong bar; instrument informative.
- Apt arm: pooled AUROC **0.401** [0.386, 0.426]; in-regime **0.458**
  [0.378, 0.523] (straddles 0.5) ⇒ registered middle branch fires: the apt
  trunk carries **no linearly-decodable reward signal**. Registered
  consequence: the legibility account shifts from "present but illegible
  to the objective" toward "not linearly present at all"; wording change
  only, no archived verdict flips (behavioral nulls already archived).
- Out-regime leg structurally unevaluable (0 rewarded out-of-regime
  frames, leak-free probeset); in-regime leg low-powered (6 negatives).
- **[POST-READ observation, no wording licensed]**: apt pooled AUROC is
  significantly BELOW chance (0.40, CI excludes 0.5, in BOTH substrate
  classes: original-weights 0.4017 [0.3816, 0.4383], fresh 0.4000
  [0.3771, 0.4116]) — apt features actively rank rewarded frames LOWER.
  Anti-predictive, not merely uninformative. Mechanism lead for the
  dual-lead section; would need its own registration to claim.

## Finger Δ_dom (consumed by P-C3, `PREREG_cup_refit_20260808`)

Δ_dom(task − apt, pooled AUROC) = **+0.534** [+0.483, +0.557], cluster
sign-flip perm p = .008 (resolution-limited at 2^8 patterns). CI entirely
> 0 ⇒ the finger-violation leg of P-C3 is established.

## Substrate sensitivity

Primary (all 32) and both true-substrate classes agree (Amendment 2 §3);
the `sensitivity_fresh_only` block inside `read.json` is the superseded
date-based split — see Amendment 2.

## Replication rider (descriptive)

Task-arm h0 NLL panel is **bit-identical** to the archived 14-Jul panel
(s0: 1.807/0.086/0.242; s1: 1.023/0.108/0.191 — in/out/all), which is the
weight-recovery proof, not a population draw: the rider caught the
provenance story (Amendment 2). Population-replication evidence for
regeneration comes from the cup s1 rider instead (see cup RECORD).
