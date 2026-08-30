# Deployed license read (se_lic_probe, rev 2) — 26 Aug 2026
Engineering-mode read (NFI repair program, deployed rung). RCC caslake
job 55992108, COMPLETED 17:46, 12/12 cells (4 gauss se_alea_s120-123 +
8 det se_cheetah_seed10-17), one host, CPU (single-device by
construction). Instrument sha f9a6f540…, driver 27121ffd…, built from
the reviewed rev-2 (REVIEW_1 findings F1-F15 all applied; the
review's freeze evidence reproduced). Replay supply verified 12/12
pre-submission (1200-1440 chunks/run).

## The four legs, all twelve runs

**L1 (h=1 licensed primary, L_stochnorm behind the null gate): the
license does NOT reprice the misprice — outcomes are 8× ANTI-FIRES,
4× NULL-GATE-FAILED, zero fires.** θ₁_lic 11.0-19.0 vs θ₁_raw
4.2-5.7 vs the stoch-only reference 1.2-2.0. Where the reading is
adjudicable it goes the WRONG direction (the pre-registered
ANTI-FIRES row), and on the distractor specifically the transplant
null reproduces the license (per-key delta ~0.001-0.004 vs tau up to
0.0085) — the instrument's own gate declares the h=1 distractor
channel fidelity-confounded rather than manufacturing a conclusion.
Both outcomes were pre-registered post-review; neither is a fire.

**Substrate (the headline number): deter_frac = 0.97-0.99 on EVERY
key in ALL 12 runs.** ~97-99% of the deployed ensemble's decoded
disagreement is member disagreement about the DETERMINISTIC recurrent
path — a function of inputs every member already holds. The deployed
"epistemic" signal is almost entirely approximation error in
composition, not uncertainty about anything an observation could
teach. This single number reframes the misprice program's mechanism
at scale.

**L3 (horizon persistence, the environment-law prediction): a clean
three-way dissociation, uniform across 12/12 runs.** At h=15 the
licensed distractor persistence is 0.138-0.162 vs velocity 0.35-0.59
vs position 0.76-0.92. The AR(1) law predicts 0.052: the learned WMs
encode the distractor's transience only PARTIALLY — they believe it
~3× more persistent than it is, in every run, both arms. **The
misprice co-occurs with a quantified wrong-world error about
persistence** — NFI's wrong-world class observed in the wild, with a
number attached.

**L4 (cross-seed min-license): the corrdef impossibility prediction
confirms at scale.** Min-vs-second gap 0.0005-0.0006 (every seed
shares the defect — shared-architecture/shared-recipe ensembles are
the maximally correlated case); min-licensed θ₁ 2.35 (det) vs raw
5.18 — attenuation, not removal, exactly as the boundary predicts
with no sane member; the refusal certificate correctly reads
"cannot certify" (distractor min-license 0.070 with the level-not-
spread caveat carried).

**L2 (obs-space aleatoric account):** the distractor's projected
aleatoric share is small (~0.057 vs real keys 0.008-0.029 in
gauss_s120) — the per-key aleatoric account does not absorb the
misprice either, consistent with B4.

## What the deployed rung decides

1. **The fold decision (per the standing plan): NFI-SECTION, not
   standalone.** The deployed fire did not occur; the constructive
   act folds into NFI as: in-world removal (XEDL) + the unified
   self-consistency boundary + the deployed read showing WHY the
   boundary binds at scale (non-epistemic disagreement composition;
   shared-defect ensembles; wrong-world persistence).
2. The deployed negative is STRONGER than B4's: B4 showed one repair
   fails; this read shows the one-step round-trip license — the
   scoring-time family's deployable form at this ensemble unit —
   cannot even adjudicate the distractor channel (fidelity
   confound), anti-fires where it can, and the reason is measured
   (0.97-0.99 deter-carried claims). Scope carried from review F7:
   this is ONE point of the self-consistency family (shared-
   component members, no independent-ensemble min, no ledger
   layers), consistent with — not a test of — the unified law.
3. Three new deployed facts for the papers: the deter-fraction
   composition; the ~3× persistence miscalibration (uniform,
   12/12); the confirmed maximal correlation of shared-recipe
   ensembles (the refusal certificate as the honest deliverable at
   scale).

Artifacts: 12 per-run jsons + 2 collates (this dir); npz on RCC in
each run dir's se_lic_probe/. Instrument + driver staged in the repo
tree (uncommitted): uncfield/se_lic_probe.py,
scripts/se_lic_probe.sbatch; design/review record in
research_notes/paper5_uncertainty_field/repair_ideation_20260825/.
