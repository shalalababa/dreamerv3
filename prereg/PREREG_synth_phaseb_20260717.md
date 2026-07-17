# Synth predictable-heads Phase B registration (draft 2026-07-17; freeze after Phase-A cluster smoke)

Registers the Paper-1 slice of the synthetic predictable-heads domain
(execution plan `research_notes/Plan_PredictableHeads_20260717.md`;
triage #6) **before any synth fit or adapt outcome exists**. Question:
do the two core phenomena — the occupancy × reward-supervision
interaction and the shuffle (binding) collapse — reproduce in a domain
where occupancy, label density, and label–state binding are engineered
dials rather than emergent properties of DMC exploration?

## Domain and buffers (built, validated)

- Env `synth_reach` (`embodied/envs/synthpred.py`, suite registered in
  `dreamerv3/main.py`, kwargs `env.synth` in configs.yaml): 2-D point
  mass + 8 OU distractor dims (decoupled, coupling=0), sparse reward
  inside radius 0.1 of a fixed goal; regime spec 'synth'
  (`probing/regimes.py`) ≡ the reward condition — the finger structure
  (occ ≡ reward), by construction.
- Buffers synthesized directly (`probing/synth_buffers.py build-pair`,
  no collector training): pair `axis1_synth/q1`, side1 occ 0.30 /
  side0 occ 0.03 (mirroring finger q1's 0.32/0.05), 100 episodes ×
  1000 steps per side, density f=1.0, predictability p=1.0, amplitude
  1.0, seed 0. Trajectories are label-independent; labels painted
  post-hoc. Generator selfcheck PASS (realized occ within 0.01 of
  dials; label accounting exact; determinism; relabel-chain reuse).
- Shuffle arm: `relabel_replay transform --kind shuffle` on the built
  pair → `axis1_synth/q1_sh` (same reuse as P3).

**Validation status at registration:** full pipeline validated locally
(2026-07-17): task-arm and apt-arm debug fits ingest the synthesized
chunks (task shows `rew` in the loss set, apt does not), frozen-readout
adapt loads exactly `^(enc|dyn|dec)/` and produces scores.jsonl in the
frozen format. Debug dirs deleted; a cluster smoke (1 fit + 1 adapt,
seed 99, 20K updates, deleted after) is required before the wave, per
the P0/TD-MPC2 pattern.

## Cells and jobs (seeds 1–8 paired; 48 jobs)

| arm | submit | mode strings |
|---|---|---|
| task | `AXIS1_DOMAINS=synth AXIS1_QUADS=q1 AXIS1_EXPL_MODE=task axis1-bundles` | `ax1q1s{0,1}` (domain synth) |
| apt | same with `AXIS1_EXPL_MODE=apt` | `ax1fq1s{0,1}` |
| sh | `AXIS1_DOMAINS=synth AXIS1_TRANSFORM=sh axis1-factorial-bundles` | `ax1shq1s{0,1}` |

Fit 500K updates, adapt 125K steps, QC ≥20 eps ≤100K — the frozen
Axis-1 protocol unchanged. Synth rows never pool with any DMC
population (domain column separates them; frozen `--modes` filters
unaffected).

## Outcomes and decision rule

B_arm(k) = AUC100k(s1) − AUC100k(s0) from the untouched
`analysis/adaptation_auc.py`; CIs cluster-bootstrap percentile 95%
(B=10,000, default_rng seed 0); decisions on CI alone; sensitivity
suite robustness-only. Read script frozen pre-outcome:
`analysis/synth_phaseb_read.py` (selfcheck PASS, branch map recovered).

- **PRIMARY-1 (interaction):** mean [B_task − B_apt], fires iff CI > 0.
- **PRIMARY-2 (binding collapse):** mean [B_sh − B_task], fires iff
  CI < 0.
- S1: B_apt simple effect — prediction: null (the fourth,
  engineered-domain form of the reward-free null).
- **Registered branch map:** both fire ⇒ synth = Paper-1 third domain
  (ground-truth version of the law), Phase C authorized for scoping;
  interaction-only ⇒ density-suffices boundary vs the DMC binding
  result; collapse-only ⇒ occupancy does not gate transfer in synth,
  diagnose before Phase C; neither ⇒ major boundary condition — the
  law depends on something DMC has and the synthetic lacks; reported
  as-is (informative and cheap), no Phase C without redesign.
- "Replicates too easily" caveat (registered): Phase B at p=1, f=1 is
  the existence check only; the causal-law claims live in Phase C's
  held-out-prediction structure (calibrate threshold cells, predict
  others), which gets its own registration.

## Disclosure and ordering

Known at freeze: all DreamerV3/TD-MPC2 outcomes through 17 Jul 2026 —
the design deliberately engineers the finger structure, so the
*direction* of both primaries is an informed prediction; the test is
prospective only through the unknown synth outcomes. Unknown: every
synth outcome (no synth fit or adapt beyond the deleted local debug
and the to-be-deleted cluster smoke, both excluded by rule; the smoke
precedes freeze and contributes only plumbing validation). Ordering:
env + generator + submit_all changes + `analysis/synth_phaseb_read.py`
+ this file are committed together BEFORE the Phase-B wave is
submitted. Phase C (density × predictability × control-relevance ×
binding × gradient-magnitude × head-identity factorial, ~200–400 jobs,
hosts P-B3/P-B4) is NOT registered here; it freezes its own file after
the Phase-B read.
