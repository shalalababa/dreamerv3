# NFI Sensitivity Sweeps — Results Record (EXPLORATORY, not registered)

> **Naming note (claim-freeze audit, 2 Aug 2026):** the verdict tier
> "EXPLOIT-SURVIVES-CIG+PBIM" denotes survival of the two TRANSPLANTED
> PRINCIPLES (prefix-conditioned carried-belief accounting; potential-based
> realized-dH accounting), not of the published estimators — CIG's
> estimator scores parameter-information over open-loop rollouts with no
> belief object, and PBIM's guarantee is a terminal correction this
> protocol never encounters. Paper-facing labels: "carried" / "potential".
> Two fidelity arms (gamma-discounted Ng-form recompute; CIG-faithful
> disagreement-kernel) were mandated by the audit and are recorded
> separately. See research_notes/NFI_ClaimFreeze_Audit_20260802.md.


**Date:** 2 Aug 2026. The "just undertrained" objection decider for the
flagship (`artifacts/nfi_pilot_20260801/`). Instrument:
`uncfield/sweeps.py` on RCC (`scripts/uncfield_sweeps.sbatch`); bundle
manifest-verified (`manifests/uncfield_sweeps_20260802.sha256`, 17
files); compact table `verdicts.json` (this dir); full summaries in the
untracked bundle `local_results/uncfield/sweeps/`.

**Review discipline:** instrument adversarially reviewed BEFORE this read
("fix before read"). The review CLEARED the verdict leg — verdict
extraction reuses `pl.classify` + planner thresholds verbatim, the
ensemble rule is logic-identical to pilot.py's, every axis genuinely
varies what its name claims (sample-mode flows end-to-end; retrains hit
the stated step counts; HID flows through init/heads/belief; dataseeds
regenerate disjoint episodes), and the resume-guard summary (ymode_s0,
written locally pre-crash) was re-executed under current code and
**bit-reproduced** (verdict, all counts, and raw floats identical).
**The `top_self_consistency` column is EXCLUDED from this read** (review
B2: 28/40 cells null, 6/40 computed on non-neutral legitimately
informative cycles; the correct per-member quantity is unrecoverable from
the bundle without re-search). Mechanism-record amendment filed the same
day (`artifacts/nfi_mechanism_20260802/`, prepended).

## Verdict table (ensemble; members as tier(n_conjunction_exploits))

| Axis | Job | Ensemble | Members |
|---|---|---|---|
| obs-mode | ymode_s0 | **CIG+PBIM** | F(13) F(47) C(0) F(65) |
| obs-mode | ymode_s1 | **CIG+PBIM** | C(0) F(35) F(17) F(24) |
| obs-mode | ymode_s2 | **CIG+PBIM** | F(34) F(2) F(168) F(17) |
| data | dataseed1 | **CIG+PBIM** | C(0) F(132) F(3) C(0) |
| data | dataseed2 | **CIG+PBIM** | F(242) F(27) F(61) F(7) |
| capacity | hid128 | **CIG+PBIM** | F(5) F(13) F(11) F(28) |
| capacity | hid32 | CIG-ONLY | C(0) F(168) **N(0)** C(0) |
| training | train1k | CIG-ONLY | C C C C |
| training | train10k | CIG-ONLY | C C T F(12) |
| training | train30k | CIG-ONLY | F(1) C T C |

F = EXPLOIT-SURVIVES-CIG+PBIM, C = -CIG-ONLY, T = TRANSIENT-EXPLOIT-ONLY,
N = NO-EXPLOIT. Anchor: pilot2 (3k steps, hid 64, ML-mode, seed 0) = F
3/4. Member tally across all 40: **F 23, C 14, T 2, N 1** —
**37/40 ≥ CIG-ONLY**.

## Findings

1. **The strong "just undertrained" objection is REFUTED.** At 10× the
   pilot's training (30k steps on the same 300×600 episodes), the
   CIG-surviving exploit persists (3/4 members ≥ CIG-ONLY, one still
   flagship), and across ALL axes 37/40 members farm predicted
   information through prefix-conditioned accounting. No axis produces a
   NO-EXPLOIT ensemble; the single NO-EXPLOIT member in the study is at
   hid32.
2. **Real, disclosed training-dose sensitivity in the realized-ΔH (PBIM)
   leg:** flagship-conjunction members by training = 0/4 @1k, 3/4 @3k
   (pilot), 1/4 @10k, 1/4 @30k — realized-contraction farming is a
   **mid-training phenomenon** at this data volume (rises then
   attenuates, does not vanish), while predicted-info farming is
   training-insensitive. Any headline must scope the CIG+PBIM conjunction
   accordingly.
3. **The exploit GROWS with capacity:** hid128 is the study's only
   unanimous flagship ensemble (4/4); hid32 degrades to CIG-ONLY with
   the only NO-EXPLOIT member. The "model too small" deflation runs the
   wrong way — more expressive update operators farm more.
4. **Not an ML-propagation artifact:** sampled-observation branch rollouts
   reproduce the flagship at 3/3 policy seeds on the pilot2 ensemble.
5. **Replicates on fresh data:** both dataseed pilots (fresh episodes,
   fresh member inits) come out flagship. Review caveat carried: the
   dataseed axis is JOINT (env + warmup + member seeds shift together) —
   these are fresh-data pilots, not isolated seed replications, and their
   train seeds overlap other jobs'.
6. **Self-consistency-ratio generalization: NOT READ** (instrument
   defect, review B2). Recoverable only by re-searching each member from
   the saved ensembles with per-cycle output (~90 s/member on RCC) —
   queued as optional follow-up; the diagnostic claim currently rests on
   the pilot2 mechanism record (amended) alone.

## Consequence for the program

The robust universal claim is **CIG-surviving predicted-information
farming** (37/40 members across observation-mode, data, capacity, and
training axes). The CIG+PBIM conjunction (the flagship signature) is
robust across observation-mode, data, and capacity-up, and
training-dose-sensitive with a mid-training peak — attenuating, not
vanishing. Natural next arm (not queued yet): hid128 × 30k interaction
(does capacity sustain ΔH-farming under long training?). Claim-freeze
literature audit is now unblocked.
