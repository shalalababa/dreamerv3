# SE STAGE-2 M3 BEHAVIORAL READ (registered) — 16 Aug 2026

**Governance:** Amendment 2 §2 (frozen; reader `uncfield/se_m3_read.py`
built + selfchecked + reviewed #23 BEFORE Stage-2 compute; gate pinned
pre-compute at position[0] > −0.13009691 = pre-Stage-1 smoke median);
ONE execution on bundle `uncfield_se_arms_20260816_224710` (manifest
committed+verified; gated seeds 20–23 / ungated 24–27, 8/8 DONE,
full-replay occupancy scans ~497–500k steps per run).

## VERDICT: PRIMARY DOES NOT FIRE — NO DIVERSION TOWARD THE GATE REGION

Registered primary (occupancy(gated) − occupancy(ungated) > 0, exact
one-sided permutation over 70 assignments, fires iff p ≤ .05):
**Δ = −0.0791, p = 1.0000 — the registered directional hypothesis is
refuted at the test's floor.** No behavioral-diversion claim is
licensed. Registered secondary (task return, two-sided): Δ = −4.3,
p = .371 — null.

**Descriptive observation (post-hoc, NOT a registered claim): the
separation is COMPLETE in the opposite direction.** Every gated run
occupies the gate-open region less than every ungated run (gated
0.467–0.493, mean 0.481 BCa [0.470, 0.490]; ungated 0.521–0.625, mean
0.560 [0.524, 0.611]); the observed assignment is the unique minimum
of all 70 (a reversed-direction test would sit at its 1/70 floor —
stated for calibration only; the direction was not registered).

## Reading (mechanism-coherent, discussion-level)

The dissociation chain is now complete and consistent across all
three levels: the objective MISPRICES the stochastic channel massively
and persistently (P-SE1: 5.3×, 8/8; M4: formed by 25%, flat), yet the
planner's ranking does not chase it (P-SE2: negative delta) and
behavior does not divert toward where it emits (M3: null-with-
reversal). The naive noisy-TV behavioral prediction — the agent parks
where the noise is — FAILS at this scale even though the accounting
pathology is enormous. Candidate boring mechanism for the reversal
(unresolved by an occupancy fraction alone): the OU disagreement is a
~state-independent pedestal inside the open region, so it contributes
level, not gradient; the gated arm's real-key disagreement landscape
differs from the ungated arm's (channels snap to zeros across the
boundary), and n=4+4 occupancy cannot separate posture/gait
confounds. Flagged for design review before any follow-up arm.

## Gates

Instrument CLEAN: 0 invalid; arm identity config-derived (both
wrappers' gate_key verified per arm; registered threshold matched to
1e-9) ✓; seeds exactly {20..23}/{24..27} ✓; occupancy floor ≥1e5
steps ×8 (realized ~497–500k) ✓; fit counters OK ×8; gate constants
recorded in the output json.

## Licensed / not licensed

Licensed: "the mispriced channel does not divert behavior toward its
emission region (registered one-sided test, p = 1.0, n = 4+4); task
return is unaffected (p = .37)" + the descriptive reversal WITH its
post-hoc label. NOT licensed: any claim that gating repels behavior
(unregistered direction), or any mechanism claim for the reversal.
