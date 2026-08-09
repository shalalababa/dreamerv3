## ⚠ AMENDMENT (9 Aug 2026) — drift-baseline artifact in the conjunction statistic

Read through this note first. The γ-arm's per-member statements about CONJUNCTION
cycle sets (e.g. "34/61 of m3's conjunction cycles farm level-neutrally") quantify
over the drift-ADJUSTED conjunction sets, which the 9-Aug review showed are
credit-manufactured in family 1 (raw counts 0/0/0/1; see
reviews/Review_FullRecord_and_Draft_20260809.md §1 and the pilot record's amendment).
The γ-invariance of the telescoped RETURN PREFERENCE and the stream-tax law are
statements about preference/stream structure and are unaffected as mechanics; their
application to "exploit" sets should be re-read against carried-leg exploit sets.
The kernel arm (carried/EIG accounting space) is unaffected.

---

# NFI Defense-Fidelity Arms — Results Record (EXPLORATORY, not registered)

**Date:** 2 Aug 2026. The two pre-freeze verification arms mandated by the
claim-freeze audit (`research_notes/NFI_ClaimFreeze_Audit_20260802.md`
§1-C1). **Read under the AMENDED protocol registered pre-read in memo
§6.3** after the arms review ("fix before read": machinery certified
bit-exact; both originally-registered read quantities replaced).
Instruments: `uncfield/gamma_rescore.py` (machine records
`local_results/uncfield/gamma_rescore/pilot2_m{0..3}.json`) and
`uncfield/cig_kernel.py` (`local_results/uncfield/cig_kernel/report.json`);
`preference_margins.json` in this dir.

## Arm 1 — γ-discounted Ng-form shaping (REQUIRED arm): READ

Validation gates: **0.0 deviation on all 8 gates** (γ=1 rate and adj
columns reproduce the saved pilot2 table bitwise, all 4 members × 626
cycles). Per §6.3 the γ<1 conjunction COUNTS are not reportable
(planted-calibration failure); the protocol quantities:

**1. Stream decomposition (adj₁ vs the (1−γ) level tax):**

| member | conj cycles @γ=1 | adj₁ range | level term @0.995 | @0.99 | stream>EPS @0.995 | @0.99 |
|---|---|---|---|---|---|---|
| m0 | 13 | .082–.098 | −.30..−.19 | −.59..−.38 | **0/13** | **0/13** |
| m1 | 61 | .080–.098 | −.27..−.13 | −.54..−.25 | **0/61** | **0/61** |
| m2 | 0 | — | — | — | — | — |
| m3 | 61 | .081–.223 | −.24..+.25 | −.47..+.52 | **36/61** | **34/61** |

The discounted reward STREAM self-limits in proportion to how far the
exploit drives the belief's entropy level below the move-only baseline
population — m0/m1's exploit streams are fully taxed away at γ ≤ 0.995,
**but the effect is member-heterogeneous: a third of m3's conjunction
exploits farm without net level displacement and their streams survive
discounting outright (34/61 at γ=0.99, stable across γ)**. The
self-limiting law quantifies PBIM's own within-episode-transient
concession; it is NOT a kill.

**2. Return preference (telescoped, vs walk-matched move twins) —
γ-INVARIANT in sign, positive for every exploited member:**

| member | exploit vs twin | γ=1 | γ=0.995 | γ=0.99 |
|---|---|---|---|---|
| m0 | c554 vs c555 | +3.64 | +0.66 | +0.12 |
| m1 | c447 vs c450 | +8.13 | +0.83 | +0.08 |
| m3 | c184 vs c185 | +2.72 | +0.72 | +0.19 |

**Headline (both readings together): the exploit is a return-preference
phenomenon, not an unbounded-stream phenomenon.** A discounted planner
still prefers the farming future at every γ (the choice a planner
actually makes), while the harvested reward stream is taxed at rate
(1−γ)·(entropy displacement) — self-limiting for level-displacing
exploits (m0/m1), untaxed for level-neutral ones (m3). Mandatory
caveats carried from §6.3: the level term scales with burn-window
placement; per-γ drift baselines are the unique gauge-invariant
accounting; the planted-positive/planted-null calibration at γ<1
supports rates/preferences only, not counts.

## Arm 2 — CIG-faithful disagreement kernel: READ (re-run with matched twins)

First-run facts established under review + §6.3: outcome falls in the
registered THIRD class — **estimator-insensitive-in-regime**. No
exploit-vs-control contrast (move-only walks carry comparable per-loop
surplus; the flagship exhibit REVERSES per action: c554 0.019 vs twin
c555 0.025), and no Prop-2(ii) decay across 30 identical wall-clock
repeats (per-loop surplus flat-to-rising) even though the planted
analytic selfcheck proves the machinery decays 1/k on actually-repeated
rows — the kernel is dominated by non-periodic open-loop GRU divergence,
not cycle information content. **Conclusion: supports C1's scoping claim
(CIG's estimator, as transplanted, does not operate in — and is not
sensitive to — the belief-state accounting space where the exploit
lives); an in-scope-refutation reading is forbidden from this data.**
Scope sentences (binding): single cross-member kernel object (per-member
flagship structure untestable here); open-loop sense steps lie outside
both CIG's estimator semantics and the model's training support
(measured non-inflation K_tt ratio ≈ 1.0 — robustness note, not
attribution); σ² floor ~66% driven by the never-sensed z₃ dim,
transients inflate σ² 7% (not outcome-gating). Matched-twin re-run (post-fix, completed): per-action steady surpluses
all sit in one narrow band (0.016–0.045/action) with NO systematic
exploit-vs-twin ordering — twins beat their exploits as often as not
(m1: twin c450 +0.045 vs exploit c447 +0.018; m0: twin c555 +0.025 vs
exploit c554 +0.019; m3: twin c185 +0.021 vs exploit c184 +0.016) —
confirming the no-contrast/insensitive outcome on the corrected
controls. Table in `local_results/uncfield/cig_kernel/report.json`.

## Consequence for the freeze (memo §5 items 3)

Both both-outcomes-publish arms landed on publishable outcomes:
- γ-arm → the two-reading split becomes a REGISTERED analysis: stream
  self-limiting law + γ-invariant preference. C1's flagship survives
  discounting in the decision-relevant (preference) sense at every γ,
  with the stream tax disclosed and quantified — and m3 as the
  stream-surviving existence case.
- kernel arm → C1's principle-vs-estimator scoping is now EMPIRICALLY
  grounded, not just definitional: the published estimator is measurably
  insensitive to the exploit's accounting space.
- planner docstring corrected (the "O(1−γ), disclosed" line was
  falsified: measured 5–8× the exploit margin at the registered window).

Remaining pre-freeze arm: P1 (PRIME aleatoric penalty, memo §6.2 spec).
