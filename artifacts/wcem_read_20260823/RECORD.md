# P2 WCEM — CEM consumer, ONE read (23 Aug 2026)

Registration: PREREG_p2_cem_consumer_20260821.md (frozen 8828ab6f;
execution sequenced behind the Wave-1 read — `--after_wave1`
satisfied by artifacts/p2_wave1_read_20260823/read.json). Bundle:
p2_wcem_20260823_210000 (manifest 31/31 OK; 8 cells + 2 dup gates;
duplicate-null EXACTLY zero; all reader gates clean).

**Execution regime, approved 23 Aug + fully honored (bundle NOTES)**:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.18, 5 tasks/GPU, uniform across the
smoke + both dups + all 8 cells; smoke ran first under the cap; the
dup gate passed under the production regime — exactly the four
conditions of the sign-off. (The user waived the dups-before-cells
ORDERING to run all 10 concurrently; the gate itself was untouched
and passed.)

## Verdict: **CEM-ACTOR-INDISTINGUISHABLE**

| quantity | value |
|---|---|
| planner-competence gate | **PASS** — pooled +4.13 internal units (per-cell all > 0; the CEM genuinely out-optimizes the actor's candidate on the model's own signal; the degenerate-planner NO-CALL did not trigger) |
| CEM − actor, real return (PRIMARY) | **−0.480 [−1.155, +0.001], perm p = .221** — no fire |
| CEM harvest vs candidate mean | −0.468 [−1.176, +0.022], p = .221 |
| realized MDE80 | 0.872 |

## What this settles

- **A dedicated planner that provably optimizes the model's value
  signal harder gains NOTHING in real return** — the sharpest
  remaining Paper-2 question, answered. The internal optimization is
  real (+4.13 pooled, competence-gated), the realized return is not
  merely flat but points mildly negative, with the BCa upper bound
  grazing zero from below ([−1.155, +0.001]). Directionally this
  LEANS anti-harvest but ANTI-HARVESTS is not licensed (no fire);
  reported symmetrically.
- **The Paper-2 constructive story is now complete and coherent**:
  the estimator chain is calibrated end-to-end (Stage A + Stage B at
  slope 1); real, discriminable decision value exists off the policy
  candidate manifold (WDC, exogenous uniform probes vs ground
  truth); yet the model's OWN value surface cannot steer to it — the
  amortized actor already sits at the reachable optimum of the
  signal the planner maximizes, and pushing that signal harder
  converts to nothing. Value is present and measurable, but not
  harvestable through the model's value estimates.
- **Item 3 of the value-head triage (reward-only CEM) stays parked
  permanently**: its registered trigger was a WCEM anti-harvest
  fire, which did not occur.
- With this, every registered Paper-2 wave except Wave-2's labeling
  chain is read.
