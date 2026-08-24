# Attribution-landscape diagnostic (Step 1 of the planner-rescue
decision) — 22 Aug 2026

**Status: EXPLORATORY DIAGNOSTIC** (`analysis/planner_landscape.py`;
thresholds declared in the script header BEFORE computation; data =
the executed planner wave's own npz, 8 runs × 5 arms × 2000 realized
steps; zero new compute; licenses no registered claim).

## Primary metric: AMBIGUOUS by the declared rule

Headroom H = mean(top-10% pooled attr)/mean(policy attr) per run:
1.49, 1.73, 1.40, 1.33, 1.38, 1.50, 1.26, 1.24 — n_peak(≥1.5)=2,
n_flat(≤1.2)=0 → neither 6/8 rule met → decided by the declared
secondary structure metrics.

## Secondary metrics: the ambiguity resolves to FLAT — the
landscape's maximum COINCIDES with the deployed policy's visitation

- **Top-decile composition:** 34.8% of the pooled top-decile states
  are POLICY states (cem_distractor 11.6%, cem_disag 2.8%, cem_real
  0.7%, random 0.2%). The policy is ~7% of a 5-arm pool, so it is
  ~3.5× over-represented at the top; the nominal "headroom" above
  the policy MEAN is the policy's own upper tail, not an unreached
  region.
- **Best single episode:** a POLICY episode in 7/8 runs (seed11's
  cem_distractor episode the lone exception, 1.27×). Policy
  episode-to-episode spread is wide (0.011–0.043) — the landscape's
  peak is policy-episode heterogeneity.
- **Dwellability:** attr lag-1 autocorrelation 0.73–0.95 within
  policy episodes — high-attribution states are sustained, and they
  are the states the policy already occupies.
- **Spatial structure:** attribution is mildly HIGHER out-of-region
  for every arm (in/out ratios 0.77–0.96) — the gate coordinate is
  not the attribution axis.

## Decision

**The trained-probe rescue wave is NOT worth running for a
TRANSMISSION revival.** Its realistic best case is matching the
policy (re-confirming flatness-at-the-top); nothing in the visited
landscape suggests a reachable peak the planners failed to find.
Honest limit: unvisited support is unknowable from this data — the
statement is about the union of five diverse visitation
distributions, disclosed as such.

**What we gain instead (the strongest-claim path):** the terminal
reading of UNSTEERABLE upgrades from "our planner was too weak" to a
mechanism statement — *the attribution maximum coincides with the
deployed policy's own visitation; deployment-time optimization
cannot amplify exposure because the amortized policy already sits at
the reachable maximum of the field it mispriced.* For the Track-B
transmission paragraph this is descriptive; a CHEAP registered
upgrade exists if wanted with the Track-B batch: pre-register
"policy-coincident maximum" (policy tops every arm's attribution,
per-run) on fresh seeds — the present wave shows exactly that
pattern (7/8 per-run ceilings negative; policy arm-mean highest in
all pooled means), so the prediction is well-grounded and one 8-run
wave away from being claimable.
