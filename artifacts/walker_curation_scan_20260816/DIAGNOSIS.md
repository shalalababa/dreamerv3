# Why the walker search landed at side0 occ = 0.0802 — and why 0.05 is unreachable

**2026-08-16, ops.** TODO item 132 step (1). Code reading plus one read-only
enumeration on the RCC login node. Nothing rebuilt, nothing submitted, no
registered output touched.

**Result: root cause found, and the rebuild target is infeasible as specified.**
Of the 2715 pairs that already satisfy every registered criterion, **none has a
low side at or below the 0.05 bar; the reachable floor is 0.05318.**

## Correction to an earlier draft

The first version of this document attributed `pairs.json` to
`probing/gate0_compose.py`. That was wrong: it is written by
**`probing/build_controlled_replay.py`**. The mechanism described was right and
holds verbatim in the correct file; only the file and line numbers were wrong.
They are corrected below. `gate0.py` / `gate0_compose.py` carry the same
acceptance test but did not produce this artifact.

## The mechanism

**1. The acceptance test has no absolute occupancy bound**
([build_controlled_replay.py:359](../../probing/build_controlled_replay.py#L359)):

```python
if dcov <= cov_tol and docc >= occ_sep_min:
```

Coverage *matched*, occupancy *separated*. Where either side sits in absolute
terms is unconstrained.

**2. Among accepted pairs it maximizes separation** (line 361):

```python
if best['q1'] is None or docc > best['q1']['docc']:
```

The winner: side0 **0.0802**, side1 0.1548, docc 0.0746 — against an
`occ_sep_min` of 0.0250, a gap 3× larger than required.

**The read's 0.05 bar was never an input to the search.** Not a
misconfiguration: a specification gap between a curation step that optimizes
*separation* and a feasibility gate that tests an absolute *level*.

**A plain re-run is useless.** `np.random.default_rng(args.seed)` with the
recorded `seed: 0` makes candidate generation deterministic — same 400
Dirichlet(0.3) weight-sets, same 388 feasible candidates, same 2715 accepted
pairs, same argmax winner, bit-for-bit.

## The enumeration

`enumerate_q1_pairs.py` (this directory) imports `build_controlled_replay` and
calls its own `_candidate_pool` / `_overlap`. Nothing is edited, monkeypatched
or re-implemented; the acceptance test is copy-identical to `cmd_search`. The
sole difference is that `cmd_search` keeps the argmax and discards the rest,
while this keeps all of them. **Selection is not changed** — the argmax is
recomputed only as a self-check.

**Self-check passed — the pool reproduced bit-for-bit**, which is what makes
the rest meaningful (the script exits 2 without reporting otherwise):

| | frozen `pairs.json` | recomputed |
|---|---|---|
| accepted pairs | 2715 | **2715** |
| side0 occ | 0.080200 | **0.080200** |
| side1 occ | 0.154755 | **0.154755** |
| docc | 0.074555 | **0.074555** |

### Result

```
low-side occupancy over 2715 accepted pairs:
  min 0.05318   p5 0.08020   median 0.09532   max 0.13310
accepted pairs with low side <= 0.05:  0
```

The six lowest-low-side accepted pairs:

| lo_occ | hi_occ | docc | dcov | overlap |
|---|---|---|---|---|
| **0.05318** | 0.07859 | 0.02541 | 0.01136 | 0.020 |
| 0.06429 | 0.09696 | 0.03267 | 0.01172 | 0.040 |
| 0.06753 | 0.09374 | 0.02621 | 0.00066 | 0.030 |
| 0.06753 | 0.10465 | 0.03712 | 0.00547 | 0.060 |
| 0.06753 | 0.09606 | 0.02853 | 0.00423 | 0.045 |
| 0.06753 | 0.11881 | 0.05128 | 0.00257 | 0.025 |

## What this settles, and what it does not

**Settled: no selection rule can rescue this pool.** The earlier draft framed
the fix as "change argmax-docc to something that prefers a low side0". Even a
perfect argmin-low rule returns 0.05318 — still above the bar. The failure is
not in how the winner is *chosen*; it is that no compliant candidate *pair
exists* in the pool.

**Settled: the feasibility scan overstated the case.** `RECORD.md` concluded
"COMPLIANCE IS TRIVIALLY ACHIEVABLE" from the marginal per-episode distribution
(3430 episodes ≤ 0.05, unconstrained floor 0.0067). Those are raw episodes. Once
composition into 200-episode candidates, coverage matching (`cov_tol` 0.01246 ≈
1% of the ~1.25 coverage level), separation (≥ 0.0250) and the overlap cap are
applied, the reachable floor rises by ~8×, from 0.0067 to 0.05318. The
occupancy/coverage coupling in the pool is why — Pearson r(source coverage,
source mean occ) = **+0.522** across the 18 sources, and the entire `random`
family (coverage 1.121–1.131, the cheapest low-occupancy material) tops out at
occ 0.1079 with **zero** episodes at side1's level.

**NOT settled: whether a wider search could clear the bar.** The floor of
0.05318 is the floor **of this sampled candidate pool** — 400 Dirichlet(0.3)
weight-sets at `seed: 0`, of which 388 were feasible. It is not a proof of
impossibility. The miss is small: 0.05318 is **+0.32 pp over the bar, 1.064×**.
A different `--seed`, a larger `--n_candidates`, or a less concentrated
`--dirichlet` samples a different set of mixtures and could plausibly reach
below 0.05. That is a re-run of the registered tool with different sampling
knobs — cheap, CPU-only, and it does not touch the criteria.

**Also relevant, and not in the earlier draft:** `build_controlled_replay`
already ships two modes beyond `search` — `search-dose`, which computes the
Q1-feasible occupancy span directly, and `search-rpair`, which selects "docc as
close as feasible to `--target_docc`" and emits `pairs.json`-shaped output that
`build --which q1` materializes unchanged. So a non-argmax selection is already
a registered capability, not a code change. It just cannot help *this* pool.

## For the owning chat

Ordered by cost, all decisions the Papers-1–3 chat owns:

1. **Re-sample the candidate pool** (`--seed`, `--n_candidates`,
   `--dirichlet`) and re-enumerate. Cheapest, touches no criteria, and the
   0.32 pp miss makes it plausible. Whether re-sampling until the bar is met
   constitutes selection-on-the-gate is a registration question, not an ops one.
2. **Relax a criterion** — `cov_match_frac` or `max_overlap`. An amendment, and
   it weakens the matched-pair control the design rests on.
3. **Carry P-D2 as not-tested**, reacher alone as the generality leg.

One consideration that argues against forcing option 1 or 2: the lowest-low
pair available (lo 0.05318 / hi 0.07859) carries **docc 0.02541**, barely over
the 0.0250 floor and about a third of the built pair's 0.0746. Even if the
absolute bar were met, that arm contrast would be far weaker than the design
assumed — so "clears the gate" and "worth running" are not the same question
here.

## Status

* Step (1) diagnose — **DONE**.
* Step (2) re-curation — **BLOCKED**: no compliant pair exists in the current
  pool; the naive low-floor target is unreachable.
* Step (4) 32 fits + 32 adapts — **NOT SUBMITTED**. The input buffer does not
  exist, and building the registered fits on the current side0 would reproduce
  exactly the configuration the read already refused.

Artifacts: `enumerate_q1_pairs.py` (this dir),
`$RUNROOT/axis1_walker_enum/q1_pairs.json` (all 2715 pairs with
lo/hi occ, cov, docc, dcov, overlap).
