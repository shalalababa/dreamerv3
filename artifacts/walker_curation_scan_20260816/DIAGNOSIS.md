# Why the walker search landed at side0 occ = 0.0802

**2026-08-16, ops.** TODO item 132 step (1). Pure code + artifact reading; no
compute, nothing rebuilt, nothing submitted.

**Result: root cause found, and it blocks step (4).** A plain re-run of the
registered search reproduces 0.0802 exactly. The rebuild cannot proceed until
a decision is made that touches a registered instrument.

## The mechanism

Two facts from `probing/gate0_compose.py`, the tool that wrote
`axis1_walker/pairs.json`:

**1. The acceptance test has no absolute occupancy bound** (line 260):

```python
if dcov <= cov_tol and docc >= occ_sep_min:
```

Coverage *matched*, occupancy *separated*. Where the two sides sit in absolute
terms is unconstrained. `gate0.py:189` uses the identical test.

**2. Among accepted pairs it maximizes separation** (line 262):

```python
if best_q1 is None or docc > best_q1[2]:
```

So of the 2715 accepted q1 candidates it returns the one with the largest gap,
with no preference over where that gap sits.

The winner: side0 occ **0.0802**, side1 **0.1548**, docc 0.0746 — against
`occ_sep_min` 0.0250, i.e. a gap 3× larger than required.

**The read's 0.05 bar was never an input to the search.** The search satisfied
every criterion it was given. This is not a misconfiguration and not a bug; it
is a specification gap between a curation step that optimizes *separation* and
a feasibility gate that tests an *absolute level*.

## Consequences for the rebuild plan

**Re-running the search unchanged is useless.** `rng = np.random.default_rng(args.seed)`
with `seed: 0` (recorded in `pairs.json.criteria`) makes candidate generation
deterministic: same 400 weight-sets, same 2715 accepted pairs, same
argmax-docc winner, same 0.0802. Bit-for-bit.

**TODO step (2) as written — "side0 re-curation targeting the low floor
(~0.007–0.02)" — is not a re-run; it is a change to how the pair is chosen.**
Either the acceptance test gains an absolute cap, or the selection rule stops
being pure argmax-docc. Both are edits to a registered instrument, so the call
belongs to the Papers-1–3 chat, not to ops.

**The feasibility scan overstated the case.** `RECORD.md` concludes
"COMPLIANCE IS TRIVIALLY ACHIEVABLE" from the marginal occupancy distribution
(3430 episodes ≤ 0.05, unconstrained floor 0.0067). That ignores the
coverage-matching constraint the curation must also satisfy, and occupancy and
coverage are coupled in this pool:

| source family | coverage | mean occ | n ≥ 0.1548 |
|---|---|---|---|
| `random*` (5) | 1.121–1.131 | 0.034 | **0** |
| `apt*` (5) | 1.214–1.280 | 0.077–0.090 | 355 total |
| `p2e*` (5) | 1.302–1.370 | 0.100–0.128 | 649 total |

Pearson r(source coverage, source mean occ) = **+0.522** over the 18 sources.
The whole `random` family — the cheapest low-occupancy material — tops out at
occ 0.1079 and contains **zero** episodes at side1's level.

So a low side0 cannot be bought at arbitrary coverage, and `cov_tol` is tight:
0.01246, about 1% of the ~1.25 coverage level.

*Caveat, stated because it bounds this argument:* the search's coverage is a
set-level kNN quantity computed over the composed mixture, not the mean of its
sources' coverages. The table above is a source-level proxy. It establishes
that the constraint is live and that the scan omitted it; it does **not** prove
a compliant pair is unreachable.

## What would settle it, cheaply

Re-run `gate0_compose.py` on the existing index with candidate enumeration —
report, for every accepted q1 pair, `(side0 occ, side1 occ, docc, dcov)`
instead of only the argmax. That answers the one open question: does an
accepted pair with side0 ≤ 0.05 exist, and how low does side0 go while docc
stays ≥ 0.0250?

It is CPU-only, minutes, reads the frozen index, and writes to a side path —
no registered output touched. The 3× slack between the winner's docc (0.0746)
and the floor (0.0250) suggests room to trade separation for a lower side0,
but that is a guess until enumerated.

**Ops has not run it**, because the enumeration is one line from being the
selection-rule change itself, and the decision about that rule is the owning
chat's.

## Status

* Step (1) diagnose — **DONE** (this document).
* Step (2) re-curation — **BLOCKED** on an instrument decision.
* Step (4) 32 fits + 32 adapts — **NOT SUBMITTED**. Its input buffer does not
  exist. Submitting the registered fits against the *current* side0 would
  rebuild the exact configuration the read already refused.
