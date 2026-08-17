# SE D-ONLY INTERFERENCE READ (registered) — 16 Aug 2026

**Governance:** Amendment 2 §1 (frozen; reader `uncfield/se_donly_read.py`
built + selfchecked + reviewed #23 BEFORE the arm's compute); ONE
execution on bundle `uncfield_se_arms_20260816_224710` (11290/11290
sha OK, committed manifest; wave se_arms, Midway3; 4/4 runs DONE).

## VERDICT: NO-INTERFERENCE (clean instrument)

**With the distractor removed entirely, the duplicates STILL price at
exact source-parity** — dup0 θ₁ = 1.0001, BCa [0.9991, 1.0006] (vs
Stage-1's 1.0000 [0.9989, 1.0004] with N present: indistinguishable);
0/4 indicators on every D channel. **Stage-1's D-null stands
UNQUALIFIED: N did not suppress duplicate attribution — there is
simply no redundancy farming in this substrate, with or without a
competing stochastic channel.**

Registered detail: dup1 θ₁ 0.9988 [0.9982, 0.9994]; dup2 0.9820
[0.9812, 0.9832] (min/max 0.9808/0.9835) — the ε-ladder's slight
DEPRESSION at ε=0.5 replicates Stage-1's 0.973 (the noise inflates the
per-dim normalizer, not the disagreement), now in an N-free
environment. The registered outcome-map cell is (a): dup0 θ₁ CI
entirely within the [0.90, 1.10] materiality band AND no channel at
4/4 indicators (max observed 0/4 — below even the coin-null
expectation).

## Gates

Seeds 10–13 distinct (config-derived) ✓; S=512 ×4 ✓; calibration max
0.036 (bar 0.5) ✓; provenance p-equality ×12 channels ✓; fit counters
OK ×4 (metrics last step ≥ 494k of 5e5, snapshot witnesses ~495–499k);
distractor-absence asserted ✓. Reader ckpt_flag = UNVERIFIED (full
ckpt dirs deliberately not bundled — BUNDLE_NOTE.md), discharged
MANUALLY: probed ckpt basename == shipped `ckpt/latest` content in
4/4 runs (recorded above; the frozen reader was not edited).

## Licensed wording

"Removing the stochastic channel does not release any redundancy
farming: exact duplicates remain priced at par with their source
(θ₁ = 1.000 [0.999, 1.001], n=4) — the fictitious-value mechanism
requires irreducible stochasticity, and the Stage-1 duplicate null is
not an artifact of cross-channel competition." Registered robustness
check (n=4), reported as such.
