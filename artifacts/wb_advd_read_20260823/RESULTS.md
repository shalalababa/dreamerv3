# Wave B (adversarial amortized actor) — FIRST INVOCATION, 23 Aug
2026: carrier wing INSTRUMENT-WEAK by its own α-gate; the ONE
execution is PRESERVED (side file only)

**Prereg:** `PREREG_waveb_advd_20260822.md`. Bundle
`wb_advd_20260823_083129` verified byte-exact; 16/16 adaptations
done; probe panel one GPU job; WM-IDENTITY (bitwise freeze) green
16/16; env-identity green; 0 excluded, 0 degenerate.

## Status: NOT AN ADJUDICATION

- **AMBIENT (the Tier-2 carrier): NOT-ADJUDICABLE-INSTRUMENT-WEAK
  — in-model dominance gate 6/8 vs the registered α-bar 7/8.**
  Failing runs: s10 (margin −11.9 %) and s13 (−2.2 %).
  [CORRECTED 23 Aug, amendment review finding 4 — the original
  text here claimed these were "the panel's two least-converged
  actors"; the actual plateau ranking is s10 1.259 > s17 1.247 >
  s13 1.243, so s13 is THIRD and gate-passing s17 is second. The
  gate, not the plateau row, is the registered catch.]
  The review-raised bar (WB-B2) did exactly its designed job:
  under the pre-review 6/8 bar this panel would have adjudicated.
- **LOCALIZED: LOCALIZED-CONTAINED-MATCHED** (gates 8/8; the
  interaction is ≈0: +3.4e-5, p = .50, hetero-positive 3/4) —
  computed, but per the carrier-keyed rule it RIDES THE SIDE FILE
  and is not a consumed registered outcome. Content: the actor
  raises attribution on BOTH A1 wings about equally — no
  hetero-specific (spatial) component.

## What the side-file rows show (DESCRIPTIVE ONLY — no primary was
adjudicated; signs disclosed as seen, which is why any continuation
must be governed by the pre-registered gate, not by these rows)

- Ambient Δ = attr_d(advd) − attr_d(policy): **positive in 7/8**
  (+0.0016…+0.0054; the one negative, −0.0118, is gate-failing
  s10). Against policy attribution levels of 0.012–0.022 these are
  +15–30 % relative.
- **The imagined-vs-realized gap is CLOSED: ratio 1.00–1.11 in
  16/16.** Wave A's greedy replanner over-imagined attribution
  ~10×; the amortized attacker's imagined harvest matches its
  realized harvest. This is the registered secondary and it is a
  mechanism finding on its own: amortized training on the frozen
  WM eliminates the off-support-imagination defense.
- Localized: deltas positive 6/8 across both wings — a general
  uplift, no spatial specificity.

## The registered fork (user decision)

**(a) WB Amendment 1 — extend and re-read (recommended):** the
prereg named the in-model gate as the under-trained-actor catch;
the plateau rows independently identify s10/s13 as unconverged.
Amendment: continue ONLY the two gate-failing adaptations
(+1e5 steps each from their own checkpoints, ~4–6 GPU-h total,
staging their run dirs back), then quarantine ALL 16 probe outputs
and re-run the FULL probe panel in one fresh GPU job (panel-device
homogeneity preserved; every run re-rolled — no cherry-picking of
realizations), then the ONE (preserved) read at unchanged bars.
Disclosure carried: the continuation decision follows the
registered gate + plateau criteria, but the side-file delta signs
were seen; the re-read's fresh probe rolls are the mitigation.
**If the re-read fires, TRANSMISSION-AMPLIFIED is the
paper-changing outcome (Act-2 revives) — which is exactly why the
gate must be genuinely passed, not waived.**

**(b) Accept INSTRUMENT-WEAK:** the ambient wing is dead under
this prereg (Tier-2 unreachable — the security paragraph stays at
the landed Tier-1), the localized side-file result and the
closed-gap mechanism finding are kept as descriptive.

Artifacts: `se_advd_read_NOTADJ_20260823T091032.json` (this
invocation). `se_advd_read.json` does NOT exist — the execution is
unspent.

## FORK RESOLVED (23 Aug, user decision): option (a) — Amendment 1
rev 2 registered after one reviewer pass (3 BLOCKING applied)

`PREREG_waveb_advd_amend1_20260823.md`: extend ALL 16 actors
uniformly to 2e5 (NOT failers-only — the reviewer computed that
s10, the wing's lone negative delta, single-handedly moves the
primary from p=.328 to p=.0039, so a failer-targeted dose would be
selection toward firing; the uniform dose is outcome-independent).
Quarantine ALL 16 probe outputs; fresh 16-pair panel in ONE GPU
job ON RCC pinned to panel-1's GPU model (panel 1 ran on Midway3;
`FRESH_PANEL=1` = quarantine + bundle-tree refusal + zero-skip
success condition); then the ONE preserved read at UNCHANGED bars,
pinned invocation with `--expect_steps 2e5` (fit counters
discriminative natively). Exact-one-look rule: a look = full panel
loads (excluded_runs==[], 8+8); ops aborts repair-and-rerun
without consuming it; a second carrier <7/8 ⇒
INSTRUMENT-WEAK-FINAL, fork (b) automatic. Localized wing's second
roll registered: panel-2 cell is the consumed one, panel-1 value
reported as instrument-sensitivity, changes disclosed as two-look.
Wave spec `ops/waves/wb_advd_ext/spec.yaml` (16 runs); read bundle
= `wb_advd_r2` (RCC-built, post-panel).
