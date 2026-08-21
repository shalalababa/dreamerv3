# Listen-or-Leap exact-referee read — ONE execution (21 Aug 2026)

Registration: `prereg/PREREG_p2_lol_20260821.md` (frozen at commit
8828ab6f; RUN_SEED 20260827 after the registered rotation). Executed
once; output `lol_read.json`.

## Verdict: **CHAIN-TRACKS-REFEREE**

- **Recovery slope 1.0063 [0.896, 1.110]** (percentile bootstrap over
  36 cells, B=4000 — the decisive statistic): excludes 0, covers 1.
  Pooled recovered opportunity 2.371 vs exact referee 2.340.
- Bias diagnostic (adjudicates nothing; registered null OC: 15%/11.7%
  anti-conservative fire rates): +0.031 [−0.145, +0.218], perm p=.74 —
  split/CRN integrity clean.
- Substrate gate PASSED: exact EVSI > 0 at every config —
  q=.6: 3.46/2.83/0.94 (c=.05/.3/1.0); q=.75: 4.88/4.53/3.67;
  q=.9: 5.49/6.21/5.00 (dose–response in q and c as designed;
  visitation-conditional, descriptive).

## Meaning (P2-appendix leg)

On a substrate where value of information PROVABLY exists (exact
belief-MDP referee, verified against an independent history-space
solver), the program's estimator chain — split selection on even
repeats, odd-half evaluation, VoI-blind plug-in consumer, CRN repeats
— measures true VoI with slope ≈ 1 and no detectable bias. Combined
with the Tier-1 bundle (tm2 positive control fires; dv3 planted-signal
detection at 0.142 opportunity units), the "your instrument found
nothing because it cannot find anything" objection is now answered
three independent ways. The DMC nulls are statements about the
substrate and consumers, not about the chain.
