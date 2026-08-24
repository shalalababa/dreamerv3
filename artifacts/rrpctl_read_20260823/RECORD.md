# Routed-repair control wave (rrpctl) — ONE read (23 Aug 2026)

Registration: PREREG_rrpctl_20260822.md (frozen 6ec78ee2, BEFORE any
job; bundle commit b10233db after). Bundle: rrpctl_20260823_140000
(manifest OK; witness problems NONE; all review-hardened gates clean —
rgo-arm pins on both fit sets, init-sha EQUALITY verified per seed,
phase-2 recipe pins + realized replay ...axis1_finger/q1/side1,
16 adapt witnesses at counters 125000 under four-signal evidence,
modal n_ep pinned 96).

## Verdict: **NO-CALL-UNDERPOWERED at E1** (registered branch; E2
## descriptive)

| quantity | value |
|---|---|
| repair (pinned, seeds 1–8) | 291.44 |
| **rca** (fresh rgo 500k, seeds 17–24) | **394.66** |
| **rcb** (matched-budget rgo→rgo 1M) | **365.88** |
| E1 repair − rca | **−103.22 [−213.52, +8.95], p = .111** — no fire, CI not inside ±100.54 |
| E2 repair − rcb (descriptive) | −74.43 [−184.67, +25.32], p = .207 |
| E3 paired rcb − rca (descriptive) | −28.78 [−173.2, +83.6] — the second 500k rgo phase adds nothing |
| realized MDE80 (E1) | 170.1 |

## What this settles — and what it newly reveals

- **The salvage claim is dead in practice.** E1 did not fire positive,
  so SALVAGE-CONFIRMED was never reachable and E2 carries no verdict
  weight. Worse for the story: the POINT estimate says a fresh rgo fit
  BEATS the repair arm by ~103 (CI barely includes 0). Nothing
  licenses "the sunk trunk adds value"; the registered NO-CALL means
  even "refit-don't-finetune" is not formally certified (that needed
  the bounded branch).
- **What survives**: the routed-repair WITHIN-WAVE result stands
  untouched — repair beat plain fine-tuning of the same donors by
  +201* in a paired same-invocation design. The licensed ladder is
  now: "routing rescues relative to fine-tuning a sunk trunk (paired,
  solid); whether the sunk trunk adds anything over a fresh rgo
  restart is UNDETERMINED, with the point leaning no-or-negative."
- **The wave's most important by-product is methodological**: rca
  (394.7) and the valuefree read's q1s1 (295.5) are the SAME fits
  under the SAME adapt protocol in different invocation batches —
  a ~+99 cross-invocation shift on nominally identical cells. At n=8
  this instability is as large as the effects under study. It (a)
  explains the NO-CALL, (b) validates the prereg's cross-cohort
  disclosure as load-bearing, and (c) CAUTIONS every cross-cohort
  unfrozen-AUC comparison in the program — within-invocation pairing
  (the carrier/routedrepair design) is the only currency that has
  held. Consistent with the 1-Aug device-nondeterminism memory, now
  measured at adapt level.
- E3: the second 500k rgo phase is worthless (−28.8 ns) — no support
  for the parked short-refit dose idea from the budget side either.
- Follow-up option (a NEW registration, user's call): a single-batch
  within-invocation three-arm wave (repair′/rca′/rcb′ paired by seed,
  one venue, one batch) would decide E1 cleanly at ~75 GPU-h. Given
  two adverse point directions, the honest prior on salvage is now
  low; the algorithm paper can ship on the within-wave refit claim
  alone.
