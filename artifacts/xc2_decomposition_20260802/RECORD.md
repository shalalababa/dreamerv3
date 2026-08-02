# XC2 excursion decomposition — EXPLORATORY post-read (2026-08-02)

Decomposes the xc read's registered estimand
(`artifacts/r3_xc_read_20260802/`, verdict MIXED — off-map XC2
negative excursion) into its two chooser terms. EXPLORATORY: cannot
alter the verdict; informs whether any follow-up deserves
registration. Plan (the two terms, per-domain splits, agreement
structure, opportunity concentration) was fixed in chat BEFORE
execution; the script + full json live here (`decompose.py` rev 3 —
rev notes in its docstring; rev 2/3 fixed only the broken
concentration leg; term/anchor values identical across revs).

Identity: `d_pair = real_term − now_term` with
`real_term = g_all[m_real_x] − g_all[m_real]` (probe-pick value change
under consumer heads) and `now_term = g_all[m_now_x] − g_all[m_now]`
(plug-in-pick value change). Anchor: reconstructed pooled d_pair
matches the read's primary points EXACTLY on both sides (asserted).

## Headline: the excursion's significant carrier is the PROBE-PICK term

Run-clustered (B=10K, rng 0, 32 clusters/side — the registered
estimand's own weighting), eval=early side (consumer = late heads):

- **real_term −0.152 [−0.269, −0.052] — CI entirely negative.**
- now_term +0.004 [−0.112, +0.119] — tight null.

So at the estimand's weighting the "harm story" is confirmed: late
heads make the reality-probe chooser pick genuinely lower-value
actions on early-agent states. The "self-consistency story" (better
mature plug-in shrinking the margin) nets to ZERO at run level.

## The asymmetry (strongest new fact)

The late-eval mirror (consumer = early heads): real_term +0.018
[−0.110, +0.144] — early heads' probe picks are value-NEUTRAL on late
states, on floors and non-floors alike (+0.018 both strata). The harm
is DIRECTIONAL: late→early damages, early→late does not. Generic
swap noise cannot produce this (compare now_term, which IS
side-symmetric — below). Consistent with directional co-adaptation:
mature heads rank by payoffs realized only under mature-policy
occupancy; on early states their preferences do not cash out.

## State-level anatomy (pooled, descriptive)

Floors dominate: 93.0% of early-side states (88.5% late) have
opportunity ≤ 0 (max(g_all) ≤ g_now; 17/32 early runs have constant
opp — matches the ladder read's floor concentration). NOTE: d_pair is
NOT identically 0 on floors (candidates still differ below g_now).

| stratum (early side) | share | d_pair | real_term | now_term |
|---|---|---|---|---|
| floor | 93.0% | +0.155 | −0.067 | −0.223 |
| non-floor (opp≈19.6) | 7.0% (449 states) | −4.272 | −1.267 | +3.005 |

- The action is concentrated in the small non-floor minority, where
  BOTH terms are large and opposite.
- **now_term is side-SYMMETRIC** (non-floor: +3.00 early side, +2.19
  late side; floors: −0.22 / −0.30) ⇒ mechanical, not knowledge: on
  states where candidates genuinely differ, the agent's own argmax
  suffers optimizer's curse against its own value errors, so ANY
  decorrelated estimator's argmax evaluates better under true g_all;
  on floors the own ranking stays closer to the least-bad candidates.
  This term is not evidence about transfer in either direction — and
  it cancels at run level anyway.
- **real_term is side-ASYMMETRIC** (−1.27 vs +0.02 non-floor) — the
  directional fact above.
- Floor states contribute +0.155 to d_pair (via the consumer plug-in
  degrading, now −0.22), so floors DILUTE the excursion; without them
  the pooled contrast would be ≈ −0.30, not −0.155.

## Concentration

16/32 early-side runs have any non-degenerate opportunity;
finger/e1 dominates (7 runs) and holds the worst per-run real_term
(−1.15, −0.93, −0.88, −0.74 all finger/e1; next cup/e4 −0.62).
Per-domain run-clustered real_term: finger −0.240 [−0.452, −0.059],
cup −0.063 [−0.155, +0.005]. Chooser agreement ~25% (75% changed
picks); on changed picks real_term −0.194. Per-run
spearman(d_pair, opp) mean −0.26 (15 valid runs) — more negative
where opportunity is larger, as the stratum table shows.

## Consequences (recommendation, not registered)

1. The excursion is NOT an artifact of the achieved functional's
   baseline: the harmful component is the probe pick itself, and it
   is directional. Worth a real sentence in Route-B §6, with the
   registered interface-mismatch caveat attached (feature-era
   mismatch can still produce exactly this sign — that is what an
   interface-controlled swap would separate).
2. An interface-controlled swap (aligner-mediated head transplant) is
   now WELL-POSED: target prediction = the aligned swap removes the
   real_term deficit if interface mismatch, not if value-knowledge
   mismatch. NEW registration + instrument + reviewer if pursued;
   basis for effect is thin (finger-e1-heavy, 16 informative runs) —
   scale expectations accordingly.
3. Default remains: fold into §6 as disclosed MIXED + excursion +
   this anatomy; no new compute unless the user GOes the swap.

Everything here is exploratory/descriptive; nothing enters any paper
as a confirmatory statistic.
