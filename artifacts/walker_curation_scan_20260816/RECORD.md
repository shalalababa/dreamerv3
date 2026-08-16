# Walker curation feasibility scan (16 Aug 2026, CPU-local, non-registered)

Question (from the P-D2 drop-vs-rebuild decision): can ANY side0
curation of the walker collector pool meet the registered
proxy-separation bar (mean occ <= 0.05)? Data: the pool's own per-episode
occupancies from $RUNROOT/axis1_walker/episodes.json (the registered
cmd_index output, pulled via rcc-connect; regime horizontal_speed,
threshold 1.0, direction above, ep_len 1001). Pure analysis of existing
index data — zero new compute, no episode files touched.

## Result — COMPLIANCE IS TRIVIALLY ACHIEVABLE; the pool was never the problem

- Pool: 7552 modal-length episodes. **3430 have occ <= 0.05**;
  1052 <= 0.02; 7 exactly 0. Pool median 0.056.
- Unconstrained side0 floor (mean of 200 lowest): **0.0067** — 7.4x
  under the bar. Largest N with mean <= 0.05: **5877**.
- High side has headroom too: top-200 mean **0.336** vs the built
  side1's 0.1548 (which passed its gate); 1144 episodes >= 0.1548.
- The built side0 (occ 0.0802 = occ_search, mixture dominated by
  */mid strata, only 5 zero-bin episodes) therefore reflects the
  SEARCH/CURATION step — matched-pair + covariate constraints
  compressing separation, or a misconfigured stratum target — not pool
  poverty.

## Consequence

The rebuild-risk argument ("might refuse again") is dead: a side0-only
re-curation has enormous slack. Before re-running, ops should diagnose
WHY the registered search landed at 0.0802 with 3430 compliant episodes
available (search invocation / pairs.json constraints) — otherwise the
same result recurs. If side1 is kept byte-identical, the rebuild costs
one re-curation + 32 fits + 32 adapts (RCC background, no rented
instances) + a fresh read attempt (the ONE execution was not consumed).
