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

---

## SUPERSESSION NOTE (16 Aug night, owning chat)

The headline above — "COMPLIANCE IS TRIVIALLY ACHIEVABLE; the pool was
never the problem" — is **WRONG as stated** and superseded by
`DIAGNOSIS.md` (ops): it was computed on the MARGINAL per-episode
distribution, ignoring the composed constraints. Under the registered
criteria (200-episode candidates, coverage matching, separation,
overlap cap) the reachable low-side floor is **0.05318** — zero of
2715 accepted pairs meet the 0.05 bar — because occupancy and coverage
are coupled in this pool (r = +0.522; the low-occupancy `random`
family is also the lowest-coverage material). The scan's numbers stand
as facts about the marginal distribution; the feasibility conclusion
drawn from them does not survive the constraint set.

**DECISION (owning chat, 16 Aug): P-D2 carried as
MEASURED-INFEASIBLE-AT-SPEC** — no rebuild, no criterion relaxation,
no pool re-sampling. Rationale: (a) re-sampling until a pair clears
the bar is selection-on-the-gate; (b) even the best available pair
(docc 0.0254 ≈ ⅓ of the built pair, ~11× below reacher's realized
0.279 dose) would make the wave near-certainly an uninterpretable
under-dosed null; (c) relaxing coverage matching would re-admit the
confound the matched-pair design exists to kill. The paper's walker
sentence upgrades from "refused at the gate" to "refused at the gate,
with an archived constrained enumeration proving no compliant pair
exists in the collected pool — a domain-structural occupancy–coverage
coupling", and reacher carries the third-domain generality leg alone.
Lesson recorded to memory: marginal feasibility ≠ constrained
feasibility — enumerate under the full constraint set before
concluding a curation target is reachable.
