# Routed repair kill (algorithm Tier-1) — ONE read (22 Aug 2026)

Registration: PREREG_routedrepair_20260820.md + Amendments 1–2. Bundle:
routedrepair_20260822_180310 (manifest verified 0 failed; witness
problems NONE; 8/8 refit counters == 500000; donor_sha pairing intact
at every seed across all three arms; _meta.donor_generation =
REFIT_20260808+seed5_completion_20260820, latest_ckpt_500k_8of8 true;
donor_provenance 5 july_resave + 3 fresh_aug_refit recorded in the
committed ops/waves/routedrepair/donor_provenance.json; modal n_ep 96).

## Verdict: **PROCEED** (both frozen criteria met, decisively)

| estimand | value | rule |
|---|---|---|
| D1 paired [repair − apt-unfrozen control], n=8 | **+201.09 [+162.26, +250.93], perm p = .0078** (d_z 2.96) | fires at α=.05 (exact 2^8 sign-flip minimum) |
| D2 repair vs scratch anchor 147.6823 | repair mean **291.44**; Δ = +143.76 [+99.25, +203.18] | kill bar 122.6823 — met, and the arm sits ~2× ABOVE scratch |

Arms: repair 291.44 · apt-unfrozen control 90.35 · shrink-and-perturb
125.57 (descriptive) · scratch anchor 147.68 · July pin 89.90
(descriptive, cross-era).

## What this settles

- An rgo-configured re-fit REPAIRS a sunk reward-free trunk that full
  fine-tuning cannot: the same 8 donors that lose to scratch under
  plain fine-tune (90.4, replicating the cross-era July pin 89.9
  almost exactly) reach 291.4 after the routed re-fit — above scratch,
  in fresh-rgo territory.
- S&P (125.6) recovers part of the gap but stays below scratch — the
  published-generic-remedy baseline does not explain the routed effect.
- **Fences carried verbatim**: amend1 M14 — this wave cannot separate
  "repaired the sunk trunk" from "overwrote it" (a 500k re-fit on the
  same buffer approaches a fresh rgo fit, and repair ≈ fresh-rgo
  levels is consistent with overwrite). PROCEED licenses the follow-up
  registrations (short-refit dose curve; rgo-from-random-init control;
  matched-budget comparison) — never the deployable salvage claim by
  itself. Self-scooping fence: standalone-algorithm-paper material.
- The apt-control replication (90.35 vs pin 89.90) is itself a
  noteworthy stability fact across eras/adapt invocations.
