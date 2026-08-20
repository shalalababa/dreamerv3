# AMENDMENT 2 — PREREG_routedrepair_20260820 (20 Aug 2026, pre-outcome)

**Scope**: ONE ops-procedure clause of Amendment 1's donor-generation
declaration — the seed5 completion METHOD. No estimand, decision rule,
gate, or reader constant changes. Nothing has been submitted; the wave
starts in ~2 days when the fleet frees.

## The finding (ops, byte-for-byte donor comparison)

| fq1s1 donor | July | August | relation |
|---|---|---|---|
| seeds 1–4 | 500000 | 500000 | **bitwise identical** |
| seed 5 | **500000 complete** | 200000 aborted | differ |
| seeds 6–8 | (seed6: 200000) | 500000 | fresh August fits |

Matches `PREREG_refit_reads_amend2_20260810`: most fq1 "regenerations"
were bitwise re-saves under August names; only six apt fits were
genuinely fresh. So the "August generation" is factually a MIXED set —
seeds 1–4 are the July weights, 6–8 are fresh draws — and seed5's July
500000 is complete while its August entry is an aborted fresh fit.

## Decision (supersedes Amendment 1's "re-run the refit recipe for seed5")

**Complete seed5 the way seeds 1–4 were completed: re-save its July
500000 checkpoint under the August name** (seconds, zero compute). This
leaves seeds 1–5 the same kind of object (July weights) rather than
making seed5 the only fresh draw among them; seeds 6–8 remain fresh
either way, so provenance is mixed under both options and the estimand
never sees it — every decision quantity is a within-donor paired
contrast (D1) or an arm mean over the same donor set (D2), and the
in-wave apt-unfrozen control shares each donor exactly.

## Disclosures

1. The witness `_meta` additionally records `donor_provenance`: a
   per-seed map with values `july_resave` (seeds 1–5) / `fresh_aug_refit`
   (seeds 6–8). RECORDED, not gated — the existing `donor_sha` pairing
   already pins the exact bytes per seed; this field makes the mixture
   auditable.
2. The `donor_generation` string stays
   `REFIT_20260808+seed5_completion_20260820` — clarified semantics: it
   names the RESOLVED 500000 checkpoint set under `latest_ckpt()`, not a
   claim that the set was freshly trained in August.
3. Amendment 1's B13 framing softens for seeds 1–5 and is corrected
   here: the July fq1s1 WEIGHTS survive (as bitwise re-saves) — what was
   lost is the July-era ADAPT runs the 89.8984625 pin was computed from.
   The pin therefore stays descriptive (adapt-era differs; Midway3
   adaptation is not job-to-job deterministic) and the in-wave control
   stays the decision baseline — unchanged, now with the sharper reason.

**Ride-along commit**: this file + STUDY_LEDGER.md.
