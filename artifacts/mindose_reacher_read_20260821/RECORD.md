# MIN-DOSE reacher kill test — ONE read (21 Aug 2026)

Registration: PREREG_mindose_reacher_20260820.md + Amendment 1.
Bundle: mindose_reacher_20260821_192326 (rebuild after the 172503
bundle's registered refusal — first_draw attestation was absent; ops
then verified the attestation with a full evidence chain: single slurm
search job 53746920 + single build job, the pre-fit dated FILL naming
the invocation, build-dose proven RNG-free/deterministic, dose.json +
manifest timestamps inside the job windows, byte-identical
source_manifest_sha across both bundles — the refused bundle always
pointed at the registered draw; only the signature was missing. The
superseded bundle is retained as _SUPERSEDED_* with both manifests
tracked; an accidental `git rm --cached` of the refused bundle's
manifest was self-caught and restored in 8ad0328b). This bundle:
manifest verified, problems NONE, first_draw TRUE, occupancy 0.097837
in [0.08, 0.20], holdout_disposition no-probeset-exists (verified
empty), 8/8 fit counters 500000.

## Verdict: **PROCEED**

dose1 − scratch (reacher, seeds 1–8 paired protocol):
**+193.53 [+90.60, +340.88], perm p = .0066** (label-permutation +
BCa, capdescent lineage); realized MDE80 = 180.8. Both conjunction
legs fire.

## What this settles — and its registered fence

A level-1 curated buffer at ~10% occupancy (0.098) supports adaptation
that beats from-scratch training by ~194 AUC on reacher — the
minimal-dose direction SURVIVES its kill test, with the effect roughly
at the detection floor's scale (MDE80 180.8 < 193.5, so the wave was
adequately powered for what it found). Registered consequence (amend1
M13 budget-asymmetry, carried verbatim): **PROCEED licenses only a
matched-budget follow-up, never a recipe claim** — the dose arm's
curation cost is not budget-matched against scratch in this design,
so no efficiency headline may be written from this read.
