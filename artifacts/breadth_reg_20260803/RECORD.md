# breadth-causal wave — registration record (2026-08-03)

Registration-time evidence for `prereg/PREREG_breadth_wave_20260803.md`
(matched-f_R diversity pair, quad `q1d`), adjudicating the theory
freeze `prereg/PREREG_breadth_theory_20260803.md` (commit `481aced9`,
made BEFORE the feasibility scan ran — ordering verified in git).
Committed BEFORE any q1d buffer, fit, or adaptation exists.

## Chain so far

1. **3 Aug, GO** (user; reviewers pre-approved in advance).
2. **Theory freeze** committed `481aced9` pre-feasibility.
3. **Feasibility** (RCC login node, value-blind): scan 7424 episodes
   (288 unavailable / 3264 missing chunk files — availability rule
   active), dim 524; search OK: side0 occ 0.3038 / PR 7.495, side1 occ
   0.3037 / PR 12.336, separation 4.841 (pool 9.844, random-K 9.689);
   pair straddles the frozen reference 9.2300.
4. **Feature-space audit**: scan obs_keys ≡ frozen instrument obs_keys
   ([dist_to_target, dyn/deter, position, target_position, touch,
   velocity], 524 dims H=1 / 4192 windowed) across the reference json,
   both q1f jsons, and the scan — "support breadth" has included
   collector-latent breadth since spectral_v1_1 was registered; all
   compared cells share the space. Disclosed in the wave prereg.
5. **Wave registration + frozen reader built** (this record).
   `analysis/breadth_read.py` selfcheck PASS. Run-name family `bdq1d`
   glob-checked CLEAN against runroot_cleanup DELETE-SAFE at design
   time.

Files here:
- `freeze_shas.txt` — sha256 of the registered files (pre-review;
  post-review finals appended after the reviewer pass).
- (to be added by the user BEFORE any fit, per the prereg):
  `div_pairs.json` + `div_moments.meta.json` copied from scratch —
  the bit-identical build input and the scan provenance.

## Review (2026-08-03, ONE adversarial reviewer Opus 5, pre-approved)

**2 BLOCKING / 4 MAJOR / 8 MINOR / 5 NIT — all B+M fixed +
machine-verified** (full itemization in the prereg's Review-provenance
section). The BLOCKINGs:

- **B1 — the manipulated index is 97.7% collector-latent.**
  diversity_pr's 524-dim space is dominated by `dyn/deter` (512 dims)
  — the COLLECTOR's stored recurrent state, which the fitted WM never
  encodes (not in obs_space; 1-frame replay-carry overwritten by
  Replay.update; excluded by the repo's own probe-set builder as a
  collection artifact). A "BREADTH-CAUSAL CONFIRMED" could have fired
  from a manipulation of a quantity invisible to the trained model.
  Fixed: `curate_frew block-pr` (moments + replay modes, selfchecked
  exact to 1e-9 against the instrument on built buffers), an
  OBSERVABLE-SUPPORT gate (proprio-block PR separation ≥ 0.5, HALT +
  reader-enforced), pre-freeze block disclosure registered, latent
  descriptives recorded, verdict vocabulary re-worded to "observable
  support".
- **B2 — rebuild silently doubles a side; every registered gate was
  duplication-invariant.** The builder appends into an existing quad;
  f_rewarded and PR are frame-multiset-duplication-invariant, so a
  crash-then-rebuild (the exact chunkgap incident this pool produced
  on 2 Aug) would have passed all HALT gates and burned 32 jobs before
  the read's n_episodes assert. Fixed: `test ! -e q1d` + `check-pairs`
  preflight (sha identity + decision-OK + member-chunk availability
  re-check; selfcheck trips on sha mismatch / OFF-TARGET / purged
  member) + `n_episodes == 200` promoted into the HALT rule +
  registered recovery rule (delete the quad entirely; never
  re-search).

MAJORs: M3 wave identity pinned (div_pairs.json committed + sha-
enforced; feasibility commands registered verbatim); M4 vacuous f-band
fixtures replaced (both-sides-moved + message-text assertions); M5
P-BD3 scoped as a gross-collapse screen (finger reward ≡ regime
indicator ⇒ trivial-predictor member; nll_out descriptives added); M6
P-BD2 curation-offset reading rule (+49.7/+56.9 prior). Mutation run:
**12/12 mutants CAUGHT on mirror copies** (band-widening, bound
deletion, point-vs-CI both directions, abs() on both separations,
percentile change, domain-filter drop, ref-pin loosening, proprio-gate
zeroing, cohort shrink), clean-copy control PASS. Both selfchecks
re-PASS at the post-review shas.

## Round-trip (2026-08-03, executed — READY TO FREEZE)

`div_pairs.json` (sha256 `48032a7598ac36c7651db060c34cf5bd68ffe6b442`
`848824aa32791f15184ba0`, pinned in the prereg's check-pairs
`--expect_sha` and in `freeze_shas.txt`) + `div_moments.meta.json` +
both block-pr disclosure outputs archived here.

- **Observable-support pre-check PASSES**: pair proprio-block PRs
  3.0359 (lo) / 3.7718 (hi), separation **0.736 ≥ 0.5**; n_rew
  60,776 / 60,760 (Δ 0.03% — sample-size confound dead).
- **Reference (q1v200/side1, replay mode): pr_full =
  9.230033291492585 — BIT-EXACT reproduction of the frozen instrument
  pin on real data** (live validation of block-pr ≡ spectral_measure);
  pr_proprio 3.2500 — the pair's proprio ordering straddles it
  (3.036 < 3.250 < 3.772), directionally coherent with the full index
  at roughly half its relative swing.
- Latent blocks (descriptive): PRs 93.4 / 110.2, trace shares
  0.380 / 0.467 — the full-space separation is latent-majority by
  trace, which is what the gate guards and why the licensed vocabulary
  is "observable support".

## Status

READY FOR FREEZE-COMMIT (prereg finalized with the sha pin + block
disclosure; both selfchecks PASS at the final shas in
`freeze_shas.txt`). After the commit: preflight → build → spectral +
block instruments → HALT gates → 16 fits + 16 adapts → E4 → ONE read.
