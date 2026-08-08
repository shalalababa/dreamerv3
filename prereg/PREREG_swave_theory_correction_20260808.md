# PREREG: s/w_r theory correction + wave design constraints — 2026-08-08

Frozen BEFORE any s-axis or w_r buffer/config exists (user GO on the
s×w_r pair, 2026-08-08). This file registers the theory-side correction
and the design constraints the review (§6/§27/§33) showed are load-bearing;
the wave's own primaries/power freeze in a separate
`PREREG_swave_<date>.md` once the buffers are materialized.

## Registered theory correction (must precede the read)

`Theory_SpectralTransfer_20260717.tex` derives the label-energy term
`a² ≈ s²f` **for a squared-error reward loss**. The apparatus trains
`symexp_twohot` cross-entropy (255 bins uniform in symlog space, spacing
20/127): the CE gradient is L1-bounded by 2 regardless of label magnitude,
and label scale moves only the TARGET SEPARATION, logarithmically —
in-regime vs zero target separation grows as `ln(1+s)/ln 2` (binary
reward). **Registered instrument-form prediction: the apparatus responds
like `ln(1+s)²·f`, not `s²·f`** — relative response at s=10 vs s=1 is
(ln 11/ln 2)² ≈ **11.97×** vs the raw form's 100× (CORRECTED per batch
review M9: the earlier 5.75 was ln(11)² without the ln(2)² baseline —
mixed normalization). A null on `s`
under the raw-units reading is NOT a refutation of the spectral model;
adjudication uses the instrument form. (Never surfaced before because
s ≡ 1 in every buffer ever fit.)

## Registered design constraints (violations invalidate the wave)

1. **`s` must be paired with `w_r`** (`agent.loss_scales.rew` override —
   pure fit-pressure knob, published non-null in the online setting:
   HarmonyDream w_r ∈ {1,10,100} ≡ s ∈ {1,3.16,10} under their MSE loss).
   Running `s` alone is registered-uninterpretable: inert-`s` ⇔ "label
   energy is not the mechanism" OR "twohot is nearly scale-free by
   construction". The pair separates label-energy from fit-pressure.
2. **Own-label scoring**: any `rew_nll` on an s-scaled fit is measured
   against SCALED labels via
   `relabel_replay transform-probeset --kind scale --scale <s>`
   (instrument built + selfchecked 2026-08-08) — the srd override-panel
   precedent; true-label NLL on a scaled fit conflates mislocation with
   calibration.
3. **Buffer identity**: the `scale` kind preserves every non-reward key
   byte-for-byte and the reward SUPPORT exactly (selfcheck-enforced), so
   the s-arm is the study's only fully composition-invariant dose axis —
   including the stored `dyn/*` context (neutralizes review D9). The
   w_r arm needs no relabeling at all (config override on the SAME
   frozen buffer).
4. **Dose points**: s ∈ {1, 2.45, 4.46} on the lo (f = 0.054) side —
   2.45 = raw-units occupancy-match √(0.323/0.054), 4.46 = head-space
   match (symlog-separation doubling is s = 3; √6-separation ⇒ 4.46);
   w_r ∈ {1, 10, 100} matching HarmonyDream's published span. Arm rgo
   (the confirmed carrier path), 8 seeds/cell. Grid subject to the wave
   prereg's power section; these constants are frozen as the
   theory-relevant crossings.
5. **vgo repair leg (optional but pre-authorized)**: one vgo cell at the
   top dose — `valnorm.impl: none` means the value path is scale-live
   (record §4.21 correction), so a waking vgo converts P-B2 "inert" to
   "below threshold" (theory repair). If dropped for budget, drop is
   disclosed in the wave prereg.
6. **Registered outcome map (theory adjudication only; behavioral
   primaries freeze in the wave prereg):**
   - rew_nll (own-label) tracks `ln(1+s)²` and w_r independently ⇒
     instrument form confirmed; theory keeps a corrected a²-term.
   - `s` inert at every dose while w_r moves ⇒ label ENERGY is not the
     mechanism — the count of rewarded frames is (first positive account
     of the volume anomaly; feeds Paper-3 D3).
   - both inert ⇒ the a²-term is refuted in the tested range (registered
     honest negative for the theory).
   - behaviour moves without membership moving (or vice versa) ⇒ another
     inclusion≠usefulness point, on a new axis.

Freeze ordering: commit this file BEFORE any scaled buffer or w_r config
is created; the wave prereg cites it and may not weaken constraints 1–3.
