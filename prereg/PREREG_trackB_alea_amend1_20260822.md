# AMENDMENT 1 to PREREG_trackB_alea_20260822, 22 Aug 2026
(user GO 22 Aug "GO with option a, reviewers pre-approved";
revised same-day per the amendment review R-A1 — 5 BLOCKING + 9
MAJOR + 9 minor, all adjudicated pre-freeze; both selfchecks PASS
post-revision; freeze = commit of this file + the revised
instruments — the wave stays HELD until then)

**Trigger (registered §0, fired as written):** the det se_mask
baseline on the 8 Stage-1 checkpoints
(`artifacts/det_semask_baseline_20260822/NOTE.md` + its R-A1-M12
correction) shows the masked-delta form NULL on det runs — total
intrinsic |Δ| ≤ 0.36% of base in 8/8, mixed signs; own-dim decoded
delta 4/8 negative with 7/8 CIs straddling zero and the single
zero-excluding CI POSITIVE (seed 11: +2.04e-4, BCa [+4.5e-5,
+3.8e-4]). **Corrected reading (R-A1-M12): the null is GUARANTEED
BY CONSTRUCTION** — with gates/mod off the distractor is a pure
exogenous AR(1), so batch permutation across anchors is
law-preserving and the population delta is zero BY EXCHANGEABILITY
for ANY statistic, regardless of pricing. §0 detected a
form/estimand mismatch, not a mechanism. Consequence unchanged:
the masked-delta manipulation check and primary are re-specified.

## Re-specified design (share form, decoder-projected)

**Instrument (`uncfield/se_alea_mask.py`), at the same S=512
anchors:**
- **share_raw(key)** — key's share of decoder-projected,
  data-variance-normalized ensemble variance of the RAW members
  (the θ₁ machinery; the normalizer floor uses the se_probe real
  convention — PLANTED_KEYS excluded — R-A1-M6).
- **share_norm(key)** — the same projection of the
  ALEATORIC-WHITENED members. **R-A1-B3.1: the divisor is
  scale-normalized to unit geometric mean per anchor,
  σ̃ = σ / gmean_dims(σ)** — the whitening carries ONLY the
  cross-dim reallocation. Under uniform σ, σ̃ ≡ 1 EXACTLY and the
  transform is the identity up to decompose/reassemble roundoff
  (~2e-7 f32, perturbing shares at O(1e-10)) — so the R3-B1
  constant-rescaling pathology is excluded by construction
  (unit-tested: `whiten_members`, uniform-σ mutant). Residual
  decoder-curvature sensitivity is bounded by the recorded gain
  diagnostic. **R-A1-B3.2:** whitened-prob clip saturation
  (`share_clip_frac`) and whitening gain recorded per anchor and
  GATED (below).
- **REGISTERED SHARE UNIVERSE (R-A1-B5):** {distractor,
  planted_dup0, planted_dup1, planted_dup2, position, velocity}.
  `planted_const` is EXCLUDED from the simplex, the aliveness
  sums, and the bootstrap (se_probe review #21 B3: const is the
  decoder-projection floor diagnostic and never enters a fire
  decision); its rows are still saved. The universe is recorded in
  the json and reader-asserted.
- Per-anchor per-key rows in the npz (the reader's paired
  bootstrap); masked-variant machinery retained: **the distractor
  mask is a CONSTRUCTION-INERT exchangeability control** (a
  materially nonzero value indicates an instrument defect); the
  velocity and dup masks remain informative. Scope (R-A1-m23): the
  det §0 baseline carries only distractor/dup channels, so only
  those rows have a det counterpart.

**Re-specified read (`uncfield/se_b4_read.py`), in order:**
0. Validity as in the parent, PLUS: probe/mask seed == 0 and probe
   n_eval == 512 (same-anchors identity, R-A1-M8);
   cross-instrument share provenance — mask raw share ratios must
   reproduce se_probe's `key_share` ratios to 1e-4 (R-A1-M7a,
   tolerance per f32/f64 divergence R-A1-m21); θ₁ recomputed from
   the npz (R-A1-M7b); no universe key floor-bound (R-A1-m16).
   **A NOT-ADJUDICABLE partial wave writes a timestamped side file
   and does NOT consume the one execution (R-A1-M13).**
   Descriptives (share table, θ₁ rows, masked rows, diagnostics)
   are emitted BEFORE any branch (R-A1-m17).
1. DEGENERACY → REPAIR-BY-DEATH (unchanged; base_raw leg).
2. **INSTRUMENT-OUT-OF-SUPPORT (R-A1-B3):** `share_clip_frac` >
   0.10 or logvar-floor fraction > 0.25 in any run → validity
   cell, never a repair cell.
3. **MISPRICE cells (R-A1-M9):** θ₁ ≤ 1 in any run (healthy
   levels) → **MISPRICE-DISSOLVED-IN-TRAINING** — a positive,
   reportable behavioural outcome (the actor's visitation under
   the normalized reward dissolved the misprice in the learned
   model; the accounting repair is then untestable); 1 < θ₁ < 2.36
   (= 0.5 × the Stage-1 minimum 4.72) in any run →
   **MISPRICE-ATTENUATED** (the raw share is too small for the
   ratio to be identified).
4. **ALIVENESS (R-A1-B1, renormalization-invariant):** on
   CONDITIONAL shares q(k) = share(k)/(1 − share(distractor)) over
   the 5 non-distractor universe keys: (a)
   q_norm(velocity) ∈ [0.5×, 2×] q_raw(velocity) — exactly 1.0
   under a distractor-only repair (the naive unconditional band
   would have FAILED a perfect repair at the measured Stage-1
   geometry: distractor share 0.43–0.49 ⇒ mechanical inflation
   1.75–1.98× vs a 2.0 ceiling — and would have PASSED global
   flattening); (b) spread conservation
   Σ|q_norm − 1/5| / Σ|q_raw − 1/5| ≥ 0.5 (→ 0 under flattening);
   (c) anchor-cv floor cv(Σ whitened) ≥ 0.25 × cv(Σ raw),
   non-positive means an explicit failure (R-A1-m22). Any failure
   → REPAIR-BY-FLATTENING.
5. **PRIMARY:** ratio = share_norm(distractor)/share_raw(
   distractor). **DE-REPAIR-INVERTED iff ≥ 1 run's anchor-resample
   2.5th percentile > 1.0** (R-A1-B2 — a point ratio a hair above
   1 is PARTIAL, not INVERTED; percentile-bootstrap interval
   disclosed for this functional in place of house BCa,
   R-A1-m15). Else **REPAIRED iff ratio < 0.5 in 4/4**; else
   PARTIAL-REPAIR. **Precedence (R-A1-M14): DE-REPAIR-INVERTED >
   REPAIRED/PARTIAL; a mixed wave reports INVERTED with the full
   per-run vector.** Power note: 4/4 unanimity ≈ 1/16 null rate,
   no gradation.

**Supersession scope (R-A1-M11):** the parent §3 ALIVENESS GATE
(R3-B3: velocity-control delta_norm BCa/sign leg + cv leg) is
superseded IN FULL — its velocity leg is built on the masked form
that M12 shows is construction-inert for the exogenous distractor
and near-inert generally; the flattening concern it addressed is
carried by legs (a)/(b)/(c) above. The parent's §3 masked-delta
manipulation check and R_rel primary are likewise superseded in
full. Everything else in the parent (wave, head design, BIND-CHECK,
degeneracy, fences) stands.

**Unchanged:** the wave itself (4 runs, seeds 120–123, gauss head,
specs), the toy-verified head design, `scripts/se_b4_probe.sbatch`
(its skip rule now also requires the share block — a pre-amendment
instrument output re-runs instead of being skipped, R-A1-m20).

**Knock-ons registered elsewhere (R-A1-M12):** the B5 TM2 prereg's
fire channel is re-specified (permutation is construction-inert
there too — see PREREG_trackB_tm2_20260822 rev 2); the B2
APT-IMMUNE scope sentence for the writing chat is re-worded (the
distractor leg of the mask suite was construction-inert; APT's
immunity claim rests on the coupling-alive velocity contrast and
the level-form θ₁ comparison, which has no APT analog).

Freeze = this file + `uncfield/se_alea_mask.py` +
`uncfield/se_b4_read.py` + `scripts/se_b4_probe.sbatch` committed —
then ops releases the held wave.
