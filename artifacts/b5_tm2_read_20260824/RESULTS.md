# B5 (TD-MPC2 column, decoder-free Q-mask attribution) — REGISTERED
READ, 24 Aug 2026: **i-TM2-PRICES-DISTRACTOR — the primary FIRES
8/8, channel-specific**

**Prereg:** `PREREG_trackB_tm2_20260822.md` rev 2 (mean-substitution
fire channel per R-A1-M12; permutation retained as the
exchangeability-null calibration row). **Reader:**
`uncfield/tm2_qmask_read.py` (frozen, selfcheck PASS at freeze),
ONE execution 24 Aug 09:59 local (stamp git db44b3aa = HEAD),
bundle `b5_tm2_20260822_203207` re-verified against the committed
manifest immediately before the read (sha256 -c, 0 mismatches);
glob `runroot/tm2_b5/tm2se_s13[2-9]` — 8/8 loaded, 0 excluded, the
smoke dir untouched.

## Primary

**Δ(Q-ensemble std) under distractor MEAN-SUBSTITUTION < 0 in 8/8
registered runs** (bar ≥7/8, binomial 9/256 = .035; attained 8/8 =
.0039). Per-run deltas: −0.0094, −0.0107, −0.0033, −0.0120,
−0.0042, −0.0093, −0.0060, −0.0036. **FIRES ⇒
i-TM2-PRICES-DISTRACTOR**: TD-MPC2's uncertainty proxy — the
K=5 Q-head std — depends on the planted distractor's realized
values. Removing them (law-changing substitution by the
anchor-population mean) REDUCES value-ensemble disagreement: the
distractor inflates the proxy.

## Specificity (the registered iii-cell does NOT trigger)

Form-matched velocity mean-substitution: nonspec_runs = 1/8
(threshold: majority with |Δ_vel| > 0.5·|Δ_dist|) ⇒
**nonspecific = false**. Stronger than the registered test: the
velocity comparator is OPPOSITE-SIGNED — positive in 7/8
(+0.020…+0.088; one −0.003) — so "any off-manifold substitution
lowers the statistic" is refuted by sign, not just magnitude.
Substituting a real state-coupled channel RAISES Q-disagreement;
only the distractor's removal lowers it.

## Calibration and nulls (all as constructed)

- **Exchangeability-null row (rev-2's M12 repair, validated in
  data): distractor batch permutation per-run |Δ| ≤ 0.0035, zero
  defect flags** — the law-preserving intervention is inert exactly
  as the theory says, while the law-changing one fires. This is
  the in-data proof that the rev-1 design could never have fired
  and the rev-2 re-specification was the correct repair.
- Dup substitution + fresh-resample: |Δ| ~ 1e-4–1e-3 (≈0, the
  duplicate channels carry nothing).
- Velocity batch permutation (coupling teeth): +0.057…+0.216 in
  8/8 — the state-coupled channel responds to permutation as it
  must, confirming the instrument has teeth.
- Validity: 8/8 audit + instrument pins green (ckpt_late, S=512,
  num_q=5, mpc=True, extra-key order, realized steps ≥ 100k), dup0
  vector-zero, float64 provenance, no partial flag.

## Consequences

- **The cross-family misprice matrix's second architecture column
  fills with a PRICES verdict**: dreamer-disagreement farms the
  planted stochasticity (θ₁ 5.26×, duplicate-immune); APT is
  immune to both (B2, sharp); **TD-MPC2's Q-ensemble std prices
  the distractor — channel-specifically and in the deflation
  direction** (its realized values inflate value disagreement).
  The misprice is therefore NOT a reconstruction/decoder artifact:
  a decoder-free, value-side uncertainty proxy in a different
  architecture family carries its own version of it.
- The decoder-free attribution instrument (tm2_qmask) is the
  registered contribution this column was gated on — first use is
  clean (all calibration rows behave).
- Registered caveat carried: Q-std measures VALUE disagreement;
  this cell is about the uncertainty proxy's accounting, and TM2
  as configured has no intrinsic bonus — no exploration-behavior
  claim is licensed either way.
- [writing chat] taxonomy table: TM2 column = PRICES (deflationary
  sign, specificity by sign reversal); B2's APT-IMMUNE restatement
  (M12) and this row land together; the "misprice is
  architecture-general but proxy-dependent" framing is now
  evidence-backed on three families.

Artifacts: `tm2_qmask_read.json` (the consumed execution).
