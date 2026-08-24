# NOBOOT (bootstrap decomposition) registered read — 22 Aug 2026

**Prereg:** `PREREG_nfi_scale_exhibit_amend4_20260820.md` §2 (review
#26). ONE execution of the frozen `se_noboot_read` ON RCC, bundle
`se_noboot_20260822_121232` (manifest 9,049/9,049 OK). This is the
RE-RUN after the 21-Aug incident (the first wave's
`DISAG_BOOTSTRAP=False` was silently unread on a stale producer
checkout; spoiled runs quarantined `_SPOILED_BOOTON`; incident note
`ops/waves/se_noboot/INCIDENT_20260821.md`; this wave ran with the
BIND-CHECK live). Instrument CLEAN, ckpt gates OK, degeneracy gate
clean (levels/intrinsic comfortably above 1/10 of the Stage-1
minima: intrinsic 0.0021–0.0025 vs ref-min 0.0019), fit OK ×4.

## Outcome: **cell (a) — MISPRICE-SURVIVES-LARGE**

- **Primary:** distractor θ₁ with bootstrap OFF (members differ by
  initialization only — the published Plan2Explore convention):
  **4.971 [4.804, 5.265]**, per-run 4.73 / 5.39 / 4.88 / 4.88,
  **4/4 per-run p < .05**. CI entirely above the 2.0 band; the
  point estimate sits essentially AT the bootstrap-on Stage-1
  anchor (5.26 [5.02, 5.46], descriptive cross-wave).
- **Secondary:** dup0 θ₁ = **0.9998 [0.9986, 1.0008]** — inside the
  par band; `parity_bootstrap_independent = True`.

## What this settles

1. **Review #25's A2 blocker is discharged empirically.** The
  undisclosed `disag_bootstrap: True` was flagged as potentially
  load-bearing ("independent 80% subsampling manufactures the
  aleatoric floor"). Answer: the 5.26× misprice is **NOT
  bootstrap-manufactured** — turning bootstrap off moves θ₁ from
  5.26 to 4.97, a ~6% change against a 5× effect. The disclosure
  stays in the paper; the caveat dies. Initialization diversity
  alone sustains the full misprice.
2. **The duplicate exact-parity null is also convention-independent**
  (0.9998 at par with bootstrap off) — the two-sided signature
  (farms irreducible stochasticity, immune to redundancy) is a
  property of ensemble disagreement per se, not of the resampling
  trick.
3. **B3 taxonomy cell filled (free, per the plan):** the
  bootstrap axis is a near-inert column of the channels × proxies
  matrix.
4. Paper-5 writing consequence: the flagship exhibit's A2
  disclosure paragraph takes the "robust" branch; the venue call
  no longer waits on anything from this axis.
