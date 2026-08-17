# LeWM / JEPA-class wave — registered ONE read (16 Aug 2026)

Registration: PREREG_lewm_20260812 (freeze 7b51ad58) + Amendment 1
(d1b797b2, verify-interface) + **Amendment 2 (PREREG_lewm_amend2_20260816,
this session, pre-outcome): same-device pe anchor panel** — the reader's
PE_E4_SHA256 pin swapped from the historical csv (9907d733…, device
unrecoverable) to e4_fingerpx_v1_pe_rtx6000.csv (sha 8bd7a683…),
collated by the registered stratified_error collate from the 16-cell
re-measure bundle pe_e4_remeasure_20260816_174813 (49 files sha-OK; all
cells midway3-0282 Quadro RTX 6000, per-cell device sidecars, job
53421265 — same device class as the jp E4 panel). Full-panel diff vs
historical: mean −0.172 nats, per-cell sd 0.32, range [−0.82, +0.26] —
cell-level scatter, not a clean offset; the same-device swap removes it
by construction. Direction-disclosed conservative (pe shifts DOWN ⇒
disfavors the fire branch). jp outcomes unobserved until this
execution; reader selfcheck re-PASS after the pin swap.

Main bundle lewm_20260816_121536 (124 files sha-OK): jp fit counters +
ckpt steps + enc_checks + integrity sweep green; fidelity 16/16 ≥ 0.5
floor (holdout R² ≈ 0.75); probes device-uniform (fpx panel on one Vast
RTX 5060 Ti after the devcheck re-unification; lewm/embed probes per
bundle NOTES). Conformance note: the bundle's flattened fidelity
filenames were mechanically restored to the registered
lewm_distill_finger_s*_seed*/fidelity.json layout (inputs/fidelity/).

## VERDICT — P-J1 PARTIAL (exactly one representation leg fires: emb)

- **p_j1_rep (graft leg, paired [pe − jp] rew_nll_in): NO-SEPARATION**
  — −0.905 [−4.61, +3.28], perm p=.6875, n=8. jp seed-cluster level
  23.76 [21.90, 25.91] — AT the pixel-anchor level; bands:
  inclusion_restored=False, below_swamping_band=False. The distilled
  LeWM encoder grafted into the WM does NOT escape the pixel null.
- **p_j1_emb (embedding probe, paired lewm − fpx AUROC): EMB-MORE-LEGIBLE**
  — +0.0297 [+0.0083, +0.0570], perm p=.032, d_z 0.58, n=16
  (lewm 0.5326 vs fpx 0.5028 — a small but significant legibility edge
  at near-chance absolute levels).
- **p_j1_beh: FLOOR-CONSISTENT** — −15.4 [−32.5, +2.4] ns vs the X2
  baseline; no behavioral lift.

Registered reading: the predictive-latent objective shows a
representation-level legibility edge that does NOT survive the graft
route into a consuming WM and has no behavioral consequence at this
substrate — the reward-free pixel null is objective-general at every
consequential level tested, with the emb-level vestige reported.
FB/successor-features conditional: this is neither the
both-legs-fire branch nor the clean strengthened-negative — the
FB decision (deferred on this read) now takes the PARTIAL outcome as
input; user call queued in TODO.
