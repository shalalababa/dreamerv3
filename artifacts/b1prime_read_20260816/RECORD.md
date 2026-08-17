# B1' 600k capacity point — registered ONE read (16 Aug 2026)

Registration: PREREG_b1prime_20260815 (freeze f8f1fcef 15 Aug, pre-wave).
Reader b1prime_read.py (frozen same commit, clean), selfcheck PASS
pre-execution. Bundle b1prime_20260816_174500: 3622 files sha-OK; 16
fits (counters 16/16 update==total; done-ckpts 16/16 @500000; b1prime's
4 pull-clobbered witnesses were repaired with authentic producer bytes
by ops pre-bundle) + 16 readout64_frozen adapts (collate 16/16 QC pass,
modal n_ep 96).

DEVICE PROVENANCE (the gated question, resolved): all 16 panel ridges
measured on midway3-0278 Tesla V100 — the anchor panel's own node —
with --platform cuda (registered contingency; decision note
artifacts/b1prime_platform_decision_20260816.md; per-cell device
sidecars in bundle). Anchor gate PASSED pre-read: 16/16 anchors
re-measured bitwise-identical to the pinned literals (max |delta| = 0;
artifacts/b1prime_platform_20260816/anchor_gate_result.md) — node, env,
platform-flag, and the unlogged ep_batch all proven inert.

## VERDICT — DISSOCIATION-REPEATS

"Behavior floors while legibility is not significantly declined — the
B1-300k shape recurs at 600k; cliff in (600k, 1m]; no ordering claim."

- P-C2 floor: task AUC mean 101.07 < floor 113.682 ⇒ NOT passed
  (behavioral collapse at 600k).
- Side-stratified legibility vs pinned per-run anchors: side0 0.9281 vs
  anchor 0.9027 (diff +0.025 [−0.002, +0.071] ns); side1 0.9890 vs
  0.9668 (+0.022 [+0.009, +0.056]) — NO decline on either side (slight
  rises; side-1 rise = the registered benign expected note).
- apt control: CONTROL-OK (apt_leg 0.4199 vs anchor 0.40107, within
  margin; drift +0.019 toward chance is mechanism-consistent).
- Interaction descriptive i600k +7.9 [−27.1, +53.7] ns.

Registered consequence: the capacity-side convergent test does NOT
fire — the sigma-ladder + B2 pair (now plus the rescue beta-side
confirmation) stands alone as the mechanism evidence; no two-lever
capacity sentence. Scientifically: the capacity lever and the noise
lever dissociate in erosion ORDER (capacity: behavior first, legibility
intact; noise: legibility first) — reported descriptively, no ordering
claim licensed.
