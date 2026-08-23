# AMENDMENT 1 to PREREG_trackB_onset_20260822 — POOLED CONTINUATION
after the one-instance breach, 23 Aug 2026

**Incident:** instance 15 was released after round 1 completed
(data pulled + verified; same release-rule gap as the A2 incident,
same ops fix). Round 1 was, by the registered 4×4 Latin square,
**exactly one run of every scale**: se_ons25_s96, se_ons30_s100,
se_ons34_s104, se_ons38_s108.

**Decision: POOL, under a registered condition.** The one-instance
rule guards against device ≡ scale correlation on a monotone
cross-run primary. Because round 1 is SCALE-COMPLETE, pooling it
with rounds 2–4 makes device a BALANCED BLOCK: every scale
contributes exactly 1 instance-15 run and 3 continuation-box runs,
so a device main effect (level shift between boxes) is orthogonal
to the Spearman primary by construction — it widens the null, it
cannot manufacture the trend. This is categorically different from
the A2 case (device ≡ arm), which is restarted.

**Registered conditions and duties:**
1. **CONDITION (ops verifies before continuing):** the completed
   instance-15 set is EXACTLY the four runs above — one per scale,
   all four passing their done_when gates. Any other composition
   (a fifth run, a missing scale, a failed gate) VOIDS this
   amendment and the wave restarts in full.
2. The continuation = rounds 2–4 of the declared Latin square (12
   runs) on ONE box (any 4-lane instance; the declaration order is
   unchanged — each continuation round remains scale-complete, and
   no lane repeats a scale within the continuation). Exactly TWO
   devices total in the pooled panel.
3. **Disclosed residual risk:** a device × scale INTERACTION is
   not excluded by the balance (only the main effect is). Carried
   as a registered disclosure in the read artifact.
4. **Registered read-time sensitivity (SEPARATE dated descriptive
   computation — the frozen reader `uncfield/se_onset_read.py` is
   NOT edited):** after the one registered execution, a
   continuation-box-only Spearman on the 12 same-box runs (3 per
   scale) is computed and reported as descriptive-never-a-fire
   (12 < the registered MIN_USABLE, so it cannot be a primary; it
   is the same-device consistency row). The read artifact must
   state per-run training device.
5. Every reader gate, bar, and cell is UNCHANGED; the probe passes
   were always one CPU job and are unaffected.

Sunk cost avoided: ~7 GPU-h (the restart alternative); nothing
discarded.
