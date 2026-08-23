# AMENDMENT 1 to PREREG_trackA_scarecrow_20260822 — FULL RESTART
after the one-instance breach, 23 Aug 2026

**Incident:** instance 13 was released after round 1 completed
(data pulled + verified — the release rule checked data safety
only, not apparatus membership; ops has added the fourth check).
Round 1 was, by the spec's per-stage lane packing, **the entire
sc_ctrl arm (4/4 control runs)**.

**Decision: RESTART ALL 20. Pooling is REJECTED** — with the whole
control arm on one device and every treatment arm on another,
device ≡ control for every registered cross-arm contrast; under the
1-Aug non-determinism rule (and the 16-Aug measured cross-GPU
shifts) that confound is unrecoverable, and the control arm anchors
EVERY primary and secondary. This is precisely the case the
one-instance registration exists for. Sunk: ~28 GPU-h.

**Registered consequences:**
1. The four completed `sc_ctrl_s76..79` run dirs on RCC are
   QUARANTINED as `_SUPERSEDED_INST13_sc_ctrl_s76..79` **BEFORE any
   restart pull lands** (the fresh runs reuse the same run_ids and
   seeds — an unrenamed old copy is a name-collision hazard for the
   pull and a poison hazard for the read). They never enter the
   registered read; the reader's denominator counts only the fresh
   20.
2. Seeds 76–95 are REUSED as registered (fresh hardware, fresh
   streams; no analysis pools old with new, so no pairing hazard).
3. **Layout fix (registered operational change; estimand and the
   frozen reader untouched):** the restart spec replaces the five
   per-arm stages with ONE stage and an explicit cyclic 4×5 runs
   list — round r, lane l → arm (r+l) mod 5 — so every ROUND holds
   4 DISTINCT arms (no round is ever arm-pure again; the incident
   showed round 1 ≡ ctrl, which was also a latent within-box
   time≡arm hazard) and every LANE runs each arm exactly once (the
   original lane balance, preserved). Shared done_when keeps the
   common gates; ARM identity is enforced by the frozen reader's
   fatal per-arm pins (the #34-reviewed authority), unchanged.
4. Instance allocation: the restart takes instance 18 (~35 h).

Nothing else in the parent registration changes.
