# rde pixel arm — registration record (2026-08-02)

Registration-time evidence for `prereg/PREREG_rde_pixel_20260802.md`
(recon-detached pixel arm — the second, complementary swamping-lever
test after the pe arm's NO-RELIEF). Committed BEFORE any rde run exists.

Files here:
- `rde_grad_check.py` + `grad_check_stdout.txt` — the local CPU
  verification harness, REV 2 post-review (traces the REAL `Agent.loss`
  through `nj.pure` under training=True with synthetic pixel spaces;
  ALL EIGHT loss components compared across both flag settings):
  - recon_grad=True: image-loss grads enc 114.14 / dyn 7528.4 / dec 3000.68
  - recon_grad=False: image→enc **0.0 exact**, image→dyn **0.0 exact**,
    image→dec 3000.68 unchanged (decoder still trains)
  - every non-image component (con/dyn/rep/rew/repval/policy/value)
    **bit-identical** across settings; rew (enc 14.19 / dyn 3220.85)
    and repval (enc 8.44 / dyn 2291.07) trunk paths verified OPEN
  - the SHIPPED `_combine_deter_moments` ≡ np.std under uneven batches
    + 1e3 offset; constant input → 0
  - harness-only rew/value/policy outscale 0→1 overrides
    (zero-initialized output kernels block feature grads at init,
    which made the rev-1 comparison vacuous for those paths — reviewer
    finding m5; real runs are past init)
- `freeze_shas.txt` — sha256 of every registered file at freeze
  (post-review finals).

Review (2 Aug, ONE adversarial reviewer Opus 5, user-approved):
**2 BLOCKING / 6 MAJOR / 6 MINOR / 5 NIT — all B+M fixed + re-verified**
(full itemization in the prereg's Review-provenance section). The two
BLOCKINGs: a DELETE-SAFE cleanup glob (`ax1wm_finger_rd*`) that matched
every rde run — any cleanup between submit and read would have deleted
the wave; and the trivial-constant floor sitting INSIDE the fire region
[0.6713, 2.0) — a marginal head would have read COMPETITION-CONFIRMED
(the pre-review selfcheck's own Case-1 fixture was an instance). Both
now closed with machine-verified fixes (digit-anchored globs;
floor-decisional NOT-BETTER-THAN-CONSTANT branch). Six decision-rule
mutants verified CAUGHT on mirror copies (incl. the four that survived
the pre-review selfcheck), clean copy PASS.

Design decision recorded: the task-trained-donor variant was REJECTED
(near-tautological — P-SW1 already measured the X2 task trunk's reward
illegibility); the discriminating lever is recon-detachment with a
trainable trunk. The masquerade trap (collapsed trunk → marginal head →
false INCLUSION-RESTORED) is closed by the deter_std latent-alive
witness calibrated on the fpxpx fits (whose KEEP-PENDING hold transfers
to this wave's calibration pass in `runroot_cleanup.sh`).

Reader selfcheck at freeze: PASS (confirmed / masquerade / borderline /
included-useless / partial / no-relief(+collapse-qualified) branches,
RD3 legs, 10 guard trips).

User GO context: "I would proceed if the new registration can make the
story clearer" — both live branches end the pixel arc decisively
(COMPETITION-CONFIRMED converts the boundary into a causal
demonstration of the headline; NO-RELIEF retires the causal competition
account with both lever directions refuted, leaving a clean
modality-level scope fact).
