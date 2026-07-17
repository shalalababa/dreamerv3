# vgo wiring audit (2026-07-17) — required before P-B2 is tested

`prereg/PREREG_theory_predictions_20260717.md` P-B2 registers a
conditionality: whether the historical vgo arm's value-path targets
were (a) imagination-internal only, or (b) replay-grounded in true
environment rewards, must be determined from the code/configs BEFORE
P-B2 is tested. This audit answers it from the training code the vgo
fits ran (dreamerv3/agent.py, unchanged since the P3 wave).

## Finding: vgo is case (b) — replay-grounded true-reward targets

- The repval loss block (`agent.py` ~370–385) runs for every task-mode
  fit with `repval_loss: True`; under vgo (`reward_grad False,
  repval_grad True`) the stop-gradient is SKIPPED
  (`feat = sg(repfeat, skip=repval_grad)`), so this loss shapes the
  trunk.
- Its targets come from `repl_loss(last, term, rew, boot, ...)` where
  `rew = obs['reward']` — the replay batch's TRUE logged rewards — and
  `boot` = imagination returns at the replay states.
- `repl_loss` (~735–762) computes
  `ret = lambda_return(last, term, rew, tarval, boot, disc, lam)`:
  the true replay rewards enter the lambda-return targets directly
  (`interm = rew[:, 1:] + ...`), mixed with slow-value/imagination
  bootstraps. The value head is regressed on `sg(ret)`, and its input
  gradients flow into the trunk under vgo.

So the vgo arm's representation DID receive out-of-subspace
true-reward signal through the value path — Prop 3 case (a)'s
structural zero (vgo ≡ sgb by wiring) does NOT apply to the historical
vgo arm.

## Consequence for P-B2 (per the registered conditionality)

P-B2 shifts to the registered case-(b) form: the theory must explain
the observed vgo null (−15.8 P0-era / −1.0 P3, dead at 500K updates)
as ATTENUATION — the true-reward signal reaches the trunk only through
a scalar bootstrapped, valnorm-normalized regression target, a much
weaker gradient channel than the direct reward-head path — and
predicts attenuated-but-nonzero vgo signal recoverable at longer fit
budgets. The registered discriminator is therefore an EXTENDED-FIT vgo
arm (same buffers, larger update budget): vgo benefit emerging with
budget supports the attenuation account; a persistent hard null at
extended budget is evidence AGAINST the spectral-competition value-path
story as written (Prop 3 needs revision, not just recalibration).

Note for the same discriminator: `boot` uses the imagination reward
head, which in vgo is trained head-only (reward_grad False) — the
bootstrap leg is informative but trunk-gradient-free; only the `rew`
leg carries the out-of-subspace signal into the trunk.

No new registration is made here; this file records the audit outcome
the theory prereg required. Any extended-fit vgo wave freezes its own
prereg before submission.
