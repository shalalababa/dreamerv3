# Amendment 1 to PREREG_fb_20260817 — no_grad wrapper on the act() call (16 Aug 2026)

**Scope: one calling-convention fix in `probing/fb_zeroshot.py`
(rollout), nothing else.** Registered pre-outcome in the strongest
sense: the wave HALTED at the registered cluster smoke gate
(`fb_zeroshot --selfcheck` FAIL, `artifacts/fb_smoke_20260816/`) with
**zero exports, zero fits, zero estimands** — the gate existed for
exactly this.

## The defect and the fix

Upstream `FBDDPGAgent.act()` ends `return action.cpu().numpy()[0]` and
never detaches; it relies on the caller. Every upstream call site (six,
enumerated in the ops RECORD: pretrain.py ×3, anytrain.py,
train_online.py, play_behaviors.py) wraps the call in
`torch.no_grad()` with the agent in eval mode. Our `rollout()` called
it bare → `RuntimeError: Can't call numpy() on Tensor that requires
grad`.

Fix: wrap the `agent.act(...)` call in `torch.no_grad()` — matching
the upstream calling convention exactly. Eval mode is already applied
permanently by `fb_fit.load_agent` (nets `.eval()`). No numerics
change: `no_grad` disables graph construction only; the forward values
are identical. The random-floor mode shares `rollout()` and is
unaffected semantically (its stub policy has no graph).

## Discipline notes

- The frozen reader is untouched. The prereg body is untouched.
- The registered smoke sequence REMAINS the gate: `fb_zeroshot
  --selfcheck` must PASS cluster-side after this fix before the chain
  proceeds (then random_floor → timed smoke fit → fits).
- Freeze-commit ride-along: `git add probing/fb_zeroshot.py
  prereg/PREREG_fb_amend1_20260816.md STUDY_LEDGER.md`.
