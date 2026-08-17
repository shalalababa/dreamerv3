# FB wave HALTED at the registered smoke gate — `fb_zeroshot` selfcheck fails

**2026-08-16, ops, instance 12 (fresh 4× RTX 5060 Ti, vast 47905279).**
The prereg's ops chain requires all three torch-side selfchecks to PASS
cluster-side "before any fit". One fails. **No export, no random_floor, no
smoke fit, no fits submitted, and no env installed on instances 1/5/7.**

## Where the chain got to

| step | result |
|---|---|
| repo push (`f9cef148`) | OK |
| `scripts/fb_env_setup.sh` | **OK** — checkout at the pinned `5a9950b0…`, py3.10, `import OK; torch 2.13.0+cu130 cuda True`, `cuda forward OK … RTX 5060 Ti` |
| `fb_fit --selfcheck` | **PASS** (tiny fit + ckpt roundtrip + B shape + same-seed determinism + wrong-dim refusal) |
| `fb_embed --selfcheck` | **PASS** (raw + fb modes, shapes + manifest) |
| `fb_zeroshot --selfcheck` | **FAIL** ← gate |
| everything downstream | **not run** |

## The failure

```
File "/workspace/dreamerv3/probing/fb_zeroshot.py", line 75, in rollout
    action = agent.act(ts.observation, meta, step=0, eval_mode=True)
File "/workspace/controllable_agent/url_benchmark/agent/fb_ddpg.py", line 281, in act
    return action.cpu().numpy()[0]
RuntimeError: Can't call numpy() on Tensor that requires grad.
```

Full log: `fb_zeroshot_selfcheck_FAIL.log` (checkout pin confirmed in-line).

## Diagnosis — ours, not the environment's

Upstream's `act()` ends with `return action.cpu().numpy()[0]` and never
detaches. It relies on the caller. **Every** upstream call site wraps it
identically:

```python
with torch.no_grad(), utils.eval_mode(self.agent):
    action = self.agent.act(time_step.observation, ...)
```

(`pretrain.py:353`, `:396`, `:628`; `anytrain.py:91`; `train_online.py:86`;
`play_behaviors.py:134` — six for six.)

`probing/fb_zeroshot.py:75` calls it **bare**. So this is a defect in our
registered instrument, not a torch-version or environment problem, and the
instrument's own selfcheck caught it exactly as designed. The other two
selfchecks passing, plus the env's own import+CUDA gate passing, localizes it
precisely.

### There are two gaps, and only the first one errors

1. **Missing `torch.no_grad()`** — hard failure, the traceback above.
2. **Missing `utils.eval_mode(agent)`** — **silent**. `utils.eval_mode`
   (`utils.py:34`) calls `model.train(False)`. Our `eval_mode=True` argument is
   a *different thing*: it is `act()`'s action-selection flag choosing
   `dist.mean` over `dist.sample()`. It does not touch module training state.

   Fixing only (1) would make the selfcheck pass and leave (2) in place, where
   it can no longer announce itself.

   **Materiality check, run here:** `grep -E "BatchNorm|Dropout|LayerNorm"` over
   the pinned `fb_ddpg.py` returns **nothing**, so on this checkout there is no
   train/eval-sensitive layer and (2) is very likely inert *numerically*. It is
   still a deviation from upstream's eval protocol, and "inert today" rests on
   a grep rather than on the design. Also checked: our `AGENT_KWARGS` sets
   `additional_metric=False`, so `act()`'s eval branch does not take the extra
   forward passes that would otherwise also need `no_grad`.

## Why ops stopped here

The fix is an edit to `probing/fb_zeroshot.py`, which the prereg freezes
alongside `analysis/fb_read.py`. It is also not cosmetic: `fb_zeroshot`
produces **both** the flagship P-FB1 zero-shot leg **and** the in-protocol
`random_floor` control whose [35, 110] band gates the whole wave. Changing how
actions are drawn during evaluation is squarely a registered-instrument
decision, so it belongs to the Papers-1–3 chat.

Standing rules applied: do not invent a fix for a failure you did not
anticipate; never change a knob to make a retry pass; escalate.

## Suggested fix, for whoever owns the edit

Match upstream verbatim at `fb_zeroshot.py:75`:

```python
from url_benchmark import utils as ca_utils
...
with torch.no_grad(), ca_utils.eval_mode(agent):
    action = agent.act(ts.observation, meta, step=0, eval_mode=True)
```

This addresses both gaps at once and copies the reference implementation rather
than inventing a variant. After the edit, the gate is simply re-run — the env
on instance 12 is built and the other two selfchecks already pass, so
revalidation is minutes.

## State left behind

* Instance 12: FB env at `/workspace/conda_envs/fb`, pinned checkout at
  `/workspace/controllable_agent`, repo at `f9cef148`. Idle, nothing queued.
* Instances 1/5/7: **untouched** — env not installed, per the user's
  "if that works" precondition. 1/5/7 are still running the SE Stage-2 arms.
* Nothing exported, no `fb_data/`, no fits, no runroot writes at all.
