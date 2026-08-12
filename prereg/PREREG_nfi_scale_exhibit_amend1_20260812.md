# PREREG AMENDMENT 1 — NFI Scale Exhibit (SE wave), 12 Aug 2026 (late)

**Parent:** `PREREG_nfi_scale_exhibit_20260812.md` v2.2 (frozen commit
fc85fdee, 12 Aug 2026). **Timing: PRE-READ, PRE-OUTCOME** — the 8
registered runs (seeds 10–17) are training; no run has been probed, no
attribution number from any registered run has been observed (the only
executed probe passes are on the smoke run, seed 0, which is outside
the registered family).

## What changes

`uncfield/se_probe.py` gains **additive, reporting-only** output
fields. Parent §5 (P-SE2) registers: "the per-family (D-family vs N)
share breakdown reported" — but the frozen probe stored only the
aggregate fire-channel rollout share, so the frozen reader
(`uncfield/se_read.py`, a registered pre-read build) could not deliver
that registered reporting obligation from its inputs. The amendment:

- json `pse2.families = {dfam: {top_share, base_share}, noise: {…}}`
  (D-family = dup0+dup1+dup2; noise = distractor; same denominator,
  same intrinsic ranking, same top-k as the registered statistic);
- npz `rollout_share_dfam`, `rollout_share_noise` (per-rollout family
  shares);
- selfcheck consistency assert: family top/base shares sum exactly to
  the registered aggregate top/base shares.

## What does NOT change

No registered statistic, fire rule, permutation scheme, normalizer,
window, or seed constant. **No rng draw is added or reordered** — the
family numerators are accumulated inside the existing loop from the
same already-computed per-key variances, so every previously defined
output (shares, per-channel p's, P-SE2 top/base/p_rank, determinism
witnesses) is bitwise unchanged. Selfcheck re-run PASS post-edit
(recorded in `artifacts/nfi_assessment_response_20260812/RESULTS.md`
§9).

## Governance

Registered per the frozen-instrument amendment discipline (precedent:
`PREREG_nfi_family2_amend2_20260809.md`): dated amendment file, filed
BEFORE any registered read, motivated by a registered reporting
obligation the frozen instrument could not meet — not by any observed
outcome.
