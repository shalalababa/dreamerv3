# Value-free vs online pretraining kill test — ONE read (23 Aug 2026)

Registration: PREREG_valuefree_online_20260821.md (v3; frozen
17a44e1f). Bundle: valuefree_20260823_123000 (manifest OK; witness
problems NONE; online counters realized 494528 updates
[metrics_updates] / 496496 env steps — both in the registered bands;
16 rgo fit counters == 500000 with the value-free cfg pins
repval_grad False / reward_grad True verified; adapt→ckpt linkage
adapt_ckpt_ok TRUE; modal n_ep 48/96/112 across the ladder; all
refusal legs clean).

## Verdict: **ONLINE-SUPERIOR** (uniform across the window ladder)

| window | online − q1s1 (PRIMARY) | anchor q1s1 − scratch |
|---|---|---|
| auc50k | +169.7 [+101.4, +262.7], p = .0005 | +83.8*, p = .0049 |
| auc100k (PRIMARY) | **+159.0 [+65.0, +272.2], p = .0084** | **+119.8 [+51.0, +165.4], p = .0043** |
| auc125k | +148.8*, p = .0186 | (anchor holds) |

Means (auc100k): online 454.5 · q1s1 295.5 · q1s0 283.1 · scratch
175.7. p_match = p_noninf = .036 (against the .95 bar — neither
certificate approaches).

## What this settles

- **The value-free recipe does NOT match online task pretraining at
  matched updates (500k/500k).** The registered ASYMMETRIC consequence
  applies verbatim: attribution stays ambiguous among {onlineness,
  task-directed data, 2.5× unique frames} — only "no match at these
  budgets" is licensed, never "the value head is necessary."
- **The advantage does not fade with adaptation length**: the gap is
  significant at auc50k, auc100k AND auc125k (+170 → +159 → +149,
  same sign, monotone but shallow decline) — the ladder's
  longer-window leg answers the "maybe value pays off later" question
  within this budget: no crossover in sight.
- **The anchor gate fired**: the rgo recipe's value over scratch is
  real (+119.8* at the primary window) — the comparison was live, not
  floor-censored. These are the FIRST rgo-unfrozen adapt numbers in
  the program: the reward-gradient-only trunk transfers under the
  unfrozen protocol at ~1.7× scratch.
- **Spillover fact for the rrpctl registration (disclosed there
  pre-outcome)**: q1s1 (adapts of ax1wm_finger_rgoq1s1_seed{17..24},
  the same fits + protocol as rrpctl's rca arm under different
  run_ids) means 295.5 — vs the routed-repair arm's pinned 291.4.
  The rrpctl prereg's disclosure anticipated this reveal ("the
  control-arm LEVEL becomes partially known — immaterial, thresholds
  frozen"); on these numbers E1 points toward
  OVERWRITE-SUFFICIENT-BOUNDED, i.e. the "refit-don't-finetune"
  demotion branch. The wave still adjudicates (fresh invocations;
  rcb/matched-budget unmeasured).
