# PREREG: anti-harvest zero-compute stage (TM2, executed W1 labels) — 2026-08-11

User GO 2026-08-11 ("explore all explanations"). Context: the W1 tm2 read
(`artifacts/w1_read_20260811/`) found a genuine-but-small split-selected
opportunity (+0.062*) and a significantly NEGATIVE deployed-planner
harvest (−0.301 [−0.745, −0.123], p=.008) — the warm-started MPPI
planner's action evaluates worse under ground truth than the mean of 8
policy-prior candidates, under IDENTICAL continuation (the ground-truth
follower is the same planner-agent for every branch, so
continuation/handoff mismatch is structurally excluded). This stage
extracts every mechanism discriminator already present in the EXECUTED
W1 label arrays — zero new compute. The new-pass discriminators are
registered separately (`PREREG_antiharvest_mech_20260811.md`).

## Honesty block — value-awareness (disclosed; review B3 corrected form)

The W1 labels are read data, and one Z-quantity is CLOSER to the
executed read than a naive account admits: the labeler's stored `m_now`
IS `plugin_choice(qfull)` = the Q-argmax candidate, i.e. **P-Z2's
`sel_q` ≡ m_now by construction** (the reader asserts this identity as
a load gate), and the executed W1 read consumed `ho(m_now)` as the
BASELINE arm of P-W1a, reporting split-opportunity +0.0617* =
"the empirically-best-selectable candidate beats the Q-argmax candidate
by +0.062". That known fact is materially informative about P-Z2's
sign: a priori it makes **Q-HARVESTS the more likely branch** (if the
Q-argmax candidate were far below the candidate mean, the +0.062 gap to
the empirical best would likely have been larger). Registered
against-interest note: P-Z2 landing Q-HARVESTS is therefore the
EXPECTED outcome and is graded with that prior disclosed; CURSE-AT-Q
landing would be the surprising branch. What was genuinely never
examined by any read: the prior-MODE column (candidate 0) — P-Z1 —
and every `qstd`/disagreement quantity — P-Z3. Known at freeze: the
pooled W1 numbers (incl. harvest −0.301) and everything archived.
Unknown: every quantity below as a NUMBER.

## Substrate

The 32 executed tm2 W1 label files
(`w1_labels_20260811_090727/w1_labels/w1tm2_{cup|finger}_{e1|e4}_seed{51..58}_late.npz`,
sha-manifest verified; dup-gate files excluded and asserted untouched).
Arrays per state (200/cell): `g_all_rep (R=8, M=8)` ground-truth
returns per candidate; `g_plan_rep (8,)` deployed planner action;
`qfull (5, 8)` per-candidate Q-ensemble values (two-hot-inv); `cands`,
`m_now`, `dup_cand`. Candidate 0 = the policy-prior MODE (labeler
construction), candidates 1–7 = prior samples.

## Registered decision rules (pooled over 32 cells, cluster = cell; BCa 95% B=10K rng 0 + cluster sign-flip permutation; fires need CI excluding 0 AND p < .05)

Per state: g_plan = mean_r g_plan_rep; g_cand(m) = mean_r g_all_rep[:,m];
gbar = mean_m g_cand; q(m) = mean over the 5 Q-heads of qfull;
qstd(m) = std over heads.

- **P-Z1 (prior-mode contrast)**: d_mode = g_plan − g_cand(0), cell-mean
  pooled.
  - CI < 0 ⇒ **PLANNER-BELOW-MODE**: six MPPI iterations produce an
    action worse than the un-optimized prior mode — optimization is
    strictly harmful at the first step.
  - CI > 0 ⇒ **MODE-BELOW-PLANNER**: the anti-harvest is driven by the
    prior SAMPLES outperforming both (the registered alternative — a
    stochastic-exploration account, not an optimization-pathology one).
  - else INDETERMINATE.
- **P-Z2 (model-value selection curse, the sharp dissociator)**:
  sel_q = argmax_m q(m); d_qsel = g_cand(sel_q) − gbar. Selection is
  purely model-side (no G involved ⇒ no selection-on-noise bias; no
  split needed).
  - CI < 0 ⇒ **CURSE-AT-Q**: the learned value function misranks its
    own prior's candidates — value-error, not planner search, is
    sufficient for anti-harvest.
  - CI > 0 ⇒ **Q-HARVESTS**: the value head ranks candidates USEFULLY
    ⇒ the planner's deficit is planner-search-specific (MPPI pathology
    or warm-start), NOT value-error — redirects the mech wave's weight
    to P-N2/P-N3.
  - else NULL (uninformative).
- **P-Z3 (disagreement-penalty repair, constructive)**:
  sel_p = argmax_m [q(m) − 1.0·qstd(m)] (λ=1 registered primary;
  λ ∈ {0.5, 2} descriptive): d_rep = g_cand(sel_p) − g_cand(sel_q).
  - CI > 0 ⇒ **PENALTY-IMPROVES** (ensemble disagreement recovers real
    value — the constructive lever, coheres with the Paper-5 penalty
    theme). CI < 0 ⇒ PENALTY-HURTS. else NULL.
- **Descriptives (no verdicts)**: pooled per-state Spearman(q, g_cand)
  (the candidate-level model-truth ranking signal); d_rep vs g_plan;
  per-domain (cup/finger) splits of all three.

Gates (fail-closed): exactly 32 non-dup files; per-file pins
repeats=8, states=200, actions=8, env_seed=20260809, labeler_version
suffix `_w1`, dup_cand all False; qfull shape (200, 5, 8); NaN-free.

Reader: `analysis/antiharvest_zc_read.py`, frozen with this file,
selfcheck (planted fixtures per branch; gates are load-time asserts —
shape/pin/NaN/dup-count/sel_q≡m_now — exercised at execution, fail-
closed) before execution. qstd uses ddof=0 (registered). Scope notes
(review NIT 23): estimands are realization-conditional (one env
snapshot per state, the W1 m1 disclosure carries over); argmax ties
break toward index 0 = the prior mode (numpy convention, registered).
ONE read execution (local — the labels are already here). Reviewer:
covered by the 2026-08-11 batch review (ONE reviewer, user-approved)
BEFORE freeze-commit + execution.
