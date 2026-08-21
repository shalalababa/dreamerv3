# PREREG — MIN-CLIP re-scoring (Paper-5 constructive section), 21 Aug 2026

**Context:** the algorithm-ideation panel's candidate 1 (user GO
21 Aug). Credit each cycle with **min(operator's promise, heads'
Bayes license)** instead of the operator's carried EIG alone. LEGAL
NOTE (registered): the veto/classifier form ("reject cycles with
ratio > τ") is a registered non-claim (record :721-726 — no
thresholdable cycle-level discriminator exists); min-clip RE-SCORES
credit, it never classifies cycles. That distinction is binding on
all write-ups.

**Pre-computation disclosure (binding):** the panel pre-computed the
clip on the 16 TRACED exploits in `artifacts/nfi_mechanism_20260802/
mechanism.json`: m0/m1/m2's traced exploits all fall below
EPS_PRED = 0.08 under the clip (m0 margin only ~1.3× — "collapses,
not far"); **m3's two traced exploits survive at 3.6×/5.3× above
threshold**. P-MC2 below is therefore PARTIALLY pre-computed (the 3
traced cycles per member; its full-bank quantifier over the 81–253
untraced neutral exploits per member is a PREDICTION — #29 M10).
Everything else (the full-bank tier primary, the retention guard,
rank fidelity, LOO) is NOT pre-computed — the mechanism artifact
holds licensed gains for 3 exploits + 1 TV + 3 controls per pilot
member only, and none for informative cycles.

## Frozen predictions (before any full-bank computation)

- **P-MC1 (primary, #29 B3/M11 form):** ≥ 50% of the DISTINCT-MODEL
  carried-exploit-tier members lose tier membership under the clip
  (tier = any neutral cycle with clipped rate > EPS_PRED = .08).
  **Bank = 32 DISTINCT MODELS** (8 ensembles × 4 members; the three
  ymode_s* jobs re-score the pilot2 ensemble and are reported
  separately as VIEW SENSITIVITY — the house scorings-vs-models
  convention from review #13 §2.6). **Pinned denominator, verified
  from the frozen ratio_research tables before any clip computation:
  29 of 32 distinct models are carried-exploit tier** (misses:
  hid32_m2, train10k_m2, train30k_m2) ⇒ P-MC1 fires iff ≥ 15 of 29
  lose tier; the collate step hard-asserts the observed denominator
  equals 29 (drift gate), and each task asserts its re-run search
  REPRODUCES the frozen table. FULL SCOPE ACCOUNTING (#29 B4): in =
  32 of the paper's 60 distinct models; out = dataseed3/4 +
  hid128×t10k/t30k (16, JOBCONFIGS extension pending) + family-2
  (12, world=dcfield; lg_gru IS pilot2). **CHANNEL SCOPE (#29 M9):
  the clip touches ONLY the carried-EIG channel — the realized-ΔH
  (carried_dh) tier is untouched by construction and is recorded per
  model; the licensed headline is "the carried-EIG channel is
  repaired", never "farming is repaired" (32/44 frozen scorings are
  carried_dh tier).**
- **P-MC2 (PARTIALLY pre-computed, #29 M10): 3 traced cycles per
  member are pre-computed; the full-bank quantifier (81–253 neutral
  carried-exploits per member) is a PREDICTION.** Predicted: pilot2
  m3 retains clipped exploits; pilot2 m0/m1/m2 do not. Mechanism
  correction (#29 M6): **m3's residual is a LARGE FINITE license**
  (heads license 0.29/0.42 nats/loop via variance re-inflation) —
  NOT the infinite-license path, which is empirically untriggered in
  the anchor (frac_r_implied_negative ≡ 0 on every traced cycle).
  The registered mechanism claim: the repair works where the
  OPERATOR is the liar and fails where the HEADS are complicit — the
  predicted partial failure is the paper's own law confirming
  itself, and the residual class is what CEI's interventional audits
  exist for.
- **P-MC3 (guard, frozen; #29 B1/B2 population fix):** retention is
  measured ONLY where credit is at stake — the top cycle among
  {non-neutral, true rate > 0.1, carried > EPS_PRED}. The planner
  itself declares carried_eig non-comparable on dynamic sensors, and
  23/44 frozen tables have carried ≤ 0 on the naive top-true cycle
  (where the clip is a NO-OP); the naive guard would have FAILED BY
  ARITHMETIC there. Models with no eligible cycle ABSTAIN explicitly
  (never a vacuous pass); all-abstain ⇒ P-MC3 NOT-ADJUDICABLE.
  Guard fires iff every MEASURED model retains ≥ 0.8× carried.
- **P-MC4 (secondary, exploratory-registered):** rank fidelity —
  Spearman(clipped, true) vs Spearman(carried, true) per model;
  direction hoped (clipped ≥ carried in ≥ 60% of models) but NOT
  load-bearing; reported either way.
- **LOO rows (defense-battery input, descriptive):** each member's
  top clipped exploit scored by the other 3 members from their own
  warmup states (candidate-8 data; analysis descriptive).

## Instrument

`uncfield/minclip_rescore.py` (built + selfchecked BEFORE any
full-bank run: clip semantics; **3 mechanism-artifact ORACLES exact
to 1e-9 incl. the m3 large-finite residual (#29 M13)**; collate
fixtures for pmc1 fire/no-fire at 15-vs-14, the P-MC2 pattern,
all-abstain, nan-fidelity, denominator-drift and TBD-output gates
(#29 M12); determinism). **License WINDOW (#29 M8, disclosed): the
mechanism-artifact convention (trace n_loops=4, last-2-loops
summary) — adjacent to, not identical with, steady_rate's loops
35–38; kept for P-MC2 comparability.** ANY inconsistent sense makes
the loop's license unbounded (#29 M7 per-sense semantics; the
mechanism.py ≤0-vs-1e-12 sliver rides the inf path, disclosed);
frac_inconsistent + n_senses persisted per cycle (#29 B5).
Execution: `scripts/uncfield_minclip.sbatch` (RCC array 0–43,
--time 02:00:00 = >6× the measured 4–20 min/task — #29 finding 14
double-sourced — with a completion counter per the 7-Aug rule) →
results-sync v2 bundle + manifest of local_results/uncfield/minclip/
→ `--collate` = the frozen read, ONE execution, explicit --output
(TBD refused). LOO rows are DESCRIPTIVE ONLY (own-max selection bias
+ mixed clipped/carried criterion, #29 m21).

## Placement

Paper-5 constructive section (with the canary/scope-card protocol);
zero GPU. Fences: the ratio-diagnostic CONCEPT belongs to the BCC
holder (Imran) — claim the clip USE and calibration methodology only.

Freeze = review #29 (min-clip half) ADJUDICATED (5B/9M/9m ALL
adopted 21 Aug — incl. the P-MC3 population fix that would have
FAILED BY ARITHMETIC on ≥23/44 models, the distinct-models
denominator re-pin, and the frozen-table reproduce gate) + commit
of: this file, `uncfield/minclip_rescore.py`,
`scripts/uncfield_minclip.sbatch` — then the RCC array.
