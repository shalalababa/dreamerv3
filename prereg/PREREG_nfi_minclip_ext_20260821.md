# PREREG — MIN-CLIP EXTENSION bank (registered addendum), 21 Aug 2026
(review #31 adjudicated same day: 2B/9M/11m ALL applied)

**Parent:** `PREREG_nfi_minclip_20260821.md` (adjudicated under review
#29; parent code sha `771445ab`). The parent's scope accounting listed
16 distinct models as "out: dataseed3/4 + hid128×t10k/t30k (JOBCONFIGS
extension pending)". This addendum registers that extension
PRE-OUTCOME (no clip has been computed on any extension model),
completing coverage to 48 of the paper's 60 distinct models.
**Residual scope, corrected (#31 M9):** out = dc_gru/dc_lstm (8
models, world=dcfield — genuine porting) and **lg_lstm (4 models,
world=lg — same world, needs NO porting; deferred by resource choice,
not by capability; it even ships per-member per-cycle npz tables that
would support the parent-strength gate).**

## Bank (pinned)

`minclip_rescore.JOBCONFIGS_EXT` = dataseed3 (search seed 3),
dataseed4 (seed 4), hid128t10k (seed 0), hid128t30k (seed 0), all
y_mode=ml, × 4 members = **16 distinct models** (no view scorings).
Ensembles are the FROZEN `local_results/uncfield/sweeps/<job>/
ensemble.pkl` (already trained; zero training compute). Scoring
pipeline is the parent's own functions (`licensed_rate`, `clip`,
EPS_PRED=.08, retention guard, fidelity, license-window and
inconsistency semantics) with ONE registered omission: **LOO rows are
not computed for the extension (#31 m18)** — the parent's LOO is
descriptive-only defense-battery input and is not replicated here.
Outputs go to a PARALLEL root (`local_results/uncfield/minclip_ext/`)
and a PARALLEL read (`--collate_ext` → `minclip_ext_read.json`).

**Isolation, stated accurately (#31 B2):** the two banks share NO
output files and NO collate — but they SHARE THE SOURCE FILE
`uncfield/minclip_rescore.py`, i.e. the reviewed frozen instrument
was extended in place after its review. The audit evidence: the
extension diff against parent sha `771445ab` is additive — 289
insertions / 2 deletions, both deletions inside `selfcheck()`
(review #31 verified `JOBCONFIGS`, `run_task`, `collate`, `ORACLES`,
`_fake_rec`, `_write_bank` byte-identical) — plus this batch's
further review-adjudication edits, all confined to the ext functions,
the selfcheck, and `main()` dispatch. **SEQUENCING RULE (registered):
the main 0–43 array is submitted either entirely before the extension
freeze commit or entirely after it, never across it** (the per-task
`stamp` records the commit; `collate_ext` hard-asserts stamp
homogeneity within the ext bank — the frozen parent `collate` cannot
check this for the main bank, which is disclosed here (#31 M11)).

**Pre-submission integrity (#31 M10):** before the array submits,
`sha256sum -c local_results/uncfield/sweeps/MANIFEST.sha256` (or the
synced copy's manifest) MUST pass on the execution host for the 8
files this bank depends on (4× ensemble.pkl + 4× summary.json) — the
results-sync v2 verify-before-reads rule applied at the input side.

## Drift gate (#31 B1 form — what is and is not pinned)

Each ext task hard-asserts its re-run search REPRODUCES the frozen
`sweeps/<job>/summary.json` member row across **every
model-dependent frozen field**: the five exploit counts
(n_exploit_naive/cig/pbim/both/transient), the verdict, the
top-cycle IDENTITY (name), and top_true / top_eig / top_dh_adj /
drift_per_action to **1e-6 ABSOLUTE** (drift_per_action is a median
over ~100 move-only cycles — a broad numerical fingerprint; the
1e-6 envelope is MEASURED, not chosen: a live 21-Aug re-run of
hid128t30k_m1 locally vs the frozen cluster summary showed
top_dh_adj Δ=8e-8 and drift Δ=1.6e-7 with every count and identity
field exact — a 1e-9 gate would have failed on healthy data).
n_cycles/n_neutral are asserted too but are model-independent
harness constants (disclosed, not billed as discrimination);
top_self_consistency is NOT gated (it would need an extra trace).
**Honest comparison to the parent:** the parent gate pins all 626
cycles' (name, neutral, carried_eig) per member; this gate pins
count-family + top-cycle identity + 4 floats. The per-member
tier_clipped bits (the P-MC1-EXT numerator) remain protected only
indirectly (a numerics shift that preserves every gated quantity
while moving which near-threshold cycles clear EPS_PRED is not
excluded) — this residual is the price of the missing per-cycle
tables and is disclosed rather than papered over.
**Gate-failure disposition (registered pre-outcome, #31 M6):** ANY
ext task failing its gate ⇒ the extension read is NOT-ADJUDICABLE
(collate_ext refuses on an incomplete bank); the failing (job,
member, field, values) are disclosed; there is NO partial-bank
substitute and NO re-pin.

## Frozen predictions (before any clip computation on this bank)

- **P-MC1-EXT (replication of the parent primary):** ≥ 50% of the
  extension carried-exploit-tier members lose tier membership under
  the clip. **Pinned denominator, verified from the frozen sweep
  summaries: 15 of 16 (miss: hid128t30k_m1, n_exploit_cig = 0)** ⇒
  fires iff ≥ 8 of 15 lose tier; collate_ext hard-asserts the
  observed denominator equals 15. **Clustering caveat (#31 M7): the
  15 units come from only 4 ensembles; if collapse is
  ensemble-driven the effective n is ~4. The per-ensemble 4-member
  breakdown is a MANDATORY disclosure and a Wilson 95% interval on
  the collapse fraction is emitted (descriptive, per the 8-Aug
  interval rule).**
- **Joint disposition with the parent (pinned 2×2, #31 M8):**
  (parent fires, ext fires) → "the carried-EIG repair generalizes
  across data seeds and the capacity×training arm";
  (parent fires, ext no-fire) → "the repair is bank-limited — the
  extension BOUNDS the parent's generality claim to its 32 models";
  (parent no-fire, ext fires) → **the ext result may NOT rescue
  P-MC1** — licensed sentence is only "collapse appears in the
  extension arm; the registered primary did not fire";
  (neither fires) → the clip's collapse claim is not supported at
  either bank. The two denominators are never pooled.
- **P-MC3-EXT (guard, parent form):** retention ≥ 0.8× carried on
  every MEASURED model (credit-at-stake population); ABSTAIN
  explicit; all-abstain ⇒ NOT-ADJUDICABLE.
- **P-MC4-EXT (exploratory-registered):** rank fidelity per model,
  reported either way. Parity disclosure (#31 m12): the nan-filter
  inherits the parent's `or np.nan` idiom, under which a legitimate
  rho of exactly 0.0 is dropped from the pair count — kept for
  cross-bank parity, disclosed here.
- **No P-MC2 analogue** (mechanism traces cover pilot2 only).
- **Interaction table (registered DESCRIPTIVE, pinned pre-outcome,
  #31 M5):** the 2×3 capacity×training table {hid64, hid128} ×
  {3k, 10k, 30k} with cells sourced pilot2 / train10k / train30k /
  hid128 / hid128t10k / hid128t30k and per-cell statistic = members
  going tier_carried → NOT tier_clipped, of 4. The main-bank cells
  require the main read's task jsons; collate_ext emits the table
  with `MAIN-BANK-PENDING` placeholders until they exist. "Never
  pooled" is scoped to the P-MC1 DENOMINATORS — this table is the
  registered cross-bank descriptive exception. The pre-clip half of
  the table is already visible in the frozen summaries (disclosed).

## Execution

`scripts/uncfield_minclip_ext.sbatch` (RCC CPU array 0–15,
--time 02:00:00; completion counter; collate_ext refuses on an
incomplete bank) → results-sync v2 bundle + manifest of
`local_results/uncfield/minclip_ext/` → ONE `--collate_ext`
execution with explicit `--output` (TBD refused, trailing-slash
proof). **Bundle-spec warning (#31 m22): `minclip/` and
`minclip_ext/` share a path prefix — never write a bundle or
archive glob as `local_results/uncfield/minclip*`; name the two
directories separately.**

## Placement

Same as parent: Paper-5 constructive section — the extension rows
widen the clip's evidence from 32 to 48 distinct models; the
licensed headline stays "the carried-EIG channel is repaired"
(carried_dh untouched by construction, recorded per model — its
reproduction is covered by the gated n_exploit_pbim count, #31 M3).
Fences unchanged.

Freeze = review #31 adjudicated (this file) + commit of the
`minclip_rescore.py` extension (selfcheck PASS incl. the
strengthened-gate schema checks, per-ensemble/Wilson/stamp/
interaction fixtures) + `scripts/uncfield_minclip_ext.sbatch` —
then the manifest check on the execution host, then the 16-task
array (respecting the sequencing rule above).
