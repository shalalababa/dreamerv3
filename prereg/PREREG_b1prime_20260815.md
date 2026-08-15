# PREREG: B1' — the 600k capacity point (capacity-side convergent test) — 2026-08-15

User GO 2026-08-15, executing the TRIGGERED 14-Aug conditional: the
σ-ladder returned EXCESS-TASK-LOSS at both loads
(`artifacts/sigma_ladder_read_20260815/`), confirming the
competition-with-swamped-rescue account on the NOISE lever of g_k.
This wave tests the same mechanism on the CAPACITY lever: **frozen
prediction — as capacity falls (g_k rises), the marginally-included
RESCUED reward-relevant support erodes FIRST, so at an intermediate
capacity the task arm's reward-legibility has already declined while
its behavioral competence is still alive** (the capacity-side analog
of the ladder's knee-before-floor ordering). The reward-free arm's
finger_v1 legibility is anchored at/below chance (0.401) — it has no
headroom on this probe and serves as a registered NO-MOVE CONTROL,
not a contrast arm (the two-arm normalized contrast of the σ-ladder
does NOT transfer to this axis; disclosed).

## Honesty block (value-aware, disclosed; review B1/B2/B3/M3 applied)

Known at freeze: ALL executed reads. **The capacity–legibility
relationship is SIDE-STRATIFIED (review B1)** — executed task
finger_v1 ridge AUROC by side:

| size | task side0 | task side1 | pooled |
|------|-----------|-----------|--------|
| 1m   | 0.9027 (sd .051) | 0.9668 (sd .031) | 0.93477 |
| 300k | 0.7101 | 0.9838 | 0.8470 |
| 100k | 0.4005 | 0.9942 | 0.6973 |

The entire executed decline is a **side-0 phenomenon**; side-1
legibility RISES monotonically as capacity falls. The primary is
therefore per-side against side-matched anchors, and a single-side
fire on side 0 is the A-PRIORI-LIKELY outcome (its qualified branch
is registered below). **The anchors are ESTIMATES, not constants
(review B3)**: the 16 per-run 1m values (sd .052 pooled, SE .013)
are pinned as literal lists in the reader and the primary is a
TWO-SAMPLE test, so anchor uncertainty enters the null. Other pinned
anchors: apt legibility 0.40107 (below chance — no-move control
only); behavioral floor 113.682; B1 apt 0.413/0.392, task behavior
73.6/97.5 (both floor-FAIL); 1m behavioral task ≈188.2 / apt ≈79.9;
the σ-ladder verdict. Unknown: every 600k quantity. Registered
risks: (i) knee below 600k ⇒ KNEE-NOT-REACHED (descriptive; no finer
grid authorized; 25m cap stands); (ii) cliff above 600k ⇒ ordering
not establishable, cliff relocates to (600k, 1m] (a new fact vs B1);
(iii) **MDE (review B2, on the EXECUTED sds — the earlier 0.03–0.05
basis was wrong)**: per-fit sds ran 0.052 (1m) / 0.202 (300k) /
0.318 (100k); with anchor uncertainty included, MDE80 ranges
**≈0.06 (1m-like dispersion) to ≈0.20 (300k-like)** — under
300k-like dispersion this wave is UNPOWERED for any sub-collapse
decline and INDETERMINATE-by-power is a live outcome; realized MDE
recomputed in-read per side; (iv) **protocol asymmetry (review
M3)**: the 600k fits train on the context-resized `q1_ctxk6` buffers
(deter/stoch replay contexts zero-initialized) while the 1m anchor
fits trained on the raw `q1` pair — the primary contrast carries
this capacity×ctx-zeroing confound as a disclosed caveat (the
resize tool re-encodes stepid so `Replay.update()` refreshes
contexts during the fit — second-order, not zero); the
within-protocol 300k-vs-600k comparison (both ctx-zeroed) is the
confound-free descriptive and will be quoted alongside in any use.

## Design — 1 size × 2 arms × 2 sides × 4 seeds = 16 fit+adapt jobs

- **New preset (this freeze-commit, `dreamerv3/configs.yaml`)**:
  `size600k` = rssm {deter 384, hidden 48, classes 4}, depth 2,
  units 48 — between size300k (floor-FAIL) and size1m (floor-pass);
  classes stays 4 so stored stoch latents keep shape (ctx pair
  resizes deter only). Whitelists extended in `scripts/axis1.sbatch`
  + `scripts/submit_all.sh`. Realized param count recorded at smoke:
  [FILL].
- **Buffer pair ([YOU], login node)** — the _ctx convention:
  `python -m probing.resize_replay_context resize --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_ctxk6 --deter 384 --stoch 32 4`
  Operator gate (B1 rev-3 F10 form): the output `manifest.json`'s
  `context_resize.sides.*.source_latent_shapes` must show deter [512]
  and stoch [32, 4]; a mismatch ⇒ delete the output and STOP.
- **Adapt-readout pin (carried from B1, the mediator repair)**: all
  16 adapts run `AXIS1_ADAPT_CONFIG=readout64_frozen` (head widths
  pinned at the size1m value 64); the reader gates
  `agent.policy.units == 64` per adapt.
- **Smoke gate**: ONE fit (task, sk6, side0 seed99) — config audit
  (rssm deter 384, expl.mode task), param count into the FILL slot,
  smoke dir REMOVED BY HAND (seed99 lesson).
- **Registered submissions** (names follow the submit_all
  `wm_infix = ID_PREFIX#ax1` mechanics, verified for the sk lineage):
  - task: `AXIS1_EXPL_MODE=task AXIS1_ID_PREFIX=ax1sk6 AXIS1_QUAD_SUFFIX=_ctxk6 AXIS1_SIZE=size600k AXIS1_ADAPT_CONFIG=readout64_frozen AXIS1_DOMAINS=finger AXIS1_QUADS=q1 AXIS1_SEEDS="1 2 3 4" ./scripts/submit_all.sh axis1-bundles`
  - apt: same with `AXIS1_EXPL_MODE=apt AXIS1_ID_PREFIX=ax1fsk6`.
  Names: WMs `ax1wm_finger_{,f}sk6q1s{0,1}_seed{1..4}`; adapts
  `adapt_ax1{,f}sk6q1s{side}_finger_seed{k}_ckpt500000` (UPDATES
  500000, STEPS 1.25e5 — protocol identical to B1 except size).
- **Measures per fit**: `ridge_probe` measure on `finger_v1` (default
  protocol), outputs renamed `<wm_run>.json`; adaptation-AUC collate
  over the 16 adapts; the E4 default pass runs and its csv ships in
  the bundle for LABELED analyses only (the frozen reader does not
  consume it). Witness producer = the LITERAL capdescent block with
  BOTH load-specific literals swapped (the m6 lesson): glob
  `root + '/ax1wm_finger_*sk6q1s*'` AND regex
  `ax1wm_finger_f?sk6q1s[01]_seed[1-4]$` (assert len == 16, not 32).
- Cleanup KEEPs (this freeze-commit, seed-restricted):
  `ax1wm_finger_*sk6q1s*_seed[1-4]` `adapt_ax1*sk6q1s*_seed[1-4]_*`
  (disjoint from the sk[13] globs; ctx pair lives under
  `axis1_finger` = KEEP_SUBSTRATE).
- Bundle: auc csv + 16 renamed ridge jsons + e4 csv + witness jsons +
  per-run config.yamls (fits AND adapts — the reader gates both).

## Registered decision rules (reader: `analysis/b1prime_read.py`)

Gates (fail-closed): 16-name dual fit witness (update==total, ckpt
500000); per-fit config gate (deter == 384, expl mode by arm);
per-adapt gate (from_checkpoint names the registered WM, frozen_wm,
**policy.units == 64**); auc gates (finger domain, milestone 500000,
qc_pass, STRICT modal n_ep — any off-modal row REFUSES, exact-16
inventory, duplicates REFUSE); ridge gates (run_id↔filename,
probeset finger_v1, no reward_override, witness_match not False,
finite auroc, exact-16).

- **P-C1 (PRIMARY, side-stratified — reviews B1 + B3)**: per side
  s ∈ {0, 1}: `two_sample(task_600k_side_s (4 fits),
  ANCHOR_Ss (the 8 pinned per-run 1m values of that side))` — house
  BCa 95% B=10K rng 0 + label permutation (the shared core in
  `analysis/capdescent_read.py`, frozen jointly; changes need dated
  amendments to every dependent reader). Multiplicity: step-up BH
  over the two side p's (kmax = max rank with p_(r) ≤ .05·r/2;
  ranked[:kmax] reject, classified by CI sign). A side DECLINES iff
  BH-rejected ∧ CI upper < 0; a side RISES iff BH-rejected ∧ CI
  lower > 0. **A side-0 rise ⇒ LEG-RISE-ANOMALY(side0)** (against
  the executed capacity series; investigate before any wording; the
  floor result is reported inside the string). **A side-1 rise is
  NOT an anomaly** — it is the expected extrapolation of the
  executed side-1 series (0.967→0.984→0.994) and is appended as a
  benign note.
- **P-C2 (floor)**: task mean auc100k > **113.682** (point rule, the
  B1 constant) ⇒ behavior ALIVE at 600k.
- **Apt no-move control (direction-split flag, not a branch selector
  — review M1)**: one-sample on the 8 apt fits' (AUROC − 0.40107).
  Whole CI ⊂ (−0.10, +0.10) ⇒ CONTROL-OK. CI entirely ABOVE +0.10
  (movement TOWARD chance — the anchor is BELOW 0.5, so attenuating
  anti-alignment is mechanism-consistent, not instrument doubt) ⇒
  **CONTROL-DRIFT-TOWARD-CHANCE** (descriptive). CI entirely BELOW
  −0.10 (MORE anti-predictive) ⇒ **CONTROL-SUSPECT**. Else
  CONTROL-INDETERMINATE (wide; note only). Registered semantics: a
  violation flags that the reward-free arm's probe relation moved;
  it does NOT invalidate the task-arm estimate — the label rides on
  the verdict string as information, never as a branch selector.
- **Verdict map**:
  - BOTH sides decline ∧ P-C2 ⇒ **MECHANISM-CONVERGENT** —
    reward-legibility erodes on both sides while behavioral
    competence is intact: the knee-before-floor ordering on the
    capacity axis; the competition mechanism gains its SECOND
    independent g_k lever.
  - EXACTLY ONE side declines ∧ P-C2 ⇒
    **MECHANISM-CONVERGENT-SINGLE-SIDE(side s)** — the qualified
    form; every mechanism sentence written from it MUST carry the
    side qualifier. (A side-0-only fire is the a-priori-likely
    outcome given the executed stratification — disclosed above.)
  - NO decline ∧ P-C2 ⇒ **KNEE-NOT-REACHED** — no decline detected
    at 600k; any knee would lie below 600k on the DESCRIPTIVE 300k
    side-0 value (0.710 — itself never tested against the 1m
    anchor; review M2 wording); convergence unadjudicated here.
  - decline(s) ∧ ¬P-C2 ⇒ **DECLINE-WITH-CLIFF** — ordering not
    establishable at this size; the learnability cliff relocates to
    (600k, 1m] (a new fact vs B1's (300k, 1m]).
  - none ∧ ¬P-C2 ⇒ **DISSOCIATION-REPEATS** — the B1-300k shape
    recurs at 600k; no ordering claim.
- Descriptives (no rules): behavioral interaction I(600k) =
  two_sample(task, apt aucs) with executed anchors quoted
  value-aware; apt behavioral panel; per-side MDE recomputed in-read
  INCLUDING anchor-sample uncertainty; E4 panels via labeled
  post-read analysis only.
- **Read procedure gates (registered)**: bundle MANIFEST.sha256
  verified against the committed manifest BEFORE any file is opened
  (the standing rule, made explicit here — review minor 8); reader
  selfcheck re-run immediately before the ONE execution.
- **Consequence mapping (frozen)**: MECHANISM-CONVERGENT ⇒ the paper
  may state the two-lever form ("crowding and capacity contraction
  both excise the rescued reward-relevant support first — the same
  exclusion ordering under two independent g_k manipulations"),
  citing this wave + the σ-ladder as its predicted tests, WITH the
  M3 ctx-zeroing caveat carried. The SINGLE-SIDE form supports the
  same sentence side-qualified. KNEE-NOT-REACHED /
  DISSOCIATION-REPEATS / DECLINE-WITH-CLIFF ⇒ facts reported; the
  σ-ladder + B2 pair stands alone as the mechanism evidence; no
  capacity-side wording. This wave does not touch the Part-B λ
  adjudication or the capacity-robustness flagship.

Registered per-fit measure/collate forms (review minor 6; arg names
verified against the current CLIs): ridge —
`python -m probing.ridge_probe measure --run_logdir <wm_dir> --probeset $RUNROOT/e4_probesets/finger_v1 --output <wm_dir>/ridge_probe_finger_v1.json`
(on Vast lanes add `--platform cuda` — the σ-ladder operational
lesson: the default `gpu` is not a backend name in that JAX build),
then renamed `<wm_run>.json` into the bundle; AUC —
`python -m analysis.adaptation_auc --runroot $RUNROOT --output <collate_dir>`
(`--output` is a DIRECTORY; the csv inside ships in the bundle; the
collate scans the whole runroot — the reader filters by the
seed-anchored sk6 regex and enforces the exact-16 inventory, and a
seed99 smoke adapt row is SKIPPED by the regex, which is the
registered behavior). The witness block's trailing `print` is edited
to `16/16 witnessed` along with the glob/regex/count (review minor 7).

Reader frozen with this file, selfcheck PASS pre-freeze (seven branch
fixtures incl. single-side and both rise semantics + the three
control labels; 18 refusal legs incl. the deter WALK, all four head
pins, and the full ridge battery; anchor pins asserted to the digit
against the pooled 0.9347681667927606). ONE read execution.
Reviewer: ONE (Opus 5, standing auto-approval) — sequential review
BEFORE freeze-commit; verdict was NOT-FREEZE-READY with 3 BLOCKING
(B1 side stratification absent from the estimand; B2 MDE contradicted
by executed sds; B3 anchor treated as a constant) + 3 MAJOR (M1
control semantics; M2 untested knee-interval claim; M3 ctx-zeroing
asymmetry) + 9 minor — ALL applied in this file + the reader;
reviewer verified plumbing/naming/globs/CLI literalness/config
composition clean (incl. that readout64_frozen lands LAST and pins
all four heads, and that no existing glob collides with sk6);
post-fix selfcheck re-PASS. FREEZE-READY in this form.
