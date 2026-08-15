# PREREG: rescue-under-load (the a²-term boundary test) — 2026-08-15

User GO 2026-08-15 ("let's do the rescue-under-load"). Resolves the
standing s×w_r inert-a² tension (`swave_read_20260810`:
BOTH-AXES-INERT — scaling the reward-head loss weight moved nothing
in the tested range) by testing the a²-rescue's magnitude WHERE the
theory says magnitude matters: at the inclusion boundary. The
σ-ladder located that boundary — the rescued reward-relevant support
dies below σ=2 (labeled fit: knee σ\* point 1.92; bootstrap 95% CI
**[0.42, 2.06]** — quoted in full, review M7: the soft lower tail
means σ=2 may sit FAR past the boundary, not just past it, in which
case a null admits a too-small-dose reading — w_r=100 arriving after
deep exclusion — alongside the lever-does-not-reach reading; both
are folded into the NO-RESCUE branch wording below). **Frozen
prediction lattice**:

- **Saturation-off-boundary account** (favored a priori; the σ-ladder
  supports it — at σ=1 task legibility is ~0.91, features included
  with slack, so extra w_r buys nothing observable): under load σ=2,
  w_r=100 raises the rescued eigenvalue λ_rew(1+β·a²·f(w_r)) back
  above the floor ⇒ task legibility at σ=2 RISES vs w_r=1.
- **Gradient-geometric account**: a² is about gradient direction, not
  loss magnitude (normalization absorbs w_r; the lever never reaches
  the spectrum) ⇒ no change even at the boundary.
- Either fire resolves the tension; the null resolves it as-scoped.

## Honesty block (value-aware, disclosed)

Known at freeze: all executed reads incl. the σ-ladder (task nzs2
pooled 0.7542; per-side 0.7091/0.7992; per-fit sd 0.0925 — the
mid-knee cell has the ladder's largest task spread, and it is
side-stratified like everything else), and the LABELED parametric
fit whose knee estimate motivated the σ=2 design choice (a design
input, not a claim; the prediction above is what freezes here).
Unknown: every w_r=100-under-load quantity (s×w_r never combined
w_r with distractor load). Design decisions against known hazards:
(i) BOTH cells run fresh in-wave (the σ-ladder's executed w=1 cells
ran on Vast; reusing them would confound arm with hardware/era — the
executed values enter only as a cross-hardware replication
DESCRIPTIVE); (ii) the primary is PAIRED per (side, seed) — the
side stratification (0.71 vs 0.80 at this load) cancels in the
delta; (iii) MDE (review M3 — computed by SIMULATION of the registered
conjunction, not the normal approximation, on the pinned per-fit
basis: paired-delta sd 0.1206): **paired MDE80 ≈ 0.15**; the
full-rescue reference (+0.16 = the executed σ1→σ2 task drop) is
detectable at ~87%, a half rescue (~0.08) at ~35% — NOT detectable,
under ANY pairing (variance decomposition of the pinned values: the
side fixed effect is only ~8% of the variance; the earlier "if
pairing removes most of the side variance" hedge was arithmetically
unreachable and is withdrawn). A null therefore discriminates
against FULL-magnitude rescues only, and the NO-RESCUE verdict
string carries the realized MDE inline; (iv) ceiling censoring
(review minor 7): one pinned w1 anchor is 0.954 (s1/seed2, 0.046
headroom) — if reproduced, that pair's delta is capped and the mean
full-rescue effect attenuates ~0.146 vs 0.16; disclosed; (v) the
adapts run at rew=1.0 in BOTH cells (AXIS1_WR reaches the FIT
invocation only — verified; review minor 2) under the default
frozen_readout adapt config — adapt scores must never be read as
"behavior under w_r=100".

## Design — 2 w_r cells × 2 sides × 4 seeds = 16 fit+adapt jobs

- Substrate REUSED, nothing new built: buffers `q1_nzs2` (σ=2, the
  σ-ladder pair; the reader re-gates its manifest incl. the pinned
  B2-era holdout sha) + probeset `finger_nzs2_v1` + config token
  `dzs2` — all frozen and cluster-resident.
- Both cells TASK arm (AXIS1_WR is task-mode-only by construction);
  w_r enters as `AXIS1_WR` → `--agent.loss_scales.rew` (the s×w_r
  plumbing, verified live: the saved config records
  `agent.loss_scales.rew`, default 1.0). w_r=1 is passed EXPLICITLY
  so the knob is recorded on both cells.
- **Smoke gate (literal direct form — review minor 1: the
  axis1-bundles route cannot produce a single-side seed99 short fit;
  it loops both sides at full updates)**:
  `RUN_ID=adapt_ruwsmoke WM_RUN=ax1wm_finger_ruw100q1s0_seed99 TASK=dmc_finger_turn_hard SEED=99 REPLAY=$RUNROOT/axis1_finger/q1_nzs2/side0 UPDATES=2 STEPS=1e3 AXIS1_EXPL_MODE=task AXIS1_WR=100 AXIS1_BASE_CONFIG=dzs2 bash scripts/axis1.sbatch`
  must run past update 1; config audit (`loss_scales.rew: 100.0`,
  `distractor.basesd: 2.0`); BOTH smoke dirs (fit + adapt) removed
  BY HAND.
- **Registered submissions** (names verified against the submit_all
  `wm_infix = ID_PREFIX#ax1` mechanics; the reader's regex
  disambiguates ruw1 from ruw100 — selfchecked):
  - w1: `AXIS1_EXPL_MODE=task AXIS1_WR=1 AXIS1_ID_PREFIX=ax1ruw1 AXIS1_QUAD_SUFFIX=_nzs2 AXIS1_BASE_CONFIG=dzs2 AXIS1_DOMAINS=finger AXIS1_QUADS=q1 AXIS1_SEEDS="1 2 3 4" ./scripts/submit_all.sh axis1-bundles`
  - w100: same with `AXIS1_WR=100 AXIS1_ID_PREFIX=ax1ruw100`.
  Names: WMs `ax1wm_finger_ruw{1,100}q1s{0,1}_seed{1..4}`; adapts
  `adapt_ax1ruw{1,100}q1s{side}_finger_seed{k}_ckpt500000` (UPDATES
  500000, STEPS 1.25e5; adapts DESCRIPTIVE-ONLY, no behavioral
  rule).
- **Measures per fit**: `ridge_probe measure --run_logdir <wm_dir>
  --probeset $RUNROOT/e4_probesets/finger_nzs2_v1 --output
  <wm_dir>/ridge_probe.json` (Vast lanes add `--platform cuda`),
  renamed `<wm_run>.json`. Witness producer = the LITERAL capdescent
  block with BOTH literals swapped: glob
  `root + '/ax1wm_finger_ruw*q1s*'`, regex
  `ax1wm_finger_ruw(1|100)q1s[01]_seed[1-4]$`, `len == 16`, print
  `16/16 witnessed`. Bundle MUST include `q1_nzs2/manifest.json`.
- **Fit→buffer LINKAGE (review B1/M2 — this is the ONLY witness that
  the fits consumed the σ=2 buffer; config.yaml records the env
  token, never `--static_replay`; a `_nzs8` mis-submission passes
  every other gate). LITERAL block, run on the cluster BEFORE any
  cleanup; `ruw_replay_linkage.json` ships in the bundle and the
  read is REFUSED if it is absent**:

```
python - <<'PY'
import csv, json, os, re
rows = {}
with open(os.environ['MANIFEST']) as f:
    for r in csv.reader(f):
        if r and r[0].startswith('ax1wm_finger_ruw'):
            rows[r[0]] = r
rx = re.compile(r'^ax1wm_finger_ruw(1|100)q1s([01])_seed[1-4]$')
out, n = {}, 0
for name, r in sorted(rows.items()):
    m = rx.match(name)
    if not m:
        continue
    replay = r[6]
    want = f'q1_nzs2/side{m.group(2)}'
    assert replay.rstrip('/').endswith(want), (name, replay, want)
    out[name] = replay
    n += 1
assert n == 16, (n, sorted(out))
json.dump(out, open('ruw_replay_linkage.json', 'w'), indent=1)
print('16/16 replay linkage OK')
PY
```

  (row schema: REPLAY at field 7 / index 6, the axis1.sbatch
  DONE-rewrite form; on the ops/waves route the per-instance
  runs.csv files are concatenated first, the σ-ladder conformance
  precedent — and on that route `AXIS1_WR` must be present in each
  task's cmd env, it being the ONLY thing distinguishing the cells;
  the reader's both-directions `loss_scales.rew` gate is the
  backstop.)
- Cleanup KEEPs (this freeze-commit):
  `ax1wm_finger_ruw*q1s*_seed[1-4]` `adapt_ax1ruw*q1s*_seed[1-4]_*`
  (token `ruw` verified against every existing KEEP/DELETE glob and
  reader regex — disjoint).

## Registered decision rules (reader: `analysis/rescue_load_read.py`)

Gates (fail-closed): `q1_nzs2` manifest (σ=2, dims 16, holdout 16,
tool, source, **the pinned B2-era holdout sha**);
**`ruw_replay_linkage.json` loaded via `--linkage` — exactly 16 rows,
every row's REPLAY path ending `q1_nzs2/side<side>` for its run name;
absent or wrong-buffer ⇒ REFUSE (review B1)**; 16-name dual witness;
per-fit config gate (expl task; dzs2 distractor block dim/basesd/
theta; **`loss_scales.rew` == the cell's w_r, both directions**);
ridge gates (run_id↔filename, probeset finger_nzs2_v1, no
reward_override, witness_match not False, finite auroc); exact-16;
duplicates REFUSE; bundle MANIFEST.sha256 verified against the
committed manifest before any file is opened; reader selfcheck
re-run immediately before the ONE execution.

- **P-R1 (PRIMARY, paired)**: Δ_(side,seed) = AUROC(w100) −
  AUROC(w1); one-sample on the 8 deltas (house BCa 95% + EXACT
  sign-flip 2^8; the shared `one_sample` core in
  `analysis/domains_read.py` — frozen jointly; changes need dated
  amendments to all dependents).
  - CI lower > 0 ∧ p < .05 ⇒ **RESCUE-CONFIRMED** — the a²-rescue
    magnitude binds at the boundary; the s×w_r inertness is
    reconciled as off-boundary saturation; the loss-weight lever
    reaches the spectrum; β becomes experimentally manipulable.
  - CI upper < 0 ∧ p < .05 ⇒ **RESCUE-INVERTED** — outside both
    accounts; fact reported, no further wording.
  - else ⇒ **NO-RESCUE-DETECTED** — the lever does not move the
    boundary at this dose; favors the gradient-geometric reading
    AGAINST FULL-MAGNITUDE rescues only (the verdict STRING carries
    the realized 80%-power MDE inline — review M4; rescues below it
    are NOT excluded, and the knee CI's lower tail additionally
    admits a too-small-dose reading — review M7). NOT an equivalence
    claim.
- **Cross-hardware replication (DESCRIPTIVE, never a verdict
  input)**: in-wave w1 cell vs the 8 PINNED executed σ-ladder nzs2
  task per-fit values (pinned as literals in the reader; a large
  discrepancy is an instrument note).
- Per-side panels descriptive.
- **Consequence mapping (frozen)**: RESCUE-CONFIRMED ⇒ the paper's
  theory section writes the saturation reconciliation as a
  CONFIRMED mechanism sentence (with this wave as its predicted
  test) and may treat β as measurable; a follow-up β dose-response
  (w_r ladder under load) becomes a registrable option, NOT
  authorized here. NO-RESCUE ⇒ the s×w_r tension is written as
  boundary-tested inertness (gradient-geometric reading favored,
  stated with the MDE bound). INVERTED ⇒ reported; no account
  adopted. No branch touches the σ-ladder/B2 mechanism claims or
  Part-B.

Reader frozen with this file, selfcheck PASS pre-freeze (three
branch fixtures + the DIAGNOSTIC pairing leg — paired fires while
the unpaired comparator is ns on the same side-gap fixture — +
ruw1/ruw100 disambiguation; 18 refusal legs incl. per-cell rew BOTH
directions, the B2-era holdout sha, foreign-run_id, rename, and the
wrong-buffer/short linkage legs). ONE read execution. Reviewer: ONE
(Opus 5, standing auto-approval) — sequential review BEFORE
freeze-commit; verdict was NOT-FREEZE-READY with 1 BLOCKING (B1 no
binding buffer witness — the linkage promoted to a fail-closed
reader gate) + 6 MAJOR (M2 literal linkage block; M3 simulated-
conjunction MDE 3.4×, half-rescue hedge withdrawn; M4 MDE in the
verdict string; M5 diagnostic pairing leg; M6 foreign-run_id leg;
M7 full knee CI + too-small-dose reading) + 9 minor (literal smoke
form, adapt rew=1.0 disclosure, ceiling censoring, per-fit values
stored in read.json, manifest source gate, probeset equality) — ALL
applied; reviewer verified naming/globs/plumbing/substrate-reuse
clean (AXIS1_WR unrestricted + float-coerced; pairing legitimate,
~8% variance gain; all 8 pinned anchors to the digit); post-fix
selfcheck re-PASS. FREEZE-READY in this form.
