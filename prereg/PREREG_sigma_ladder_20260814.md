# PREREG: sigma-ladder extension of B2 (crowding-account discriminator) — 2026-08-14

User GO 2026-08-14 ("for the account b you've mentioned, I think a
sigma ladder should be good to GO"). Discriminates the two candidate
accounts of the executed B2 NEGATIVE-DID
(`artifacts/b2_nuisance_read_20260813/`, verdict outside both
registered models):

- **(a) proportional-headroom compression**: any monotone stressor
  shrinks each arm's legibility toward chance in proportion to its
  headroom; the B2 DID appeared only because the task arm had more to
  lose.
- **(b) competition-with-swamped-rescue** (the spectral model in the
  g-swamped regime): reward-relevant support is low-variance, so a
  rising noise floor g(σ) excludes it FIRST once it crosses the
  rescued eigenvalue λ_rew(1+βa²); the task arm's NORMALIZED loss
  exceeds the reward-free arm's, increasingly with σ, until joint
  collapse.

**Frozen predictions (this file, before any σ∈{2,8} quantity
exists)**: (b) ⇒ the headroom-normalized task−apt drop contrast is ≈0
below the knee and POSITIVE past it, growing with σ up to joint
collapse; at extreme σ a task-collapse-with-apt-retained
configuration is (b)-consistent and (a)-violating. (a) ⇒ the
normalized contrast is 0 at EVERY load.

## Honesty block (value-aware, disclosed)

Known at freeze: ALL executed reads, in particular the B2 read this
wave extends — nz1 means task 0.9133 / apt 0.7180, nz2 means task
0.6794 / apt 0.6805. The value-aware motivating anchor: at σ=4 the
realized normalized drops are 0.566 (task) vs 0.172 (apt); the
proportional-null statistic S(4) = raw drop_task − r·raw drop_apt =
**+0.163** (r = headroom ratio 1.896; on RAW drops — the equivalent
normalized contrast is S(4)/HR_task = 0.394, review m7) — the
executed data already strain account (a) descriptively AT σ=4. That
is WHY the primary tests FRESH loads only (σ = 2 and 8). Role of the
executed cells (review M3, stated precisely): σ=4 enters ONLY as a
descriptive curve anchor; **σ=1 enters as the PINNED normalization
baseline of the primary** (numerator origin and per-arm scale) — a
known, value-aware constant frozen in reader code before any σ∈{2,8}
quantity exists, leaving no researcher degree of freedom. Unknown: every
σ∈{2,8} quantity. Registered risks: (i) σ=8 may JOINT-COLLAPSE both
arms (the proportional null is trivially satisfied at total collapse —
handled by the floor classification, and both-loads collapse ⇒
INSTRUMENT-LIMITED); (ii) the equivalence margin (±0.25 normalized) is
weak at n=8 — PROPORTIONAL-CONSISTENT is hard to reach and
INDETERMINATE is the expected outcome if truth is null; disclosed.
Symlog compression disclosure (B2 delta-rev F8 lineage): obs-space σ
ratios 1:2:4:8 compress to ≈1:1.6:2.3:3.2 in symlog loss-sd units —
the ladder is denser in loss units than in σ, disclosed.

## Design — 2 new loads × 2 arms × 2 sides × 4 seeds = 32 fit+adapt jobs

Protocol-identical to B2 except basesd. New single-token configs
(this freeze-commit, `dreamerv3/configs.yaml`): `dzs2` / `dzs8` =
dmc_proprio + `distractor: {dim: 16, basesd: 2.0|8.0, scale: 1.0,
theta: 1.0}` (tokens carry σ explicitly so ladder order stays
readable: dz1, dzs2, dz2, dzs8 = σ 1, 2, 4, 8). Whitelists extended
in `scripts/axis1.sbatch` + `scripts/submit_all.sh`.

- **Buffers ([YOU], login node)** — same tool, same source, same
  holdout construction. The matched-probe-experience invariant the
  PRIMARY needs is that the new holdout episodes are THE SAME set the
  pinned B2 baselines were probed on; within-pair filename equality
  alone is nearly vacuous (the rng-0 draw depends only on the source
  LISTING at build time — review B2), so the reader gates the
  canonical holdout-set sha against the pinned B2-era digest
  `B2_HOLDOUT_SHA = 0c76043733f0bd9bab13b94dc2d659c4b124b73220fe1f709012390026c0e97a`
  (computed from BOTH committed B2 buffer manifests, which agree;
  verified independently 14 Aug). If the q1 source listing changed
  since 12 Aug the read REFUSES — do not "fix" that by rebuilding
  anything; it is a dated-amendment situation. Commands:
  `python -m probing.nuisance_replay build --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_nzs2 --sigma 2.0 --dims 16 --holdout 16`
  `python -m probing.nuisance_replay build --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_nzs8 --sigma 8.0 --dims 16 --holdout 16`
- **Probesets ([YOU], after buffers; label=dir form)**:
  `python -m probing.stratified_error build-probeset --task dmc_finger_turn_hard --source s0=$RUNROOT/axis1_finger/q1_nzs2/heldout/side0 s1=$RUNROOT/axis1_finger/q1_nzs2/heldout/side1 --n_episodes 16 --seed 0 --output $RUNROOT/e4_probesets/finger_nzs2_v1`
  and the same with nzs8, then frozen.
- **Smoke gate**: ONE fit (task, nzs8, side0 seed99 — the HIGHEST
  load, the numerically riskiest cell) must run past update 1 AND its
  `config.yaml` must record `distractor: {dim: 16, basesd: 8.0}`;
  smoke dir removed BY HAND (seed99 lesson: never rely on cleanup).
- **Registered submissions** (axis1-bundles; names verified against
  the submit_all naming code `wm_infix = ID_PREFIX#ax1`):
  - task σ2: `AXIS1_EXPL_MODE=task AXIS1_ID_PREFIX=ax1nzs2 AXIS1_QUAD_SUFFIX=_nzs2 AXIS1_BASE_CONFIG=dzs2 AXIS1_DOMAINS=finger AXIS1_QUADS=q1 AXIS1_SEEDS="1 2 3 4" ./scripts/submit_all.sh axis1-bundles`
  - apt σ2: same with `AXIS1_EXPL_MODE=apt AXIS1_ID_PREFIX=ax1fnzs2`
  - σ8: both again with `AXIS1_ID_PREFIX=ax1nzs8/ax1fnzs8 AXIS1_QUAD_SUFFIX=_nzs8 AXIS1_BASE_CONFIG=dzs8`.
  Names: WMs `ax1wm_finger_{,f}nzs{2,8}q1s{0,1}_seed{1..4}`; adapts
  `adapt_ax1{,f}nzs{2,8}q1s{side}_finger_seed{k}_ckpt500000`
  (UPDATES 500000, STEPS 1.25e5; adapts remain DESCRIPTIVE-ONLY as in
  B2 — no behavioral rule).
- **Measures per fit**: `ridge_probe` measure on that fit's LOAD
  probeset (`finger_nzs2_v1`/`finger_nzs8_v1`, default protocol);
  each output json renamed `<wm_run>.json` in the bundle (reader
  gates the rename). Witness producer = the LITERAL capdescent block
  with BOTH load-specific literals swapped (review m6): the glob to
  `root + '/ax1wm_finger_*nzs[28]q1s*'` AND the regex to
  `ax1wm_finger_f?nzs[28]q1s[01]_seed[1-4]$`.
  **Bundle MUST include `q1_nzs2/manifest.json` +
  `q1_nzs8/manifest.json`** (replay-side σ gates + the pinned
  B2-era holdout-sha gate).
- **Fit→buffer LINKAGE GATE (review M5 — literal, run on the cluster
  BEFORE any cleanup, output committed with the read artifact)**: the
  fit config records no replay path (`--static_replay` never enters
  config.yaml; the path lives only in the cluster runs manifest), and
  this wave has four confusable tokens (`_nz1/_nzs2/_nz2/_nzs8`), so
  the linkage is gated by this block, not prose:

```
python - <<'PY'
import csv, json, os, re
rows = {}
with open(os.environ['MANIFEST']) as f:
    for r in csv.reader(f):
        if r and r[0].startswith('ax1wm_finger_') and 'nzs' in r[0]:
            rows[r[0]] = r
rx = re.compile(r'^ax1wm_finger_f?(nzs[28])q1s([01])_seed[1-4]$')
out, n = {}, 0
for name, r in sorted(rows.items()):
    m = rx.match(name)
    if not m:
        continue
    replay = r[6]
    want = f'q1_{m.group(1)}/side{m.group(2)}'
    assert replay.rstrip('/').endswith(want), (name, replay, want)
    out[name] = replay
    n += 1
assert n == 32, (n, sorted(out))
json.dump(out, open('sgl_replay_linkage.json', 'w'), indent=1)
print('32/32 replay linkage OK')
PY
```

  (row schema: the axis1.sbatch DONE-rewrite puts REPLAY at
  field 7 / index 6 — `run,stage,task,static,seed,updates,REPLAY,dir,
  status,`; if the manifest schema differs at run time, conform the
  index and record the conformance in the bundle notes, refusing on
  any endswith mismatch. `sgl_replay_linkage.json` ships in the
  bundle; the read is REFUSED if it is absent.)
- Cleanup KEEPs (this freeze-commit, seed-restricted):
  `ax1wm_finger_*nzs[28]q1s*_seed[1-4]`
  `adapt_ax1*nzs[28]q1s*_seed[1-4]_*` (verified: the existing
  `nz[12]` globs cannot match nzs names, and no DELETE glob matches).

## Registered decision rules (reader: `analysis/sigma_ladder_read.py`)

Gates (fail-closed): 32-name dual fit witness (update==total, ckpt
500000); per-fit config gate on the RECORDED knob — `distractor.dim
== 16` ∧ `basesd == σ_of(load)` ∧ **`theta == 1.0`**; expl mode
matches arm; both replay manifests gated (σ, dims,
`holdout_per_side == 16`, `tool == nuisance_replay_v1`, equal
`source` across the pair, holdout-set equality across the pair, AND
**the canonical holdout-set sha == the pinned B2-era
`B2_HOLDOUT_SHA`** — the cross-era matched-probe-experience witness
the pinned baselines actually require, review B2); ridge-json gates
(run_id↔filename, load-matched probeset_id, no reward_override,
`witness_match is not False`, finite non-null auroc — review m8);
exact-32 inventory; duplicates REFUSE; `sgl_replay_linkage.json`
present in the bundle (the M5 gate's committed output).

Legibility scalar per fit: `probe.alpha_0.001.auroc`. Normalized drop
per fit: `(BASE_arm − auroc) / (BASE_arm − 0.5)` with the PINNED
executed-B2 baselines `BASE_TASK = 0.9133080491668023`, `BASE_APT =
0.7180130325674831` (recipe: b2 read.json panel_means task_nz1 /
apt_nz1; headroom ratio r = 1.8958… derived in-code).

- **Floor classification per load** (raw means, point rule, bar 0.55 —
  same conventional provenance as B2, disclosed): both ≥ 0.55 ⇒
  INFORMATIVE; task < 0.55 ≤ apt ⇒ **TASK-COLLAPSE-APT-RETAINED**;
  apt < 0.55 ≤ task ⇒ APT-COLLAPSE-TASK-RETAINED; both < 0.55 ⇒
  JOINT-COLLAPSE.
- **P-S1 (PRIMARY)**: per informative load, contrast = mean normalized
  task drop − mean normalized apt drop, house `two_sample` (BCa 95%
  B=10K rng 0 + label permutation; the shared statistical core in
  `analysis/capdescent_read.py` — frozen jointly; changes require
  dated amendments to every dependent reader). Multiplicity:
  **step-up BH** over the informative new loads (m = their count;
  kmax = max rank with p_(r) ≤ α·r/m at α=.05; ALL of ranked[:kmax]
  reject, then classify by CI sign — review M4: the per-rank variant
  is coherence-broken and could mis-attribute loads).
  - Verdict order: (1) both loads JOINT-COLLAPSE ⇒
    **INSTRUMENT-LIMITED** (proportional trivially satisfied at total
    collapse; no discrimination). (2) any TASK-COLLAPSE-APT-RETAINED
    OR any informative load with BH-rejected p ∧ CI > 0 ⇒
    **EXCESS-TASK-LOSS** — account (b) supported; proportional
    rejected at those loads. The task-side collapse branch is
    (a)-IMPOSSIBLE under the pins, not a convention: task < 0.55 ≤
    apt forces normalized task drop > 0.879 while proportionality
    caps the apt-retained band's common retention at f ≥ 0.229 ⇒
    normalized drops ≤ 0.771 — the configuration cannot arise under
    (a) (review B1). (If the opposite direction ALSO fires at
    another load ⇒ **MIXED-DIRECTIONS**, outside both accounts, fact
    reported.) (3) BH-rejected p ∧ CI < 0 at any informative load ⇒
    **EXCESS-APT-LOSS** — the originally-frozen λ direction at these
    σ; fact reported, no further wording. **APT-COLLAPSE-TASK-
    RETAINED triggers NOTHING** (review B1: under proportional
    compression the apt arm crosses the raw 0.55 bar at common
    retention f = .229 while the task arm crosses only at f = .121,
    so an apt-only crossing is the null's own generic mid-collapse
    signature — it is reported as floor censoring in `fired.
    collapse_negative` and carries no verdict weight). (4) every
    informative load ns with its whole CI inside ±0.25 normalized ⇒
    **PROPORTIONAL-CONSISTENT** — compression suffices at the named
    informative loads (the verdict string names them — review m12).
    Coverage disclosure (review m13): the BCa CI alone realizes ~90%
    coverage on the B2 sd basis (the p∧CI directional conjunction is
    calibrated at ~4.7%; the equivalence branch, which consumes the
    CI alone, is correspondingly somewhat EASIER to reach than a
    nominal-95% interval would imply — disclosed, opposite direction
    from the n=8 weakness note above). (5) else **INDETERMINATE**
    (licenses nothing).
- **P-S2 (descriptive, no rule)**: the 4-σ curve {1, 2, 4, 8} of raw
  arm means and S(σ) = RAW drop_task − r·RAW drop_apt (review m7:
  raw-drop units; S_raw = HR_task × normalized contrast), with
  σ∈{1,4} as the pinned executed anchors (S(1)=0 by construction,
  S(4)=+0.163 value-aware); knee localization narrative only.
- MDE disclosure: on the B2 sd basis (raw 0.051 ⇒ normalized ~0.123
  task / ~0.234 apt), contrast se ≈ 0.094 at n=8/group; 80%-power
  MDE ≈ 0.26 normalized units — the executed σ=4 anchor (0.395
  normalized) is ~1.5× this MDE. Pinned-anchor uncertainty (se
  ~0.031/0.059 normalized) is NOT propagated — disclosed. Realized
  MDE recomputed in-read.
- **Consequence mapping (frozen)**: EXCESS-TASK-LOSS ⇒ the B2 anomaly
  gets a competition-mechanism account (the spectral model in the
  g-swamped regime: predicted here, then confirmed) — paper wording
  may claim the mechanism WITH this wave cited as its predicted test;
  R1's scope note is written from that account. PROPORTIONAL-
  CONSISTENT ⇒ the anomaly is bounded as a compression artifact at
  the new loads; the spectral-regime story stays labeled reanalysis.
  EXCESS-APT-LOSS / MIXED ⇒ facts reported; no model adopted. This
  wave does NOT reopen the Part-B λ adjudication (that pair is
  complete and split); it adjudicates the ACCOUNT of B2's anomaly.

Reader frozen with this file, selfcheck PASS pre-freeze (branch
fixtures incl. both EXCESS routes, the apt-collapse REPORTED-ONLY leg
planted from the EXACT proportional null at f=0.20, proportional,
joint-collapse, indeterminate; 17 refusal legs incl. the B2-era
holdout sha, seed99/unmatched run_id, NaN auroc, rename, duplicate,
missing-json, missing-distractor, ckpt-step). ONE read execution.
Reviewer: ONE (Opus 5, standing auto-approval) — sequential build
review BEFORE freeze-commit; verdict was NOT-FREEZE-READY with 2
BLOCKING (B1 apt-collapse null-signature branch; B2 vacuous holdout
gate) + 3 MAJOR (M3 honesty wording, M4 per-rank BH, M5 linkage
gate) + 8 minor — ALL applied in this file + the reader; reviewer
adjudicated naming/globs/configs/CLI literalness/pins/calibration
clean (directional rule Type-I 4.7% at the B2 sds, incl. under the
normalization-induced heteroscedasticity); post-fix selfcheck
re-PASS. FREEZE-READY in this form.
