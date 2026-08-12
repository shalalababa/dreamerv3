# PREREG: B2 nuisance-injection λ trial (crowding test) — 2026-08-11 (post-review REDESIGNED form)

User GO 2026-08-11. The second registered λ-discriminating trial of
`PREREG_theory_R1_gating_20260811` **Part B** (+ Amendment 1 mapping):
under added high-variance distractor channels, **λ-competition ⇒ a
load×arm interaction** (reward-free arms lose reward-legible support
to crowding; the task arm's a² term rescues it); **R1 independent
gating ⇒ a main effect of load at most, NO load×arm interaction**.
Mediator (distractor variance) set BY CONSTRUCTION. Inclusion-level
measurement (reward-legibility of the fitted WM), not behavior. **This
file replaces the first draft in full** — reviewer-3 findings 1/2/3/9:
the draft's replay-only `nuisance` key could never reach the model
(obs spaces are env-derived; every fit would crash at update 1), its
config gate was vacuous on real configs, its probeset command was
unexecutable, and its informativeness bar cited the wrong field. The
redesign uses the **D0 distractor wrapper** — exactly the instrument
the frozen Part-B text names.

## Honesty block + registered design risk

Known at freeze: all executed reads — in particular the dv3 apt ridge
AUROC on finger_v1, **pooled 0.401** (the field this wave's gate
consumes: `probe.alpha_0.001.auroc`; the in-regime variant is 0.458 —
both quoted, reviewer-3 F9). **Registered informativeness risk**: if
the reward-free arm is at/near chance at LOW load, crowding cannot
reduce it further and the DID is uninformative. Handled by the gate
below. **The 0.55 bar's provenance is honestly conventional** (chance
0.5 + 0.05 margin), NOT derived from a zero-load reference on this
wave's probe distribution — no clean zero-load reference exists (the
existing q1 fits trained ON the holdout episodes), and this is
disclosed rather than dressed up; the a-priori chance the gate fires
is MATERIAL given the 0.401 anchor (different probeset, same field).
Unknown: every quantity on distractor-trained fits and nz probesets.

## Design — 2 loads × 2 arms × 2 sides × 4 seeds = 32 fit+adapt jobs

- **Injection mechanism (two halves, both required)**: (a) the env
  side — new single-token configs `dz1`/`dz2` (this freeze-commit,
  `dreamerv3/configs.yaml`): dmc_proprio base (yaml-inherited) +
  `distractor: {dim: 16, basesd: 1.0|4.0, scale: 1.0}` — the D0
  wrapper declares the `distractor` obs key in the env-derived obs
  space, passed as `AXIS1_BASE_CONFIG=dz<L>` so BOTH stages see it
  (fit: the space; adapt: the space AND live env-generated distractor
  obs — adapts are protocol-coherent but remain DESCRIPTIVE-ONLY as
  registered). The dz configs set `theta: 1.0` so the wrapper emits
  i.i.d. N(0, basesd²) — distributionally identical to the replay-side
  channels (delta-rev F3: the wrapper default is AR(1) θ=0.1, which
  would have mismatched the two halves' temporal statistics). Honesty
  note (delta-rev F8): σ is set by construction in OBS space; under
  the symlog decoder loss the 1→4 sd contrast compresses to ~2.4× in
  loss-sd terms — the dose is large but not 16× in loss units,
  disclosed. (b) the replay side — `probing/nuisance_replay.py`
  (frozen; selfcheck PASS) appends the SAME key `distractor` [T, 16]
  ~ N(0, σ²) to the fit buffers:
  `python -m probing.nuisance_replay build --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_nz1 --sigma 1.0 --dims 16 --holdout 16`
  `python -m probing.nuisance_replay build --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_nz2 --sigma 4.0 --dims 16 --holdout 16`
  Same holdout FILENAMES at both levels (level-invariant, verified in
  selfcheck) ⇒ matched probe experience across loads. No load-0 fits
  (the existing q1 fits trained on the holdout episodes — quoted as
  value-aware context only).
- **Probesets ([YOU], after buffers; label=dir form, reviewer-3 F3)**:
  `python -m probing.stratified_error build-probeset --task dmc_finger_turn_hard --source s0=$RUNROOT/axis1_finger/q1_nz<L>/heldout/side0 s1=$RUNROOT/axis1_finger/q1_nz<L>/heldout/side1 --n_episodes 16 --seed 0 --output $RUNROOT/e4_probesets/finger_nz<L>_v1`
  for L ∈ {1, 2}, then frozen.
- **Smoke gate (knob-validity)**: ONE fit (task, nz1, side0 seed99)
  must run past update 1 (the finding-1 crash mode is a first-update
  space-mismatch assert) AND its `config.yaml` must record
  `distractor: {dim: 16, basesd: 1.0}`; smoke dir removed BY HAND.
- **Registered submissions** (axis1-bundles; the export list carries
  AXIS1_BASE_CONFIG already, AXIS1_ADAPT_CONFIG via the F16 fix):
  - task nz1: `AXIS1_EXPL_MODE=task AXIS1_ID_PREFIX=ax1nz1 AXIS1_QUAD_SUFFIX=_nz1 AXIS1_BASE_CONFIG=dz1 AXIS1_DOMAINS=finger AXIS1_QUADS=q1 AXIS1_SEEDS="1 2 3 4" ./scripts/submit_all.sh axis1-bundles`
  - apt nz1: same with `AXIS1_EXPL_MODE=apt AXIS1_ID_PREFIX=ax1fnz1`
  - nz2: both again with `AXIS1_ID_PREFIX=ax1nz2/ax1fnz2 AXIS1_QUAD_SUFFIX=_nz2 AXIS1_BASE_CONFIG=dz2`.
  Names: WMs `ax1wm_finger_{,f}nz{1,2}q1s{0,1}_seed{1..4}`; adapts
  `adapt_ax1{,f}nz{1,2}q1s{side}_finger_seed{k}_ckpt500000`.
- **Measures per fit**: `ridge_probe` measure on that fit's LOAD
  probeset (`finger_nz1_v1`/`finger_nz2_v1`; default protocol, no
  override); each output json renamed `<wm_run>.json` in the bundle
  (ridge_probe writes identical basenames per fit dir; the reader's
  filename↔run_id gate checks the rename — delta-rev F5). Witness
  producer = the LITERAL capdescent block with the regex swapped to
  `ax1wm_finger_f?nz[12]q1s[01]_seed[1-4]$`. **Bundle MUST include
  `q1_nz1/manifest.json` + `q1_nz2/manifest.json`** — the reader
  gates the REPLAY-side σ/dims on them (delta-rev F2: the replay
  channels are the mediator; the env knob alone cannot witness which
  buffer the fits consumed). Operator pre-read linkage check
  (registered): for every fit row in the cluster runs manifest, the
  recorded REPLAY path must end `q1_nz<L>/side<side>` with L matching
  the run's name infix — any mismatch refuses the wave. Bundle: 32
  renamed ridge jsons + witness jsons + fit config.yamls + the two nz
  manifests (+ optional adapt-AUC csv, descriptive).
- Cleanup KEEPs (this freeze-commit, seed-restricted):
  `ax1wm_finger_*nz[12]q1s*_seed[1-4]` `adapt_ax1*nz[12]q1s*_seed[1-4]_*`
  (buffers/probesets live under `axis1_finger`/`e4_probesets` =
  KEEP_SUBSTRATE).

## Registered decision rules

Gates (fail-closed): 32-name dual fit witness; per-fit config gate
(reviewer-3 F2 form — on the RECORDED knob, not obs-space regexes:
`config.distractor.dim == 16` ∧ `basesd == σ_of(load)`; expl mode
matches arm); ridge-json gates (run_id↔filename, probeset_id == the
fit's load probeset — cross-load scoring REFUSES; no reward_override;
`witness_match is not False`); exact-32 inventory; duplicates REFUSE.

Legibility scalar per fit: `probe.alpha_0.001.auroc`.

- **Informativeness gate (registered)**: mean apt AUROC at nz1 must be
  ≥ **0.55**; else **APT-FLOOR-CENSORED ⇒ INSTRUMENT-LIMITED** — the
  reward-free arm has no legibility headroom to lose; the wave reports
  panels only (itself a registered descriptive fact).
- **P-N2a (PRIMARY)**: DID = [AUROC_task − AUROC_apt](nz2) −
  [AUROC_task − AUROC_apt](nz1), group bootstrap BCa 95% + within-arm
  load-label permutation p on **main-effect-ALIGNED values** (B=10K
  rng 0; the shared `capdescent_read.did_stats` — carries the rev-3 F4
  alignment fix; a load main effect is expected and must not deflate
  the permutation).
  - CI > 0 ∧ p < .05 ⇒ **LAMBDA-SUPPORTED**: crowding hits the
    reward-free arm harder — competition confirmed in the forced
    regime (Part-B adjudication jointly with B1 per the Amendment-1
    mapping).
  - CI < 0 ∧ p < .05 ⇒ **NEGATIVE-DID**: the task arm lost MORE —
    outside both models' registered predictions; fact reported.
  - else **INDETERMINATE** (in-read MDE note recorded from the
    realized AUROC spread; licenses nothing).
- **λ-negative mapping** (Amendment 1): only `NEGATIVE-DID` counts as
  λ-negative for the joint rule; `INDETERMINATE`/`APT-FLOOR-CENSORED`
  count as neither.
- Panels (descriptive): per-arm×load AUROC means; task-arm load slope
  (reported in ALL branches incl. the censored one); adapt AUC
  descriptives (UNINTERPRETED).

Reader: `analysis/nuisance_read.py` (redesigned gates; selfcheck PASS
incl. missing-distractor-block and basesd-mismatch refusals on
REAL-shaped configs), imports the DID machinery from
`analysis/capdescent_read.py`. ONE read execution. Reviewer: 2026-08-11
B1/B2 batch review (ONE, Opus 5) — B2 verdict was NEEDS-REDESIGN; this
file is the redesign; a delta re-review runs BEFORE freeze-commit.
