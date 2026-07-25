# Deviations from PREREG_phase5a.md

Any change to the frozen spec gets an entry here with date, what changed,
and why. (Registration record for new stages: tracked `prereg/` directory,
immutable dated files; see plan v4.)

## Deviations

- **2026-07-24 (later) — FULL CODE AUDIT: three further instrument
  issues found and fixed pre-freeze (one labels-affecting, two
  latent).** Systematic audit of the labeling/env/read stack after the
  morning's invalidation. (1) **xpol trajectory prevact
  mis-attribution** (AFFECTS the 23/24-Jul xpol labels — a third
  defect in those cells): with a behavior driver, the eval carry's
  prevact was the eval agent's own counterfactual sample while the env
  executed the behavior action ⇒ eval belief mis-conditioned on every
  xpol trajectory step. Fixed (`with_prevact` substitution +
  behavior-driven selfcheck section). (2) **Distractor OU state not in
  snapshot_env** (latent for e1 waves — wrapper not applied; REAL for
  any dosed labeling: R2's old e4 cells carried it, on top of the
  carry defect): rng captured but not the OU value/Welford triple ⇒
  branch CRN broken in distractor dims. Fixed
  (Distractor.oracle_get/set_state, exact round-trip verified,
  rng-only restore demonstrated inexact; snapshot custom slot now a
  list — was last-writer-wins). (3) **FromDM._done not restored**
  (latent — verified never fired: episodes 1000 wrapper steps, labels
  ≤ 875, branch reach ≤ 975): an episode ending inside a branch would
  have auto-reset the next branch and the resumed trajectory. Fixed
  (done-latch snapshot/restore + selfcheck). AUDITED-CLEAN (no
  change): d0 signal alignment (qfull columns ↔ cands rows within a
  call, row-major reshape verified), d1_shift_read pairing/meta
  asserts, adapt-run naming vs all three band-ledger read regexes
  (ax1ufz/ax1og/ax1x match the prereg submit loops exactly, milestone
  asserts correct), pixel_repl_read MODE_RE vs the running X2 modes,
  scaling_read MODES vs the Scaling-B naming (+ its selfcheck),
  baseline csv paths exist, orthreward wrapper (canonical Physics
  call, is_first guard, stateless), cluster-CI bootstrap, remaining
  wrapper chain stateless (NormalizeAction/UnifyDtypes/CheckSpaces/
  ClipAction). All fixes folded into the not-yet-committed relabel
  registration (PREREG_d1_relabel_20260724.md "Audit additions") —
  zero corrected labels existed at fix time. Labeler selfcheck PASS;
  distractor round-trip EXACT.

- **2026-07-24 — D1 REAL-OP LABELER: TWO IMPLEMENTATION DEFECTS
  CONFIRMED ⇒ ALL D1 decision-value conclusions reclassified
  INSTRUMENT-INVALIDATED (Gate-D1, R2, ladder, shift, context
  constants).** External review (`research_notes/other research/
  D1_GPT_Analysis_20260724.md`) claimed and code inspection CONFIRMED:
  (1) candidate branches evaluated obs_{t+1} from the PRE-obs_t carry
  with prevact = a_{t-1} instead of the candidate (`d0_eval`'s updated
  carry discarded at the label site; op_real and rollout_return both
  affected) — the labels are a well-defined but UNINTENDED estimand
  (returns under a belief-corrupted follower): nulls attenuated,
  positives unanchored; (2) eval-mode policy SAMPLES with seeds from
  the global n_actions counter, never restored across branches — the
  "exact CRN" claim covered env+belief only, not follower noise.
  Independently verified: finger 22/23 base cells are literal reward
  floor (all 200 G ≡ 0); small-cluster t-intervals are materially
  wider than the registered percentile bootstrap (xpol [−1.16,+1.15],
  phys [−0.60,+0.49] vs [−0.48,+0.33]); the ladder's "up to the full
  belief state" wording overclaims (L3 = deter512 + scalars, linear —
  no stoch, no qfull, no nonlinear). SCOPE: everything downstream of
  delta_real/G labels. NOT affected: Stage-0 attractor/phase-change,
  Stage-1A density residual (renamed "density-deconfounded
  disagreement residual" — diagnostic, not per-state gate), all of
  Paper 1 (no Paper-1 pipeline consumes D1 labels), belief features
  (deter was taken post-assimilation, correct). ACTIONS: labeler
  REPAIRED (candidate-conditioned branch carries + policy-RNG
  marks + g_all per-candidate returns; labeler_version d1fix_20260724;
  selfcheck extended with a carry-sensitive recurrent mock whose
  closed-form scores the stale convention provably fails — PASS);
  relabel campaign REGISTERED pre-outcome
  (prereg/PREREG_d1_relabel_20260724.md; 18 passes, existing shift
  pilots, d1_labels_fix/, defective labels preserved); invalidation
  addenda added to all four affected artifact records; original
  artifacts preserved unchanged as provenance. Paper-2 status
  reverted: experimental program NOT complete; "closure" claims of
  23–24 Jul withdrawn (plan v4 = EVPI_Plan_Revision_20260724.md).

- **2026-07-23 — R2 probe: imag-op estimand DEFECT confirmed pre-read;
  rule tightened by dated amendment; two process deviations recorded.**
  (1) External review (`research_notes/D1_GPT_Analysis_20260723.md`)
  claimed and code inspection CONFIRMED (before any R2 label was read)
  that `op_imag` averages Q matrices column-wise across `d0_eval`
  calls that each sample M−1 FRESH candidates — only the mode column
  is action-aligned, so the imagined-purchase Δ is a muddled estimand.
  `prereg/PREREG_gate_d1_r2probe_amend1_20260723.md` (written BEFORE
  executing the frozen read) restricts the decision rule to real-op
  cells (6 looks, strict tightening) and demotes all imag numbers to
  descriptive-defective. Retroactive: Gate-D1's imag cells carry the
  same defect (verdict unchanged — real cells also failed; imag ≈ 0
  reclassified uninterpretable). (2) Process deviation: the two
  transfer-guard patches (`fd66addb`, `f2798309`, plumbing only,
  diff-verified) were applied between smoke and labels WITHOUT the
  dated amendment the prereg required; disclosed one day late in
  Amendment 1 §B. (3) Attempt 1 of the probe was discarded
  outcome-blind (early snapshots at ~41.4K steps, outside the
  registered [10K,40K] window; step metadata only) and the full grid
  rerun with denser saving (`save60`); Amendment 1 §C.

- **2026-07-18 — Option C realized design ≠ registered ladder (side
  inversion).** The registered volume rider ("ladder on the hi side,
  occupancy matched") was not realized: the plain `search` subcommand
  does not enforce the side0=low convention, and the v400 pair came
  out high-first — so the `*v400s1*` cells ran on the LOW-occ (0.094)
  400-episode side while v200s1 is the hi-occ (0.323) 200-episode
  side. Volume is confounded with occupancy and composition; the
  registered directional statement ("no volume effect at fixed
  fraction") is NOT adjudicable from these cells and P-B4 is NOT
  adjudicated. Read + full audit in
  `artifacts/scaling_optc_20260718/RESULTS.md` (registered direction
  fails descriptively; the anomaly — task at occ .094/400 eps beating
  occ .323/200 eps 433 vs 204 with fewer in-regime frames — is
  recorded as unexplained). Cheap corrective on disk: the built
  `q1v400/side0` (occ .290) was never run; {task, apt} × 6 seeds = 12
  jobs would give the occ-matched volume AND fixed-volume occupancy
  contrasts; requires a dated amendment BEFORE those outcomes exist.
  Also noted: the synth seed-99 smoke fits (20K updates) were not
  deleted and were swept into the diagnosis e4 glob (n=9/side in the
  cluster-side read); the canonical diagnosis read excludes them by
  rule (verdicts unchanged).

- **2026-07-18 — DISCOVERED IMPLEMENTATION DEVIATION: the stamp
  functions consumed dyn/* context latents.** Surfaced by the E4
  override build ("probe set lacks stamp obs keys ['dyn/deter',
  'dyn/stoch']"): `stamp_obs_keys` selected ALL float keys of the
  buffer chunks, so g₀/g₁ ran on ~654 columns of which ~640 are the
  SOURCE agents' stored dyn/deter+dyn/stoch latents — the prereg said
  "flattened obs vector". With per-dim standardization the latent block
  dominates g, so the stamp direction largely lives in source-latent
  space, not instantaneous proprio. Consequences: (1) the registered
  behavioral PRIMARY is unaffected (labels deterministic, exact-count
  marginal-matched, still frame-bound and task-unaligned; the 18-Jul
  null and P-B1 branch stand as read); (2) the prereg scope statement
  "trivially predictable at h=0 from the frame's obs" does NOT hold as
  written — predictability of the stamp by the fitted trunk is now an
  empirical question that the pending stamp-NLL panel answers: low
  stamp-NLL ⇒ original "learnable-but-unaligned" reading stands; high
  stamp-NLL ⇒ srd degrades toward a hard-to-learn scalar and the P-B1
  claim narrows to "unaligned scalars of this class" (record at that
  panel's read); (3) fixes in `probing/relabel_replay.py` (selfcheck
  extended + PASS): `stamp_obs_keys` now excludes namespaced extras
  (future stamp waves get obs-only inputs; existing manifests are
  reproduced from their recorded obs_keys, bit-unchanged), and
  `stamp-probeset --latent_replay <pilot replay dirs>` recovers dyn/*
  for probe frames by full-20-byte stepid join (probe sets preserve
  original stepids), with the recovery recorded in the override meta.
  Probe-frame stamp labels use the probe pilots' own stored latents —
  the natural extension of g, noted here because the panel is
  descriptive-only.

- **2026-07-17 — scaling-pilot + synth-domain code notes (pre-outcome;
  registrations = `prereg/PREREG_scaling_pilot_20260717.md` [freeze
  after timing smoke] and `prereg/PREREG_synth_phaseb_20260717.md`
  [freeze after cluster smoke]).** (1) AXIS1_SIZE plumbing:
  `scripts/axis1.sbatch` applies an optional size config to BOTH stages
  (validated whitelist size1m..size400m; empty = historical default,
  bit-unchanged); `scripts/submit_all.sh` passes AXIS1_SIZE through the
  bundle export list and the non-bundled axis1 case (DRYRUN-verified;
  submit-time guard). New mode strings reserved: `ax1s12*`/`ax1fs12*`
  (12m task/apt; note wm_infix strips `ax1` so WM dirs are
  `ax1wm_finger_{s12|fs12}...` and E4 globs need `ax1wm_finger_*s12*`),
  `ax1v2*`/`ax1v4*` (volume rider) — all parse under the frozen RUN_RE,
  excluded from population models by the frozen `--modes` filters.
  (2) NEW `synth` suite: `embodied/envs/synthpred.py` + main.py ctor +
  `env.synth` config block + `probing/regimes.py` 'synth' spec
  (threshold 0.1 == default radius; regime ≡ reward condition) +
  `probing/synth_buffers.py` (direct pair synthesis, selfcheck PASS:
  occ within 0.01 of dials after calibration pass, exact label
  accounting, determinism, relabel-chain reuse) + `synth) task=
  synth_reach ;;` in all five submit_all domain cases. Synth rows carry
  domain='synth' and never pool with DMC populations. Full pipeline
  validated locally 17 Jul (task+apt debug fits, frozen-readout adapt,
  scores.jsonl frozen format; debug dirs deleted). (3) Read scripts
  frozen pre-outcome: `analysis/scaling_read.py` (bit-checks 1m strata
  vs frozen paired JSONs at read time), `analysis/synth_phaseb_read.py`,
  `probing/reward_direction_rank.py` (scaling-prereg prior probe) — all
  selfcheck PASS. (4) vgo wiring audit recorded
  (`artifacts/vgo_wiring_audit_20260717/AUDIT.md`): vgo = case (b)
  replay-grounded; P-B2 shifts to its registered case-(b) form.
  (5) 18-Jul smoke finding: size12m fits crash on the frozen q1 pair's
  stored size1m `dyn/*` replay-context latents (compiled-shape check;
  zero training progress, so no outcome information). Fix, pre-freeze:
  NEW `probing/resize_replay_context.py` (zero-init context copy at
  target dims, selfcheck PASS) + `AXIS1_QUAD_SUFFIX` in the
  axis1/axis1-bundles buffer-root lines (changes only the replay path,
  never run naming; DRYRUN-verified). The 12m cells fit `q1_ctx12`;
  disclosed in `PREREG_scaling_pilot_20260717.md` as a stratum-internal
  protocol difference.
- **2026-07-17 — stamping-wave code notes (pre-outcome; registration =
  `prereg/PREREG_stamping_20260717.md`).** (1) `probing/relabel_replay.py`
  gains stamp kinds `stamp_rand`/`stamp_iid` (+`--fn_seed`, `stamp-probeset`
  subcommand, selfcheck extensions; shuffle/relocate path untouched,
  selfcheck re-PASS). (2) `scripts/submit_all.sh` factorial transform
  whitelist `sh|rl` → `sh|rl|srd0|srd1|sid` (DRYRUN-verified: run ids
  `adapt_ax1srd0q1s<side>_*`, WM `ax1wm_finger_srd0q1s<side>_*`, buffer
  roots `axis1_<dom>/q1_<code>`; new mode strings parse under the frozen
  `RUN_RE` and are excluded from every population model by the frozen
  `--modes` filters). (3) INSTRUMENT EXTENSION, descriptive-only:
  `probing/stratified_error.py measure` gains optional `--reward_override`
  (replaces the reward-head NLL TARGET with an externally supplied array;
  strata remain true-reward-based; outputs go to `e4_<probeset>_ov-*`
  dirs so registered E4 outputs are never touched or pooled — collate
  only reads `e4_<probeset_id>/`). Default behavior bit-unchanged;
  numpy selfcheck re-PASS. (4) `analysis/stamping_read.py` frozen
  pre-outcome (selfcheck PASS, both branches recovered).
- **2026-07-11 — DISCOVERED IMPLEMENTATION DEVIATION: Axis-1 offline WM
  fits were reward-aware.** Found by the 11-Jul editorial code audit,
  verified same day. All 64 Axis-1 fits ran `offline_fit --configs
  dmc_proprio` with default `expl.mode: task` (scripts/axis1.sbatch; every
  saved `wm_audit/*/config.yaml`), which trains a reward head on logged
  task rewards with representation gradients (`reward_grad: true`) plus
  the replay-value loss (`repval_loss/repval_grad: true`) — whereas the
  Phase-4 online pretrains (`expl.mode` p2e/apt/random) skip both losses
  (`agent.py`: `reward_free`). In finger the regime is definitionally the
  reward condition (regimes.py), so reward supervision was differential by
  intervention side. **Consequence:** the 10-Jul Axis-1 read
  (`artifacts/phase6_axis1_20260710/`) is reclassified as the
  *reward-aware arm*; no reward-free causal claim rests on it. Corrective
  protocol frozen the same day in
  `prereg/PREREG_axis1_corrective_20260711.md` (reward-free `expl.mode
  apt` refit of the same buffers, seeds 1–8, run ids `adapt_ax1f*`;
  2×2 occupancy × reward-supervision read). E3 submissions and the
  seed-9–16 extension are paused behind the corrective read; queued
  reward-aware jobs cancelled. `axis1.sbatch` and `submit_all.sh` now
  require an explicit `AXIS1_EXPL_MODE` and derive distinct run-id
  prefixes, so the arm is always visible in run names and saved configs.
  Buffers, searches, and all Phase-5a observational results are unaffected
  (the deviation is in the fitting objective only). Plan v4
  (`research_notes/Research_Plan_v4_20260711.tex`) reorganizes the
  program around this; disclosure category: *corrective replication*.
  **Same-day operational follow-ups (v4 re-review, pre-outcome):**
  (1) bundle-path defect confirmed and fixed — `submit_axis1_bundle`'s
  explicit `--export` list lacked `AXIS1_EXPL_MODE` (bundle children would
  have died at the guard); added + value-validated (`apt|task`) + threaded
  through the bundle child; (2) flag-path defect caught by the required
  smoke test — `--expl.mode` is not a config key; corrected to
  `--agent.expl.mode`; (3) smoke test PASS on a real 400-step debug replay
  (apt: losses {con,dyn,position,rep,velocity}, no `rew` head; task: same
  + `rew`; saved configs record the mode; evidence in
  `artifacts/smoke_axis1_expl_20260711/`); per-run audit rule added.
  (4) `prereg/PREREG_axis1_corrective_amendment1_20260711.md` freezes the
  inferential hierarchy (finger Q1 sole confirmatory, cup Q2 secondary,
  bootstrap CI is the only decision criterion) and redefines the E3v2
  replication unit as the collector run (within-run cov-matched high/low
  pairs; feasibility measured from the frozen indices: finger 15/15 runs
  Δocc 0.20–0.36, cup 10/15 at 0.11–0.37; exact sign-flip permutation
  across collector runs is the primary small-cluster inference; `ax1d*`/
  `ax1r*` demoted to buffer resamples; mode strings `ax1w<collector>s<side>`
  reserved).

## Code notes and registrations (non-deviations)

- **2026-07-24 — RELABEL AMENDMENT 2: config schema-drift fix after
  Amendment-1 smoke failure.** The d1pilot smoke crashed at agent
  construction (`AttributeError: model_obs`): the 18-Jul pilot configs
  predate the `agent.model_obs` / `env.synth.*` / `orthreward.task`
  schema additions. NOT a labeler defect; no label was produced (the
  ordering guard held). FIX: `d0/sweep.py load_config` backfills
  absent keys from current `configs.yaml` defaults (saved values win;
  every backfilled key printed to the pass log). Validated on the
  retained true 18-Jul pilot config: exactly 8 keys backfill, all
  inert for dmc_proprio labeling; `model_obs='.*'` reproduces the
  pre-key include-everything behavior; strict no-op on
  schema-current configs. Registered:
  `prereg/PREREG_d1_relabel_amend2_20260724.md`; smoke re-runs from
  the top before labels.

- **2026-07-24 (post-audit session) — RELABEL AMENDMENT 1 + THREE NEW
  REGISTRATIONS/BUILDS (all pre-outcome).** (1) **D1 relabel
  Amendment 1** (`prereg/PREREG_d1_relabel_amend1_20260724.md`): the
  owner confirmed the six Gate-D1 cluster pilots
  (`d1pilot_{cup,finger}_e1_seed{1,2,3}`) are alive ⇒ +18 corrected
  passes registered (same 1e5 maturity as the shift pilots — adds
  CLUSTERS, not a maturity axis). `analysis/d1_shift_read.py` extended
  BEFORE any corrected label is read: cohort-aware loader
  (all-or-nothing d1pilot cohort), frozen d1s primary byte-for-byte in
  semantics (still solely decides `fired_arms`), new
  replication/pooled-12 outputs; extended selfcheck PASS (cohort-split
  grid separates primary from replication; missing cells trip in both
  cohorts). New companion driver `scripts/d1pilot_relabel_local.sh`
  (pull/smoke/labels). (2) **Comp×capacity theory predictions
  registered** (`prereg/PREREG_compcapacity_theory_20260724.md`,
  derivations `research_notes/Theory_CompCapacity_Addendum_20260724.tex`):
  P-E1 ordered un-nulling (forbidden pattern: apt off floor while
  task-lo floored), P-E2 task-lo-first, P-E3 membership-vs-budget
  split, P-E4 diversity-as-λ; frozen with zero 12m outcomes known;
  MUST be committed before the Scaling B read. (3) **Spectral
  domain-contrast pass built + registered**
  (`probing/spectral_measure.py`, selfcheck PASS;
  `prereg/PREREG_spectral_domains_20260724.md`): P-SM1 = cup
  reward-direction variance rank < finger on every matched side;
  design note — per-dim standardization was rejected mid-build in
  favor of symlog (the encoder's own transform) because standardizing
  erases the variance-salience quantity λ measures. (4) **Paper-4
  collusion harness built** (`collusion/env.py` + `collusion/pilot.py`,
  selfcheck PASS incl. planted punish-then-forgive fingerprint;
  computed p^N=1.4729/p^M=1.9250 match the published Calvano pair;
  ~20 s per 2M-iter session) + pilot design memo
  (`research_notes/Design_Collusion_Pilot_20260724.md`) frozen before
  any production session.

- **2026-07-24 (late night) — UPSTREAM-SURFACES AUDIT (completes the
  audit series): clean; one arm-invariant nuance recorded.** Surfaces
  previously relied on without re-derivation, now read: (1)
  **offline_fit replay is read-only by construction** — the checkpoint
  attaches only step+agent (never the replay), `replay.load` ingests
  chunks in-memory, no env interaction ⇒ the shared frozen buffers
  cannot be mutated by fits (no cross-arm contamination path exists);
  (2) **replay_context=1 nuance (recorded)**: first-visit chunk
  prefixes initialize the RSSM carry from latent entries STORED IN THE
  BUFFER — i.e. the collector agent's latents — then `replay.update`
  overwrites them with the fit agent's own after each visit; a
  transient foreign-carry initialization, identical for every arm and
  side (same buffers, same mechanism) ⇒ not a contrast confound; the
  12m fits' context shapes were handled by the registered ctx12
  rebuild; (3) **return math is canonical upstream DreamerV3** —
  lambda_return recursion (reward offset rew[:,1:], term cuts
  discounting, last cuts λ-mixing), imag_loss (all targets sg'd,
  valnorm updates gated on training), repl_loss (weight=~last, boot =
  imagination λ-return per replay step, indexing aligned via
  imgloss_out['ret'][:,0]), head_value_loss never updates shared
  normalizers; (4) **AUC provenance**: episode/score = sum of
  wrapper-emitted rewards over ONLINE train-env episodes (embodied/
  run/train.py aggregator) ⇒ the registered adaptation metric, and the
  orthogonal wave's AUC therefore measures spin reward as intended;
  (5) **RSSM one-step imagine** (single=True, action tensor) applies
  the candidate at the current posterior and prior-samples s' ⇒
  d0-signal Q = r̂(s')+γ·ĉ·v̂(s') is exactly the intended estimand,
  consistent with the observe() prevact convention the labeler fix
  restored; (6) **Consec stream** chunking matches
  _apply_replay_context's consec==0 convention; replay sample/update
  standard (uniform sampler, KeyError-tolerant in-memory latent
  refresh); (7) **E4 err_diff literally out_minus_in** in
  stratified_error (matches every recorded panel convention); (8)
  **Stage-1A density_residualize is genuinely out-of-fold** (stream-
  cluster folds via seeded permutation, isotonic direction chosen by
  SSE on TRAIN folds only, degenerate-fold guard) — the surviving
  Paper-2 diagnostic is leakage-free as registered. With this, the
  full path physics → wrappers → replay → training → labels → reads
  has been read adversarially; open defects: none known.

- **2026-07-24 (night) — TRAINING-SIDE CODE AUDIT: clean; one recorded
  scoping caveat, no code changes.** Systematic audit of the
  training/loss/adapt stack (companion to the same-day labeling-stack
  audit). VERIFIED CLEAN: (1) arm gradient wiring exact — rew loss
  input `sg(..., skip=reward_grad)`, repval `sg(..., skip=
  repval_grad)`, axis1.sbatch arm→flag mapping matches registered
  semantics (sgb both False / rgo repval False / vgo reward False),
  `ac_grads: False` so actor-critic gradients never reach the trunk;
  (2) 'reward' hard-excluded from enc/dec spaces independent of
  model_obs — reward-free arms structurally reward-blind; (3)
  measurement-only guarantees hold — valens imag + replay-grounding
  losses on sg'd inputs with `update=False` on the shared valnorm,
  Disag.loss internally sg's feat/action/target (passive-probe claim
  correct); (4) intrinsic rewards (apt kNN, p2e disag) are sg'd; (5)
  labeler Q denormalization (`pred*vscale+voffset`) matches the
  imag_loss target convention exactly; (6) adapt stage passes NO
  replay pointer — readout heads train on freshly collected online
  episodes only (REPLAY feeds only the offline fit), so the
  orthogonal-objective wave has no stored-reward contamination; (7)
  frozen/unfrozen readout configs + `load(regex)` partial-checkpoint
  semantics correct (heads left at fresh init); (8) runtime invariants
  (losses↔scales key-set assert, (B,T) shape assert) catch arm
  misconfiguration at launch. RECORDED CAVEAT (immaterial for DMC, no
  change): the continuation-head loss input is NOT sg'd, so con
  gradients reach the trunk in every non-random arm — in DMC episodes
  never terminate early, the con target is constant, and the path is
  task-information-free and arm-invariant; if this pipeline is ever
  ported to terminating envs (e.g. Atari), con becomes a
  task-information gradient path into the trunk in ALL arms including
  sgb, and the stopped-gradient-baseline claim would need the sg
  extended (do NOT change mid-study — it would alter every existing
  arm's semantics). Upstream-standard surfaces (replay sampling,
  Consec streams, return computation, slow-value updates) relied on
  upstream correctness + cross-wave empirical consistency, not
  re-derived.

- **2026-07-24 — E4 sgb COVERAGE GAP CLOSED: fresh-batch separation
  replicates at n=16 both arms.** Repeat pass
  (`local_results/e4_sgbq1_20260724_101758/`, 32 sgb fits sides×seeds
  1–16): fresh sgb (9–16) rew_nll_in h0 = s1 2.03±0.10 / s0
  3.78±0.21 vs rgo fresh 1.07/1.92 — the rgo/sgb level separation
  (≈1.0 nat hi side) that anchors the E4 mechanism now stands at
  n=16 per arm per side on the Amendment-1 seeds. Common-mode batch
  drift noted (both arms slightly lower in fresh batch; separation
  batch-stable). err_diff arm- and batch-invariant at n=16. Integrity:
  64 overlapping seed-1–8 rows match the 23-Jul csv EXACTLY.
  Descriptive panels only. Record:
  `artifacts/e4_amend1_optionc_20260723/RESULTS.md` §A′ + full csv
  copied there.

- **2026-07-24 — D1 SHIFT READ EXECUTED: NEITHER ARM FIRES ⇒
  consequence leg CLOSED; Paper-2 experimental program COMPLETE.**
  Frozen read on the 18 label passes
  (`local_results/d1_shift_20260724_091746/`): xpol D −0.006
  [−0.853, +0.772], phys D −0.055 [−0.483, +0.325]; per-domain signs
  opposite within each arm (cup/finger mirror). Secondary C1
  calibration fails in all three arms (F1 ρ ≤ 0.06) — state-
  illegibility extends to shifted support/dynamics. Registered
  consequence applied: Route B rests on mechanism + boundary; no
  further shift arms without new registration; 24+24 frozen, Route A
  closed, imag retired — all unchanged. Integrity: 18/18 cells n=200,
  meta conforms (rotation, mass_scale, dials), results-dir code
  snapshot diffs clean vs repo, freeze ordering verified
  (registration d68277f1 → amend-1 fix commits → labels 24 Jul →
  read). Power caveat recorded: xpol CI wide (excludes only > +0.77);
  phys bounds effects > +0.33. Descriptive: finger 22/23 base cells
  at exact 0.000 (reward floor at 1e5 pilots) depress finger paired
  sensitivity. Record: `artifacts/d1_shift_20260724/`.

- **2026-07-23 (night) — D1 shift smoke fix, ON-TIME dated amendment.**
  xpol smoke crashed on the transfer-guard gotcha in its two-agent
  form: the behavior agent's `make_agent` re-arms the guard
  (embodied/jax/internal.py 'disallow') AFTER the 'allow' line the R2
  run placed post-first-load. Fix = move the allow to after the LAST
  agent construction (`d0/oracle_labels.py` main_real; plumbing only,
  labeler_version unchanged, selfcheck PASS).
  `prereg/PREREG_d1_shift_amend1_20260723.md` written BEFORE any shift
  label exists — the registered smoke→fix→amendment→labels flow,
  followed on time this round. Resume at the smoke stage.

- **2026-07-23 (night) — Paper-1 band-ledger BUILD WAVE: three
  registration packages frozen-ready pre-outcome.** (1) **Unfrozen
  calibration** (PREREG_unfrozen_calib_20260723.md +
  analysis/unfrozen_calib_read.py selfcheck PASS): 32 adapt-only jobs,
  {rgo,sgb}×{s0,s1}×seeds 1–8 from the existing 500K fits under NEW
  config `unfrozen_readout` (same enc/dyn/dec init, nothing frozen);
  PRIMARY = seed-level pooled rgo−sgb, cluster CI>0; frozen baselines
  = committed auc_pooled_1_16.csv (disclosed). (2) **Orthogonal
  objective** (PREREG_orthogonal_obj_20260723.md +
  analysis/orthogonal_obj_read.py selfcheck PASS + NEW
  embodied/envs/orthreward.py): finger:spin sparse reward evaluated ON
  the turn_hard env (same obs space ⇒ ckpts load exactly; wrapper ==
  dm_control Spin.get_reward on shared physics, validated locally
  pre-freeze incl. injected-velocity readout + make_env config path);
  frozen-readout adapt, grid seeds 1–16 conditional on sgb-9–16 fit
  existence (fallback 1–8, existence ≠ outcome); registered FLOOR
  gate (pooled AUC < 20 ⇒ uninformative); PRIMARY = pooled rgo−sgb on
  spin, CI>0; both outcomes land the band leg (interpretation map
  frozen). (3) **vgo extended-fit discriminator**
  (PREREG_vgo_extended_20260723.md + analysis/vgo_extended_read.py
  selfcheck PASS): FRESH 1.5M-update fits {vgo,sgb}×seeds 1–8×s1 (new
  WM dirs, frozen 500K fits untouched) + standard adapt; PRIMARY =
  diff-in-diff vs the committed 500K baselines (sgb = generic-
  extension control), CI>0 ⇒ P-B2 attenuation form. Plumbing (all
  smoked): configs `unfrozen_readout`/`orth_spin`/`orth_spin_frozen`,
  orthreward wiring in make_env, `AXIS1_ADAPT_CONFIG` knob in
  axis1.sbatch (default frozen_readout — already-queued jobs read
  their submission-time script copy; behavior unchanged when unset).
  Ordering: freeze-commit everything BEFORE any submission.

- **2026-07-23 (evening) — SYNTH B″ READ EXECUTED: NULL ⇒ SYNTH
  INTERACTION LEG CLOSED; + E4 panels for Amendment-1/Option-C.**
  (1) Frozen `synth_phasebpp_read` on the 96-job grid: I = +30.5
  [−18.7, +100.2], perm p .53 ⇒ does not fire; registered consequence
  = no Phase C, synth stays shuffle-collapse-only; power honest
  (realized CI ±60 vs registered ±87 prediction); local read ==
  cluster json exactly (artifacts/synth_phasebpp_20260723/). Apt
  simple nominally negative (−24.5 [−48.6, −0.8], descriptive).
  (2) E4 descriptive panels (artifacts/e4_amend1_optionc_20260723/):
  Amendment-1 rgo fresh batch REPLICATES the mechanism level (hi-side
  rew_nll_in h0 = 1.07 ± 0.10 vs original 1.21 ± 0.20; err_diff
  batch-invariant) — mechanism + behavior now replicate on the same
  fresh seeds; **COVERAGE GAP: sgb csv contains seeds 1–8 only, the
  Amendment-1 sgb fits (9–16) were not measured** (repeat pass needed
  if fits persist); Option-C volume: occupancy-matched 2× volume
  0.83→0.73 with sd collapse 0.16→0.02 (stability > level; P-B4
  note), and the 18-Jul behavioral anomaly has NO rep-level
  counterpart (anomalous cell worst+noisiest, 0.94 ± 0.29).

- **2026-07-23 (evening) — LADDER READ EXECUTED: NO COMPRESSION
  SIGNATURE; ensemble arm stays gated.** Owner ran the frozen
  `d1_ladder_read` after the freeze commit (`d68277f1` 13:51 → read
  15:01; ordering verified). L3 belief-state ridge |ρ| ≤ 0.042 in all
  4 cells (two CIs exclude 0 at trivial magnitude, one negative) ⇒
  the D1/R2 null is no-exploitable-heterogeneity up to the agent's
  full belief state, NOT a scalar-compression artifact; level-5
  (privileged state) not stored = recorded gap. New with-CI
  descriptives: always-buy Δ_real early×e1 −0.208 [−0.472, −0.003]
  (harmful), late×e4 +0.313 [+0.051, +0.627] (valuable) — value is
  context-legible, state-illegible. Registered resource consequence
  executed: support-diverse-ensemble arm NOT authorized. Artifact:
  `artifacts/d1_ladder_20260723/`.

- **2026-07-23 (later) — Paper-2 post-closure build wave: ladder
  addendum + shift-consequence probe, both frozen-ready pre-outcome.**
  (1) Actionability ladder (`prereg/PREREG_d1_ladder_20260723.md` +
  `analysis/d1_ladder_read.py`, selfcheck PASS): registered-descriptive
  L0–L3 ladder on the EXISTING R2 labels, real op only; new quantity =
  L3 nested-LORO ridge on [deter 512 + F1] (λ ∈ 1e1..1e4 inner-CV);
  known R2 scalar outcomes disclosed at freeze; non-decisional
  "compression signature" heuristic (L3 ρ ≥ .2 CI>0 & L2 < .1) whose
  only consequence is resource-side (ensemble-arm design). (2) Shift
  probe (`prereg/PREREG_d1_shift_20260723.md` +
  `analysis/d1_shift_read.py` selfcheck PASS +
  `scripts/d1shift_local.sh` + labeler shift extension
  `shift_ext_20260723`: `--behavior_checkpoint` cross-policy driver +
  `--mass_scale` dm_control body-mass scaling, behavior=None path
  bit-identical to r2_ext; labeler selfcheck PASS): 6 fresh e1 pilots
  seeds 21–23 × {base, xpol sibling-rotation, phys ×1.3} = 18 label
  passes, final ckpts only; PRIMARY per arm = paired pooled
  mean-Δ_real difference cluster CI > 0 (2 looks); calibration
  descriptors never decisional; explicitly NOT a Route-A revival.
  Ordering: freeze-commit → ladder read (labels exist) and pilots →
  smoke → labels → shift read.

- **2026-07-23 — R2 probe READ EXECUTED: NULL ⇒ ROUTE A CLOSED.**
  Frozen `analysis/gate_d1_r2_read.py` run unchanged on the 24 rerun
  (save60) labels; 16/16 cell×op×fset fail C1 (best ρ +0.085,
  early×e1 real F1; threshold 0.3); rule-robust under the original
  12-look and Amendment-1 6-look rules; boundary cell late×e1
  replicates Gate-D1 near-zero (local port validated, no anomaly).
  R1 adjudicated: udyn_resid |ρ| ≤ .029 as ÊVSI feature. Registered
  consequence executed: Route A CLOSED, Route B = Paper 2, 24+24
  frozen permanently. Descriptives: pooled Δ_real early NEGATIVE
  (−0.21 e1 / −0.15 e4) vs late positive (+0.07 e1 / **+0.31 e4**),
  change_rate ≈ .85 everywhere. Artifact:
  `artifacts/gate_d1_r2_20260723/`. Plan v3 (gitignored):
  `research_notes/other research/EVPI_Plan_Revision_20260723.md`.

- **2026-07-22 — Route-A R2 decision probe BUILT + frozen-ready
  (pre-outcome; Paper-2 track, local-5090 execution).**
  `prereg/PREREG_gate_d1_r2probe_20260722.md` +
  `analysis/gate_d1_r2_read.py` (selfcheck PASS ×6; C1/C2/C3 imported
  from the frozen 14-Jul gate_d1_read) + `scripts/d1r2_local.sh` +
  `d0/oracle_labels.py` R2 extension (labeler_version r2_ext_20260722:
  udyn + belief deter + strided latent reference + dose meta +
  bootstrapped-G columns; selfcheck PASS; Gate-D1-era column semantics
  unchanged). Design: 12 fresh local pilots {e1, e4=d0_dose3} ×
  {cup, finger} × seeds 11–13 (platform-uniform, all-prospective) ×
  {early≈25K, late=100K} ckpts ⇒ 24 label passes; dials PINNED this
  time (200/100/25/8/16/seed0/ref_stride5; kNN k=10 excl ±10).
  Registered rule: R2 fires iff C1 (ρ≥0.3, cluster CI>0) in ≥1
  predicted cell {early×e1, early×e4, late×e4}, either op, either
  feature set (F0 = five signals; F1 = +udyn/logdens/udyn_resid =
  the Stage-1A asset, R1 riding along); late×e1 = registered
  boundary-replication check (predicted FAIL). Consequence: fires ⇒
  Route A revived as boundary-law paper (own wave + prereg required;
  24+24 stay frozen either way); null ⇒ Route A CLOSED, Route B
  absorbs. Ordering: freeze-commit BEFORE pilots; smoke before labels.

- **2026-07-22 — TD-MPC2 F2 read executed: P-F2a FIRES +1.717
  [+1.650, +1.775], lottery_fraction 0.000 ⇒ NEVER-INCLUDED-LIKE**
  (`artifacts/tdmpc2_f2_20260722/`). Cluster code snapshot
  bit-identical to the repo frozen instrument/read/prereg; 64/64 grid,
  single probeset sha, smoke=0 throughout; the registered no-outcome
  smoke ran 4× (all random-init, nothing revealed); no instrument
  amendment was needed. Free-latent probe R² NEGATIVE (−0.24/−0.18 vs
  aware +0.63/+0.76); ln-distributions fully disjoint (gap +0.47;
  geometric ratio 5.6×) ⇒ consistency-only training deterministically
  EXCLUDES the reward direction — **Prop A2's α→0 lottery account
  WEAKENED per the registered map; the family boundary is
  representational**; F1's free +12.5 ns is not
  reward-representational. P-F2b NOT invariant-like (−0.61: free ~1.8×
  better at consistency — registered qualitative difference from
  Dreamer's arm-invariant d_errin). G-F3 stays NO-GO (dissociation
  present, no paper-blocking ambiguity). Bit-check note: local vs
  cluster read json differ in 4 float fields at the last 1–2 ulp
  (platform BLAS); every decision field and reported digit identical —
  local json is canonical.

- **2026-07-20 — two follow-on registrations BUILT + frozen-ready
  (pre-outcome): synth Phase-B″ and TD-MPC2 F2.** (1) **Phase-B″**
  (authorized by the re-diagnosis gate, share .865):
  `prereg/PREREG_synth_phasebpp_20260720.md` + frozen
  `analysis/synth_phasebpp_read.py` (selfcheck PASS: fires /
  lottery-null / missing-cell / rediag-mode-exclusion). 96 adapt-only
  jobs, {task, apt} × sides × fit seeds 1–8 × FRESH adapt seeds
  104–106 (101–103 were revealed in the re-diagnosis; no revealed
  outcome enters B″); modes `ax1rb(f?)<fit>q1s<side>`; fit-level
  AUC100k = 3-seed mean; primary = interaction CI>0; registered power
  statement (decides ≳ ~90; between-fit floor ≈±49) and consequence
  map (null ⇒ synth interaction leg CLOSED, no Phase-C). (2) **F2**
  (G-F2 default GO on the coherent F1 read):
  `probing/tdmpc2_stratified_error.py` (NEW instrument: closed-form
  ridge reward probe on frozen latents at h=0 — required because the
  free arm's native reward head is untrained (reward_coef=0) and
  cannot report trunk content; one-step consistency d_errin analog;
  descriptive native-head MSE; frozen finger_v1 probe set sha-pinned;
  even/odd episode split, λ=1e-3·n_train; numpy-core selfcheck PASS;
  `--smoke_random_init` = registered no-outcome API smoke) +
  `analysis/tdmpc2_f2_read.py` (selfcheck PASS: lottery /
  never-included / null / missing-cell) +
  `prereg/PREREG_tdmpc2_f2_20260720.md` (P-F2a = mean ln-ratio
  free/aware in-regime probe-MSE over 32 pairs, seed-clustered
  bootstrap, fires iff CI>0; P-F2b consistency invariance band 0.5 +
  lottery_fraction ≥ 0.15 map, all descriptors non-decisional).
  Ordering: both preregs + scripts must be committed BEFORE the 96
  B″ jobs / the F2 smoke+measures run.

- **2026-07-20 — four reads executed (Option C corrective / synth
  re-diagnosis / TD-MPC2 F1 / stamping true-E4 panel).** All three
  frozen reads ran on cluster csvs with local canonical AUC recompute
  bit-identical (36/36 + 64/64 rows) and local read jsons bit-identical
  to the cluster-side jsons. (1) **Option C corrective**
  (`artifacts/optc_corrective_20260719/`): C1 occupancy contrast at
  fixed 400-ep volume +23.3 [−248, +302] spans 0 (registered:
  occupancy saturates below .094 at this volume); **C2 volume contrast
  at ≈fixed fraction +251.9 [+120, +396], 6/6 ⇒ P-B4 fraction-not-count
  strong form REFUTED** (first refuted adjudicated prediction; theory
  needs a volume/estimation-floor term; composition caveat as
  registered). (2) **Synth re-diagnosis**
  (`artifacts/synth_rediag_20260719/`): pooled within-share **0.865 ≥
  0.5 ⇒ adapt-lottery CONFIRMED ⇒ Phase-B″ (adapt-averaged re-test)
  AUTHORIZED**; side0 share 1.000 (σ²_b clamps to 0), adapt-seed swings
  up to ~30× on a fixed fit. (3) **TD-MPC2 F1**
  (`artifacts/tdmpc2_f1_20260719/`): PRIMARY pooled 1–16 +22.9 [−19.9,
  +68.9] does NOT fire ⇒ family-scope limitation stands at doubled
  power; aware simple +35.5 CI>0 confirmed; **P-C1 PARTIAL** (C1a share
  .646 fails <0.6; C1b unresolved, free +12.5 point ≥ +10 spans 0; C1c
  pass); audit 32/32; **G-F2 not incoherent ⇒ F2 (E4 mechanism port)
  default GO**, reframed as lottery-vs-never-included discriminator.
  (4) **Stamping true-label E4 panel (b)** appended to
  `artifacts/stamping_20260718/RESULTS.md` (descriptive): srd heads'
  true-reward in-regime NLL 3.9–15.1 ≫ sgb 2.30 while stamp-NLL was
  low ⇒ included-but-useless signature complete from both directions;
  no alignment leakage through the dyn/* deviation; err_diff stays in
  the arm-invariant band.

- **2026-07-19 — pixel X1 registration package BUILT + frozen-ready
  (pre-outcome); G-X1 CLOSED on the X0 closeout.** X0 smoke
  (`local_results/pixel_x0_closeout_20260719_131300/`): seed-99 fit+adapt
  green on both sides of the image-only pair `pxq1_img` (20K updates in
  16–40 min ⇒ 500K ≈ 7–17 h ≪ 36 h; pair storage 31 MB; adapt reaches
  episode score ≈966/974 from pixels — learnability risk dead; loss set
  {con,dyn,image,rep,rew}, `model_obs: image` in saved configs; 686,070
  params). First attempt failed with `RuntimeError: 'image'` — the
  original `pxq1` pair drew from image-LESS episodes (only 6 of 16
  sources rendered); the user rebuilt on an image-only index. That
  gotcha + the smoke pair's collector⊗occupancy confound (side0 = one
  pure collector vs side1 = five others) drive the X1 design:
  NEW `build_controlled_replay search-matched` (pooled high side, low
  side repeats the high side's per-collector episode counts ⇒
  collector_l1 = 0 by construction, side0 = low by construction;
  `selfcheck-matched` PASS; `build` passes `collector_l1` into the
  manifest confound deltas; note the binned `source_l1` metric
  mechanically saturates for any occupancy pair, so `collector_l1` is
  the controlled quantity). Registered: `prereg/PREREG_pixel_repl_20260719.md`
  (K ladder 20/16/12 + buffer quality gate occ ≥.30/≤.05, fill record
  before first submission, 32-job X2 with modes `ax1(f)pxpxq1ms*`,
  P-C2 band [+84.0, +98.7] directional) + frozen
  `analysis/pixel_repl_read.py` (selfcheck PASS: fires/null/
  missing-cell/smoke-mode-exclusion). Smoke score sightings disclosed
  in the prereg's ordering statement; no pxq1m pair, no X1/X2 outcome
  exists. The 18-Jul plan's "50–100 eps/side" aspiration is superseded:
  the image pool holds only ~30 high-occ episodes, so K=20 is the
  registered ceiling.

- **2026-07-18 night — decisions recorded + two correctives frozen
  (pre-outcome).** (1) **Goodhart DEMOTED** (owner decision, 18 Jul):
  the track is reported as a negative/boundary result with the 3-seed
  evaluator ensemble as its closing control (spearman .136→.186, top-1
  regret unchanged 571.9); no further Goodhart runs. (2)
  `prereg/PREREG_scaling_optc_amend1_20260718.md` + frozen
  `analysis/optc_amend_read.py` (selfcheck PASS ×3): 12 corrective
  jobs on the never-run `q1v400/side0`; C1 fixed-volume occupancy
  contrast + C2 ≈fraction-matched volume contrast (the intended P-B4
  test); v200s1/v400s1 from the archived artifact csv, bit-checked.
  (3) `prereg/PREREG_synth_rediag_20260718.md` + frozen
  `analysis/synth_rediag_read.py` (selfcheck PASS ×2): 24 adapt-only
  jobs (task arm, 4 fits × 3 fresh adapt seeds × 2 sides, reusing the
  existing 500K ckpts), one-way variance decomposition, adapt-lottery
  prediction within-share ≥ 0.5, gate → Phase-B″. (4) Route-A revision
  analysis at `research_notes/D1_RouteA_Revision_Options_20260718.md`
  (assessment only; recommends one cheap R2 decision probe before the
  fork closes).

- **2026-07-18 pm — four reads executed (Option C / stamping-E4-override
  / synth diagnosis / pixel X0 status).** (1) Option C: see the
  deviation entry above. (2) Stamping override panel
  (descriptive; appended to `artifacts/stamping_20260718/RESULTS.md`):
  32/32 override measures, stepid-join latent recovery recorded in
  every meta, probeset sha matches; stamp-NLL 0.24–1.49 (all-frames,
  3/4 cells at-or-below the included band full 1.02/rgo 1.21; sgb
  reference 2.30) ⇒ stamps substantially LEARNED on held-out probes +
  null transfer = **inclusion-without-transfer signature; P-B1 keeps
  its full form**, function-draw heterogeneity noted; true-label pass
  on srd fits not yet run. (3) Synth diagnosis
  (`artifacts/synth_diagnosis_20260718/`): P-D1 PASS (ranks synth
  [3,2] vs finger [6,5]) but P-D2 FAIL (rew-NLL separation 3.93 ≈ 77×
  finger's) and P-D3 FAIL (to_target ratio 3.74/0.63) ⇒ **gate
  CLOSED, no Phase-B′**; in-subspace refuted at model level; post-hoc
  leading hypothesis = adapt-stage variance bottleneck. (4) Pixel X0
  partial: source run done (model_obs: image in saved config; 100K
  steps; fps/train ≈ 10.9K frames/s ⇒ ≈13 h per 500K-update fit ≪ 36 h
  envelope), 80-ep index occ 0–0.79, but single-source q1 search
  returns None (expected degeneracy) — fit/adapt smoke blocked on the
  8 additional source runs now in flight; G-X1 provisionally green on
  throughput/storage, final after the multi-source pair + smoke.

- **2026-07-18 — synth diagnosis package BUILT + registered
  (`prereg/PREREG_synth_diagnosis_20260718.md`, descriptive-only,
  gates Phase-B′ scoping).** New `analysis/synth_diagnosis_read.py`
  (selfcheck PASS both branches) summarizes P-D1/P-D2/P-D3.
  **Instrument revision (validation-discovered, pre-cluster-outcome):
  `reward_direction_rank` now ranks the cross-correlation direction
  ν = E[r·x]** — the min-norm regression direction of the 17-Jul draft
  falls into the correlation null space under exact obs collinearity
  (synth `to_target = GOAL − position`: lstsq keeps near-zero singular
  values; observed var≈0/rank 13-of-14 in validation); regression
  direction kept as labeled secondary; selfcheck gained an
  exact-collinearity case. The tool had never touched real cluster
  data (both prereg slots that cite it — scaling prior, this
  diagnosis — remain unfilled), so no reproduction concern. Local e2e
  validated the whole chain (tiny generator pair → 30-update task/apt
  fits → build-probeset (FROZEN) → measure → summarizer); the
  validation previewed the synth half of P-D1 at debug scale
  (rank 2/14) — disclosed in the prereg.

- **2026-07-18 — synth Phase B read executed as registered
  (`artifacts/synth_phaseb_20260718/`).** Pre-frozen
  `analysis/synth_phaseb_read.py`, unmodified; audit 48/48; local
  canonical AUC (48/48 QC). **PRIMARY-1 interaction +44.2
  [−88.0, +178.4] does NOT fire; PRIMARY-2 shuffle collapse −226.6
  [−341.5, −96.6] FIRES ⇒ registered collapse-only branch: Phase C NOT
  authorized pending diagnosis.** S1 apt +12.6 ns (4th reward-free
  null). Notable descriptives: shuffle INVERTS the occupancy benefit
  (sh-lo 216.8 = highest cell; different signature from finger's
  attenuation); task-arm per-seed deltas span −256..+375. Leading
  post-hoc diagnosis (labeled post-hoc): `to_target` in the synth obs
  puts the reward direction in-subspace (theory's cup/E5 cell) ⇒ no
  interaction expected by the theory itself; checkable via an E4-style
  reward-NLL pass over the existing 48 synth ckpts, fixable via
  `agent.model_obs` (Phase-B′ would need its own registration).

- **2026-07-18 — pixel X0 pre-work BUILT (Plan_PixelReplication;
  pre-outcome: no pixel study run exists).** (1) `agent.model_obs`
  config regex (default `'.*'` = behavior-preserving) filters which obs
  keys enter enc/dec; new `pixel_wm` named config (size1m-anchored,
  `agent.model_obs: image`, train_ratio 256) trains the WM on pixels
  ONLY while proprio keys stay in obs/replay for regime labeling and
  pair search. (2) Plumbing: `AXIS1_BASE_CONFIG` (dmc_proprio|pixel_wm,
  whitelist-guarded, auto-forces RENDER=True for pixel) threaded through
  `axis1.sbatch` both stages + `submit_all.sh` bundle exports and
  non-bundled case; `BASE_CONFIG` likewise through `pretrain.sbatch` +
  the pretrain submit case. (3) Local rehearsal (WSL2): DMC emits
  image+proprio simultaneously; `build_controlled_replay index`/`build`
  handle pixel chunks (image byte-preserved, regime from
  dist_to_target); pixel_wm offline fit trains with loss set
  {con, dyn, image, rep, rew} and NO proprio heads (saved config
  records model_obs); frozen-readout load surface verified (58
  enc/dyn/dec tensors shape-match a fresh pixel_wm agent). Local-only
  limitation (cluster unaffected): main.py's worker-thread env stepping
  crashes EGL under WSL2, so online stages (source runs, adapt) must
  smoke on the cluster — coexistence in the main thread verified.

- **2026-07-18 — TD-MPC2 Amendment 1 frozen (pre-outcome).**
  `prereg/PREREG_tdmpc2_amend1_20260718.md` + frozen read
  `analysis/tdmpc2_amend_read.py` (selfcheck PASS: fires+LANDS branch,
  REFUTED branch, bit-check trip, missing-cell trip; seeds-1–8
  machinery reproduces the 17-Jul read bit-for-bit: +9.5 [−43.2,+51.4]
  / +35.3 / +25.8) + theory derivation
  `research_notes/Theory_SpectralTransfer_Addendum_DecoderFree_20260718.tex`.
  Registers the seeds-9–16 extension (32 jobs, design unchanged),
  pooled n=16 primary re-test on the P3-Amendment-1 pattern (seeds 1–8
  authoritative from the frozen artifact csv, overlap bit-checked), and
  **P-C1** (decoder-free limit: C1a share < 0.6 directional; C1b
  free-arm positivity CI-bearing with lands/unresolved/fails
  trichotomy; C1c aware positivity). The execution plan's provisional
  absolute-scale anchors were REJECTED at freeze as violating the base
  registration's never-compare-numerically rule; registered forms are
  within-family/dimensionless. Gate G-F2 (E4-port resource gate,
  non-inferential) recorded in the prereg. No seed-9–16 TD-MPC2 job
  exists at freeze.

- **2026-07-18 — stamping + Gate-D1 reads executed
  (`artifacts/stamping_20260718/`, `artifacts/gate_d1_20260718/`).**
  (1) Stamping read per `PREREG_stamping_20260717.md` with the
  pre-frozen `analysis/stamping_read.py`, unmodified: local canonical
  AUC recompute bit-identical to cluster csv 48/48; audit 48/48;
  **PRIMARY [B_srd−B_sid] +8.90 [−11.49,+31.85] CI includes 0 ⇒ P-B1
  strong form STANDS** (srd pooled benefit +16.5 also inside the
  registered band = the P0 apt null CI); S3 function-draw ≈0. The E4
  stamp-NLL mechanism panel was still running at read time; it is
  descriptive-only and will be appended when it lands. (2) Gate-D1
  read: NEW `analysis/gate_d1_read.py` (selfcheck PASS) implements the
  three criteria frozen in `PREREG_gate_d1_stage1a_20260714.md` §B on
  the Stage-1B label npzs. Mechanical choices post-dating label
  existence, disclosed in the script header: OLS-on-5-signals ÊVSI with
  leave-one-run-out-within-domain cross-fitting; criterion-3 cost price
  unpinned by the prereg ⇒ CI statement at price 0 + price sweep
  sensitivity. **All 4 domain×op cells FAIL all three criteria** (best
  raw feature +0.085 ≪ 0.3 bar ⇒ verdict robust to the mechanical
  choices) ⇒ registered consequence: **24+24 stays frozen**. Also
  disclosed: the labeling sweep ran at the tool's [prov.] dials
  (actions 8 / rollouts 16) without the planned dial-pinning amendment;
  dials are recorded in each npz meta and do not affect the verdict.
  The Stage-1A density-residual instrument was NOT testable (labeler
  emits no udyn/density proxy) — the one registered Route-A move left.

- **2026-07-17 — P3 + mechanism-wave reads executed as registered
  (`artifacts/p3_wave_20260717/RESULTS.md`); P3 Amendment 1 frozen.**
  P3 (pre-frozen script, unmodified; audit 80/80 on the registered
  config-flag rule; transform replay paths verified 32/32): **PRIMARY
  B_rgo−B_sgb +50.2 [−12.8,+111.2] NOT confirmed; S2 shuffle collapse
  FIRES −127.9 [−201.0,−51.5]; S3 relocate fires; S4 placebo ≈0;
  registered branch = interaction needs both gradient paths/interplay,
  with frame-level label binding necessary.** Descriptive: rgo simple
  +65.0 [+4.2,+120.9] (own CI>0), vgo −1.0 dead; ordering
  full>rgo>rl>sh>sgb>apt monotone in supervision quality. E4 factorial
  pass: reward-NLL LEVEL separates gradient arms (full 1.02/rgo 1.21 vs
  sgb 2.30/vgo 2.31; rl 3.43 worst-vs-truth yet keeps +52.8 behavioral
  ⇒ true-label predictability not the whole carrier); d_errin
  arm-invariant across all 5 arms. Cup-apt E4 (32/48 targets;
  fq2s1_seed{7,8} no ckpt): arm-invariance holds in cup (12/12) — all
  four domain×arm cells now. Lo-side battery: all lo sides mono-source;
  finger occ↔rew +1.0 both sides; cup q1 lo reward-rich (0.222 at occ
  0.013). Goodhart p2elong (registered rule): Spearman 0.209, top-1
  regret 78.4% ≈ random-pick 81.3%, top-quintile inversions 1.54×<2×
  ⇒ neither trigger fires; selection≈lottery at r≈0.2; path = one
  ensemble attempt or demote. **Amendment 1 frozen pre-outcome
  (PREREG_p3_amendment1_20260717.md): rgo+sgb seeds 9–16 (32 jobs),
  pooled n=16 primary re-test, W0 playbook.**

- **2026-07-16 — build wave (all pre-outcome).** (1) **P3 read script
  frozen before any factorial outcome exists**
  (`analysis/p3_factorial_read.py`, selfcheck PASS: recovers planted
  effects and the registered branch map; loads the disclosed B_full/
  B_apt records from the frozen paired JSONs). Audit clarification
  encoded there, pre-outcome: `offline_fit.py` computes `repval` in
  every task-mode fit (agent.py:370) but its stdout filter
  (NON_WM_LOSS) never prints it, so the prereg's "loss lines contain
  rew and repval" sentence is verified as: config flags
  (reward_grad/repval_loss/repval_grad, recorded per run) are the
  authoritative repval audit; stdout verifies `rew` presence and
  AC-key absence only. (2) **TD-MPC2 cross-family pipeline built +
  registered** (`prereg/PREREG_tdmpc2_crossfamily_20260716.md`, frozen
  before any TD-MPC2 outcome exists): official repo pinned at
  e9f5932, `probing/tdmpc2_{compat,bridge,offline_fit,adapt}.py`,
  `scripts/tdmpc2.sbatch` + `tdmpc2_env_setup.sh` + `submit_all.sh
  tdmpc2-bundles` (DRYRUN-verified, 32 jobs); bridge selfcheck PASS
  (prev-action shift + contract violations raise); cluster smoke run
  required before the wave per the prereg ordering statement. (3)
  **Stage-1B oracle labeler built** (`d0/oracle_labels.py`, selfcheck
  PASS: synthetic env recovers exact hand-computed Δ_real=+0.8/
  Δ_imag=0, restore-determinism asserted, costs metered) per
  PREREG_gate_d1_stage1a §C; supporting agent change: `_d0_signals`
  now also emits `d0/cands` (the candidate action vectors its Q
  matrices refer to) — additive output, no consumer changed. (4)
  Scaling-pilot design brief drafted
  (`research_notes/Scaling_Pilot_Design_Brief_20260716.tex`,
  disk-only): options + recommendation; prereg deliberately NOT
  frozen until the scope decision.

- **2026-07-16 — W1/W2/W3 reads executed as registered (Amendment 2
  §B/§C/§D; `artifacts/w123_e3v2_20260716/RESULTS.md`).** Audit PASS
  248/248 (arm loss sets, saved configs, registered replay paths, QC;
  bit-consistency with the P0/10-Jul/aware-1–16 records on all
  overlapping rows). **W1 PRIMARY (finger E3v2 within-collector):
  interaction REPLICATES — Δ̂ mean +84.0, 14/14 collectors positive,
  exact sign-flip perm p = 0.00012 (two-sided floor at n=14); hier boot
  CI [+48.2, +124.0]** on buffers with source_l1 ≡ 0 (diversity confound
  eliminated by design). Apt simple effect −5.1 (perm p=0.57) = third
  independent reward-free null. Registered interpretation fires:
  buffer-level causal support for the objective–data-alignment headline;
  **scaling pilot authorized**. W1 SECONDARY (cup): does not replicate
  (−140, 3/9, p=0.31); labeled descriptive mirror pattern (apt simple
  +213, task null) consistent with the battery/E4 mechanism. W2 (cup-Q1
  apt pooled 1–16): +93.1 [−33.0, +217.1] ⇒ CI includes 0 ⇒
  directional-only, not claimed (fresh 9–16 +35, attenuation again). W3
  (finger-Q2 completion, descriptive): −5.4 [−41.9, +28.3] at n=16 —
  the Q2 null stands. Feasibility disclosures: finger p2e3 and cup apt2
  pairs SHORT at build scale (pre-outcome, frozen criteria) ⇒ 14 + 9
  collector units instead of 15 + 10.

- **2026-07-14 — E4 + Goodhart reads executed as registered
  (`artifacts/e4_goodhart_20260714/RESULTS.md`).** E4 (descriptive):
  relative in-regime advantage of hi-occ buffers holds 8/8 in both arms
  and both domains, but is **arm-invariant** (apt = task at modeling the
  regime better, yet apt transfers nothing) ⇒ regime-modeling accuracy
  is NOT the interaction's carrier; **reward-head NLL tracks the
  behavioral interaction across cells** (finger Q1 improves on hi, cup
  Q1 flat, cup Q2 follows reward density) ⇒ mechanism = reward
  predictability of transferred features; registered "higher err_out"
  leg fails (hi side better everywhere, differentially in-regime);
  obs-decoder version of signature 3 fails (task's obs advantage is on
  unrewarded frames). Cup-apt measure pass outstanding. Goodhart
  (registered rule): pool n=63, Spearman +0.15, top-1/3/5 regret = 0 —
  neither advance trigger fires ⇒ continue-but-weaker (stronger
  evaluator next); descriptive fragility: M's rank-2 policy has real
  return 0.0, LOO regret 100%.

- **2026-07-14 — Four registrations frozen + Stage-1A read PASSES.** (1)
  `prereg/PREREG_e4_stratified_error_20260714.md` — E4 estimator/probe
  sets/signatures (descriptive only; `probing/stratified_error.py`,
  selfcheck PASS, measure path e2e-validated on a local W0 checkpoint;
  `submit_all.sh e4-measure`). (2)
  `prereg/PREREG_p3_gradient_path_20260714.md` — gradient-path factorial
  (arms sgb/rgo/vgo via `AXIS1_ARM`, flags derived in axis1.sbatch,
  audit = saved config) + reward-label transforms sh/rl
  (`probing/relabel_replay.py`, selfcheck PASS); primary = B_rgo − B_sgb
  fully prospective; 80 jobs reserved. (3)
  `prereg/PREREG_gate_d1_stage1a_20260714.md` — quantified Gate-D1
  unfreeze criteria + Stage-1A decision rule. (4) **Stage-1A registered
  read executed same day
  (`artifacts/latent_uq_stage1a_20260714/RESULTS.md`): density-residual
  instrument (out-of-fold isotonic residual on the density proxy,
  `latent_uq_analysis --instrument density_residual`) PASSES — 10/10
  within sub-cells flip TRACKS_DENSITY→TRACKS_ERROR at 500K, all four
  cross cells TRACKS_ERROR (finger_random improves from AMBIGUOUS),
  early-checkpoint regime preserved.** Route B's calibration fix is now
  a registered, passing asset; Gate D1 (oracle-label criteria) still
  gates the 24+24 runs.

- **2026-07-13 — W0 read executed as registered (Amendment 2 §A); n=16
  interaction CONFIRMED.** Read of the 16 apt finger-Q1 seed-9–16 runs
  (`artifacts/w0_n16_interaction_20260713/RESULTS.md`). Audit PASS 16/16
  (saved `expl.mode: apt`; loss-line component set byte-identical to the
  audited P0 finger runs, no `rew`/`value` term); QC 16/16; seeds-1–8
  deltas bit-checked against the frozen paired JSONs. Primary (decision on
  CI alone): fully-paired n=16 interaction **+98.72 [+45.81, +158.83]
  excludes 0**, 13/16 seeds, exact 2¹⁶ permutation p=0.00272, d_z 0.83,
  LOO [+81.0, +110.4]. Registered subsidiary fresh-batch 9–16: +40.17
  [−12.77, +90.68] ns — attenuation entirely in the aware arm; honest
  effect size = pooled +98.7. Secondary: reward-free simple effect pooled
  1–16 **−3.71 [−16.97, +9.76]** (tight null; all apt cells on the ~80
  floor). Buffer battery (descriptive, hi sides only; lo re-run
  requested): finger occ↔reward episode-corr = **+1.00** (occupancy ≡
  reward density) vs cup **−0.51/−0.70** (sign flip explains domain
  heterogeneity); lo sides are mono-source vs multi-source hi ⇒ source
  diversity confounded with occupancy in the original Axis-1 buffers —
  E3v2 within-collector (Amendment 2 §B) now carries de-confounded
  identification, not just replication. Correction note added to the P0
  RESULTS.md audit prose (finger/cup loss-key lists were swapped;
  verification unaffected).

- **2026-07-13 — P0 corrective read executed as registered; decision-tree
  branch 2 fires.** Read of the 64 reward-free (`ax1f*`) runs per
  `prereg/PREREG_axis1_corrective_20260711.md` + Amendment 1
  (`artifacts/p0_axis1_corrective_20260713/RESULTS.md`). Per-run audit PASS
  64/64 (saved `expl.mode: apt`; loss lines contain no `rew` term); QC
  64/64. Headline confirmatory contrast (finger Q1, AUC₁₀₀ₖ, B=10K seed-0
  bootstrap CI): **+2.04 [−22.04, +24.16] = NULL** ⇒ the registered pivot
  branch fires (occupancy-rescue unsupported as reward-free; headline moves
  to observational reversal + occupancy × reward-supervision interaction;
  E3 as designed deprioritized). Registered 2×2 interaction (same seeds,
  same buffers): finger Q1 **+157.28 [+76.51, +240.68]**, 8/8 seeds, exact
  permutation p=0.0078, d_z 1.24; other three cells null (specific).
  **Supplementary, NOT registered for this read (labeled in RESULTS.md):**
  the reward-aware seed-9–16 extension (launched before the 11-Jul pause,
  snapshot `local_results/axis1_seed_extension_20260711_201510/`) was read
  as a fresh-seed robustness check of the reward-aware simple effect:
  finger Q1 +30.71 [−15.51, +75.59] ns — strong attenuation vs seeds 1–8
  (+159.32); pooled 1–16 +95.02 [+44.20, +152.91] still excludes zero.
  Winner's-curse caveat carried on the +157 interaction magnitude; fix =
  the protocol's registered re-pointing of the extension: reward-free apt
  finger-Q1 fits, seeds 9–16 (16 jobs), for a fully-paired n=16
  interaction. Disclosure category: corrective replication (confirmatory
  part) + labeled supplementary (extension read).

- **2026-07-12 — E1 registered read; canonical-variant note; search-within
  built.** (a) Reacher critic gate PASS (held-out R² 0.3696 ≥ 0.2 ⇒ VSA
  primary). (b) The registered E1 driver-level analysis was run with the
  frozen pipeline locally (`artifacts/e1_reacher_final_20260712/RESULTS.md`):
  cov +0.318* [0.132, 0.512], occ_phys null, M0 null, M2 unstable
  (fwd/geom VIF 147), retdec degenerate (range −181..1). The cluster-side
  fit (`reacher_primary_measured_only/`) pre-filtered AUC rows to measured
  cells before z-scoring — a deviation from the frozen spec input contract;
  qualitatively identical, but the frozen local run is canonical. Driver
  rows verified identical (56/56). (c) `search-within` mode +
  `selfcheck-within` added to `build_controlled_replay.py` implementing
  Amendment 1's collector-run units (per-source cov-matched high/low
  pairs, corner + rank-weighted candidates, cov_tol semantics unchanged,
  source_l1 ≡ 0 by construction; all three prior selfchecks re-PASS);
  submit section to follow before E3v2 launch.

Changes to the analysis *code* that enforce the frozen spec rather than
alter it, logged for transparency. All entries below were made before any
adaptation outcome was read locally.

- **2026-07-06 — mode filter in `fit_mixed_effects.py`.** PREREG §1 defines
  the dose-response unit of analysis with mode ∈ {p2e, apt, random}, but the
  frozen code read every `adapt_*` row in auc.csv. Once Axis-1 intervention
  adapts (run ids `adapt_ax1<q>s<side>_<dom>_seed<k>_ckpt<updates>`, mode
  strings `ax1q1s0` etc.) land in the same runroot, they would have leaked
  into M0/M1/M2 and the within-domain z-scoring. Added `--modes` (default
  `p2e apt random`) applied before z-scoring. No formula, window, driver, or
  inference change.
- **2026-07-06 — `paired` CLI convenience.** `paired_contrast()` itself is
  unchanged (PREREG §6). The CLI now accepts auc.csv directly: optional
  `--domain` filter, `mode` used as `cond` when no `cond` column exists,
  qc_pass filter applied. Pairing remains within-seed.
- **2026-07-06 — Axis-1 run naming registered.** Phase-6 adapt runs follow
  the frozen id pattern with mode = `ax1<quadrant>s<side>`, milestone =
  offline gradient-update count (500000 = equalized to the 500K-step online
  pretrains at train_ratio 1024 ⇒ 1 update/env step). AUC extraction
  (PREREG §2) applies to them unchanged; they enter only the paired
  contrasts, never the dose-response models.
- **2026-07-07 — driver-side exclusions enumerated post-hoc into
  exclusions.csv (unblinding day).** PREREG §1/§7 exclude rows whose
  milestone has no retained checkpoint within 5,000 steps. That exclusion is
  applied *upstream* by `scripts/measure.sbatch` (skips the milestone), so
  the frozen local pipeline never sees those cells and `adaptation_auc.py`
  cannot list them. The 11 affected cells (10 `random_*` at 300K/500K plus
  `p2e_finger_seed1@500K`; enumerated in
  `artifacts/phase5a_20260707/exclusions.csv`) were appended to
  exclusions.csv by a one-off script after the frozen collation ran.
  Exclusion *rule* unchanged; this only makes the pre-registered log
  complete. Note the corresponding *adapt* runs used the nearest snapshot
  even when >5,000 steps off, so those 11 rows appear in M0 (milestone is
  the label) but never in M1/M2 (no driver row to merge).
- **2026-07-11 — Gate F applied; E1 outcome verification; E3 pre-submit
  fixes registered; latent-UQ stage0 read.** (a) Gate F (frozen rule, 7-Jul
  addendum): oracle mean AUC₁₀₀ₖ 68.34 < μ+2σ = 131.83 of the 5a finger rows
  ⇒ rule fires — finger stays reported-null *in the observational
  population*; interpretation per the 10-Jul amendment (registered before
  any Gate-F outcome existed): buffer deficiency, not readout ceiling
  (Axis-1 high-occ buffer adapts; p2elong 496K-step budget probe stays
  flat). `artifacts/e2_gatef_20260711_092902/DECISION.md`. (b) E1 reacher:
  75/75 adapts verified bit-identical; mode-level direction replicates
  (p2e/apt ≈95–98 vs random 42 AUC₁₀₀ₖ); registered population fit awaits
  the measure grid. (c) E3 pre-submit check
  (`artifacts/e3_precheck_20260711/PRECHECK.md`): finger dose + both finger
  rpairs + cup r1 GO; registered pre-outcome fixes — cup dose re-search
  with `--n_candidates 1200 --beam 300` adopted iff total_abs_dev improves;
  cup r2 exclusion widened (all random labels; fallback original+r1-hi
  dominants) until the pair differs from r1; control-reuse rule where an
  r-pair side is member-identical to an already-run buffer (cup r1 lo ≡
  q1 s0 ⇒ contrast ax1r1s1 vs existing ax1q1s0, no duplicate runs). These
  are all search/efficiency changes made before any E3 outcome exists; no
  frozen criterion (cov tol, occ-sep, overlap, seed) changed.
  **Redo check (same day, still pre-outcome):** cup dose re-search ADOPTED
  (dev 0.2258→0.1567; levels 0.061/0.117/0.230/0.232 — top two duplicate at
  the cup frontier, kept as a pure-error replicate). Cup r2 all-random
  exclusion returned OK but docc 0.048 = 0.30× target (1.18× the separation
  floor) — **new registered minimum-dose criterion: an r-pair is adopted
  only if docc ≥ 0.5 × target_docc**; the all-random pair fails ⇒ fallback
  exclusion (random4 apt2 random2 apt5 random3) to be searched/rebuilt; if
  that also fails the criterion, cup keeps r1 as its only r-pair and the
  all-random pair is descriptive-only (no adapt runs). Code note:
  search-dose/search-rpair now record `n_candidates` in their output
  criteria (metadata only; the frozen `search` output is untouched).
  (d) Latent-UQ
  stage0 (other study): Biased Dreams attractor bias replicates 10/10
  held-out seed-cells ⇒ D0 stays frozen
  (`artifacts/latent_uq_stage0_20260710_221232/DECISION.md`).
- **2026-07-10 — Phase-6/Axis-1 unblinded.** The 64-run paired grid was
  analyzed on the cluster with the frozen scripts (copied alongside the
  data; `local_results/axis1_analysis_20260710_213221/`) and re-derived
  locally from raw `scores.jsonl` — all 64 AUC rows and all 16 paired
  contrasts bit-identical (<1e-6). No deviation; write-up in
  `artifacts/phase6_axis1_20260710/RESULTS.md`. Ordering note: the E1–E7
  addendum (7 Jul) was frozen **before** this read, so its E3 directional
  prediction ("dose-response slope negative") predates and now conflicts
  with the Axis-1 outcome (occupancy null in cup, positive in finger).
  Per the addendum's own freeze rule the E3 prediction is NOT revised;
  E3 outcomes will be reported against the frozen 7-Jul prediction, with
  the Axis-1-informed expectation noted as post-hoc. A dated post-Axis-1
  amendment in the addendum (registered before any E3/E5/extension run
  exists) adds: finger dose-response (+20 jobs, same `ax1d*` modes),
  Axis-1 seed extension 9–16 (same run-id scheme; seeds-1–8 read remains
  the registered primary, pooled estimates labeled post-unblinding
  confirmatory extension), and the occupancy-floored finger demo — new
  mode strings `rrwc` (natural finger draw) / `rrwd` (floored draw),
  parsing under the frozen `RUN_RE` and excluded from dose-response
  models by the `--modes` filter.
- **2026-07-07 — post-5a enhancement modules pre-registered; run-naming
  registry extended.** `research_notes/Addendum_Post5a_Enhancements_20260707.tex`
  (frozen before any Axis-1 outcome was read) registers modules E1–E7 and
  the following new mode strings, all parsing under the frozen `RUN_RE` and
  all excluded from the dose-response population models by the `--modes`
  filter: `goalwm` (Gate-F finger oracle probe), `p2elong` (finger budget
  probe), `ax1d0`–`ax1d3` (occupancy dose levels, coverage-matched),
  `ax1r1s0/1`, `ax1r2s0/1` (composition-robustness pairs), `rrwa`/`rrwb`
  (occupancy-capped replay demo). Reacher joins as a replication
  population: its rows use the ordinary dose-response modes
  (p2e/apt/random) and ARE in scope of the population models when the
  pipeline is run with reacher in `--primary_domains` — reported as a
  separate population, never silently pooled into the registered
  cup+finger primary. Reacher `regimes.py` threshold 0.025 confirmed
  against `artifacts/gate0_20260702/gate0_reacher/gate0.json`
  (goal 0.433 vs reward-free 0.005–0.010).
