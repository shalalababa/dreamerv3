# PREREG — NFI Scale Exhibit (SE wave), DRAFT v2.2 (12 Aug 2026, probe-review fixes)

**Status: DRAFT v2.1 — smoke executed and verified
(`local_results/uncfield_se_smoke_20260812_155552`, manifest OK, rc=0,
all 5 planted keys live with per-key decoder losses, disag 3.17M params
trained, expl.mode=p2e confirmed in the run config); all smoke FILLs
pinned below.** Probe built + reviewed same day (review #21: 5
BLOCKING / 6 MAJOR / 7 MINOR, all adopted in this v2.2 + se_probe
fixes; adjudication in `artifacts/nfi_assessment_response_20260812/`
§8). **Freeze after the registered D-niche scoped search (§7)
returns.** v1 reviewed same
day (instrument reviewer, Fable): 4 BLOCKING / 8 MAJOR / 9 MINOR, all
adopted below (adjudication in
`artifacts/nfi_assessment_response_20260812/RESULTS.md` §2). Plan:
`research_notes/paper5_uncertainty_field/Plan_AssessmentResponse_20260812.md`
§A. Purpose: the NFI external-validity leg — the fictitious-information
chain at standard world-model scale, ground truth **by construction**.
NOT gating ICLR.

## 1. Substrate (pinned)

DreamerV3 (this repo), size1m, dmc proprio `cheetah_run` (PINNED —
smoke inventory: `position` dim 8 per-dim sd 0.024–0.143 mean 0.0976;
`velocity` dim 9 sd 0.37–3.72; overall proprio mean sd 1.215). Objective: the repo's
deployed latent-ensemble disagreement module (`dreamerv3/explore.py`,
`expl.mode: p2e`, `disag_task: False` — asserted at agent.py:104), the
UNMODIFIED training path. Training steps PINNED = 5e5 (exploration-study convention; smoke
timing 2e4 ≈ 16 min ⇒ ≈ 6.7 h/run, inside the sbatch limit). **Seeds
10–17 pre-committed** (8 runs; renumbered from 0–7 so no seed collides
with the smoke run se_smoke0/seed 0; no conditional extension clause). Arms are
independent samples — never cross-arm seed-paired (1-Aug GPU
nondeterminism rule).

## 2. Planted channels (appended observation keys)

Precise information statements (review M-1):
- **N (noisy-TV)**: `embodied/envs/distractor.py` OU wrapper, dim 8,
  scale 1.0, `basesd` PINNED = 1.215 (overall proprio mean per-dim sd from the
  smoke inventory — the Distractor's own reference convention; no
  online calibration window, so the zero statement holds from step 0), wrapper seed =
  training seed. **Zero information about task state and reward**; the
  OU latent itself is learnable, so early disagreement on N is partly
  legitimate — the registered claim is FINITE-TIME NON-DECAY (M4), not
  instant-zero.
- **D-family (redundancy ladder) — FLAGSHIP = D0**:
  - **D0**: exact duplicate (ε = 0) of the pinned source key
    `position` (dim 8; the bounded, low-sd key — velocity's 30×-larger
    sds would make the ladder semantics scale-inhomogeneous). Zero marginal information given the
    source key, and no stochastic component whatsoever — if the
    objective allocates value to D0, no noisy-TV reading survives.
  - **D1**: duplicate + i.i.d. N(0, (0.05·refsd)²); **D2**: same at
    ε = 0.5·refsd, with refsd = BASESD_PLANTED PINNED = **0.0976** (the
    SOURCE key's own mean per-dim sd — source-commensurate, so the
    ε-ladder stays a small-to-moderate perturbation OF THE SOURCE; the
    velocity-dominated overall mean 1.215 would have made D1's noise
    ≈ 60% of the source scale and wrecked the ladder semantics). The ε-ladder is the registered discriminator
    (review B3): redundancy farming predicts D0 > floor and/or
    D-share NOT ∝ ε²; noisy-TV predicts share ∝ ε² with D0 at floor.
- **C (constant ≡ 0, dim 4)**: anchor for the DECODER-PROJECTION floor
  only (its projected share is estimated and can fail); C is NOT a
  masking floor (mean-masking a constant is a bitwise no-op — review
  B1).

All channels appended in the single Stage-1 arm. Wrapper selfcheck =
construction checks (D-family emissions regress exactly onto recorded
source + registered-sd residual; N/D consume only their own RNG) +
smoke cross-correlation diagnostic (review minor 3).

## 3. Arms

**Stage 1: planted arm ONLY, seeds 10–17 = 8 runs** (the v1 unwrapped
control arm fed no registered measurement and carried a dimensionality
confound — deleted per review B4; all Stage-1 primaries are within-arm).
**Conditional D-only arm** (channels D0/D1/D2 without N, seeds 10–13,
4 runs) PRE-AUTHORIZED, trigger: N-share fires while all D-shares are
at floor (cross-channel interference check, review M-8).
**Stage 2 (behavioral, trigger: P-SE1 fires on any non-C channel):
gated vs ungated at MATCHED observation space** — both arms carry
identical appended channels; the gated arm emits them only when a
pinned proprio coordinate crosses a threshold (else 0), the ungated arm
always. Diversion contrast = gate-vs-no-gate at identical obs
dimensionality, encoder, and loss mass (review B4.2). Gated channels
remain zero-MARGINAL-information because the gating coordinate is
itself observed (stated per review minor 7). 2 × 4 seeds = 8 runs.

## 4. Attribution instruments (registered architecture, review B2)

- **PRIMARY (intervention-free, defined identically in replay and
  imagination): decoder-projected per-key disagreement.** Recon-pinned
  mechanics: `Disag.predict` (explore.py:30-33) exposes per-member
  postfeat predictions; each is split into (deter̂, probŝ) and decoded
  with SOFT stoch (the decoder consumes {deter, stoch} dicts); the
  probe selfcheck registers a calibration control — decoding the TRUE
  postfeat of held states must track decoding their actual sampled
  state (pins the soft-stoch approximation as adequate before any
  read). For each
  disag-ensemble member m, push its predicted next latent through the
  frozen decoder's per-key heads (plumbing exists: per-key heads
  agent.py:277-281; imagined-feature decoding precedent
  agent.py:469-479); d_k = Var_m(μ_k^m) averaged over key-k dims,
  normalized per key PINNED: per-dim probe-set variance in the
  decoder's target space (symlog), matching probing/latent_uq.py's
  normalization convention. Planted share S = Σ_planted d_k / Σ_all d_k,
  reported per channel. This reads out the DEPLOYED ensemble — no
  retraining (the v1 `disag_target` fallback is WITHDRAWN: it changes
  the trained objective and forfeits the "standard objective"
  wording).
- **SECONDARY (replay-only, causal): interventional masks**, signed Δ
  registered with the directional hypothesis (removal of a farmed key
  REDUCES disag); the inflation side reported separately as a
  shift-artifact diagnostic (review M-2). Mask forms pinned per
  channel: N → batch-permutation mask; D-family →
  source-substitution (D := source value, deleting exactly the
  fictitious content); masks applied over the full probe window
  (burn_in = 16, the same window as the primary), in raw obs space
  before symlog, mask statistics from a pinned replay slice (review
  M-3). **Build scope (review #21 B5): the mask instrument is built +
  selfchecked BEFORE THE READ, not before freeze** — replay-only
  secondary whose sole decision role is outcome-map cell 6, which
  cannot be adjudicated until it exists.

## 5. Registered measurements

- **P-SE1 (accounting, PRIMARY).** On N_eval = 512 replay states per
  run (validity floor: read invalid if realized S < 256) at the FINAL
  checkpoint, probe window burn_in = 16, ep_batch = 64, normalizer
  floor = 0.05 × mean real-key per-dim sd (pinned instrument
  constants; member-prediction probs simplex-projected via
  clip+renormalize): **per-channel** decoder-projected share vs the
  within-run dim-level permutation null (channel dims vs REAL dims
  only, ≥1000 permutations; crc32-keyed rng) AND vs θ₁ = the source
  key's own share (D-family) / nearest-dim real key (N).
  **Fire-eligible channels = {D0, D1, D2, N}; the constant channel is
  the decoder-projection floor DIAGNOSTIC only and never fires.**
  Registered caveat (review #21 M1): dims within a key co-move, so
  dim-level permutation is anti-conservative as a per-run test — the
  CONFIRMATORY axis is cross-seed replication:
  **CROSS-SEED PRIMARY (the registered fire rule): a channel FIRES if
  its per-run share exceeds its per-run permutation-null MEDIAN in
  ≥ 7 of 8 runs (exact binomial p = .035 under the null), with BH
  q = .05 over the 4 fire-eligible channels.** Supporting (reporting
  only): Fisher-combined per-run p's, seed-level BCa intervals, and
  the pooled fire+real share (const-excluded; near-powerless by
  construction — 32/49 pooled dims are fire dims — hence DESCRIPTIVE
  only). The FLAGSHIP wording additionally requires the D-family
  discriminator (§2).
- **P-SE2 (ranking, PRIMARY).** M = 256 imagined rollouts (trained p2e
  policy via the policy-callable form of `dyn.imagine` — rssm.py:94 —
  with the agent.py:307 sampling pattern; imag_length 15; starts = the
  probe-window anchor states, uniform over the N_eval replay windows),
  ranked by imagined UNDISCOUNTED intrinsic sum over the horizon
  (states paired with the actions taken AT them — the deployed disag
  pairing, review #21 B1); statistic = mean FIRE-channel (const
  excluded) decoder-projected share of the top-k = 10 vs the
  all-rollout mean, permutation test over rollout labels within run.
  **CROSS-SEED PRIMARY: FIRES if top-k share > all-rollout mean in
  ≥ 7 of 8 runs (binomial p = .035)**; per-run p's and the per-family
  (D-family vs N) share breakdown reported (the pooled SE2 statistic
  cannot separate outcome cells 1 vs 2 on its own).
- **M3 (behavioral, Stage 2 SECONDARY).** Occupancy share of the gate
  region, gated vs ungated arm, + task-return delta under identical
  eval. Directional: diversion toward the gate region.
- **M4 (persistence, SECONDARY).** P-SE1 statistic at 25/50/100%
  checkpoints vs the permutation-null floor at each: registered weak
  form = fire-channel share remains above the null at 100%.
  **Registered analytic choice (review #21 M5): snapshots retain
  checkpoints, not buffers — all M4 probes run on the FINAL replay
  buffer (fixed eval distribution, shared read-time normalizers).**
- **Calibration acceptance (registered, review #21 M6): the
  instrument is VALID iff the normalized soft-vs-hard decode
  discrepancy is < 0.5 on every REAL key** (smoke: 0.066/0.083 —
  wide margin); planted keys reported; const reported against its RAW
  (unfloored) normalizer.
- **Cross-seed reader (review #21 B4): `uncfield/se_read.py` — the
  frozen reader implementing the ≥7/8 fire rules + BH — is built and
  selfchecked BEFORE THE READ and executes it; per-run se_probe
  outputs are its only inputs.**
- Exploratory (unregistered): M5 coherence fingerprint; encoder-mask
  attribution comparisons.

## 6. Outcome map (frozen wordings at freeze; review M-5 — exhaustive)

Cells over {P-SE1, P-SE2} × {D-family fires w/ discriminator, N fires}:
1. **SE1 ∧ SE2 ∧ D-discriminator**: FLAGSHIP — "a standard disagreement
   objective at standard scale allocates epistemic value to channels
   carrying zero marginal information by construction — including an
   exact duplicate — and its planner's imagined-rollout ranking
   concentrates on them."
2. SE1 ∧ SE2, N only (D at floor, share ∝ ε²): model-internal
   noisy-TV accounting exhibit (beyond the behavioral literature via
   per-key attribution); flagship wording NOT licensed; triggers the
   D-only arm (§3) before writing.
3. SE1 only (accounting without ranking): "the objective misprices;
   the planner does not yet concentrate" — accounting-level exhibit,
   ranking claim dropped.
4. SE2 only (ranking without per-state accounting): report as
   surprising dissociation, no flagship claim, flag for design review.
5. All planted shares at the permutation null: informative null —
   "not exhibited by ensemble-disagreement objectives at this scale
   under these channels"; core paper unaffected.
6. Instrument-invalid: decoder projection degenerate at smoke (e.g.,
   D0 collapsed by the encoder) AND interventional masks degenerate —
   report as build failure, no claim. (D0-degeneracy alone at smoke =
   registered sub-branch: D0 dropped, flagship rests on the ε-ladder
   discriminator only.)

## 7. Differentiation clause (REGISTERED, named foils — review M-7)

Known and NOT claimed: behavioral noisy-TV attraction (Burda et al.
2019 RND + large-scale curiosity; Schmidhuber's formulation); the
claim that disagreement handles stochasticity is the FOIL, not the
finding (Pathak, Gandhi & Gupta ICML 2019; Sekar et al. 2020
Plan2Explore — the deployed objective's own robustness argument);
aleatoric-aware curiosity measuring noise-channel intrinsic reward
(Mavor-Parker et al. ICML 2022 — nearest prior; our delta = per-key
attribution certified against BY-CONSTRUCTION-zero channels incl. an
exact duplicate, inside the deployed objective's own accounting);
stochasticity-robust curiosity at scale (Jarrett et al. 2023);
ensemble-disagreement ≠ calibrated epistemic uncertainty in principle
(Bengs/Hüllermeier line — ours is the certified in-objective exhibit).
Novel content claimed ONLY as: (i) redundancy/duplicate farming (D0 +
ladder), (ii) per-key model-internal attribution certified against
by-construction-zero channels, (iii) planner-ranking concentration
(P-SE2). **Pre-freeze scoped search EXECUTED 12 Aug 2026 (web + arXiv +
citation graphs of Sekar 2020 / Mavor-Parker 2022): CLEAR — claims
(i)–(iii) unoccupied at freeze; LIMIT-2, cited for differentiation
only:** CIG (arXiv:2605.20878 — the same work the NFI paper engages as
a transplanted defense) builds an ensemble-disagreement kernel whose
"redundancy" is within-rollout state-visit repetition — TEMPORAL, not
observation-channel redundancy — with a stochastic-only distractor arm
and no per-key attribution; DreamerV3-XP (arXiv:2510.21418) deploys
ensemble-disagreement intrinsic reward inside DreamerV3 (over predicted
REWARDS, not latent dynamics), confirming the objective family is
current at our substrate scale, with no planted channels and no
attribution analysis. No 2024–26 work measures disagreement/curiosity
objectives on zero-marginal-information channels, does model-internal
per-key attribution against by-construction-zero channels, or analyzes
planner-ranking concentration.

## 8. Instruments to build (inline; review fixes folded)

BUILT + selfchecked: `embodied/envs/planted.py` (construction checks
§2); `uncfield/se_probe.py` (decoder-projected attribution, P-SE1/
P-SE2 per-run reads, calibration control, statistics mutant battery;
reviewed #21, all findings adopted); `scripts/uncfield_se.sbatch`.
TO BUILD BEFORE THE READ (registered): the §4 mask instrument;
`uncfield/se_read.py` (cross-seed frozen reader, §5). Smoke executed
12 Aug (bundle `uncfield_se_smoke_20260812_155552`) → all FILLs pinned
→ freeze commit → full submission (seeds 10–17).

## 9. Compute

Stage 1: 8 × size1m ≈ 6.7 h/run (smoke-extrapolated; sbatch limit
10 h) + CPU probe passes. Conditional D-only: +4. Stage 2: +8.
All RCC/cloud lanes.

## 10. Standing-rule compliance

Permutation-primary pinned (§5, key-label/rollout-label schemes); BCa
reporting-only; fit counters verified in the read; results-sync v2
manifests; never-delete-without-archive; arms never cross-seed-paired;
frozen reader executes the read in the Paper-5 chat.
