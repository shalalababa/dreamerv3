# NFI/CEI Assessment-Response Wave — record (12 Aug 2026)

Executes `research_notes/paper5_uncertainty_field/Plan_AssessmentResponse_20260812.md`
(user decisions: arXiv → ICLR → ICML fallback, both papers; SE wave GO).
Reviewers approved by user 12 Aug.

## 1. Held-out predictive quality vs coherence audit (plan §C.2) — DONE

`uncfield/heldout_eval.py` (selfcheck PASS: floor < trained < ceiling
ordering + determinism). 50 fresh episodes × 600 steps, env/policy
seeds 777123/888777 (disjoint from all training/warmup/imagination
seeds). Output `heldout.json` (this dir).

| model | obs-NLL (per sensed step) | z-NLL (per step) | coherence verdict |
|---|---|---|---|
| exact filter (floor) | 0.301 | −1.820 | — |
| pilot2 GRU m0–m3 | 0.384–0.395 | 0.46–1.48 | ALL exploit tier |
| LG-LSTM m0–m3 | 0.464–0.560 | 2.00–2.89 | ALL exploit tier |
| untrained GRU/LSTM ×4 seeds | 1.42–1.66 | 9.36–9.79 | (§3b: reaches CIG-ONLY) |

**The exhibit:** trained GRU members sit within 0.08–0.09 nats of the
Bayes floor per sensed observation — normal, near-optimal held-out
predictive quality — while every one of the 8 trained members carries
an exploit verdict. Ordinary predictive evaluation does not surface the
coherence defect; the untrained ceiling is ~4× the floor on obs-NLL and
~11 nats off on z-NLL. This is the assessment-§4 answer ("show the
trained models look good on normal metrics while failing the audit").
MANUSCRIPT: one table + two sentences in the untrained-control /
defenses discussion.

## 2. SE wave (plan §A) — prereg v1 REVIEWED → v2 SAME DAY

**v1 instrument review (Fable, 12 Aug): 4 BLOCKING / 8 MAJOR / 9 MINOR
— ALL ADOPTED in draft v2** (review #18, yield 18/18). Blockers, for
the record: **B1** the v1 constant-channel masking floor was vacuous
(mean-masking a constant is a bitwise no-op ⇒ θ₀ ≡ 0, P-SE1's CI clause
could never fail) → v2 uses a key-label PERMUTATION-NULL floor (also
discharges the permutation-primary standing rule). **B2** v1's M2
attribution referenced encoder-input masking in IMAGINATION where no
observations exist (disag path never calls the encoder;
agent.py:319-322, explore.py:45-48) → v2 primary = DECODER-PROJECTED
per-key disagreement (per-key heads agent.py:277-281; intervention-free,
identical in replay and imagination, reads the DEPLOYED ensemble; the
disag_target-retraining fallback WITHDRAWN); masks demoted to
replay-side causal secondary (N → permutation-mask, D →
source-substitution, signed Δ + shift diagnostic). **B3** D's ε-noise
made late-training D a small-amplitude noisy-TV (redundancy vs
ε-farming unidentified) → v2 adds **D0 = exact duplicate (flagship)** +
ε-ladder D1/D2 with a registered discriminator. **B4** the Stage-1
unwrapped control fed no registered measurement and carried a
dimensionality confound → deleted (8 pre-committed seeds instead);
Stage-2 M3 = gated-vs-ungated at MATCHED obs space. Majors folded:
precise per-channel information statements; pinned mask scope/space;
θ₁ = own source key; exhaustive 6-branch outcome map; named-foil
differentiation clause (Pathak-Gandhi-Gupta 2019, Sekar 2020, Burda
2019×2, Mavor-Parker 2022 = nearest prior, Jarrett 2023,
Bengs/Hüllermeier) + registered pre-freeze D-niche scoped search;
basesd pinned, wrapper seed = training seed; conditional D-only arm
pre-authorized. Next: build §8 instruments → smoke → freeze.

### 2-archive. Original §2 text (pre-review, for provenance)

`prereg/PREREG_nfi_scale_exhibit_20260812.md` (draft v1: channels
N/D/C with D=duplicate flagship; staged ungated→gated; P-SE1 masking
attribution + P-SE2 imagined-rollout ranking; outcome map with
informative null; registered differentiation clause; FILL slots for
smoke). Key recon facts: `dreamerv3/explore.py` + `expl.disag_*`
config already implement the Plan2Explore-style objective (from the
exploration-pretraining study); `embodied/envs/distractor.py` is the
registered noisy-TV injector. Instrument reviewer (Fable) launched
12 Aug — findings to adjudicate here before freeze.

## 3. CEI exposure theorem — v0.1 REVIEWED → v0.2 SAME DAY (MAJOR UPGRADE)

**v0.1 math review (Fable, 12 Aug; review #19, yield 19/19; numerics =
exact Gaussian KLs, no MC):** R1 CONFIRMED (Prop 1 valid; +2 explicit
hypotheses H1/H2 — private noise not entering the sensor, no unlogged
reads; internal-state marginalization = Rubin-1976 ignorability);
R2 BROKEN pointwise (κ_j(t,h) ≤ κ FAILS at a 2σ observation — the
posterior-MEAN gap under observation-dependent histories is unbounded;
1.988 > 1.808 at Y=2 in the pilot pair; also R_k → R_j^{(i)} errata);
R3 rate ρ²Δ² confirmed IN EXPECTATION (slopes 2.07/1.99) but v0.1's ρ
functional failed its own ε=0 sanity check (→ normalized
cross-covariance) and the theorem is SUPERSEDED; R4 CONFIRMED —
BH/DPI step lifts verbatim; R5 Prop 1 + decomposition identity =
classical (KCG16 Lemma 1 pre-collapse; Wald/Rubin lineage;
Nitinawarat-Atia-Veeravalli 2013 nearest) — the COMPOSITION is new.

**The reviewer's headline finding (beyond the request list): the
coupling-term architecture was the wrong shape.** Full-PATH latent
augmentation (generalizing v3.2 Thm 3's static-X device) proves
**KL(P₀^log ‖ P₁^log) ≤ κ·E_{P₀}[N_j] for ARBITRARY P₀/A/Q coupling
and ARBITRARY observation-dependent policies** — no block structure,
no staticity, no ancillarity, no coupling term. Coupled reads
redistribute evidence WITHIN the budget, never add to it (verified:
totals ≤ κE[N_j] across all ε and both adaptive-selection directions,
including regimes where the SELECTED per-read divergence exceeds κ —
0.220 vs 0.0966). Consequence: **CEI's Theorem 3 generalizes with its
restrictive assumptions DELETED from the impossibility direction** —
the exact answer to the assessment-§7 reductive review — and "read
more when surprised" is provably the unique policy-side rescue
(theorem-grade form of NFI's gate/re-admission).

v0.2 written (all repairs adopted: H1–H3 explicit; pointwise bound
deleted, budget theorem = headline; decomposition demoted to
descriptive with corrected ρ; KCG/Rubin/NAV attributions; worked-example
spec incl. the adaptive-selection demo). Remaining before manuscript
use: O1–O4 (optional-stopping step in full, sup_x measurability,
independence condition precise, worked example) + second review pass.

### 3-archive. Original §3 text (pre-review, for provenance)

`research_notes/paper5_uncertainty_field/theory/Theory_CEI_Exposure_20260812.tex`
v0.1. The chain-rule analysis yields a SHARPER structure than the
assessment's proposed two-term bound: **Proposition 1 — behavioral
leakage is exactly zero for arbitrary observation-dependent policies**
(the action kernel is hypothesis-independent, so action terms of the
KL chain rule vanish identically); the true generalization axis is
SENSOR/STATE COUPLING (Theorem 1 decomposition: direct j-reads +
coupled non-j-reads, the latter vanishing under v3.2's block
assumptions; Theorem 2 coupling bound = the open proof obligation).
Consequence if it survives review: block structure reframes from
assumed convenience to the ρ=0 point of a quantified axis, and
"reading more when surprised" becomes the theorem-grade unique rescue
channel (ties to the NFI gate/re-admission finding). Math reviewer
(Fable) launched 12 Aug with R1–R5 attack list — note is v0.1, NOT yet
consumed by any manuscript.


## 4. Instruments built (12 Aug pm)

- **SE wave**: `embodied/envs/planted.py` (Duplicate ε-ladder
  0/0.05/0.5 + Constant + Stage-2 gate; selfcheck PASS — dup0 exact,
  ladder sds within 5%, base-RNG isolated, deterministic,
  gate-independent noise stream) + `dreamerv3/main.py` wiring
  (SeedSequence 0x5E, stream keyed to training seed) + `configs.yaml`
  `planted` defaults + `scripts/uncfield_se.sbatch` (reuses the
  exploration-pretraining p2e launch path incl. the
  checkpoint-retention watcher M4 needs; SMOKE=1 prints the proprio
  inventory + candidate BASESD to fill the prereg FILLs). Remaining
  before freeze: [YOU] smoke run; [ME] `uncfield/se_probe.py`
  (decoder-projected attribution + P-SE1/P-SE2 reads + fixture
  battery) developed against the smoke checkpoint; then FILLs → freeze
  commit.
- **CEI exposure theorem**: note v0.3 discharges O1–O4 (appendix
  proofs: Wald step via i.i.d. residuals + stopping time; non-Gaussian
  supermartingale variant; emission-closed factorization lemma;
  `uncfield/exposure_example.py` selfcheck PASS = the registered
  worked example, promoted from review #19's exact-KL verification
  script, output `exposure_example.json`). Second-pass reviewer
  (Fable) launched on the new proofs + code — adjudication lands here.


## 5. ADJUDICATION (12 Aug night) — second-pass theory review; note → v0.4 READY

**Review #20 (Fable; yield 20/20): v0.3 SURVIVES — headline budget
theorem TRUE; two MEDIUM proof repairs + four minors, ALL APPLIED
(note v0.4).** The findings:
- **MEDIUM, the important one: H1′ was missing.** H1–H3 as written do
  not entail ξ ⊥ noises — reviewer counterexample: a randomization
  device set to the QUANTILE of the next j-residual satisfies the
  letter of H1/H2/H3 (ξ ~ U(0,1) under both hypotheses, affects only
  the action) yet makes the transcript KL = +∞ while κ·E[N_j] =
  0.0024 — selection on unread outcomes is a side channel. v0.4 adds
  (H1′): ξ jointly independent of (path, all measurement noises).
- **MEDIUM: O2's optional-stopping hypothesis was mis-stated**
  (bounded conditional MEANS ≠ the required conditional L¹ bounds;
  independently repairable via E[|ℓ| | x] ≤ KL(x)+2). Superseded by
  the reviewer's Repair B: at fixed T the chain rule alone gives the
  budget term-by-term with NO stopping-time or moment conditions —
  v0.4 makes that the main proof, retaining Wald ONLY for the
  stopped-τ extension (with the filtration and stopping-time
  justification written out; k-read outcomes in the decision stream
  are handled because they are functions of hypothesis-free
  quantities inside the augmented σ-algebra).
- Minors applied: countable-supremum measurability; O3's z-free
  action-factor sentence (the factorization has THREE factors, the
  action factor cancels by log-measurability); stopped-τ sentence in
  the Corollary; dead placeholder deleted from exposure_example.py
  (verified inert by the reviewer; selfcheck re-PASS after removal);
  version header; Wald 1944 (Ann. Math. Statist. 15(3):283–296)
  citation + "genie-aided" device attribution.
- **Confirmations that matter for the manuscript:** v3.2 Thm 3's κ is
  the IDENTICAL conditional-on-latent object (no constant swap; its
  own step (iii) already removed predictive-variance dependence);
  d(δ,1−δ) ≥ log(1/2.4δ) so the new form is at least as tight;
  exposure_example.py audited clean (quadrature vs 4M-sample MC
  agreement; E[N_j] formulas verified; all asserts live).

**STATUS: Theory_CEI_Exposure_20260812.tex v0.4 is READY for
manuscript integration.** WRITING CHAT: lift the budget theorem +
corollary into the CEI ICLR version, replacing/generalizing Thm 3 per
the note's §7 (assumptions deleted from the impossibility direction;
block structure retained only for the constructive/count statements
and the descriptive decomposition; H1′ stated as a scope condition —
it is a REAL scope boundary, not a technicality: agents whose
randomization is coupled to unread outcomes are outside the theorem,
which is worth one honest sentence in the paper).

> **EXECUTED (12 Aug, writing chat):** integration landed in
> `writing/Paper_CEI_Draft_20260809.tex` (header change-log block
> documents the delta). Thm 3 → two-part form: (a) exposure budget
> KL ≤ κ·E[N_j] for arbitrary (P₀,A,Q)/policies/coupling, (b) the
> d(δ,1−δ) audit-cost corollary (constants unchanged). Added:
> Assumption 2 (H1/H1′/H2/H3) + H1′ scope remark (quantile-coupling
> counterexample stated as a real boundary, in the theorem's scope
> sentence AND §10); Prop 1 no-action-channel-evidence w/ classical
> credits (Wald/Rubin/KCG16 — composition-only claim); Prop 2
> exposure decomposition (descriptive) + emission-closed vanishing
> condition + pointwise-cap-is-false honesty (1.988 vs 1.808) + ρ +
> slopes 2.07/1.99; endogenous-law remark (unique-rescue = companion
> gate/re-admission hook); genie full-path proof w/ static-X as
> special case; App C (stopped-τ Wald 1944/Gut + non-Gaussian sup_x
> w/ countable-sup measurability); worked-example numbers in App B;
> App A provenance item 6 (broken-then-repaired intermediate claims
> disclosed); Chernoff-1959/NAV-2013 controlled-sensing ¶; scope §10:
> E[n] conjecture resolved in count form, dynamic-target LOWER bound
> discharged (upper bound + counting stay open), block structure
> rescoped to constructive results. Static checks green; theorem
> numbering 1–4 preserved. Flagged at file foot: the manuscript
> rendering of §7+App C postdates all manuscript reviews → next
> review round alongside the 10-Aug Thm-4 proof.


## 6. SE smoke VERIFIED + all FILLs pinned (12 Aug night) — prereg v2.1

Smoke bundle `local_results/uncfield_se_smoke_20260812_155552`
(manifest OK, 146 files; cloud lane, rc=0, 2e4 steps ≈ 16 min).
Verified live in the run: all 5 planted keys in obs space WITH per-key
decoder losses (planted_dup0's loss tracks `position` exactly —
0.38/0.38 → 0.03/0.03, as an exact duplicate must; dup2 sits at its
ε-noise floor ~1.1; the distractor OU is being learned, loss 3.1→0.5);
disag ensemble trained (3.17M params); run config confirms
expl.mode=p2e, disag_task=False, ens 8, target postfeat. Snapshots
retained at 0/8.4k/14.9k + final (M4 machinery works).

**FILLs pinned (prereg → v2.1, sbatch updated):** task cheetah_run;
SOURCE_KEY = `position` (dim 8, mean per-dim sd 0.0976); **BASESD split
into two constants** — BASESD_PLANTED = 0.0976 (source-commensurate;
the velocity-dominated overall mean 1.215 would have made D1's ε-noise
≈60% of the source scale and wrecked the ladder semantics) and
BASESD_N = 1.215 (the Distractor's own convention); STEPS = 5e5
(≈6.7 h/run, sbatch limit raised to 10 h); **Stage-1 seeds renumbered
10–17** (no collision with smoke seed 0). Probe design risks retired
by recon: `Disag.predict` exposes per-member postfeat predictions
(explore.py:30-33); `dyn.imagine` accepts a policy callable
(rssm.py:94) with the agent.py:307 sampling pattern; per-key decoder
heads confirmed by the smoke's own loss lines; loading path =
probing.collect.load_run_config/load_frozen_agent + make_agent
(latent_uq.py precedent, CPU-capable). Soft-stoch decoding of member
predictions registered with a calibration control in the probe
selfcheck.

**Remaining before freeze: `uncfield/se_probe.py`** (next focused
block; develops against the smoke checkpoint) → freeze commit → [YOU]
submit 8 runs (seeds 10–17).


## 7. se_probe BUILT + selfchecked (12 Aug night); pre-freeze reviewer in flight

`uncfield/se_probe.py`: decoder-projected per-key disagreement (per
prereg §4 — Disag.predict per-member postfeat → soft-stoch decode
through the frozen per-key heads → per-key ensemble variance,
normalized in decoder target space), P-SE1 dim-level permutation null
+ per-channel tests + θ₁ source-share comparator, P-SE2 policy-driven
imagined rollouts (dyn.imagine callable form) ranked by intrinsic
return with rollout-label permutation, calibration control
(soft-vs-hard decode), pilot-schema outputs (json + dims npz).
**Selfcheck PASS**: statistics mutants killed (inflated-planted p<0.01;
exchangeable labels null; rank-uninformative order → mean p ≈ 0.5 over
20 orders — single-order asserts are 5%-flaky by construction);
end-to-end on the smoke checkpoint; bit-deterministic under fixed seed.

**Instrument bug caught by the smoke run, fixed + pinned:** the
constant channel's normalizer (std of a constant ≡ 0, floored at 1e-6)
divided decoder round-off by 1e-12 and manufactured share 0.9999.
Fix = pinned normalizer floor, 0.05 × the mean per-dim std of the REAL
keys. Post-fix smoke-dose diagnostics (2e4 steps = 4% of full dose;
DESCRIPTIVE, not a read): key shares — planted_const 0.605, distractor
0.096, dup0 0.083, dup1 0.062, dup2 0.013, position 0.076, velocity
0.066; calibration 0.065–0.183 for real keys, 0.368 for const; P-SE2
top≈base (p 0.46) as expected for a barely-trained policy. The const
channel's large early share is itself informative (ensemble members
disagree even about a constant at this dose) and is exactly what M4's
persistence read adjudicates at full dose.

Pre-freeze instrument reviewer (Fable, auto-approved) launched on
se_probe.py + prereg v2.1 — known open questions handed to it: the
pooled permutation null's power under planted-dim majority (36/53
dims), the per-channel rng's hash() salting (cross-process
determinism), and the not-yet-implemented SECONDARY masks
(implement-pre-freeze vs re-scope). FREEZE after adjudication.


## 8. ADJUDICATION (12 Aug night) — probe review #21 + D-niche search; SE wave FREEZE-READY

**Review #21 (probe + prereg pre-freeze; Fable; yield 21/21): 5 BLOCKING
/ 6 MAJOR / 7 MINOR — ALL ADOPTED same session** (se_probe fixes +
prereg v2.2). The blockers:
- **B1 rollout pairing off-by-one**: dyn.imagine returns (s_{i+1}, a_i)
  pairs; the deployed disag pairing is (s_i, a_i) → states rebuilt from
  the anchor carry (latent_uq.py:281-293 pattern). Post-fix the smoke
  P-SE2 FLIPPED from null (p 0.455) to top-rollouts-carry-more-fire-
  channel-share (0.264 vs 0.242, p 0.010) — the mispairing was masking
  the signal. (Smoke dose; descriptive.)
- **B2 process-salted hash()** in the per-channel rng (three fresh
  interpreters: 3 different streams; smoke distractor p sat at 0.0465,
  exactly at the α boundary — a rerun could have flipped a registered
  fire) → crc32; determinism assert extended to per-channel p's.
- **B3 const contamination**: planted_const (share 0.605 at smoke) was
  inside the pooled statistic and the P-SE2 numerator against the
  prereg's own scoping → FIRE_KEYS = {D0,D1,D2,N}; const = projection-
  floor diagnostic only.
- **B4 cross-seed aggregation unpinned** → registered fire rule:
  per-channel share > per-run null MEDIAN in ≥7/8 runs (binomial
  p=.035), BH q=.05 over the 4 fire-eligible channels; Fisher/BCa
  reporting-only; `uncfield/se_read.py` = frozen reader built before
  the read (registered).
- **B5 prereg not freeze-ready** (2 residual FILLs; §8 promised masks
  the probe lacks) → FILLs filled (mask window = probe window 16;
  walltime 6.7h/10h); masks re-scoped to built-before-READ (their only
  decision role is outcome cell 6).
Majors: dim-exchangeability caveat registered (confirmatory weight on
the cross-seed axis); pooled share demoted to descriptive
(near-powerless at 32/49 fire dims); BH multiplicity; burn_in=16 /
ep_batch=64 / S≥256 floor pinned; M4 runs on the FINAL replay buffer
(registered analytic choice); calibration acceptance = soft-vs-hard
< 0.5 on every REAL key (smoke 0.066/0.083). Verified-correct list:
anchor/action alignment, postsplit order/dims, decoder soft-stoch
consumption, feat2tensor, nj purity, statistics mechanics.
se_probe selfcheck re-PASS post-fixes (per-channel determinism incl.).

**D-niche scoped search (freeze precondition, §7): CLEAR — LIMIT-2.**
No 2024–26 work occupies claims (i)–(iii); CIG arXiv:2605.20878 (the
NFI paper's own transplanted-defense foil — its "redundancy" is
temporal state-revisit, not channel redundancy) and DreamerV3-XP
(arXiv:2510.21418; reward-ensemble disagreement in DreamerV3, no
channels/attribution) registered as differentiation cites. Citation
graphs of Sekar 2020 + Mavor-Parker 2022: zero matches.

**PREREG v2.2 IS FREEZE-READY** (no residual FILLs; instruments built
+ selfchecked; smoke verified; search discharged). FREEZE = [YOU]
commit: prereg/PREREG_nfi_scale_exhibit_20260812.md,
embodied/envs/planted.py, uncfield/se_probe.py, dreamerv3/main.py,
dreamerv3/configs.yaml, scripts/uncfield_se.sbatch,
manifests/uncfield_se_smoke_20260812_155552.sha256, this record — then
submit the 8 runs (seeds 10–17).

## 9. Pre-read instrument batch (12 Aug, late-late) — mask secondary + frozen reader + Amendment A1; review #22 adjudicated

Freeze commit fc85fdee landed and the 8 runs (seeds 10–17) were
submitted; the two registered pre-read builds (prereg §§4–5, review
#21 B4/B5) were executed while they train, plus one governed amendment:

**(a) Amendment A1** (`prereg/PREREG_nfi_scale_exhibit_amend1_20260812.md`,
pre-read/pre-outcome): prereg §5 registers a P-SE2 per-family
(D-family vs N) breakdown, but the frozen probe stored only the
aggregate fire-share — the frozen reader could not deliver it from its
inputs. Additive reporting-only fields (json `pse2.families`, npz
`rollout_share_dfam/noise`, sum-consistency asserts); NO rng draw
added or reordered. Selfcheck re-PASS with IDENTICAL registered
numbers (planted_share 0.628, p 0.6944, cal max 0.368) = bitwise-inert
witness. Reviewer verdict: VERIFIED INERT.

**(b) `uncfield/se_mask.py`** — the §4 SECONDARY. Pinned forms: N →
batch-permutation of whole window trajectories (marginals preserved
exactly); D-family → source-substitution in RAW obs space over the
full probe window; pinned replay slice = identical collect_windows rng
as the probe. dup0 substitution = bitwise no-op by construction
(hard-asserted replay equality; delta ≡ 0 reported with noop flag).
Directional sign-flip permutation (removal REDUCES disag = p_reduce)
+ inflation side reported separately (review M-2); BCa reporting-only.
Smoke: deltas tiny vs base intrinsic 3.6e-4 (2e4-step model), dup2
own-key reduction directional p=.037 — instrument responds. Selfcheck
PASS (stats mutants, marginal preservation, noop, determinism).

**(c) `uncfield/se_read.py`** — the FROZEN cross-seed reader (executes
the registered read; per-run se_probe outputs = only statistical
inputs). Implements: P-SE1 ≥7/8-vs-null-median rule with exact
binomial (9/256) + BH q=.05 over the 4 fire-eligible channels; P-SE2
≥7/8 without BH; invalid runs (S<256 or calibration ≥0.5 on a real
key) = conservative fire FAILURES with denominator pinned at 8;
provenance gate = per-channel permutation p recomputed from the npz
with bit-identical float32 arithmetic + crc32 rng and asserted EQUAL
to the stored p (verified exact on the real smoke output); Fisher +
seed-level BCa + pooled descriptive + θ₁ vs_source + D-ladder
diagnostics + fit counters + outcome-cell suggestion (§6 map).
NOTE (registered-rule arithmetic): a LONE 7/8 channel (p=.0352) dies
under BH at rank 1 (.0125) and even at rank 2 (.025) — it survives
only when ≥3 channels sit at p≤.0375 or itself is 8/8. This is the
registered rule's own step-up math, documented pre-read.

**Review #22** (ONE Fable instrument reviewer, pre-approved): se_mask
+ A1 = FREEZE-READY; se_read = NOT-READY with 1 BLOCKING + 5 MAJOR +
9 minor — ALL 15 ADOPTED:
- B1 final-checkpoint gate: an M4 snapshot probe pass with default
  --output would silently clobber <run>/se_probe (json+npz clobber
  together, so the provenance gate alone cannot catch it) → reader now
  hard-asserts probed ckpt == run's final ckpt (flag UNVERIFIED when
  ckpt dir unreachable, never silent); M4 protocol: snapshot passes
  must use --output <run>/se_probe_snap<pct>.
- M2 θ₁ (vs_source) was registered but dropped by the reader → carried
  per-run + cross-seed BCa. M3 calibration reporting completed
  (planted keys per run; const vs RAW normalizer recovered from npz —
  raw mean exactly 0.0 for the constant channel, reported as
  None+0.0 = the registered honesty). M4 run identity: probe seed is 0
  on every run → training seed read from config.yaml, duplicates
  FATAL, out-of-family flagged. M5 selfcheck missing-run scenario
  (kills the binom_tail(succ, len(recs)) denominator mutant:
  7/7 → 1/128 vs registered 9/256). M6 fixture provenance de-circularized:
  stored p now generated by se_probe's OWN perm_null (cross-implementation
  test) + real-smoke check embedded in selfcheck.
- Minors: probe_seed rename; BCa on the registered share_vs_real (not
  key_share); weak-gate (all-p==1.0) note + n_perm recorded; se_mask
  summary fixture; --output now required; se_mask docstring/assert
  honesty; per-(channel,stat) rng keying in se_mask; probe-pairing
  cross-check (seed/S/ckpt) when the probe json exists; fit-counter
  +NO-CONFIG flag.
Reviewer also verified-sound: provenance-gate bit-identity (incl. rng
offsets 7383/11329/32251/7060/38118 — no collision with 7/13/31/77/
101/211/999), BH tie behavior, real_keys membership identity, se_mask
machinery verbatim vs frozen probe, BCa formula.

Both selfchecks re-PASS post-fixes. One selfcheck-expectation bug was
mine, not the code's (BH: 7/8 next to a single 8/8 still dies at rank
2 — assertion corrected to the true step-up math + rank-3 case added).

**State: ALL pre-read instruments BUILT + REVIEWED + selfchecked.**
[YOU] commit: prereg/PREREG_nfi_scale_exhibit_amend1_20260812.md,
uncfield/se_probe.py, uncfield/se_mask.py, uncfield/se_read.py, this
record, STUDY_LEDGER.md. [ME] next: the registered read when the 8-run
bundle lands (per-run se_probe ×8 → se_read; se_mask only if cell-6
adjudication needs it; M4 snapshot passes with --output se_probe_snap<pct>).
