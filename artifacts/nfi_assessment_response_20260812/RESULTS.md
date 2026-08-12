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
