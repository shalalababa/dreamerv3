# PREREG — NFI confirmatory wave + registered analyses (2 Aug 2026)

Frozen per the claim-freeze audit memo (`research_notes/
NFI_ClaimFreeze_Audit_20260802.md`, §5 guidance + §6.1–6.4 closure pins;
gitignored memo — this prereg carries the binding content). Evidence base:
`artifacts/nfi_pilot_20260801/`, `nfi_mechanism_20260802/` (amended),
`nfi_sweeps_20260802/`, `nfi_gamma_kernel_20260802/` (all read
post-instrument-review). Instruments frozen at the commit carrying this
file; thresholds by reference: `uncfield/planner.py` EPS_TRUE=0.02,
EPS_PRED=0.08, N_BURN=30, N_LOOPS=8, STEADY=4; verdict tiers + ensemble
cumulative-severity rule verbatim.

## 1. Frozen claim wordings (paper-facing labels: carried / potential)

**C1 (existence/flagship, rescoped):** A planner searching a frozen
learned belief-state world model discovers referee-certified
evidence-neutral action cycles (true gain < 0.02 nats/loop) whose
predicted information return stays positive under both prefix-conditioned
(carried-belief) accounting and potential-based (realized-ΔH,
drift-adjusted) accounting on the same cycle — the two governing
principles of the published CIG and PBIM defenses. Neither published
estimator, **as specified**, operates in the belief-state accounting
space where the exploit lives (empirically grounded: the CIG-faithful
kernel is insensitive in this regime — `nfi_gamma_kernel_20260802`); the
exploit survives the
transplanted principles and falls outside the proven scope of the
published estimators. Under discounting the exploit is a
RETURN-PREFERENCE phenomenon: the telescoped preference for the farming
future is γ-invariant in sign, while the harvested stream self-limits at
rate (1−γ)·(entropy displacement), member-heterogeneously (level-neutral
exploits stream-survive γ=0.99 outright).

**C2 (scoping dissociation, rescoped):** On referee-certified
evidence-neutral cycles the two accounting legs dissociate across
training dose — **dose = gradient steps at fixed data (300×600
episodes), not data volume** — carried farming is dose-insensitive
(37/40 members; persists at 10× steps) while the conjunction is
non-monotone with a mid-training peak (0/4 → 3/4 → 1/4 → 1/4),
attenuating without vanishing; the exploit grows with the capacity of
the exploited update operator while the searching planner is held fixed.
(No "scaling law" vocabulary; single-architecture observation; Pan et
al. cited as owning the capability-axes template.)

**C3 (mechanism, rescoped, m3 split):** In three of four members the
learned update contracts belief covariance 2.6–11.6× more than the
model's own observation heads license under Bayes, heads individually
sane — localizing the defect to the update operator; the fourth member
is a mixed anatomy (head-side re-inflation dominant, 1.4–1.8×
update-side). On the zero-information channel the licensed gain is
exactly 0 vs realized +0.84 nats/loop. A planted Bayes-coherent filter
calibrates the ratio to exactly 1.000. (Positioning: Bayer 2021 +
Cremer 2018 + Young 2026 cited per memo §6.1; three "conditioning gap"
homonyms disambiguated; TV-exhibit pre-emption clause included.)

**C4 (diagnostic — CONDITIONAL, see §4):** enters the registered claims
ONLY if the pre-specified rank-correlation test fires; otherwise
discussion-level. Renamed the **Bayes-license ratio** ("self-consistency
ratio" is Schmitt et al.'s term). Never "provably", never bare
"referee-free" (differentiator = single-forward-pass / no-ground-truth /
same-network). **Fire-branch frozen wording (verbatim, freezes on
fire):** "The Bayes-license ratio (realized EIG ÷ own-heads-licensed
gain) — computable from a single forward pass with no ground truth, no
simulation ensemble, and no reference posterior — is a model-level
screen for epistemic-reward-hacking susceptibility, calibrated to
exactly 1.000 on a planted coherent filter; it instantiates, for the
update operator of a learned recurrent filter, the coherence-audit
tradition of SBC and the information-processing gap of Chen et al. We
further exhibit its known blind spot concretely: a Bayes-coherent filter
with wrong noise assumptions scores 1.000 and passes — an instance of
the classical result that coherence does not imply calibration, for
which the practical escape is interventional/referee-based audit."

## 2. Primary confirmatory experiment — fresh-data pilots

**Jobs:** `dataseed3`, `dataseed4` via the reviewed `uncfield/sweeps.py`
dataseed mechanism (zero code change; episode env-seed ranges
3·100003+e / 4·100003+e, disjoint from all prior data; member init seeds
3–6 / 4–7 — init-seed reuse across DIFFERENT data disclosed). RCC:
`sbatch --array=10-11 scripts/uncfield_sweeps.sbatch`.

**P-N1 (primary):** carried-farming universality — fires if ≥ 6 of the 8
fresh members reach tier ≥ EXPLOIT-SURVIVES-CIG-ONLY AND both ensemble
verdicts are ≥ EXPLOIT-SURVIVES-CIG-ONLY. (Tier strings denote the
transplanted principles per the planner NAMING note; paper labels
carried/potential. The ensemble clause is mathematically implied by
≥6/8 under the ≥2/4 majority rule — stated for auditability, noted as
redundant. Base rate 37/40; the claim under test is the universal
carried leg, not the conjunction.) What this wave adds beyond
dataseed1/2: those were read exploratorily inside the sweep, and the
≥6/8 criterion postdates the 37/40 observation — this is the first
PROSPECTIVE test under frozen criteria, on fully novel env/planner
seeds and partially novel init seeds (6, 7).
**P-N2 (secondary, descriptive):** conjunction tier counts at the 3k
(peak-dose) point, reported against the sweep's dose curve; no
firing criterion (dose scoping already registered in C2).
Failure branch: if P-N1 does not fire, the universality claim is
restricted to the previously-read draws (data seeds 0–2) and the paper
reports the fresh-seed miss verbatim; no re-runs, no seed additions.
Hygiene gate: landed summaries must carry `stamp.git` equal to the
freeze commit (the idempotent summary guard would otherwise accept a
stale/foreign summary silently); bundle manifest-verified before read.

## 3. Registered analysis A — P1 (PRIME aleatoric-penalty re-score)

Data: `local_results/uncfield/p1_rescore/pilot2_m{0..3}.json` (recorded,
gates 0.0×4 on the authoring GPU; UNREAD — this section governs the
read). Reference: memo §6.2 spec + §6.4 pins.

- Scorer: per sense u = eig − λ·log(1 + max(ale,0)/σ₀²) — **ale ≤ 0
  CLAMPED to zero penalty (registered), and the zero-penalty-farming
  fraction reported as a finding** (census: m3 95/816 conjunction senses;
  negative implied noise = infinite license, invisible to the penalty).
- Grids: λ ∈ {0.1, 0.25, 0.5†, 1, 2} († deployed), σ₀² ∈ {0.5†, median
  genuine R_implied — defined as the pooled-across-members median of
  per-sense ale_model over the static sensors k ∈ {0,1,2,3,7}};
  **ale_model PRIMARY** for all λ* outcomes; ale_data secondary with the
  s5 process-noise conflation (2.15×R) disclosed.
- Gate variant (secondary): u_i = eig_i − λ·log(1 + [ale_i −
  0.5·Σ_{k=1..4} 0.95^{k−1}·eig_{i+k}]₊/σ₀²), look-ahead within the
  loop, wrapping at the loop boundary. **Registered prediction: the gate
  RE-ADMITS exploit reward** (its subtractor is the corrupted score).
- Outcomes (all both-outcomes-publish): surviving conjunction counts per
  (λ, σ₀²) per member (penalized-eig replaces the carried leg; dh leg
  unchanged); TV-loop λ*; the exploited duplicate loop
  `c447_1-4-1-4-1-4-1_all` λ*; legitimate-sensing λ*; **primary claim =
  the ORDERING (legitimate sensing dies before the duplicate exploit;
  predicted factor ≈ the Bayes-license ratio)**.
- **λ* definition (registered; u is linear in λ so closed-form):** per
  cycle, λ* = (steady carried-eig rate − EPS_PRED) / (steady penalty
  rate at λ=1), evaluated at σ₀² = 0.5 with ale_model (deliberately
  broader than the memo pin: ale_model primary for ALL λ* outcomes;
  ale_data reported secondary). TV λ* = on the member's
  top-carried-eig TV cycle (non-positive λ* reported as "not farming" —
  expected for the TV-coherent members m0/m2); duplicate λ* = on
  c447 (m1's exploit); legitimate-sensing λ* = median over
  neutral==False cycles containing ≥ 1 sense action. m1's 129
  negative-ale senses (all on non-neutral dynamic cycles) reported
  alongside m3's 95/816 in the zero-penalty-farming fraction.
- Carve-out (registered scope sentence): PRIME's epistemic estimator is
  non-transplantable (parameter-IG ≡ 0 under a frozen model; counts
  floor on every repeated walk) — estimand-substitution defenses are
  untested in the accounting space where the exploit lives, by
  construction of their estimand.

## 4. Registered analysis B — Bayes-license-ratio generalization (C4 gate)

Data: the RCC ratio bundle (`uncfield/ratio_research.py` outputs, 44
cells; instrument reviewed, B1 pick fix applied). **UNREAD — this
section governs the read**; landed JSONs must carry `stamp.git` equal to
the freeze commit and the bundle must manifest-verify before the read.
**Test (pre-specified): Spearman rank correlation (average-rank ties;
n_exploit_both is tie-heavy) between the per-member ratio (tier-0/1
target cycle) and n_exploit_both, across the 40 SWEEP cells only**
(pilot2's 4 cells excluded — the diagnostic was derived there).
Cell handling (registered): tier-2 cells (target None, trace None) are
EXCLUDED from the correlation and counted in the report;
heads-license-degenerate cells (trace present with mean_bayes_gain None
or < 1e-4) are likewise excluded and counted. (The no-senses branch is
structurally unreachable for tier-0/1 targets — carried-eig exploits
require senses — noted for completeness.) Minimum-N floor: if fewer
than 30 cells remain, C4 does not fire and the exclusion census is
reported instead. **C4 fires iff ρ ≥ 0.5 with one-sided permutation
p < 0.05 (10k permutations of the y-column within the included-cell
table; p = fraction with ρ_perm ≥ ρ_obs).** Design is within-pass
throughout (no cross-invocation pairing; cluster nondeterminism).

## 5. Registered follow-up (named, not promised)

hid128×{10k,30k} interaction arm (`--array=8-9`): reported
descriptively against C2's dose curve; no registered criterion.

## 5b. Registered scope sentences (defenses outside the battery)

**LPM (arXiv 2509.25438):** its monotone-in-IG guarantee conditions on
learning dynamics that a frozen-model protocol never runs; the
training-dose sweep is the closest control. Structurally inapplicable —
registered here so the scope is written down before a reviewer finds it.
**Learnable novelty (arXiv 2607.18433):** addressed analytically — a
learnability filter kills the TV exploit by construction (license = 0)
but not the duplicate exploit (genuinely learnable sensors; the defect
is double-counting); this split separates the two exploit classes.

## 6. The paper will NOT claim

Unqualified "survives current defenses"; CIG/PBIM as *implemented*
rather than transplanted-in-principle; any impossibility theorem for
coherence audits; "first referee-free diagnostic"; the term
"self-consistency ratio"; "scaling law" for three capacity points; a
noisy-TV claim covering all members (fires 2/4); the withdrawn anatomy-3
dichotomy; any cycle-level ratio discriminator; unbounded harvest
(telescoping cap stated: total potential-based harvest ≤ H(b₀) −
H_floor); the 1.4× lower bound inside the sane-heads sentence; LPM
addressed by the registered frozen-model-inapplicability sentence;
learnable-novelty addressed analytically (kills TV by construction,
not the duplicate class) — both per §5b.

## 7. Abridgment note + wording provenance

§1's C2/C3 texts are abridged from the frozen memo wordings; binding
specifics carried here verbatim where load-bearing: C2 dose curve = 0/4
@1k → 3/4 @3k → 1/4 @10k → 1/4 @30k gradient steps on 300×600 episodes,
capacity leg hid32 (sole NO-EXPLOIT member) vs hid128 (unanimous 4/4)
in one architecture family; C3 heads-sane support = implied observation
noise strictly positive in 0/32 accounts, localization "to the update
operator, not the scorer", TV = "pure fictitious information", m3 =
"1.4–1.8× update-side excess on top". Jobs note: "zero Python change"
(the sbatch JOBS list rides this freeze commit).
