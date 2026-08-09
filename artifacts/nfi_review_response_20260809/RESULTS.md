# NFI/CEI Review Response — verifications, new analyses, remediation (9 Aug 2026)

Response record for `research_notes/paper5_uncertainty_field/reviews/
Review_FullRecord_and_Draft_20260809.md` (12-lens review of the full
record + v1 draft). Everything below was recomputed in this session from
the committed bundles/arrays; scripts inline in the session transcript,
key outputs in this file. Companion actions: amendment notes prepended to
5 artifact records; `uncfield/confirm_read.py` (retroactive confirmatory
reader) built; `prereg/PREREG_nfi_family2_amend1_20260809.md` (G-STAMP
dirty gate) registered + implemented + re-read; Theory_CEI v3.2.

## 1. Review findings VERIFIED independently (my recomputation)

**1a. Drift-baseline artifact (review §1) — CONFIRMED.** Raw vs adjusted
conjunction counts, pilot2 anchor (neutral & carried>0.08 & dh{,_adj}>0.08):

| member | adjusted (published) | RAW | raw ΔH ≤ 0 among adjusted | median credit |
|---|---|---|---|---|
| m0 | 13 | **0** | 0 | +0.062 |
| m1 | 61 | **0** | 0 | +0.054 |
| m2 | 0  | 0   | 0 | 0 |
| m3 | 61 | **1** | **52** | +0.141 |

The family-1 flagship conjunction is manufactured by the negative-drift
credit (median credits ≈ or above the 0.08 threshold). The carried leg
uses no correction and is untouched.

**1b. Family 2 drift-clean (review §2) — CONFIRMED.** dc_gru m1: 2
conjunctions raw AND adjusted; dc_lstm m3: 3 and 3; all other members 0/0.

**1c. Capacity magnitudes (review §3) — CONFIRMED.** Mean per-member
conjunction counts: hid32 [0,168,0,0]=42.0; hid64 [13,61,0,61]=33.8;
hid128 [5,13,11,28]=14.2 — magnitude runs DOWN as capacity rises; only
tier membership (1/4→3/4→4/4) rises, seed-confounded at one cell/level.

**1d. CEI near-optimality (review §11) — CONFIRMED.** Exact one-sided
power at m=628 is 0.943 (not ≥0.95); the exact 0.95-power fixed-sample
test needs m=653; Wald SPRT E[N]≈339 = 1.65× the BH bound, and the
fixed-sample audit is 1.93× the SPRT — real fixed-vs-sequential
inefficiency inside Thm 3's own auditor class. v3.1's "within 4% of the
true optimum" is RETRACTED in Theory_CEI v3.2 (correction #15; notable:
it was itself review #13's suggested upgrade — reviewer-derived fixes
also need verification).

## 2. NEW: window sensitivity — the realized-ΔH leg is a transient everywhere

Family-2's five conjunction cycles (the drift-clean ones), re-scored at
shifted windows (registered values reproduce to ≤1e-7 first):

| cycle (member) | registered rawΔH / eig | early (loops 2–5) | late (loops 37–40) |
|---|---|---|---|
| c600 only_d7 (dc_gru m1) | +0.150 / +0.254 | −0.000 / +0.017 | +0.000 / +0.036 |
| c543 only_d6_tv (dc_gru m1) | +0.106 / +0.093 | +0.000 / +0.009 | +0.007 / −0.002 |
| c210 only_d3 (dc_lstm m3) | +0.100 / +0.106 | −0.034 / −0.092 | +0.000 / +0.054 |
| c620 only_d3 (dc_lstm m3) | +0.098 / +0.104 | −0.040 / −0.094 | +0.000 / +0.042 |
| c196 only_d3 (dc_lstm m3) | +0.098 / +0.110 | −0.024 / −0.078 | +0.000 / +0.055 |

**Conjunction holds 0/5 early, 0/5 late.** With §1a: the study has NO
window-robust realized-ΔH claim in either family — the conjunction is a
mid-imagination-depth transient and must be presented as a window-scoped
descriptive, exactly the review's fallback branch ("the paper has a
realized-ΔH result nowhere and should say so").

**The carried leg's window behavior is member-heterogeneous, not
uniformly transient.** Top-5 carried exploits per dc member re-scored:
persist (>0.08) at loops 37–40 in **5/8 members** (all four dc_lstm +
dc_gru m3; dc_lstm m2's rates are bit-stable across all three windows —
true steady-state farming; dc_gru m3's late rates exceed its registered
ones) and from loop 2 in 3/8 (dc_lstm m1/m2/m3); dc_gru m0/m1's top
exploits decay. Family-1 anchors' carried leg was shown window-robust by
the review (0.25→0.19, 0.46→0.41).

## 3. NEW: the analyses that earn the title

**3a. Predicted-vs-true ranking (pilot2; verified = review §4).** Ranking
all 626 cycles by carried rate: top-1 cycle is referee-certified
FICTITIOUS in m0/m2/m3 with the entire top-10 evidence-free (top-50:
49/47/49 of 50); m1 is the honest exception (best fictitious rank 19).
**True-information regret of the top pick: ≈1.04 nats/loop foregone** in
the three fictitious-top members (vs available best true rate).

**3b. Untrained-filter control (random-init, deploy scale, NEW).**
lg GRU untrained: CIG-ONLY (96 cig, 0 dh/both/transient, raw≡adjusted,
neutral-eig median −0.010, max +0.795). dc GRU untrained: CIG-ONLY
(34 cig, 0 elsewhere, median +0.005, max +0.142). **Tier membership
alone does NOT separate trained from untrained filters** — the
carried-tier verdict is reachable by pure init noise. The trained-
specific facts are: systematic structure (member/dose/architecture
regularities, prospective 8/8 replication, m1's coherent
informative-first ranking), head-sanity (the licensed-gain anatomy —
untrained heads are incoherent everywhere), and magnitude concentration
on designed baits. Any "emerges from training" implication must be
scrubbed; the honest form is "training does not remove it" (37/40 at
10× dose).

**3c. Pooled membership headline.** 68/72 members ≥ CIG-ONLY across all
18 ensembles = 0.944, Wilson 95% [0.866, 0.978] (vs 8/8's [0.63, 1.00]).

**3d. EPS_PRED sensitivity (carried counts, raw).** No cliff anywhere:
pilot2 m1 253→211 as eps 0.08→0.16; dc members retain exploits at 2×
threshold (e.g. dc_gru m0 51→22, dc_lstm m1 194→94).

## 4. Remediation ledger (compliance findings, review §§14–15)

- **Confirm-wave frozen reader RETRO-BUILT** (`uncfield/confirm_read.py`,
  selfcheck PASS): re-executes every registered decision of
  `PREREG_nfi_confirm_20260802` from the landed bundles and hard-asserts
  the recorded outcomes. **ALL REPRODUCED**: P-N1 8/8; P1 grids to the
  unit ([13,61,0,61] / [7,39,0,22] / gate [13,61,0,42] / m3@λ2 13);
  σ₀²_med 0.050833; censuses 95/816 + m1 129 (amendment-corrected
  values); C4 ρ=−0.020893, N=30, exclusions 3+7, perm p 0.5477 (rng(0),
  10k — matches the recorded p exactly); λ* dup/tv to 5e-4. One
  definitional nuance pinned: λ*_legit = 0.170 under the registered
  last-4-loop steady (the recorded 0.176 was the all-8-loop/excluding-
  non-farming variant; ordering claim unaffected at ~4×).
- **G-STAMP dirty gate**: registered as
  `PREREG_nfi_family2_amend1_20260809`, implemented in `family2_read.py`,
  compensating control (clean-HEAD end-to-end model reproduction of
  dc_gru m1 + dc_lstm m3) executed → `reproduction.json` in the read
  record; read re-run under the amended gate (verdicts unchanged).
- **Reader hardening**: run_read-level fixture battery kills the
  review's surviving mutants (CIG_LEVEL primary flip, all→any, every
  gate deletion, dirty-gate bypass); G-REPRO's boundary stated in the
  amendment (cannot catch remote rate-computation mutations — that is
  the reproduction control's job).
- **Record errors amended** (prepended notes): reads.json 95/4080 →
  95/816; m1's 129 negative-ale senses restored; "everywhere" (5/30
  cells ≤1); p1_rescore pre-freeze-ancestor stamp disclosure; the
  sign-inverted drift sentence.
- **Raw-ΔH recovery wave BUILT** (`uncfield/raw_research.py` +
  `scripts/uncfield_raw.sbatch`, selfcheck reproduces pilot2 m0 exactly,
  census both adj/raw = 13/0): 14 RCC array tasks re-derive every sweep
  member verdict from the model (matched-device verdict_match gate —
  also closes review §15's "P-N1 rests on runner-written verdict
  strings" residual) and ship the per-cycle npz the bundles lack;
  descriptive, awaits submission; raw tables fold in here as a dated
  addendum on landing.
- **Theory v3.2**: near-optimality retraction + gap decomposition
  (recommend the sequential audit); E3/E4 footnote units fix; hazard
  exponent → ≈4–5 (CI [4.1,5.9], ℓ+1=4 not excluded); general-n ψ
  formula; Thm 4(b)/(c) scope fixes; notation notes.

## 5. Consequence — the claim hierarchy for the restructured manuscript

1. FLAGSHIP (carried leg, registered + prospective + cross-family):
   planners reliably discover referee-certified evidence-neutral cycles
   — including exactly-zero channels in the discrete family — whose
   prefix-conditioned predicted gain stays far above threshold; top-1
   ranked cycle fictitious in 3/4 anchor members with ≈1 nat/loop true
   regret; 8/8 prospective, 12/12 factorial, pooled 68/72 [0.866,0.978];
   survives 10× training; attenuated-not-removed by the aleatoric
   penalty whose gate re-admits it.
2. SCOPED: the conjunction (realized-ΔH ∧ carried) is a
   mid-imagination-depth, member-heterogeneous transient: drift-credit
   artifact in family 1 (raw 0/0/0/1), drift-clean but window-local in
   family 2 (5 cycles, 2/12 members, 0/5 at shifted windows).
3. CONTROLS/honesty: untrained filters reach the tier (3b) — claim
   "training does not remove", never "training creates"; capacity =
   "more seeds exploitable, not larger exploits"; dose peak descriptive
   (n=4); C4 null uninformative (power 0.26–0.50); TV λ* negative
   restored (deployed penalty kills the TV class, not the duplicate
   class).
