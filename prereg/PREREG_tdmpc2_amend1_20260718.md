# TD-MPC2 Amendment 1 — seeds 9–16 extension, pooled n=16 re-test, and P-C1 (frozen 2026-07-18)

Amends `PREREG_tdmpc2_crossfamily_20260716.md` after its registered
seeds-1–8 read (17 Jul: PRIMARY +9.5 [−43.2, +51.4] did NOT fire ⇒
family-scope limitation recorded; per-seed sd ≈ 72 ⇒ only ±50
detectable at n=8). Mirrors the P3-Amendment-1 pattern: a powered
pooled re-test, frozen **before any seed-9–16 TD-MPC2 job exists**.
Execution plan: `research_notes/Plan_TDMPC2_FamilyBoundary_Exec_20260718.md`
(stages F0+F1); theory derivation:
`research_notes/Theory_SpectralTransfer_Addendum_DecoderFree_20260718.tex`.

## Design (unchanged except seeds)

Seeds 9–16 × {aware, free} × {s0, s1} = 32 jobs. Identical to the
registered wave in every other respect: pinned TD-MPC2 commit
`e9f5932...`, model size 5, arm flags (free: `reward_coef 0,
value_coef 0`; aware: official 0.1/0.1; `consistency_coef 20` both),
frozen finger q1 buffer pair via the same bridge manifests, offline fit
500K updates → frozen-representation adapt 125K steps, run naming
`tm2wm_finger_<arm>q1s<side>_seed<k>` / `adapt_tm2<arm>q1s<side>_finger_seed<k>_ckpt500000`,
AUC by the untouched `analysis/adaptation_auc.py`, QC unchanged.
Submit: `TM2_SEEDS="9 10 11 12 13 14 15 16" SLURM_TIME=12:00:00
./scripts/submit_all.sh tdmpc2-bundles` (no plumbing change needed).

## Read (frozen script: `analysis/tdmpc2_amend_read.py`, committed pre-outcome)

Machinery identical to the frozen family: B_arm(k) = AUC100k(s1) −
AUC100k(s0); cluster-bootstrap percentile 95% CI, B=10,000,
`default_rng(0)`; decisions on CI alone; sensitivity suite (paired t,
exact sign-flip permutation, Wilcoxon, d_z, LOO) robustness-only.
Seeds 1–8 rows are taken from the frozen record
(`artifacts/tdmpc2_goodhart_d1pilot_20260717/auc.csv`); any 1–8 rows in
the new csv must bit-match it (assert), exactly as in the P3 amendment.

- **PRIMARY (decision CI): pooled seeds 1–16 [B_aware − B_free].**
  - CI > 0 ⇒ the cross-family interaction is **present-but-attenuated**;
    the 17-Jul family-scope limitation is revised to "attenuated in the
    decoder-free family" and the external-validity leg is restored in
    weakened form (band ledger: TD-MPC2 leg ✓-attenuated).
  - CI spans 0 ⇒ the limitation **stands** at doubled power; headline
    language remains restricted to reconstruction-based world models.
  - CI < 0 ⇒ outside all registered accounts; audit before interpreting.
- **Fresh-batch panel:** seeds 9–16 alone, reported as
  replication/winner's-curse control (as in P3 Amendment 1 and W0).
  The seeds-1–8 read of 17 Jul remains the registered record for n=8.
- **Named secondaries:** pooled B_aware simple effect; pooled B_free
  simple effect (both CI-bearing for P-C1 below); descriptive cell
  means; never compared numerically to DreamerV3 AUC (frozen rule).

## P-C1 — decoder-free-limit predictions (theory registration)

Derivation (addendum Props. A1–A2): with reconstruction weight α → 0,
the inclusion criterion βa²λ_j > α(g_k − λ_j)+ is satisfied for ANY
nonzero rewarded-frame fraction ⇒ membership is no longer contested in
the aware arm (both sides include the reward direction; the side
contrast is estimation gradation, not a threshold), and free-arm
subspace selection is a lottery (partial, seed-noisy inclusion) rather
than a variance ranking that deterministically excludes.

The plan's provisional absolute-scale anchors (point in [0,+40); CI vs
the DreamerV3 +51.5) are REJECTED at this freeze: they would violate
the frozen never-compare-numerically rule of the base registration.
Registered forms are within-family or dimensionless:

- **P-C1a (share attenuation, directional):** pooled
  [B_aware − B_free] / B_aware point estimate **< 0.6**. DreamerV3
  analog share at n=16 ≈ 1.04 (W0 pooled: +95.0 task, −3.7 apt);
  disclosed n=8 TD-MPC2 share ≈ 0.27. Dimensionless cross-family
  comparison is permitted ("contrast structure compared
  qualitatively"); 0.6 is a registered convention, midway between the
  disclosed value and the DreamerV3 share.
- **P-C1b (free-arm positivity, CI-bearing — the sharp one):** pooled
  B_free CI > 0 — the OPPOSITE of the DreamerV3 reward-free null
  (apt −3.7 tight), uniquely predicted by the lottery mechanism.
  Trichotomy: **lands** (CI > 0); **unresolved** (CI spans 0, point
  ≥ +10); **fails** (CI spans 0, point < +10 — no evidence at doubled
  power ⇒ consistency-only selection also excludes the reward
  direction, refuting the indifference proposition's relevance).
- **P-C1c (aware positivity):** pooled B_aware CI > 0 (directionally
  known at n=8, +35.3 [+4.8,+69.6]; prospective through seeds 9–16).

**Adjudication:** P-C1 LANDS iff C1a ∧ C1b-lands ∧ C1c; REFUTED iff
C1b-fails; otherwise PARTIAL. P-C1 would be the third adjudicated
prediction of the spectral-competition theory (after P-A1 landed,
P-B1 landed).

## Gate G-F2 (resource gate; non-inferential)

Per the execution plan: the E4-style mechanism port over TD-MPC2
checkpoints proceeds by default after this read; it is blocked only if
the read is INCOHERENT (fresh batch 9–16 sign-contradicts seeds 1–8 on
the primary AND the pooled aware simple effect collapses to a CI
spanning 0 with point < +10). This gate allocates build effort only; it
never alters how the present read is interpreted.

## Disclosure and ordering

Known at freeze: all DreamerV3 outcomes through 18 Jul 2026 (incl. the
stamping read — P-B1 stands — and the Gate-D1 fail), and the TD-MPC2
seeds-1–8 outcomes in full (+9.5 primary; +35.3 aware; +25.8 free;
aware ≈ 2.2× free levels; per-seed sd ≈ 72). The pooled and
fresh-batch quantities are prospective only through seeds 9–16:
**no seed-9–16 TD-MPC2 fit or adapt exists** (no
`tm2*_seed(9|1[0-6])` logdir anywhere; the only TD-MPC2 outputs are
the 32 registered seed-1–8 runs and the discarded protocol smoke).
The direction of every P-C1 component is an informed prediction from
disclosed n=8 data; the test is prospective through the new seeds.
Ordering: this file + `analysis/tdmpc2_amend_read.py` + the theory
addendum are committed BEFORE any seed-9–16 submission (no AI
co-author trailer, per standing repo rule).
