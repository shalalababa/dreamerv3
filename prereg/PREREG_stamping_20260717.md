# Synthetic reward stamping registration (frozen 2026-07-17)

Registers the stamping experiment (execution plan
`research_notes/Plan_Stamping_20260717.md`; theory prediction **P-B1
strong form**, `prereg/PREREG_theory_predictions_20260717.md`) **before
any stamped fit or adapt outcome exists**. Adjudicates
"task-aligned-direction inclusion" (spectral-competition theory) vs
"any stable scalar supervision shapes the trunk" (generic
objective-shaping / subspace expansion). Registered claim scope
(editorial framing, `Research_Branch_Ideas_Triage_20260717.tex` §3.1):
*at fixed data and behavior, stable scalar supervision acting only
through the WM trunk changes downstream transfer independently of
exploration and actor–critic learning* — explicitly differentiated from
UNREAL (auxiliary pseudo-rewards trained jointly with RL) and RaMP
(random cumulants for zero-shot value transfer).

## Design

Finger q1 pair only (the confirmed-interaction domain), seeds 1–8,
mirror of the P3 transform arms: full task objective
(`expl.mode=task, reward_grad=T, repval_loss=T, repval_grad=T`) on
relabeled copies of the frozen buffers, offline fit 500K updates, then
frozen-readout adaptation — identical stages, flags, QC, and naming to
`PREREG_p3_gradient_path_20260714.md`.

| arm | buffer | labels | learnable? | aligned? |
|---|---|---|---|---|
| srd0 | `q1_srd0` | frozen random MLP g₀(obs), thresholded | yes | no |
| srd1 | `q1_srd1` | frozen random MLP g₁(obs) (2nd draw) | yes | no |
| sid  | `q1_sid`  | state-independent exact-count placement | **no** | no |

**Label-marginal matching** (implemented in
`probing/relabel_replay.py` kinds `stamp_rand`/`stamp_iid`, selfcheck
PASS): per side, the stamped-frame COUNT equals that side's true
rewarded-frame count exactly (top-n by g for srd; uniform without
replacement for sid — exact-count/hypergeometric placement rather than
literal Bernoulli, so density is matched exactly), and the stamp
amplitude equals that side's mean positive true reward. The stamp label
second moment a² = s²f therefore matches the true labels' by
construction on both sides; realized densities/amplitudes/thresholds
are recorded in each transform manifest. g = fixed 64×2-tanh MLP on the
standardized flattened obs vector (pooled standardization over both
sides), weights from `--fn_seed` (srd0→0, srd1→1); manifests record
enough to reproduce the labels bit-for-bit
(selfcheck-verified). Deterministic given seeds.

Two function draws give function-draw variance (S3); sid is the
learnability/stability control — same marginals, nothing
state-dependent to learn.

Jobs: 3 buffers × 2 sides × 8 seeds = 48
(`AXIS1_TRANSFORM=srd0|srd1|sid ./scripts/submit_all.sh
axis1-factorial-bundles`; run ids `adapt_ax1<code>q1s<side>_finger_*`,
WM `ax1wm_finger_<code>q1s<side>_seed<k>`).

## Outcomes and decision rule

B_arm(k) = AUC100k(s1) − AUC100k(s0) within seed k, from the untouched
`analysis/adaptation_auc.py`; **B_srd(k) = mean(B_srd0(k), B_srd1(k))**.
All CIs: cluster-bootstrap percentile 95% (B=10,000, numpy default_rng
seed 0); decisions on the CI alone; sensitivity suite (paired t, exact
sign-flip permutation, Wilcoxon, d_z, LOO) robustness-only. Read script
frozen pre-outcome: `analysis/stamping_read.py` (selfcheck PASS,
recovers both branches on synthetic data).

- **PRIMARY (sole confirmatory): mean over seeds 1–8 of
  [B_srd − B_sid]** (both arms new, contemporaneous, paired by seed —
  no historical-record dependence).
  - CI includes 0 ⇒ **P-B1 strong form stands**: learnable-but-unaligned
    scalars do not transfer; Paper-1 discussion gains "not any scalar —
    aligned scalars".
  - CI > 0 ⇒ **strong form refuted**: subspace-expansion amendment to
    the theory (Theory_SpectralTransfer §1.6); stamping becomes a
    constructive-method candidate; triggers the alignment-graded
    follow-up wave.
  - CI < 0 ⇒ outside both registered accounts; audit before
    interpreting; reported as-is.
- **Named secondaries (descriptive comparators against disclosed-known
  historical records, never decision-bearing):**
  S1 = B_srd − B_apt (P0 corrective paired JSON, seeds 1–8) — "does
  stamping beat nothing"; S2 = B_srd − B_sh (P3 wave read JSON) —
  learnable-unaligned vs unlearnable-density-preserving;
  S3 = B_srd0 − B_srd1 (function-draw heterogeneity; expected ≈ 0; a
  large S3 flags function-draw luck and demotes the pooled reading).
- **Mechanism panel (E4-style, descriptive-only, no decision role):**
  reward-head NLL of the srd fits against (a) the STAMPED labels
  (`probing/relabel_replay.py stamp-probeset` →
  `probing/stratified_error.py measure --reward_override`; outputs land
  in `e4_<probeset>_ov-*` dirs, never the registered E4 dirs) — low
  NLL = the stamp direction was included (membership happened); and
  (b) the TRUE labels (standard e4-measure pass) — expected high. The
  theory's signature outcome is **inclusion-without-transfer**: low
  stamp-NLL + null primary; this makes even the null informative.
- Audit (per-run, registered rule): saved WM config flags = full-arm
  row; ADAPT_DONE; fit stdout (where retained) shows
  `Static replay: .../q1_<code>/side<side>`, contains rew, no AC keys;
  QC ≥20 eps ≤100K, same as P3.

## Scope statements and risks (registered)

- The stamp is a function of the single frame's obs; it is trivially
  predictable at h=0 but NOT Markov-consistent with imagination
  rollouts at h>0 the way true rewards are. This is fine for the fit
  (the reward head trains on frames) and is a stated scope limit of the
  arm, not a defect.
- sid uses exact-count placement (not literal Bernoulli) — registered
  here as the intended control; it removes density variance from the
  contrast.
- Fallback (kill criterion from the plan): the exact-count
  implementation matches marginals exactly by construction, so the
  "within 1%" fallback is moot; if a side had zero positive frames the
  tool refuses and a dated amendment would be required (finger q1 has
  0.05/0.32 rewarded-frame fractions — not applicable).
- The 33 h bundles of 3 mirror P3; no shared run-ids with any other
  wave (Amendment-1 runs are `ax1rgo/ax1sgb` seeds 9–16).

## Disclosure and ordering

Known at freeze: every DreamerV3 outcome through 17 Jul 2026 (P0, W0,
W1/W2/W3, P3 factorial + E4 panel, lo-side battery, Goodhart, TD-MPC2
cross-family read) — including that sh (+31.4) and rl (+52.8) behave as
they do, which motivated this design. Unknown: every stamped outcome —
no srd0/srd1/sid buffer has ever been fitted or adapted; at freeze the
transformed buffers may exist on disk (a CPU relabeling of frozen
inputs, deterministic given seeds, produces no outcome information).
Amendment-1 (rgo/sgb seeds 9–16) outcomes are also unknown and this
registration does not depend on them. Ordering: this file, the
transform code, the whitelist change, and `analysis/stamping_read.py`
are committed together BEFORE any stamped fit job is submitted; the
E4 `--reward_override` extension is likewise pre-outcome
(instrument-extension code note to be added to `analysis/DEVIATIONS.md`
at commit).
