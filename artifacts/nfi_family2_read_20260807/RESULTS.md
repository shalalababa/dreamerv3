## ⚠ AMENDMENT (9 Aug 2026) — drift-baseline artifact in the conjunction statistic

Read through this note first (full basis: reviews/Review_FullRecord_and_Draft_20260809.md
§§2,14,15 + artifacts/nfi_review_response_20260809/). (1) FAVORABLE: family 2's
conjunction statistic is drift-clean — all 5 conjunction cycles are identical on raw
and adjusted ΔH (drift credits ~1e-4), verified independently, and the dc_gru m1 cell
was end-to-end reproduced from ensemble.pkl at clean HEAD (names/neutral bitwise,
rates at float64 round-off). (2) WINDOW SENSITIVITY (9 Aug re-score, review's
recommended test): all 5 conjunctions REPRODUCE at the registered window (dev ≤1e-7)
but hold at **0/5 in the early window (loops 2–5) and 0/5 in the late window (loops
37–40)** — the realized-ΔH conjunction is a mid-imagination-depth transient here too.
Combined with the family-1 drift finding: **the study has no window-robust
realized-ΔH claim in either family**; the conjunction is a window-scoped descriptive.
The CARRIED leg's window behavior is member-heterogeneous: top-5 carried exploits
persist at loops 37–40 in 5/8 dc members (all four dc_lstm + dc_gru m3; dc_lstm m2's
rates are bit-stable across all windows) and from loop 2 in 3/8. (3) COMPLIANCE: the
prereg's §5 review record states "a dirty or wrong-commit stamp fails the G-STAMP
gate," but the frozen reader never implemented the dirty check, and all three stamps
carry dirty:true — under the registered gate as worded this read should have HALTED.
Remediated by `prereg/PREREG_nfi_family2_amend1_20260809.md` (gate implemented; dirty
discharged by the registered compensating control = clean-HEAD model-level
reproduction) and a re-read under the amendment. P-F2's verdict is unchanged by the
remediation (it was verified correct); the process defect is recorded, not hidden.
REMEDIATION COMPLETE (9 Aug, same day): the compensating control executed
device-matched (JAX_PLATFORMS=cpu; the first, GPU, attempt deviated at 1.68e-6 —
the documented jax device effect — and the gate correctly HALTED on it, disclosed
in the amendment) — dc_gru m1 and dc_lstm m3 reproduce end-to-end from ensemble.pkl
at max rate dev 5.33e-15 with names/neutral bitwise and verdicts + conjunction
counts exact (`reproduction.json`); the re-read under the amended gate passes all
gates and reproduces every verdict of this record identically.

---

# NFI Family-Generality Factorial — Registered Read (CONFIRMATORY)

**Date:** 7 Aug 2026. Read executed exactly per
`prereg/PREREG_nfi_family2_20260807.md` (freeze commit `8f6b4de8`) by the
frozen reader `uncfield/family2_read.py` (sha `1671dcae…`, byte-identical
to the freeze). Machine record `reads.json` (this dir).

**Provenance.** Bundle `local_results/uncfield/family2/` manifest-verified
(21 files) and byte-identical to the committed copy
`manifests/uncfield_family2_20260807.sha256`. Two-commit provenance
(disclosed, per the confirm-wave precedent): jobs executed at `da800742`,
an ancestor-verified descendant of the freeze with an **empty decision-path
diff** (`git diff 8f6b4de8 da800742 -- uncfield/
scripts/uncfield_family2.sbatch` = 0 lines). All three stamps: git
`da80074`, pinned thresholds (0.02/0.08/30/8/4/ml) — G-STAMP PASS.
**Disclosure:** all stamps carry `dirty: true` from the RCC checkout —
expected from untracked scratch/job files, but the flag cannot
retroactively distinguish untracked noise from modified tracked files on
the remote; mitigation = the empty decision-path diff above plus G-REPRO
(every member verdict and every ensemble verdict re-derived locally from
the raw npz rate columns by the frozen reader — all 12 + 3 match).
G-DATA PASS (dc_gru and dc_lstm episodes_sha identical, `e73563c9…`;
lg_lstm trained on the pilot2 anchor, `b7388406…`, 300 episodes).

## 1. P-F2 (PRIMARY) — **FIRES, 3/3 cells, 12/12 members**

Every new cell's ensemble verdict is EXPLOIT-SURVIVES-CIG-ONLY, and every
individual member (4 lg_lstm + 4 dc_gru + 4 dc_lstm) is ≥
EXPLOIT-SURVIVES-CIG-ONLY. The frozen fire-branch wording is licensed:

> The carried-accounting exploit is not an artifact of one world-family
> or one belief architecture: at a single inherited threshold set,
> planners farm referee-certified evidence-neutral cycles under
> prefix-conditioned accounting in a linear-Gaussian world with GRU and
> LSTM belief models, and in a discrete-chain world — categorical
> observations, exact discrete-Bayes referee, exactly-zero-gain baits —
> with GRU and LSTM belief models: four cells, two exact-referee
> families, two architectures, no retuning.

Per-member cig-exploit counts: lg_lstm 263/100/205/205; dc_gru
51/16/10/7; dc_lstm 13/194/16/60.

## 2. P-F2b (secondary) — does NOT fire; the STRONGER-RESULT branch

The conjunction-location prediction missed in BOTH directions, exactly as
the pre-specified branches anticipated:

- **dc steady conjunction: 2/8 members** (predicted 0/8). dc_gru m1
  (member verdict EXPLOIT-SURVIVES-CIG+PBIM, 2 cycles) and dc_lstm m3
  (CIG+PBIM, 3 cycles). Per the §1 pre-specified interpretation (review
  B1), **this is the flagship signature in the new family — the
  conjunction (carried-EIG AND realized-ΔH farming on the same cycle)
  reappears in a bounded-Shannon-entropy world on referee-certified
  EXACT-ZERO cycles**: dc_gru m1 farms `c600_…only_d7` (true 0.0, eig
  0.254, dh_adj 0.150 nats/loop) and the noisy-TV cycle `c543_…only_d6_tv`
  (true 0.0, eig 0.093, dh_adj 0.106 — the coin-flip channel farmed
  through BOTH accounting legs); dc_lstm m3 farms three duplicate-pair
  cycles (`only_d3`, true 0.0, eig ≈0.104–0.110, dh_adj ≈0.098–0.100).
- **lg_lstm steady conjunction: 0/4 members** (predicted ≥1, from
  GRU's 3/4 at this dose). LSTM in the LG world is eig-only at 3k steps
  — the conjunction's mid-training dose curve is
  architecture-dependent, not just world- and dose-dependent.
- **Transient tier: zero everywhere** (n_transient = n_transient_conj =
  0 for all 12 members). The bounded-entropy transient-shift prediction
  was simply wrong: where dh-farming appears in dc, it appears in the
  STEADY tier, consistent with the review-B1 arithmetic (a ≥EPS_PRED
  steady rate is sustainable for ≈69 loops against the 38-loop window —
  boundedness does not bind here).

Reported as a registered miss with the stronger-result interpretation, no
salvage wording beyond the pre-specified branch.

## 3. Registered descriptives

- **tv_max** (noisy-TV farming): lg_lstm 0.201/0.062/0.788/0.257 — the
  LSTM family-1 members include the strongest TV farmer measured in the
  study (m2, 0.788 nats/loop predicted on a zero-information channel);
  dc members 0.002–0.137, plus the dc_gru m1 TV **conjunction** above.
- **top_claimed_cum vs the 5.545-nat dc budget**: all dc top-cycle
  cumulative claims are small (|·| ≤ 0.77) — no member exceeds the world
  entropy budget; the conjunction members farm via within-window
  entropy cycling, not runaway contraction (the bounded-world absurdity
  exhibit did not materialize; consistent with §2's steady-tier
  finding).
- Raw n_exploit_pbim / n_exploit_transient counts carry the registered
  move-only-drift contamination caveat; every conjunction quantity above
  uses the eig-conjunct forms that exclude it.

## Consequence

The generality leg of the NFI paper is complete and confirmatory: the
carried-accounting exploit — the paper's universality claim — holds in
every cell of the 2×2 factorial with no retuning, and the flagship
conjunction is no longer a single-world phenomenon (2 dc members farm
certified exactly-zero cycles through both defenses' principles).
Writing-phase note: the conjunction's appearance pattern across cells
(GRU-LG 3/4, LSTM-LG 0/4, dc 2/8, all at one dose) is
member-heterogeneous in every family — present the carried leg as
universal and the conjunction as seed/architecture/dose-scoped, per the
frozen C1/C2 wordings.
