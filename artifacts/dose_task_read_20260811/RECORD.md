# READ RECORD — #21 task-arm dose wave — 2026-08-11

- Registration: `prereg/PREREG_dose_task_20260810.md` (frozen 183bb7c,
  review 1B/11M applied). Reader `analysis/dose_task_read.py` bitwise ≡
  the frozen copy (sha `0251271c…d9d7` = bundle `code_sha256.txt`);
  selfcheck PASS pre-execution. ONE execution.
- **Theory timing**: `prereg/PREREG_theory_R1_gating_20260811.md`
  (sha256
  `98015404ca386a3af53ae99c8b9961a8ce5cd1f175e2e26a22f6a87f3c9d5bb8`)
  was frozen THIS SESSION before any file of the dose bundle was
  opened; its #21 predictions (A1–A6) are graded below.
- Inputs: bundle `dose_task_20260811_203441` (sha-manifest verified, 0
  non-OK): auc.csv + e4_dose.csv + dose_manifest.json (sha recorded in
  artifact) + fit_counters.json + ckpt_steps.json + the pre-fit
  `leak_overlap.json` (M10). Gates all passed: occupancies realized
  0.0536/0.1305/0.2490/0.3234 (within ±0.02 of nominal), dual fit
  witness (counters + measured ckpt steps), modal n_ep 96, milestone
  filter, zero QC / sub-modal exclusions, n=8 per level.

## Verdict (frozen reader, verbatim)

**GRID-UNINFORMATIVE: the known occupancy effect does not reproduce on
dose-constructed buffers AT THIS POWER; P-DR1/P-DR2 reported with zero
decision weight.**

- **G1 replication gate**: level3 − level0 = **+157.6, BCa
  [+25.8, +309.5], perm p = .0663** — CI excludes 0 but the p-conjunct
  fails (gate = CI>0 ∧ p<.05). Registered M12 context applies
  verbatim: G1 power ~0.73 at a true +100 — "a G1 failure is only
  interpretable AT THIS POWER."
- Cell means (zero-weight descriptive): 146.5 → 282.5 → 273.1 → 304.1.
  Interior contrasts: level1 +136.0 (p=.072), level2 +126.6 (p=.0258 —
  misses BH crit .025 by .0008); bh_survivors = []. P-DR1 label "LATE"
  carries no weight.
- **P-DR2 shape (zero weight)**: step wins the CV race NON-decisively
  (margin .036 < calibrated crit .079) → TIE-UNDISCRIMINATED; no shape
  wording licensed.
- **P-DR3 membership (registered descriptive)**: E4 rew-NLL member
  band (≤1.5, anchor 0.827): levels 0–1 OUT (1.52 / 1.62), levels 2–3
  IN (0.99 / 1.04) — the membership transition sits between occ 0.130
  and 0.249. (Bookkeeping: membership panel counts n=9 at level 1 —
  one extra fit row in the E4 csv vs the 8 adapt rows; descriptive
  panel only, noted.)
- D1-leak overlap (M10 disclosure): tiny — 0 / 11 / 5 / … episodes per
  level of 200 members; pre-fit sha recorded.

## R1 addendum grading (PREREG_theory_R1_gating A1–A6)

| Pred | Outcome |
|---|---|
| **A1** (G1 passes) | **FAILS at the registered letter** — the gate's p-conjunct missed (.0663). Disclosed nuance: the point estimate (+157.6) exceeds the +100 the power note contemplated; the miss is a power event the prereg itself pre-declared interpretable only "at this power". The scratch stands on the scoreboard. |
| **A2** (monotone, strong) | UNADJUDICATED — zero decision weight; no activation wording licensed. |
| **A3** (no decisive step win) | UNADJUDICATED (zero weight); descriptively consistent — step's win was non-decisive. |
| **A4** (concave lean, weak) | UNADJUDICATED — TIE-UNDISCRIMINATED. |
| **A5** (scope note) | No grading (registered note). |
| **A6** (membership rises with f, descriptive) | Descriptively coherent — out-of-band → in-band with rising occupancy. |

## Consequence

The wave is instrument-clean and power-starved: every gate that failed
did so within its pre-declared MDE limits, with the G1 point estimate
nearly twice the interaction the power basis assumed. The natural next
step (decision, not licensed here) is a **P3-Amendment-1-style power
extension**: seeds 9–16 at all four levels (32 fits + 32 adapts),
pooled n=16 gates, which also gives P-DR2's shape machinery its
registered power (~STEP power 45% was quoted at n=8). Until then, #21
licenses NO dose wording, and R1's #21 predictions remain
unadjudicated except A1's recorded miss.
