# Paper-3 confirmatory 25m point — ONE read record (2026-08-01)

Registration `PREREG_compcapacity_confirm_20260726.md`. The read was
EXECUTED CLUSTER-SIDE by the staged wave driver
(`meta/COMMANDS_ALREADY_RUN.txt`: `analysis.adaptation_auc` →
`analysis.compcapacity_read --auc .../auc.csv`); this record adopts
that execution as the ONE read. Provenance verified locally: the
bundle's `meta/code_sha256.txt` shas for `analysis/compcapacity_read.py`
(166532e3…), `analysis/adaptation_auc.py` (b3acaa00…), and
`prereg/PREREG_compcapacity_confirm_20260726.md` (f8004750…) all match
the local checked-out frozen files byte-for-byte. Completion:
`done=32/32 partial=0 missing=0`; auc rows QC-pass. Bundle:
`local_results/compcapacity_25m_full_20260801_214345/` (scores-only by
sync policy — small size is by design, sufficient for this read).
`compcapacity_read.json` here = verbatim copy of the executed output.

## Registered outcome: PRIMARY DOES NOT FIRE (bounded null)

- **Primary task-lo rise (25m−12m, s0 task):** −12.2 [−45.7, +17.7]
  (t-sensitivity agrees). Verdict (verbatim): "regime A persists
  through 25m — the transition bracket widens (G1 crossing above 25m);
  reported as a bounded null with the E4 membership counterpart; no
  ordering violation."
- **Guard (forbidden pattern): NOT triggered** — apt rise +7.0
  [−10.8, +25.3]; P-E1's ordering is not refuted.
- Secondaries: the composition interaction PERSISTS at 25m
  (+113.8 [+68.9, +155.4], 8/8 positive; b_task 25m +119.2*); task-hi
  change −32.2 ns; apt simple at 25m +5.4 ns. Cell means: task s1
  203.9 / task s0 84.7 / apt ≈ 98–103 (25m).

Reading: 25m does not un-null task-lo — the capacity transition sits
above 25m, so the Paper-3 bracket is [12m, >25m] on this axis; the
composition headline is capacity-robust through 25m. The E4 membership
pass (`ax1wm_finger_*s25*`) was still RUNNING at sync — it feeds the
separate theory adjudication (`PREREG_compcapacity_theory_20260724`
P-E2/P-E3), not this confirmatory read (which consumes auc.csv only,
per the frozen CLI).

## E4 membership counterpart (registered DESCRIPTIVE, landed 2 Aug)

Bundle `local_results/e4_25m_20260801_225152/` (csv copied here);
provenance verified: `scripts/e4_measure.sbatch` + 
`probing/stratified_error.py` shas match the local checked-out files
byte-for-byte; RCC git head `782d8ed5` is a committed ancestor on
`causal-wm-transfer`. 32 runs × 4 horizons = 128 rows, complete;
frozen `finger_v1` probe set.

Registered descriptive form (prereg §E4: "A→B predicts task@25m,s0
reward-NLL moves from the 12m value 5.77 toward the membership band"):

- **task@25m,s0 in-regime rew-NLL (h=0): mean ≈ 8.06 nats, per-seed
  3.77–10.81** (seeds 1–8: 4.64, 10.67, 10.81, 10.19, 6.75, 8.96,
  3.77, 8.70) vs the 12m reference 5.77
  (`artifacts/orthogonal_obj_20260725/`). The membership form did NOT
  move toward the band — descriptively it sits slightly further out.
  **Coherent corroboration of the bounded null: regime A persists
  through 25m on BOTH metrics** (AUC flat and membership not
  approaching) — the two-point capacity result is metric-consistent.
- Apt rows: `reward_aware=0`, rew-NLL absent-by-design under the
  reward_aware gate — exactly as disclosed at 12m; the gate behaves
  as registered.
- Descriptive only, per registration: no CI, no fire, no decision
  rule touches these rows. The theory adjudication
  (`PREREG_compcapacity_theory_20260724` P-E2/P-E3) remains a
  separate future read (Scaling-B completeness blocker still open).
