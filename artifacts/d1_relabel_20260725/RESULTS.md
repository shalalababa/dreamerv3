# D1 relabel read — corrected NEGATIVE at 12 clusters; consequence leg fails (2026-07-25)

Snapshot: `local_results/d1_relabel_fix_20260725_090105/` (36 corrected
label passes, labeler `d1fix_20260724`, on the 6 shift pilots
`d1s_{cup,finger}_seed{21,22,23}` + the 6 Gate-D1 pilots
`d1pilot_{cup,finger}_e1_seed{1,2,3}`). ONE read execution over both
cohorts, exactly as registered (`prereg/PREREG_d1_relabel_20260724.md`
+ Amendment 1 cohort structure + Amendment 2 schema backfill), with the
pre-frozen `analysis/d1_shift_read.py` (selfcheck PASS).

## Pipeline integrity

- 36/36 label files present (18 d1s + 18 d1pilot), 200 states each;
  read asserts labeler_version `d1fix_20260724`; the 4 `_smoke.npz`
  files are skipped by the frozen filename rule.
- Amendment-2 backfill log: d1pilot passes backfilled ONLY registered
  keys — early passes the 7-key subset (checkout predating
  `orthreward.task`), later passes the full 8; never a foreign key.
  All backfilled keys are inert for dmc_proprio labeling (validated in
  the amendment). d1s passes needed NO backfill (23-Jul configs are
  schema-current) — consistent with the drift diagnosis.
- change_rate_real 0.755–0.905 across all 36 cells (healthy; no
  degenerate cell).

## Registered read (run-clustered bootstrap B=10K rng 0; fires iff CI > 0)

| family | arm | D_r | 95% CI | fires | t-interval (registered disclosure) |
|---|---|---|---|---|---|
| **PRIMARY (d1s, decisional)** | xpol | +0.243 | [−0.334, +0.767] | **NO** | [−0.553, +1.038] |
| **PRIMARY (d1s, decisional)** | phys | −0.189 | [−0.338, −0.060] | **NO** | [−0.391, +0.013] |
| replication (d1pilot) | xpol | +0.144 | [−0.107, +0.380] | NO | [−0.209, +0.497] |
| replication (d1pilot) | phys | +0.074 | [−0.220, +0.373] | NO | [−0.351, +0.499] |
| pooled-12 (headline magnitude) | xpol | +0.193 | [−0.108, +0.506] | NO | [−0.163, +0.550] |
| pooled-12 (headline magnitude) | phys | −0.058 | [−0.224, +0.131] | NO | [−0.268, +0.153] |

`fired_arms = []`, `fired_arms_replication = []`. Cross-cohort
consequence map (Amendment 1): **neither cohort fires ⇒ corrected
negative at 12 clusters.**

## Registered consequences applied

- The shift-consequence leg **fails ON THE CORRECTED INSTRUMENT**. The
  24-Jul defective-wave null is superseded; this negative carries the
  interpretive weight that one could not — the attenuation objection is
  discharged up to the design's power (t-intervals above bound it
  honestly). Route B rests on mechanism + boundary alone.
- Unchanged by registration: 24+24 stay frozen; Route A stays closed;
  imag stays retired; Gate-D1/R2/ladder conclusions remain
  instrument-invalidated (not re-adjudicated by this wave).

## Descriptive notes (never decisional)

- **phys d1s CI entirely negative** (−0.189 [−0.338, −0.060]): under
  the one-sided registered rule this does not fire; as description, the
  mass×1.3 shift REDUCED Δ_real in the d1s cohort — but the sign flips
  in the replication cohort (+0.074) and pools to null (−0.058), so no
  claim is made beyond "no rise anywhere."
- **Defect-impact panel** (d1s cohort, same checkpoints, old
  `shift_ext` vs new `d1fix`; distributional A/B — trajectories
  re-sample): base +0.200 → −0.010, phys +0.145 → −0.199, xpol
  +0.194 → +0.233; change_rate ~0.86 → ~0.78. The belief-corrupted
  follower systematically INFLATED base/phys purchase value; the
  corrected instrument moves the base cell to ~0.
- **Opportunity/achieved decomposition** (from `g_all`, `--oracle_all`;
  run-averaged means over states):

  | arm | domain | opportunity | achieved | implementation gap |
  |---|---|---|---|---|
  | base | cup | +0.944 | +0.139 | +0.805 |
  | base | finger | +1.147 | −0.185 | +1.333 |
  | phys | cup | +0.614 | −0.143 | +0.757 |
  | phys | finger | +1.032 | −0.018 | +1.049 |
  | xpol | cup | +1.960 | +0.161 | +1.799 |
  | xpol | finger | +6.272 | +0.180 | +6.093 |

  Oracle opportunity is large everywhere and BALLOONS under xpol
  (finger +6.3), while achieved value stays ≈ 0 in every cell — the
  consequence null is an implementation-gap fact, not an
  absence-of-opportunity fact. This is exactly the factorial R3
  (consumer competence: opportunity vs achieved value) is registered to
  separate; these numbers parameterize its design.
- C1-style calibration gates: all fail in both cohorts (spearman ≈ 0,
  no lift) — consistent with every prior wave; purchases remain
  uncalibrated at these maturities.
- Pre-existing design limit carried forward: finger 22/23 base cells'
  reward floor at 1e5-step pilots (disclosed in the base prereg).

## Provenance

- `d1_shift.json` — full read output (primary/replication/pooled +
  integrity + per-run values + calibration secondaries).
- Read command: `python -m analysis.d1_shift_read --labels
  <bundle>/runroot_light/d1shift_local/d1_labels_fix --output <dir>`.
- Defective `d1_labels/` remains preserved untouched as provenance;
  old-wave record at `artifacts/d1_shift_20260724/`.
