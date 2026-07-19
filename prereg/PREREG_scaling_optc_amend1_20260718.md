# Option C Amendment 1 — occ-matched corrective cells (frozen 2026-07-18)

Amends the Option C rider after its 18-Jul read
(`artifacts/scaling_optc_20260718/RESULTS.md`; deviation entry in
`analysis/DEVIATIONS.md`): the plain `search` does not enforce
side0=low and the v400 pair came out inverted, so the registered
"volume ladder on the hi side" was not realized — the run cells
confounded volume with occupancy (v400s1 = occ .094/400 eps vs v200s1
= occ .323/200 eps) and P-B4 was not adjudicated. This amendment
registers the 12 corrective jobs on the already-built, never-run
`q1v400/side0` (occ .290, 400 eps, same pool/search as v400s1)
**before any v400s0 outcome exists**.

## Jobs (12)

{task, apt} × seeds 1–6 on `axis1_finger/q1v400/side0`. Run ids
`adapt_ax1v4q1v400s0_finger_seed<k>` (task) /
`adapt_ax1fv4q1v400s0_finger_seed<k>` (apt) — the standard submit
loop's side-0 branch; existing side-1 logdirs make resubmission skip
s1 automatically:

```
AXIS1_ID_PREFIX=ax1v4  AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger \
  AXIS1_QUADS=q1v400 AXIS1_SEEDS="1 2 3 4 5 6" ./scripts/submit_all.sh axis1-bundles
AXIS1_ID_PREFIX=ax1fv4 AXIS1_EXPL_MODE=apt  AXIS1_DOMAINS=finger \
  AXIS1_QUADS=q1v400 AXIS1_SEEDS="1 2 3 4 5 6" ./scripts/submit_all.sh axis1-bundles
```

## Registered read (frozen script `analysis/optc_amend_read.py`, committed pre-outcome)

AUC100k, untouched pipeline; existing v200s1/v400s1 rows come from the
archived `artifacts/scaling_optc_20260718/auc.csv` (bit-checked on
overlap); cluster-bootstrap percentile 95% CI, B=10,000,
`default_rng(0)`; seed-paired contrasts; decisions on CI alone;
sensitivity suite robustness-only. Task-arm contrasts are decisional;
apt analogs descriptive.

- **C1 (fixed-volume occupancy contrast): task [v400s0 − v400s1]**
  (occ .290 vs .094, same 400-ep pool/search).
  - CI > 0 ⇒ occupancy still binds at 400 episodes.
  - CI spans 0 ⇒ occupancy saturates below .094 at this volume (the
    threshold moved with volume — consistent with an
    estimation-floor account).
  - CI < 0 ⇒ the lo-side advantage is real ⇒ composition (not
    occupancy, not volume) drives the 18-Jul anomaly; audit episode
    mixtures before interpreting.
- **C2 (≈fraction-matched volume contrast, the P-B4 test the rider
  intended): task [v400s0 − v200s1]** (occ .290/400 eps vs .323/200
  eps; fraction ≈ matched, count ×2; composition differs — stated
  limit).
  - CI > 0 ⇒ volume matters at ≈fixed fraction ⇒ **P-B4
    fraction-not-count strong form REFUTED**.
  - CI spans 0 ⇒ the v400s1 anomaly was not volume ⇒ P-B4 stands,
    18-Jul anomaly attributed to composition/occupancy mixture.
  - CI < 0 ⇒ unexpected; audit.
- Descriptive: apt analogs of C1/C2 (expected ≈ floor throughout);
  cell means; v400s1 anomaly re-stated against the new cells.

## Disclosure and ordering

Known at freeze: every outcome through 18 Jul incl. the full Option C
read (v200s1 204±91, v400s1 433±243, apt cells, side inversion).
Unknown: every v400s0 number (the side0 buffer was built during the
rider's build step and has never been fitted or adapted). Ordering:
this file + `analysis/optc_amend_read.py` are committed BEFORE the 12
jobs are submitted.
