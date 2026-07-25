# Pixel X2 read — interaction does NOT fire; G-X3 NO-GO (2026-07-24)

Snapshot: `local_results/pixel_x2_20260724_213538/` (32 fit+adapt
jobs: {task, apt} × {s0, s1} × seeds 1–8, finger, pixel_wm 1m, 500K
updates, collector-matched pxq1m pair). Read exactly as registered in
`prereg/PREREG_pixel_repl_20260719.md` with the pre-frozen
`analysis/pixel_repl_read.py` (selfcheck PASS).

## Pipeline integrity

- Local read on the snapshot csv is IDENTICAL to the cluster-side
  json (every key equal; `pixel_repl.json` here is that object).
- Config audit CLEAN 32/32: `model_obs: image` + correct `expl.mode`
  on every wm run.
- QC: frozen `adaptation_auc` rule asserted on every consumed row.
  (Bundle exclusions.csv lists 3 UNRELATED `ax1(f)s12*` Scaling-B
  adapt runs missing scores.jsonl — flagged to the runner; not pixel
  rows.)
- Pair provenance: `pair_manifest.json` (collector_l1 = 0 by
  construction; side occs per the X1 fill record).

## Registered read (B=10K seed bootstrap rng 0; decision on CI alone)

| quantity | mean | 95% CI | notes |
|---|---|---|---|
| **PRIMARY interaction (AUC100k)** | **+3.6** | **[−29.9, +40.9]** | **does NOT fire** (perm p=.867, d_z=.06, LOO range [−12.0, +13.3]) |
| task simple s1−s0 | +5.4 | [−19.6, +36.0] | null |
| apt simple s1−s0 | +1.8 | [−20.7, +22.5] | null |

P-C2 (directional secondary): **not evaluated** — its premise (a
fired primary) never engaged. Registered consequence: **G-X3
distractor-amplification NO-GO**; the registered next step is the
reconstruction-swamping diagnostic (registered same day:
`prereg/PREREG_pixel_swamping_20260724.md`).

## Descriptive context (not decisional)

Cell means, AUC100k: task s0 77.2±5.7 / s1 82.6±12.4; apt s0
81.1±10.4 / s1 82.9±6.8. All four cells sit on a common low floor —
the pattern is not "interaction absent atop working transfer" but
"no arm and no side adapts fast at this budget." Same-family proprio
cells at the same adapt protocol run far higher, so the pixel WMs'
frozen features support much slower readout learning in every cell
(descriptive cross-modality remark; no pooled statistic).

## Interpretation under the theory (mechanism-testable, not assumed)

The bar framework (Theory_CompCapacity_Addendum_20260724: bars
G0 < G1 < G2 in g-space) allows exactly this pattern when pixel
reconstruction pushes the competition bar ABOVE G2: every cell —
including task-hi — fails inclusion, so all four cells floor and the
interaction is zero. That is the "extreme-g_k" escape the X2 prereg
itself registered pre-outcome (§5 decision tree). It is now a
TESTABLE claim, not a save: the swamping diagnostic asks whether the
task arm's reward-NLL LEVEL matches the apt arm's (no membership
anywhere ⇒ swamping-consistent) or separates the way rgo≪sgb did in
proprio (membership present without behavioral transfer ⇒ the
membership→transfer link itself fails in pixels — a genuine theory
problem). Registered forms in the swamping prereg.

Adjacent registered prediction bookkeeping: the spectral pass's
out-of-sample restatement (pixel ⇒ LARGER interaction,
PREREG_spectral_domains_20260724 §out-of-sample, frozen before this
read) FAILS in its simple monotone form; the non-monotone
(swamping) regime is the surviving branch and is what the diagnostic
adjudicates. Recorded as such in the ledger.

## Paper-1 consequence (registered boundary either way, per prereg)

The pixel leg lands as a **registered scope boundary**: the
occupancy × supervision interaction is established for
proprio/reconstruction WMs at 1m/500K; at the same capacity and
budget the pixel variant shows a uniform adaptation floor. The
swamping diagnostic decides which sentence the paper writes:
"pixels raise the competition bar past every arm" (mechanism
continuous with the theory) vs "pixel membership exists without
transfer" (boundary of the membership→transfer link).

## Provenance

- `pixel_repl.json` — full read output (primary/simples/P-C2/audit).
- `auc.csv` — canonical AUC rows (pixel rows consumed; s12 rows out
  of scope here).
- `pair_manifest.json` — X1 pair audit object.
- Read command: `python -m analysis.pixel_repl_read --auc <csv>
  --audit_runroot <runroot> --output <dir>`.
