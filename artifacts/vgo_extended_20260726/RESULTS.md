# vgo extended-fit discriminator — does NOT fire; vgo absence-like (2026-07-26)

Snapshot: `local_results/vgo_extended_20260726_211521/` (16 fresh
1.5M-update fits + standard frozen-readout adapt: {vgo, sgb} × seeds
1–8 × s1 only; `ax1x*` naming; the frozen 500K fits untouched). Read
exactly as registered in `prereg/PREREG_vgo_extended_20260723.md`
with the pre-frozen `analysis/vgo_extended_read.py` (selfcheck PASS).
This was the LAST open Paper-1 experimental leg.

## Pipeline integrity

- 32 runroot entries (16 WM fits + 16 adapts) complete; read asserts
  milestone 1,500,000 per run; 500K baselines from the committed
  `auc_p3.csv` (known at freeze, disclosed).
- Bundle read code md5-identical to the frozen repo version; local
  read IDENTICAL to the cluster-side json on every key.
- exclusions.csv lists only in-flight volume-replication rows
  (`ax1v2q1v200s0` fresh seeds — the P-E4a wave already queued;
  unrelated to this read).

## Registered read (paired by seed, cluster bootstrap B=10K rng 0)

| quantity | mean | 95% CI | notes |
|---|---|---|---|
| **PRIMARY: [vgo₁.₅M − vgo₅₀₀K] − [sgb₁.₅M − sgb₅₀₀K]** | **−2.2** | **[−35.2, +40.2]** | **does NOT fire** (perm p=.93, d_z=−.04, LOO [−20.8, +5.4]) |
| vgo extension gain | +5.8 | [−15.7, +27.8] | generic-sized, ns |
| sgb extension gain (generic control) | +8.0 | [−20.3, +28.3] | ns |
| levels | vgo 81.4→87.3, sgb 74.4→82.4 | — | both arms drift up equally |

## Registered consequence applied

**P-B2's attenuation form is NOT supported at 3× updates**: the
value-gradient path, though replay-grounded (case (b), wiring audit),
produces no differential growth — vgo's extension gain is
indistinguishable from the no-representation-path control's generic
gain. vgo stays **absence-like** in the tested range; P-B2 keeps only
its weakest reading; **no further extended-fit arms without new
theory** (frozen). For the paper: the discriminator is reported
run-and-negative — the value-head route to legibility is not merely
slow, it is inert at the budgets and scales tested, which sharpens the
headline's mechanism claim (reward-prediction gradients, specifically,
are the sufficient carrier — value gradients are not, even with 3× the
time to act).

## Paper-1 program status

With this read the experimental program is COMPLETE: evidence chain +
mechanism + all four band conditions + scale-robustness + all boundary
legs (family, pixel, domain, objective) + this final discriminator.
No open Paper-1 experiment remains; the paper moves to writing
(outline ~1 Sept per plan; the volume replication and 25m point in
flight belong to Paper 3).

## Provenance

- `vgo_extended.json` — full read output.
- `auc.csv` — canonical AUC rows consumed (`ax1x*` modes).
- Read command: `python -m analysis.vgo_extended_read --auc <csv>
  --output <dir>`.
