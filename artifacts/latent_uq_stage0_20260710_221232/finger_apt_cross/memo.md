# Latent-UQ Stage 0 evidence memo — 2026-07-10

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `apt_finger_seed1`, `apt_finger_seed2`, `apt_finger_seed3`, `apt_finger_seed4`, `apt_finger_seed5` (cross-ensemble members)
- Probe set(s): finger_v1_7a32ff08 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: finger_apt_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 99072 | anchor | 1 | +0.706 (0.029) | +0.567 (0.019) | +0.613 | +0.392 | +0.221 (0.048) | TRACKS_ERROR |
| 99072 | path | 1 | +0.706 (0.029) | +0.567 (0.019) | +0.613 | +0.392 | +0.221 (0.048) | TRACKS_ERROR |
| 99072 | anchor | 5 | +0.492 (0.024) | +0.567 (0.018) | +0.339 | +0.455 | -0.117 (0.035) | TRACKS_DENSITY |
| 99072 | path | 5 | +0.656 (0.022) | +0.619 (0.016) | +0.555 | +0.500 | +0.055 (0.039) | AMBIGUOUS |
| 200896 | anchor | 1 | +0.755 (0.017) | +0.532 (0.023) | +0.704 | +0.399 | +0.304 (0.029) | TRACKS_ERROR |
| 200896 | path | 1 | +0.755 (0.017) | +0.532 (0.023) | +0.704 | +0.399 | +0.304 (0.029) | TRACKS_ERROR |
| 200896 | anchor | 5 | +0.519 (0.021) | +0.532 (0.021) | +0.409 | +0.427 | -0.018 (0.036) | AMBIGUOUS |
| 200896 | path | 5 | +0.702 (0.017) | +0.594 (0.019) | +0.644 | +0.504 | +0.140 (0.024) | TRACKS_ERROR |
| 300352 | anchor | 1 | +0.765 (0.017) | +0.499 (0.026) | +0.727 | +0.384 | +0.344 (0.030) | TRACKS_ERROR |
| 300352 | path | 1 | +0.765 (0.017) | +0.499 (0.026) | +0.727 | +0.384 | +0.344 (0.030) | TRACKS_ERROR |
| 300352 | anchor | 5 | +0.507 (0.025) | +0.499 (0.023) | +0.407 | +0.396 | +0.011 (0.038) | AMBIGUOUS |
| 300352 | path | 5 | +0.699 (0.020) | +0.580 (0.020) | +0.647 | +0.496 | +0.152 (0.026) | TRACKS_ERROR |
| 399808 | anchor | 1 | +0.763 (0.015) | +0.446 (0.023) | +0.726 | +0.301 | +0.425 (0.033) | TRACKS_ERROR |
| 399808 | path | 1 | +0.763 (0.015) | +0.446 (0.023) | +0.726 | +0.301 | +0.425 (0.033) | TRACKS_ERROR |
| 399808 | anchor | 5 | +0.505 (0.022) | +0.446 (0.024) | +0.415 | +0.332 | +0.083 (0.038) | TRACKS_ERROR |
| 399808 | path | 5 | +0.707 (0.018) | +0.529 (0.023) | +0.656 | +0.425 | +0.231 (0.030) | TRACKS_ERROR |
| 499264 | anchor | 1 | +0.806 (0.013) | +0.408 (0.029) | +0.780 | +0.258 | +0.522 (0.033) | TRACKS_ERROR |
| 499264 | path | 1 | +0.806 (0.013) | +0.408 (0.029) | +0.780 | +0.258 | +0.522 (0.033) | TRACKS_ERROR |
| 499264 | anchor | 5 | +0.530 (0.022) | +0.408 (0.027) | +0.452 | +0.281 | +0.172 (0.042) | TRACKS_ERROR |
| 499264 | path | 5 | +0.739 (0.016) | +0.515 (0.026) | +0.695 | +0.406 | +0.290 (0.034) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/finger_apt_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell finger_apt_cross --dumps apt_finger_seed1=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_finger_seed1/latent_uq/finger_v1 apt_finger_seed2=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_finger_seed2/latent_uq/finger_v1 apt_finger_seed3=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_finger_seed3/latent_uq/finger_v1 apt_finger_seed4=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_finger_seed4/latent_uq/finger_v1 apt_finger_seed5=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_finger_seed5/latent_uq/finger_v1 --horizons 1 5 --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/finger_apt_cross`
- git commit: `d102ac4fd5cf27d13eae50171d4dacc56b18d79e`
