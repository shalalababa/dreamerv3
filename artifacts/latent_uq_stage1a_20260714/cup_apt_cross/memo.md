# Latent-UQ Stage 0 evidence memo — 2026-07-14

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `apt_cup_seed1`, `apt_cup_seed2`, `apt_cup_seed3`, `apt_cup_seed4`, `apt_cup_seed5` (cross-ensemble members)
- Probe set(s): ball_in_cup_v1_9327e960 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)
- Instrument: **density_residual** (out-of-fold isotonic residual on the density proxy, 5 stream-cluster folds — Stage-1A primary, PREREG_gate_d1_stage1a_20260714.md)

## Cell: cup_apt_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 99600 | anchor | 1 | +0.225 (0.043) | -0.272 (0.037) | +0.681 | -0.691 | +1.372 (0.028) | TRACKS_ERROR |
| 99600 | path | 1 | +0.225 (0.043) | -0.272 (0.037) | +0.681 | -0.691 | +1.372 (0.028) | TRACKS_ERROR |
| 99600 | anchor | 5 | +0.152 (0.044) | -0.272 (0.035) | +0.536 | -0.569 | +1.105 (0.032) | TRACKS_ERROR |
| 99600 | path | 5 | +0.224 (0.051) | -0.281 (0.040) | +0.659 | -0.672 | +1.331 (0.031) | TRACKS_ERROR |
| 200608 | anchor | 1 | +0.131 (0.025) | -0.344 (0.025) | +0.639 | -0.685 | +1.325 (0.023) | TRACKS_ERROR |
| 200608 | path | 1 | +0.131 (0.025) | -0.344 (0.025) | +0.639 | -0.685 | +1.325 (0.023) | TRACKS_ERROR |
| 200608 | anchor | 5 | +0.015 (0.025) | -0.344 (0.025) | +0.448 | -0.543 | +0.992 (0.030) | TRACKS_ERROR |
| 200608 | path | 5 | +0.120 (0.027) | -0.325 (0.026) | +0.592 | -0.640 | +1.232 (0.028) | TRACKS_ERROR |
| 299904 | anchor | 1 | +0.213 (0.028) | -0.321 (0.028) | +0.667 | -0.692 | +1.359 (0.027) | TRACKS_ERROR |
| 299904 | path | 1 | +0.213 (0.028) | -0.321 (0.028) | +0.667 | -0.692 | +1.359 (0.027) | TRACKS_ERROR |
| 299904 | anchor | 5 | +0.120 (0.027) | -0.321 (0.029) | +0.538 | -0.594 | +1.132 (0.036) | TRACKS_ERROR |
| 299904 | path | 5 | +0.194 (0.028) | -0.297 (0.029) | +0.619 | -0.645 | +1.265 (0.032) | TRACKS_ERROR |
| 399200 | anchor | 1 | +0.377 (0.025) | -0.310 (0.033) | +0.739 | -0.722 | +1.461 (0.018) | TRACKS_ERROR |
| 399200 | path | 1 | +0.377 (0.025) | -0.310 (0.033) | +0.739 | -0.722 | +1.461 (0.018) | TRACKS_ERROR |
| 399200 | anchor | 5 | +0.262 (0.027) | -0.310 (0.034) | +0.601 | -0.617 | +1.218 (0.021) | TRACKS_ERROR |
| 399200 | path | 5 | +0.330 (0.028) | -0.295 (0.034) | +0.676 | -0.666 | +1.342 (0.020) | TRACKS_ERROR |
| 498512 | anchor | 1 | +0.354 (0.026) | -0.320 (0.024) | +0.716 | -0.707 | +1.422 (0.020) | TRACKS_ERROR |
| 498512 | path | 1 | +0.354 (0.026) | -0.320 (0.024) | +0.716 | -0.707 | +1.422 (0.020) | TRACKS_ERROR |
| 498512 | anchor | 5 | +0.256 (0.028) | -0.320 (0.026) | +0.584 | -0.606 | +1.189 (0.032) | TRACKS_ERROR |
| 498512 | path | 5 | +0.316 (0.027) | -0.315 (0.025) | +0.657 | -0.657 | +1.314 (0.025) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/cup_apt_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell cup_apt_cross --dumps apt_cup_seed1=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_cup_seed1/latent_uq/cup_v1 apt_cup_seed2=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_cup_seed2/latent_uq/cup_v1 apt_cup_seed3=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_cup_seed3/latent_uq/cup_v1 apt_cup_seed4=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_cup_seed4/latent_uq/cup_v1 apt_cup_seed5=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_cup_seed5/latent_uq/cup_v1 --horizons 1 5 --n_boot 300 --instrument density_residual --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/cup_apt_cross`
- git commit: `c92f984794e7f1b57db82ccfc3f8faae3c65df52`
