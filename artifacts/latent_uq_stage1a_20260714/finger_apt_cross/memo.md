# Latent-UQ Stage 0 evidence memo — 2026-07-14

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `apt_finger_seed1`, `apt_finger_seed2`, `apt_finger_seed3`, `apt_finger_seed4`, `apt_finger_seed5` (cross-ensemble members)
- Probe set(s): finger_v1_7a32ff08 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)
- Instrument: **density_residual** (out-of-fold isotonic residual on the density proxy, 5 stream-cluster folds — Stage-1A primary, PREREG_gate_d1_stage1a_20260714.md)

## Cell: finger_apt_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 99072 | anchor | 1 | +0.482 (0.023) | -0.143 (0.029) | +0.618 | -0.461 | +1.079 (0.042) | TRACKS_ERROR |
| 99072 | path | 1 | +0.482 (0.023) | -0.143 (0.029) | +0.618 | -0.461 | +1.079 (0.042) | TRACKS_ERROR |
| 99072 | anchor | 5 | +0.219 (0.023) | -0.143 (0.026) | +0.311 | -0.266 | +0.577 (0.036) | TRACKS_ERROR |
| 99072 | path | 5 | +0.449 (0.019) | -0.108 (0.029) | +0.548 | -0.367 | +0.915 (0.034) | TRACKS_ERROR |
| 200896 | anchor | 1 | +0.555 (0.016) | -0.155 (0.029) | +0.674 | -0.480 | +1.154 (0.031) | TRACKS_ERROR |
| 200896 | path | 1 | +0.555 (0.016) | -0.155 (0.029) | +0.674 | -0.480 | +1.154 (0.031) | TRACKS_ERROR |
| 200896 | anchor | 5 | +0.270 (0.020) | -0.155 (0.025) | +0.357 | -0.285 | +0.643 (0.034) | TRACKS_ERROR |
| 200896 | path | 5 | +0.519 (0.016) | -0.109 (0.028) | +0.606 | -0.381 | +0.987 (0.029) | TRACKS_ERROR |
| 300352 | anchor | 1 | +0.580 (0.017) | -0.138 (0.027) | +0.677 | -0.447 | +1.124 (0.032) | TRACKS_ERROR |
| 300352 | path | 1 | +0.580 (0.017) | -0.138 (0.027) | +0.677 | -0.447 | +1.124 (0.032) | TRACKS_ERROR |
| 300352 | anchor | 5 | +0.277 (0.025) | -0.138 (0.026) | +0.352 | -0.264 | +0.616 (0.035) | TRACKS_ERROR |
| 300352 | path | 5 | +0.525 (0.017) | -0.114 (0.031) | +0.609 | -0.379 | +0.988 (0.032) | TRACKS_ERROR |
| 399808 | anchor | 1 | +0.577 (0.015) | -0.150 (0.023) | +0.677 | -0.454 | +1.131 (0.035) | TRACKS_ERROR |
| 399808 | path | 1 | +0.577 (0.015) | -0.150 (0.023) | +0.677 | -0.454 | +1.131 (0.035) | TRACKS_ERROR |
| 399808 | anchor | 5 | +0.296 (0.023) | -0.150 (0.025) | +0.376 | -0.283 | +0.660 (0.031) | TRACKS_ERROR |
| 399808 | path | 5 | +0.528 (0.017) | -0.120 (0.030) | +0.613 | -0.384 | +0.997 (0.031) | TRACKS_ERROR |
| 499264 | anchor | 1 | +0.617 (0.017) | -0.142 (0.028) | +0.709 | -0.462 | +1.172 (0.031) | TRACKS_ERROR |
| 499264 | path | 1 | +0.617 (0.017) | -0.142 (0.028) | +0.709 | -0.462 | +1.172 (0.031) | TRACKS_ERROR |
| 499264 | anchor | 5 | +0.311 (0.029) | -0.142 (0.027) | +0.389 | -0.281 | +0.670 (0.033) | TRACKS_ERROR |
| 499264 | path | 5 | +0.557 (0.020) | -0.123 (0.028) | +0.645 | -0.408 | +1.053 (0.031) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/finger_apt_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell finger_apt_cross --dumps apt_finger_seed1=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_finger_seed1/latent_uq/finger_v1 apt_finger_seed2=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_finger_seed2/latent_uq/finger_v1 apt_finger_seed3=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_finger_seed3/latent_uq/finger_v1 apt_finger_seed4=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_finger_seed4/latent_uq/finger_v1 apt_finger_seed5=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_apt_finger_seed5/latent_uq/finger_v1 --horizons 1 5 --n_boot 300 --instrument density_residual --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/finger_apt_cross`
- git commit: `c92f984794e7f1b57db82ccfc3f8faae3c65df52`
