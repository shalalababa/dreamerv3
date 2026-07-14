# Latent-UQ Stage 0 evidence memo — 2026-07-14

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `random_finger_seed1`, `random_finger_seed2`, `random_finger_seed3`, `random_finger_seed4`, `random_finger_seed5` (cross-ensemble members)
- Probe set(s): finger_v1_7a32ff08 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)
- Instrument: **density_residual** (out-of-fold isotonic residual on the density proxy, 5 stream-cluster folds — Stage-1A primary, PREREG_gate_d1_stage1a_20260714.md)

## Cell: finger_random_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 96208 | anchor | 1 | +0.240 (0.035) | -0.141 (0.041) | +0.544 | -0.518 | +1.062 (0.035) | TRACKS_ERROR |
| 96208 | path | 1 | +0.240 (0.035) | -0.141 (0.041) | +0.544 | -0.518 | +1.062 (0.035) | TRACKS_ERROR |
| 96208 | anchor | 5 | +0.168 (0.032) | -0.141 (0.039) | +0.375 | -0.364 | +0.739 (0.039) | TRACKS_ERROR |
| 96208 | path | 5 | +0.304 (0.031) | -0.131 (0.038) | +0.556 | -0.502 | +1.058 (0.032) | TRACKS_ERROR |
| 199120 | anchor | 1 | +0.248 (0.042) | -0.187 (0.039) | +0.692 | -0.682 | +1.374 (0.025) | TRACKS_ERROR |
| 199120 | path | 1 | +0.248 (0.042) | -0.187 (0.039) | +0.692 | -0.682 | +1.374 (0.025) | TRACKS_ERROR |
| 199120 | anchor | 5 | +0.192 (0.043) | -0.187 (0.037) | +0.514 | -0.512 | +1.026 (0.037) | TRACKS_ERROR |
| 199120 | path | 5 | +0.292 (0.040) | -0.179 (0.037) | +0.657 | -0.632 | +1.289 (0.026) | TRACKS_ERROR |
| 302032 | anchor | 1 | +0.234 (0.042) | -0.169 (0.033) | +0.671 | -0.659 | +1.330 (0.033) | TRACKS_ERROR |
| 302032 | path | 1 | +0.234 (0.042) | -0.169 (0.033) | +0.671 | -0.659 | +1.330 (0.033) | TRACKS_ERROR |
| 302032 | anchor | 5 | +0.191 (0.045) | -0.169 (0.036) | +0.499 | -0.493 | +0.992 (0.041) | TRACKS_ERROR |
| 302032 | path | 5 | +0.298 (0.041) | -0.163 (0.035) | +0.657 | -0.627 | +1.283 (0.030) | TRACKS_ERROR |
| 396384 | anchor | 1 | +0.256 (0.046) | -0.134 (0.034) | +0.681 | -0.660 | +1.341 (0.033) | TRACKS_ERROR |
| 396384 | path | 1 | +0.256 (0.046) | -0.134 (0.034) | +0.681 | -0.660 | +1.341 (0.033) | TRACKS_ERROR |
| 396384 | anchor | 5 | +0.192 (0.040) | -0.134 (0.033) | +0.468 | -0.451 | +0.920 (0.044) | TRACKS_ERROR |
| 396384 | path | 5 | +0.325 (0.038) | -0.102 (0.030) | +0.638 | -0.586 | +1.224 (0.034) | TRACKS_ERROR |
| 499312 | anchor | 1 | +0.208 (0.045) | -0.176 (0.032) | +0.678 | -0.672 | +1.350 (0.029) | TRACKS_ERROR |
| 499312 | path | 1 | +0.208 (0.045) | -0.176 (0.032) | +0.678 | -0.672 | +1.350 (0.029) | TRACKS_ERROR |
| 499312 | anchor | 5 | +0.159 (0.042) | -0.176 (0.034) | +0.476 | -0.480 | +0.956 (0.036) | TRACKS_ERROR |
| 499312 | path | 5 | +0.312 (0.043) | -0.122 (0.036) | +0.648 | -0.606 | +1.254 (0.028) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/finger_random_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell finger_random_cross --dumps random_finger_seed1=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_finger_seed1/latent_uq/finger_v1 random_finger_seed2=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_finger_seed2/latent_uq/finger_v1 random_finger_seed3=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_finger_seed3/latent_uq/finger_v1 random_finger_seed4=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_finger_seed4/latent_uq/finger_v1 random_finger_seed5=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_finger_seed5/latent_uq/finger_v1 --horizons 1 5 --n_boot 300 --instrument density_residual --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/finger_random_cross`
- git commit: `c92f984794e7f1b57db82ccfc3f8faae3c65df52`
