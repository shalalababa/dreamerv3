# Latent-UQ Stage 0 evidence memo — 2026-07-14

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `random_cup_seed1`, `random_cup_seed2`, `random_cup_seed3`, `random_cup_seed4`, `random_cup_seed5` (cross-ensemble members)
- Probe set(s): ball_in_cup_v1_9327e960 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)
- Instrument: **density_residual** (out-of-fold isotonic residual on the density proxy, 5 stream-cluster folds — Stage-1A primary, PREREG_gate_d1_stage1a_20260714.md)

## Cell: cup_random_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 101056 | anchor | 1 | +0.181 (0.024) | -0.141 (0.026) | +0.622 | -0.615 | +1.237 (0.050) | TRACKS_ERROR |
| 101056 | path | 1 | +0.181 (0.024) | -0.141 (0.026) | +0.622 | -0.615 | +1.237 (0.050) | TRACKS_ERROR |
| 101056 | anchor | 5 | +0.101 (0.022) | -0.141 (0.025) | +0.469 | -0.477 | +0.946 (0.072) | TRACKS_ERROR |
| 101056 | path | 5 | +0.181 (0.024) | -0.112 (0.029) | +0.581 | -0.569 | +1.150 (0.063) | TRACKS_ERROR |
| 200304 | anchor | 1 | +0.138 (0.028) | -0.182 (0.028) | +0.557 | -0.566 | +1.123 (0.051) | TRACKS_ERROR |
| 200304 | path | 1 | +0.138 (0.028) | -0.182 (0.028) | +0.557 | -0.566 | +1.123 (0.051) | TRACKS_ERROR |
| 200304 | anchor | 5 | +0.046 (0.032) | -0.182 (0.029) | +0.397 | -0.428 | +0.825 (0.075) | TRACKS_ERROR |
| 200304 | path | 5 | +0.142 (0.028) | -0.140 (0.028) | +0.511 | -0.511 | +1.022 (0.067) | TRACKS_ERROR |
| 299568 | anchor | 1 | +0.353 (0.026) | -0.156 (0.034) | +0.680 | -0.633 | +1.312 (0.052) | TRACKS_ERROR |
| 299568 | path | 1 | +0.353 (0.026) | -0.156 (0.034) | +0.680 | -0.633 | +1.312 (0.052) | TRACKS_ERROR |
| 299568 | anchor | 5 | +0.254 (0.027) | -0.156 (0.034) | +0.565 | -0.539 | +1.104 (0.066) | TRACKS_ERROR |
| 299568 | path | 5 | +0.318 (0.026) | -0.138 (0.035) | +0.640 | -0.597 | +1.237 (0.058) | TRACKS_ERROR |
| 398832 | anchor | 1 | +0.252 (0.020) | -0.190 (0.030) | +0.625 | -0.611 | +1.236 (0.046) | TRACKS_ERROR |
| 398832 | path | 1 | +0.252 (0.020) | -0.190 (0.030) | +0.625 | -0.611 | +1.236 (0.046) | TRACKS_ERROR |
| 398832 | anchor | 5 | +0.162 (0.021) | -0.190 (0.029) | +0.507 | -0.514 | +1.022 (0.066) | TRACKS_ERROR |
| 398832 | path | 5 | +0.239 (0.025) | -0.166 (0.035) | +0.600 | -0.583 | +1.183 (0.057) | TRACKS_ERROR |
| 498144 | anchor | 1 | +0.388 (0.020) | -0.150 (0.035) | +0.639 | -0.564 | +1.203 (0.070) | TRACKS_ERROR |
| 498144 | path | 1 | +0.388 (0.020) | -0.150 (0.035) | +0.639 | -0.564 | +1.203 (0.070) | TRACKS_ERROR |
| 498144 | anchor | 5 | +0.299 (0.021) | -0.150 (0.038) | +0.547 | -0.498 | +1.045 (0.073) | TRACKS_ERROR |
| 498144 | path | 5 | +0.346 (0.021) | -0.150 (0.039) | +0.612 | -0.552 | +1.164 (0.062) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/cup_random_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell cup_random_cross --dumps random_cup_seed1=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_cup_seed1/latent_uq/cup_v1 random_cup_seed2=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_cup_seed2/latent_uq/cup_v1 random_cup_seed3=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_cup_seed3/latent_uq/cup_v1 random_cup_seed4=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_cup_seed4/latent_uq/cup_v1 random_cup_seed5=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_random_cup_seed5/latent_uq/cup_v1 --horizons 1 5 --n_boot 300 --instrument density_residual --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/cup_random_cross`
- git commit: `c92f984794e7f1b57db82ccfc3f8faae3c65df52`
