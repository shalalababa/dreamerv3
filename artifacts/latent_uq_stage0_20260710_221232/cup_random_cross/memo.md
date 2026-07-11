# Latent-UQ Stage 0 evidence memo — 2026-07-10

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `random_cup_seed1`, `random_cup_seed2`, `random_cup_seed3`, `random_cup_seed4`, `random_cup_seed5` (cross-ensemble members)
- Probe set(s): ball_in_cup_v1_9327e960 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: cup_random_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 101056 | anchor | 1 | +0.863 (0.021) | +0.717 (0.021) | +0.696 | -0.137 | +0.833 (0.045) | TRACKS_ERROR |
| 101056 | path | 1 | +0.863 (0.021) | +0.717 (0.021) | +0.696 | -0.137 | +0.833 (0.045) | TRACKS_ERROR |
| 101056 | anchor | 5 | +0.814 (0.023) | +0.717 (0.021) | +0.552 | +0.017 | +0.535 (0.056) | TRACKS_ERROR |
| 101056 | path | 5 | +0.850 (0.022) | +0.723 (0.023) | +0.650 | -0.082 | +0.732 (0.049) | TRACKS_ERROR |
| 200304 | anchor | 1 | +0.885 (0.018) | +0.778 (0.014) | +0.677 | +0.117 | +0.560 (0.049) | TRACKS_ERROR |
| 200304 | path | 1 | +0.885 (0.018) | +0.778 (0.014) | +0.677 | +0.117 | +0.560 (0.049) | TRACKS_ERROR |
| 200304 | anchor | 5 | +0.836 (0.021) | +0.778 (0.014) | +0.523 | +0.222 | +0.301 (0.063) | TRACKS_ERROR |
| 200304 | path | 5 | +0.878 (0.019) | +0.794 (0.013) | +0.633 | +0.170 | +0.462 (0.059) | TRACKS_ERROR |
| 299568 | anchor | 1 | +0.887 (0.014) | +0.674 (0.014) | +0.784 | +0.108 | +0.676 (0.038) | TRACKS_ERROR |
| 299568 | path | 1 | +0.887 (0.014) | +0.674 (0.014) | +0.784 | +0.108 | +0.676 (0.038) | TRACKS_ERROR |
| 299568 | anchor | 5 | +0.836 (0.017) | +0.674 (0.015) | +0.677 | +0.135 | +0.542 (0.045) | TRACKS_ERROR |
| 299568 | path | 5 | +0.881 (0.014) | +0.697 (0.014) | +0.756 | +0.122 | +0.633 (0.038) | TRACKS_ERROR |
| 398832 | anchor | 1 | +0.904 (0.014) | +0.698 (0.010) | +0.801 | +0.032 | +0.770 (0.038) | TRACKS_ERROR |
| 398832 | path | 1 | +0.904 (0.014) | +0.698 (0.010) | +0.801 | +0.032 | +0.770 (0.038) | TRACKS_ERROR |
| 398832 | anchor | 5 | +0.851 (0.017) | +0.698 (0.011) | +0.684 | +0.100 | +0.584 (0.053) | TRACKS_ERROR |
| 398832 | path | 5 | +0.896 (0.016) | +0.730 (0.010) | +0.764 | +0.106 | +0.657 (0.046) | TRACKS_ERROR |
| 498144 | anchor | 1 | +0.863 (0.016) | +0.575 (0.018) | +0.787 | +0.054 | +0.733 (0.061) | TRACKS_ERROR |
| 498144 | path | 1 | +0.863 (0.016) | +0.575 (0.018) | +0.787 | +0.054 | +0.733 (0.061) | TRACKS_ERROR |
| 498144 | anchor | 5 | +0.812 (0.019) | +0.575 (0.020) | +0.703 | +0.064 | +0.639 (0.065) | TRACKS_ERROR |
| 498144 | path | 5 | +0.859 (0.017) | +0.607 (0.019) | +0.766 | +0.075 | +0.691 (0.058) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/cup_random_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell cup_random_cross --dumps random_cup_seed1=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_cup_seed1/latent_uq/cup_v1 random_cup_seed2=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_cup_seed2/latent_uq/cup_v1 random_cup_seed3=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_cup_seed3/latent_uq/cup_v1 random_cup_seed4=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_cup_seed4/latent_uq/cup_v1 random_cup_seed5=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_cup_seed5/latent_uq/cup_v1 --horizons 1 5 --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/cup_random_cross`
- git commit: `d102ac4fd5cf27d13eae50171d4dacc56b18d79e`
