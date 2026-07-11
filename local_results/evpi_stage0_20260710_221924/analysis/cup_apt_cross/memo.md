# Latent-UQ Stage 0 evidence memo — 2026-07-10

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `apt_cup_seed1`, `apt_cup_seed2`, `apt_cup_seed3`, `apt_cup_seed4`, `apt_cup_seed5` (cross-ensemble members)
- Probe set(s): ball_in_cup_v1_9327e960 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: cup_apt_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 99600 | anchor | 1 | +0.932 (0.005) | +0.783 (0.015) | +0.835 | +0.336 | +0.499 (0.042) | TRACKS_ERROR |
| 99600 | path | 1 | +0.932 (0.005) | +0.783 (0.015) | +0.835 | +0.336 | +0.499 (0.042) | TRACKS_ERROR |
| 99600 | anchor | 5 | +0.844 (0.011) | +0.783 (0.015) | +0.639 | +0.452 | +0.187 (0.054) | TRACKS_ERROR |
| 99600 | path | 5 | +0.915 (0.007) | +0.781 (0.016) | +0.807 | +0.402 | +0.406 (0.044) | TRACKS_ERROR |
| 200608 | anchor | 1 | +0.927 (0.006) | +0.809 (0.010) | +0.818 | +0.433 | +0.385 (0.028) | TRACKS_ERROR |
| 200608 | path | 1 | +0.927 (0.006) | +0.809 (0.010) | +0.818 | +0.433 | +0.385 (0.028) | TRACKS_ERROR |
| 200608 | anchor | 5 | +0.820 (0.015) | +0.809 (0.010) | +0.542 | +0.503 | +0.039 (0.041) | AMBIGUOUS |
| 200608 | path | 5 | +0.900 (0.010) | +0.824 (0.009) | +0.748 | +0.501 | +0.247 (0.034) | TRACKS_ERROR |
| 299904 | anchor | 1 | +0.933 (0.006) | +0.786 (0.009) | +0.860 | +0.473 | +0.386 (0.023) | TRACKS_ERROR |
| 299904 | path | 1 | +0.933 (0.006) | +0.786 (0.009) | +0.860 | +0.473 | +0.386 (0.023) | TRACKS_ERROR |
| 299904 | anchor | 5 | +0.823 (0.014) | +0.786 (0.010) | +0.596 | +0.487 | +0.109 (0.035) | TRACKS_ERROR |
| 299904 | path | 5 | +0.900 (0.010) | +0.808 (0.009) | +0.776 | +0.525 | +0.251 (0.027) | TRACKS_ERROR |
| 399200 | anchor | 1 | +0.945 (0.005) | +0.589 (0.018) | +0.915 | +0.093 | +0.822 (0.026) | TRACKS_ERROR |
| 399200 | path | 1 | +0.945 (0.005) | +0.589 (0.018) | +0.915 | +0.093 | +0.822 (0.026) | TRACKS_ERROR |
| 399200 | anchor | 5 | +0.771 (0.014) | +0.589 (0.019) | +0.643 | +0.233 | +0.410 (0.030) | TRACKS_ERROR |
| 399200 | path | 5 | +0.889 (0.009) | +0.614 (0.020) | +0.822 | +0.194 | +0.628 (0.024) | TRACKS_ERROR |
| 498512 | anchor | 1 | +0.926 (0.007) | +0.600 (0.014) | +0.884 | +0.162 | +0.723 (0.023) | TRACKS_ERROR |
| 498512 | path | 1 | +0.926 (0.007) | +0.600 (0.014) | +0.884 | +0.162 | +0.723 (0.023) | TRACKS_ERROR |
| 498512 | anchor | 5 | +0.732 (0.021) | +0.600 (0.016) | +0.585 | +0.304 | +0.280 (0.046) | TRACKS_ERROR |
| 498512 | path | 5 | +0.861 (0.014) | +0.633 (0.017) | +0.780 | +0.301 | +0.479 (0.029) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/cup_apt_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell cup_apt_cross --dumps apt_cup_seed1=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_cup_seed1/latent_uq/cup_v1 apt_cup_seed2=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_cup_seed2/latent_uq/cup_v1 apt_cup_seed3=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_cup_seed3/latent_uq/cup_v1 apt_cup_seed4=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_cup_seed4/latent_uq/cup_v1 apt_cup_seed5=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_apt_cup_seed5/latent_uq/cup_v1 --horizons 1 5 --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/cup_apt_cross`
- git commit: `d102ac4fd5cf27d13eae50171d4dacc56b18d79e`
