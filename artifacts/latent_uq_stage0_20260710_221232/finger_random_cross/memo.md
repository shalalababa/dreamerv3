# Latent-UQ Stage 0 evidence memo — 2026-07-10

**Replication verdict (Biased Dreams, arXiv 2604.25416): MIXED/AMBIGUOUS: see per-cell verdicts; do not aggregate into a single claim.**

## Protocol

- Dumps: `random_finger_seed1`, `random_finger_seed2`, `random_finger_seed3`, `random_finger_seed4`, `random_finger_seed5` (cross-ensemble members)
- Probe set(s): finger_v1_7a32ff08 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: finger_random_cross

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 96208 | anchor | 1 | +0.866 (0.008) | +0.798 (0.011) | +0.659 | +0.423 | +0.236 (0.034) | TRACKS_ERROR |
| 96208 | path | 1 | +0.866 (0.008) | +0.798 (0.011) | +0.659 | +0.423 | +0.236 (0.034) | TRACKS_ERROR |
| 96208 | anchor | 5 | +0.744 (0.016) | +0.798 (0.010) | +0.435 | +0.583 | -0.149 (0.040) | TRACKS_DENSITY |
| 96208 | path | 5 | +0.835 (0.010) | +0.789 (0.011) | +0.646 | +0.524 | +0.122 (0.025) | TRACKS_ERROR |
| 199120 | anchor | 1 | +0.938 (0.006) | +0.794 (0.014) | +0.828 | +0.172 | +0.656 (0.049) | TRACKS_ERROR |
| 199120 | path | 1 | +0.938 (0.006) | +0.794 (0.014) | +0.828 | +0.172 | +0.656 (0.049) | TRACKS_ERROR |
| 199120 | anchor | 5 | +0.837 (0.014) | +0.794 (0.014) | +0.600 | +0.457 | +0.143 (0.043) | TRACKS_ERROR |
| 199120 | path | 5 | +0.908 (0.010) | +0.774 (0.017) | +0.781 | +0.333 | +0.448 (0.036) | TRACKS_ERROR |
| 302032 | anchor | 1 | +0.947 (0.008) | +0.814 (0.016) | +0.840 | +0.182 | +0.658 (0.043) | TRACKS_ERROR |
| 302032 | path | 1 | +0.947 (0.008) | +0.814 (0.016) | +0.840 | +0.182 | +0.658 (0.043) | TRACKS_ERROR |
| 302032 | anchor | 5 | +0.857 (0.013) | +0.814 (0.016) | +0.630 | +0.487 | +0.143 (0.041) | TRACKS_ERROR |
| 302032 | path | 5 | +0.918 (0.010) | +0.786 (0.020) | +0.797 | +0.345 | +0.452 (0.030) | TRACKS_ERROR |
| 396384 | anchor | 1 | +0.954 (0.005) | +0.844 (0.016) | +0.844 | +0.271 | +0.573 (0.040) | TRACKS_ERROR |
| 396384 | path | 1 | +0.954 (0.005) | +0.844 (0.016) | +0.844 | +0.271 | +0.573 (0.040) | TRACKS_ERROR |
| 396384 | anchor | 5 | +0.856 (0.013) | +0.844 (0.015) | +0.601 | +0.558 | +0.043 (0.040) | AMBIGUOUS |
| 396384 | path | 5 | +0.920 (0.010) | +0.814 (0.020) | +0.792 | +0.419 | +0.372 (0.028) | TRACKS_ERROR |
| 499312 | anchor | 1 | +0.955 (0.006) | +0.850 (0.014) | +0.841 | +0.267 | +0.574 (0.041) | TRACKS_ERROR |
| 499312 | path | 1 | +0.955 (0.006) | +0.850 (0.014) | +0.841 | +0.267 | +0.574 (0.041) | TRACKS_ERROR |
| 499312 | anchor | 5 | +0.858 (0.014) | +0.850 (0.013) | +0.599 | +0.568 | +0.031 (0.032) | AMBIGUOUS |
| 499312 | path | 5 | +0.925 (0.010) | +0.820 (0.018) | +0.799 | +0.426 | +0.374 (0.023) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **AMBIGUOUS**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/finger_random_cross/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cross --cell finger_random_cross --dumps random_finger_seed1=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_finger_seed1/latent_uq/finger_v1 random_finger_seed2=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_finger_seed2/latent_uq/finger_v1 random_finger_seed3=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_finger_seed3/latent_uq/finger_v1 random_finger_seed4=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_finger_seed4/latent_uq/finger_v1 random_finger_seed5=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_random_finger_seed5/latent_uq/finger_v1 --horizons 1 5 --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/finger_random_cross`
- git commit: `d102ac4fd5cf27d13eae50171d4dacc56b18d79e`
