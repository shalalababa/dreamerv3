# Latent-UQ Stage 0 evidence memo — 2026-07-10

**Replication verdict (Biased Dreams, arXiv 2604.25416): REPLICATES: disagreement tracks training density, not held-out error (attractor bias present).**

## Protocol

- Dumps: `p2e_cup_seed1`, `p2e_cup_seed2`, `p2e_cup_seed3`, `p2e_cup_seed4`, `p2e_cup_seed5`
- Probe set(s): ball_in_cup_v1_9327e960 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: p2e_cup_seed1

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100768 | anchor | 1 | +0.617 (0.025) | +0.316 (0.024) | +0.580 | +0.191 | +0.390 (0.034) | TRACKS_ERROR |
| 100768 | path | 1 | +0.617 (0.025) | +0.316 (0.024) | +0.580 | +0.191 | +0.390 (0.034) | TRACKS_ERROR |
| 100768 | anchor | 5 | +0.575 (0.025) | +0.316 (0.023) | +0.546 | +0.235 | +0.311 (0.030) | TRACKS_ERROR |
| 100768 | path | 5 | +0.602 (0.025) | +0.275 (0.025) | +0.577 | +0.181 | +0.396 (0.038) | TRACKS_ERROR |
| 199680 | anchor | 1 | +0.418 (0.026) | +0.600 (0.016) | +0.253 | +0.523 | -0.270 (0.048) | TRACKS_DENSITY |
| 199680 | path | 1 | +0.418 (0.026) | +0.600 (0.016) | +0.253 | +0.523 | -0.270 (0.048) | TRACKS_DENSITY |
| 199680 | anchor | 5 | +0.382 (0.021) | +0.600 (0.014) | +0.245 | +0.544 | -0.299 (0.039) | TRACKS_DENSITY |
| 199680 | path | 5 | +0.359 (0.022) | +0.542 (0.018) | +0.228 | +0.481 | -0.253 (0.044) | TRACKS_DENSITY |
| 298576 | anchor | 1 | +0.358 (0.024) | +0.614 (0.020) | +0.183 | +0.556 | -0.373 (0.038) | TRACKS_DENSITY |
| 298576 | path | 1 | +0.358 (0.024) | +0.614 (0.020) | +0.183 | +0.556 | -0.373 (0.038) | TRACKS_DENSITY |
| 298576 | anchor | 5 | +0.323 (0.021) | +0.614 (0.020) | +0.198 | +0.575 | -0.377 (0.034) | TRACKS_DENSITY |
| 298576 | path | 5 | +0.327 (0.022) | +0.566 (0.023) | +0.212 | +0.522 | -0.310 (0.042) | TRACKS_DENSITY |
| 401936 | anchor | 1 | +0.336 (0.032) | +0.668 (0.020) | +0.188 | +0.630 | -0.443 (0.046) | TRACKS_DENSITY |
| 401936 | path | 1 | +0.336 (0.032) | +0.668 (0.020) | +0.188 | +0.630 | -0.443 (0.046) | TRACKS_DENSITY |
| 401936 | anchor | 5 | +0.311 (0.027) | +0.668 (0.018) | +0.172 | +0.636 | -0.465 (0.039) | TRACKS_DENSITY |
| 401936 | path | 5 | +0.282 (0.028) | +0.622 (0.025) | +0.142 | +0.589 | -0.447 (0.051) | TRACKS_DENSITY |
| 496320 | anchor | 1 | +0.362 (0.034) | +0.480 (0.028) | +0.278 | +0.427 | -0.150 (0.034) | TRACKS_DENSITY |
| 496320 | path | 1 | +0.362 (0.034) | +0.480 (0.028) | +0.278 | +0.427 | -0.150 (0.034) | TRACKS_DENSITY |
| 496320 | anchor | 5 | +0.350 (0.032) | +0.480 (0.028) | +0.272 | +0.434 | -0.162 (0.030) | TRACKS_DENSITY |
| 496320 | path | 5 | +0.308 (0.035) | +0.431 (0.031) | +0.231 | +0.386 | -0.155 (0.042) | TRACKS_DENSITY |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_cup_seed2

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100736 | anchor | 1 | +0.424 (0.021) | +0.227 (0.026) | +0.405 | +0.183 | +0.222 (0.037) | TRACKS_ERROR |
| 100736 | path | 1 | +0.424 (0.021) | +0.227 (0.026) | +0.405 | +0.183 | +0.222 (0.037) | TRACKS_ERROR |
| 100736 | anchor | 5 | +0.418 (0.018) | +0.227 (0.027) | +0.405 | +0.199 | +0.206 (0.030) | TRACKS_ERROR |
| 100736 | path | 5 | +0.427 (0.018) | +0.181 (0.027) | +0.416 | +0.148 | +0.268 (0.031) | TRACKS_ERROR |
| 199664 | anchor | 1 | +0.349 (0.023) | +0.394 (0.035) | +0.290 | +0.345 | -0.055 (0.062) | AMBIGUOUS |
| 199664 | path | 1 | +0.349 (0.023) | +0.394 (0.035) | +0.290 | +0.345 | -0.055 (0.062) | AMBIGUOUS |
| 199664 | anchor | 5 | +0.361 (0.021) | +0.394 (0.033) | +0.329 | +0.366 | -0.037 (0.052) | AMBIGUOUS |
| 199664 | path | 5 | +0.362 (0.020) | +0.357 (0.035) | +0.331 | +0.325 | +0.006 (0.054) | AMBIGUOUS |
| 298560 | anchor | 1 | +0.294 (0.026) | +0.428 (0.036) | +0.239 | +0.396 | -0.157 (0.068) | TRACKS_DENSITY |
| 298560 | path | 1 | +0.294 (0.026) | +0.428 (0.036) | +0.239 | +0.396 | -0.157 (0.068) | TRACKS_DENSITY |
| 298560 | anchor | 5 | +0.266 (0.022) | +0.428 (0.037) | +0.213 | +0.401 | -0.189 (0.058) | TRACKS_DENSITY |
| 298560 | path | 5 | +0.276 (0.023) | +0.381 (0.040) | +0.228 | +0.351 | -0.122 (0.064) | AMBIGUOUS |
| 401936 | anchor | 1 | +0.279 (0.024) | +0.505 (0.036) | +0.158 | +0.461 | -0.302 (0.067) | TRACKS_DENSITY |
| 401936 | path | 1 | +0.279 (0.024) | +0.505 (0.036) | +0.158 | +0.461 | -0.302 (0.067) | TRACKS_DENSITY |
| 401936 | anchor | 5 | +0.236 (0.020) | +0.505 (0.036) | +0.148 | +0.478 | -0.330 (0.054) | TRACKS_DENSITY |
| 401936 | path | 5 | +0.253 (0.023) | +0.479 (0.038) | +0.171 | +0.448 | -0.277 (0.060) | TRACKS_DENSITY |
| 496336 | anchor | 1 | +0.199 (0.028) | +0.566 (0.034) | +0.059 | +0.542 | -0.483 (0.056) | TRACKS_DENSITY |
| 496336 | path | 1 | +0.199 (0.028) | +0.566 (0.034) | +0.059 | +0.542 | -0.483 (0.056) | TRACKS_DENSITY |
| 496336 | anchor | 5 | +0.187 (0.024) | +0.566 (0.034) | +0.065 | +0.546 | -0.482 (0.048) | TRACKS_DENSITY |
| 496336 | path | 5 | +0.192 (0.026) | +0.537 (0.037) | +0.078 | +0.516 | -0.437 (0.055) | TRACKS_DENSITY |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_cup_seed3

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100736 | anchor | 1 | +0.448 (0.018) | +0.135 (0.032) | +0.441 | +0.103 | +0.338 (0.033) | TRACKS_ERROR |
| 100736 | path | 1 | +0.448 (0.018) | +0.135 (0.032) | +0.441 | +0.103 | +0.338 (0.033) | TRACKS_ERROR |
| 100736 | anchor | 5 | +0.445 (0.018) | +0.135 (0.033) | +0.447 | +0.144 | +0.303 (0.033) | TRACKS_ERROR |
| 100736 | path | 5 | +0.480 (0.020) | +0.073 (0.031) | +0.480 | +0.075 | +0.406 (0.030) | TRACKS_ERROR |
| 199536 | anchor | 1 | +0.291 (0.022) | +0.216 (0.031) | +0.264 | +0.177 | +0.087 (0.038) | TRACKS_ERROR |
| 199536 | path | 1 | +0.291 (0.022) | +0.216 (0.031) | +0.264 | +0.177 | +0.087 (0.038) | TRACKS_ERROR |
| 199536 | anchor | 5 | +0.261 (0.018) | +0.216 (0.031) | +0.252 | +0.206 | +0.046 (0.036) | AMBIGUOUS |
| 199536 | path | 5 | +0.274 (0.019) | +0.148 (0.029) | +0.267 | +0.134 | +0.133 (0.032) | TRACKS_ERROR |
| 298336 | anchor | 1 | +0.334 (0.021) | +0.319 (0.034) | +0.271 | +0.251 | +0.019 (0.047) | AMBIGUOUS |
| 298336 | path | 1 | +0.334 (0.021) | +0.319 (0.034) | +0.271 | +0.251 | +0.019 (0.047) | AMBIGUOUS |
| 298336 | anchor | 5 | +0.293 (0.018) | +0.319 (0.039) | +0.240 | +0.272 | -0.032 (0.041) | AMBIGUOUS |
| 298336 | path | 5 | +0.296 (0.019) | +0.233 (0.041) | +0.257 | +0.179 | +0.078 (0.038) | TRACKS_ERROR |
| 402368 | anchor | 1 | +0.278 (0.024) | +0.532 (0.025) | +0.173 | +0.496 | -0.323 (0.028) | TRACKS_DENSITY |
| 402368 | path | 1 | +0.278 (0.024) | +0.532 (0.025) | +0.173 | +0.496 | -0.323 (0.028) | TRACKS_DENSITY |
| 402368 | anchor | 5 | +0.267 (0.029) | +0.532 (0.025) | +0.181 | +0.503 | -0.322 (0.030) | TRACKS_DENSITY |
| 402368 | path | 5 | +0.204 (0.027) | +0.462 (0.029) | +0.117 | +0.437 | -0.320 (0.033) | TRACKS_DENSITY |
| 496000 | anchor | 1 | +0.294 (0.027) | +0.573 (0.023) | +0.186 | +0.539 | -0.353 (0.039) | TRACKS_DENSITY |
| 496000 | path | 1 | +0.294 (0.027) | +0.573 (0.023) | +0.186 | +0.539 | -0.353 (0.039) | TRACKS_DENSITY |
| 496000 | anchor | 5 | +0.304 (0.027) | +0.573 (0.022) | +0.191 | +0.536 | -0.345 (0.038) | TRACKS_DENSITY |
| 496000 | path | 5 | +0.277 (0.025) | +0.510 (0.023) | +0.170 | +0.471 | -0.301 (0.034) | TRACKS_DENSITY |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_cup_seed4

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100736 | anchor | 1 | +0.468 (0.023) | +0.148 (0.037) | +0.458 | +0.104 | +0.354 (0.032) | TRACKS_ERROR |
| 100736 | path | 1 | +0.468 (0.023) | +0.148 (0.037) | +0.458 | +0.104 | +0.354 (0.032) | TRACKS_ERROR |
| 100736 | anchor | 5 | +0.450 (0.020) | +0.148 (0.038) | +0.442 | +0.116 | +0.326 (0.031) | TRACKS_ERROR |
| 100736 | path | 5 | +0.492 (0.022) | +0.088 (0.038) | +0.488 | +0.045 | +0.442 (0.032) | TRACKS_ERROR |
| 199632 | anchor | 1 | +0.329 (0.022) | +0.412 (0.036) | +0.238 | +0.349 | -0.111 (0.045) | TRACKS_DENSITY |
| 199632 | path | 1 | +0.329 (0.022) | +0.412 (0.036) | +0.238 | +0.349 | -0.111 (0.045) | TRACKS_DENSITY |
| 199632 | anchor | 5 | +0.303 (0.021) | +0.412 (0.038) | +0.216 | +0.357 | -0.141 (0.041) | TRACKS_DENSITY |
| 199632 | path | 5 | +0.320 (0.023) | +0.363 (0.040) | +0.245 | +0.301 | -0.056 (0.043) | AMBIGUOUS |
| 298688 | anchor | 1 | +0.373 (0.029) | +0.393 (0.033) | +0.301 | +0.327 | -0.026 (0.049) | AMBIGUOUS |
| 298688 | path | 1 | +0.373 (0.029) | +0.393 (0.033) | +0.301 | +0.327 | -0.026 (0.049) | AMBIGUOUS |
| 298688 | anchor | 5 | +0.358 (0.026) | +0.393 (0.038) | +0.294 | +0.338 | -0.044 (0.044) | AMBIGUOUS |
| 298688 | path | 5 | +0.333 (0.027) | +0.314 (0.041) | +0.279 | +0.254 | +0.025 (0.047) | AMBIGUOUS |
| 397712 | anchor | 1 | +0.268 (0.035) | +0.620 (0.027) | +0.127 | +0.590 | -0.463 (0.042) | TRACKS_DENSITY |
| 397712 | path | 1 | +0.268 (0.035) | +0.620 (0.027) | +0.127 | +0.590 | -0.463 (0.042) | TRACKS_DENSITY |
| 397712 | anchor | 5 | +0.233 (0.034) | +0.620 (0.027) | +0.108 | +0.598 | -0.490 (0.040) | TRACKS_DENSITY |
| 397712 | path | 5 | +0.195 (0.032) | +0.590 (0.030) | +0.065 | +0.570 | -0.505 (0.039) | TRACKS_DENSITY |
| 496768 | anchor | 1 | +0.370 (0.035) | +0.534 (0.029) | +0.272 | +0.483 | -0.211 (0.043) | TRACKS_DENSITY |
| 496768 | path | 1 | +0.370 (0.035) | +0.534 (0.029) | +0.272 | +0.483 | -0.211 (0.043) | TRACKS_DENSITY |
| 496768 | anchor | 5 | +0.351 (0.035) | +0.534 (0.031) | +0.262 | +0.491 | -0.229 (0.045) | TRACKS_DENSITY |
| 496768 | path | 5 | +0.332 (0.036) | +0.487 (0.035) | +0.246 | +0.442 | -0.195 (0.048) | TRACKS_DENSITY |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_cup_seed5

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100832 | anchor | 1 | +0.520 (0.019) | +0.022 (0.032) | +0.520 | -0.010 | +0.530 (0.034) | TRACKS_ERROR |
| 100832 | path | 1 | +0.520 (0.019) | +0.022 (0.032) | +0.520 | -0.010 | +0.530 (0.034) | TRACKS_ERROR |
| 100832 | anchor | 5 | +0.516 (0.014) | +0.022 (0.032) | +0.516 | +0.030 | +0.486 (0.031) | TRACKS_ERROR |
| 100832 | path | 5 | +0.529 (0.015) | -0.031 (0.031) | +0.529 | -0.033 | +0.562 (0.027) | TRACKS_ERROR |
| 199856 | anchor | 1 | +0.332 (0.022) | +0.161 (0.045) | +0.301 | +0.065 | +0.236 (0.056) | TRACKS_ERROR |
| 199856 | path | 1 | +0.332 (0.022) | +0.161 (0.045) | +0.301 | +0.065 | +0.236 (0.056) | TRACKS_ERROR |
| 199856 | anchor | 5 | +0.280 (0.022) | +0.161 (0.047) | +0.254 | +0.107 | +0.147 (0.053) | TRACKS_ERROR |
| 199856 | path | 5 | +0.265 (0.023) | +0.088 (0.045) | +0.252 | +0.033 | +0.220 (0.051) | TRACKS_ERROR |
| 298944 | anchor | 1 | +0.267 (0.029) | +0.270 (0.043) | +0.179 | +0.183 | -0.004 (0.057) | AMBIGUOUS |
| 298944 | path | 1 | +0.267 (0.029) | +0.270 (0.043) | +0.179 | +0.183 | -0.004 (0.057) | AMBIGUOUS |
| 298944 | anchor | 5 | +0.238 (0.024) | +0.270 (0.048) | +0.166 | +0.210 | -0.044 (0.051) | AMBIGUOUS |
| 298944 | path | 5 | +0.234 (0.025) | +0.178 (0.051) | +0.190 | +0.111 | +0.078 (0.049) | AMBIGUOUS |
| 397984 | anchor | 1 | +0.316 (0.029) | +0.547 (0.035) | +0.108 | +0.481 | -0.372 (0.051) | TRACKS_DENSITY |
| 397984 | path | 1 | +0.316 (0.029) | +0.547 (0.035) | +0.108 | +0.481 | -0.372 (0.051) | TRACKS_DENSITY |
| 397984 | anchor | 5 | +0.302 (0.028) | +0.547 (0.034) | +0.135 | +0.493 | -0.358 (0.043) | TRACKS_DENSITY |
| 397984 | path | 5 | +0.284 (0.030) | +0.489 (0.039) | +0.133 | +0.432 | -0.299 (0.044) | TRACKS_DENSITY |
| 497040 | anchor | 1 | +0.266 (0.033) | +0.539 (0.032) | +0.093 | +0.493 | -0.400 (0.037) | TRACKS_DENSITY |
| 497040 | path | 1 | +0.266 (0.033) | +0.539 (0.032) | +0.093 | +0.493 | -0.400 (0.037) | TRACKS_DENSITY |
| 497040 | anchor | 5 | +0.261 (0.031) | +0.539 (0.035) | +0.115 | +0.499 | -0.384 (0.038) | TRACKS_DENSITY |
| 497040 | path | 5 | +0.235 (0.035) | +0.490 (0.040) | +0.097 | +0.451 | -0.354 (0.043) | TRACKS_DENSITY |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/cup_p2e/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cell cup_p2e --dumps p2e_cup_seed1=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_cup_seed1/latent_uq/cup_v1 p2e_cup_seed2=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_cup_seed2/latent_uq/cup_v1 p2e_cup_seed3=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_cup_seed3/latent_uq/cup_v1 p2e_cup_seed4=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_cup_seed4/latent_uq/cup_v1 p2e_cup_seed5=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_cup_seed5/latent_uq/cup_v1 --horizons 1 5 --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/cup_p2e`
- git commit: `d102ac4fd5cf27d13eae50171d4dacc56b18d79e`
