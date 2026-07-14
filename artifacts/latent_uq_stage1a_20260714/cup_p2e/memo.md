# Latent-UQ Stage 0 evidence memo — 2026-07-14

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `p2e_cup_seed1`, `p2e_cup_seed2`, `p2e_cup_seed3`, `p2e_cup_seed4`, `p2e_cup_seed5`
- Probe set(s): ball_in_cup_v1_9327e960 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)
- Instrument: **density_residual** (out-of-fold isotonic residual on the density proxy, 5 stream-cluster folds — Stage-1A primary, PREREG_gate_d1_stage1a_20260714.md)

## Cell: p2e_cup_seed1

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100768 | anchor | 1 | +0.522 (0.028) | -0.082 (0.031) | +0.569 | -0.278 | +0.847 (0.031) | TRACKS_ERROR |
| 100768 | path | 1 | +0.522 (0.028) | -0.082 (0.031) | +0.569 | -0.278 | +0.847 (0.031) | TRACKS_ERROR |
| 100768 | anchor | 5 | +0.502 (0.027) | -0.082 (0.030) | +0.535 | -0.230 | +0.765 (0.031) | TRACKS_ERROR |
| 100768 | path | 5 | +0.540 (0.024) | -0.052 (0.026) | +0.566 | -0.210 | +0.777 (0.035) | TRACKS_ERROR |
| 199680 | anchor | 1 | +0.177 (0.028) | -0.147 (0.031) | +0.256 | -0.237 | +0.493 (0.045) | TRACKS_ERROR |
| 199680 | path | 1 | +0.177 (0.028) | -0.147 (0.031) | +0.256 | -0.237 | +0.493 (0.045) | TRACKS_ERROR |
| 199680 | anchor | 5 | +0.193 (0.023) | -0.147 (0.029) | +0.258 | -0.227 | +0.485 (0.041) | TRACKS_ERROR |
| 199680 | path | 5 | +0.183 (0.023) | -0.112 (0.025) | +0.234 | -0.185 | +0.420 (0.041) | TRACKS_ERROR |
| 298576 | anchor | 1 | +0.122 (0.019) | -0.162 (0.030) | +0.197 | -0.223 | +0.420 (0.040) | TRACKS_ERROR |
| 298576 | path | 1 | +0.122 (0.019) | -0.162 (0.030) | +0.197 | -0.223 | +0.420 (0.040) | TRACKS_ERROR |
| 298576 | anchor | 5 | +0.149 (0.018) | -0.162 (0.034) | +0.206 | -0.215 | +0.421 (0.040) | TRACKS_ERROR |
| 298576 | path | 5 | +0.154 (0.019) | -0.143 (0.035) | +0.205 | -0.197 | +0.402 (0.041) | TRACKS_ERROR |
| 401936 | anchor | 1 | +0.142 (0.028) | -0.138 (0.028) | +0.195 | -0.192 | +0.387 (0.037) | TRACKS_ERROR |
| 401936 | path | 1 | +0.142 (0.028) | -0.138 (0.028) | +0.195 | -0.192 | +0.387 (0.037) | TRACKS_ERROR |
| 401936 | anchor | 5 | +0.137 (0.027) | -0.138 (0.028) | +0.185 | -0.186 | +0.371 (0.038) | TRACKS_ERROR |
| 401936 | path | 5 | +0.092 (0.028) | -0.148 (0.026) | +0.141 | -0.182 | +0.323 (0.046) | TRACKS_ERROR |
| 496320 | anchor | 1 | +0.228 (0.029) | -0.099 (0.030) | +0.266 | -0.171 | +0.436 (0.042) | TRACKS_ERROR |
| 496320 | path | 1 | +0.228 (0.029) | -0.099 (0.030) | +0.266 | -0.171 | +0.436 (0.042) | TRACKS_ERROR |
| 496320 | anchor | 5 | +0.234 (0.028) | -0.099 (0.027) | +0.267 | -0.166 | +0.434 (0.042) | TRACKS_ERROR |
| 496320 | path | 5 | +0.177 (0.032) | -0.093 (0.030) | +0.207 | -0.143 | +0.350 (0.050) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_cup_seed2

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100736 | anchor | 1 | +0.376 (0.022) | -0.057 (0.037) | +0.390 | -0.123 | +0.513 (0.045) | TRACKS_ERROR |
| 100736 | path | 1 | +0.376 (0.022) | -0.057 (0.037) | +0.390 | -0.123 | +0.513 (0.045) | TRACKS_ERROR |
| 100736 | anchor | 5 | +0.382 (0.020) | -0.057 (0.035) | +0.392 | -0.109 | +0.500 (0.039) | TRACKS_ERROR |
| 100736 | path | 5 | +0.399 (0.020) | -0.046 (0.036) | +0.407 | -0.099 | +0.507 (0.041) | TRACKS_ERROR |
| 199664 | anchor | 1 | +0.220 (0.028) | -0.130 (0.040) | +0.258 | -0.189 | +0.447 (0.062) | TRACKS_ERROR |
| 199664 | path | 1 | +0.220 (0.028) | -0.130 (0.040) | +0.258 | -0.189 | +0.447 (0.062) | TRACKS_ERROR |
| 199664 | anchor | 5 | +0.289 (0.022) | -0.130 (0.036) | +0.316 | -0.185 | +0.501 (0.056) | TRACKS_ERROR |
| 199664 | path | 5 | +0.306 (0.019) | -0.056 (0.041) | +0.320 | -0.111 | +0.431 (0.060) | TRACKS_ERROR |
| 298560 | anchor | 1 | +0.182 (0.028) | -0.070 (0.044) | +0.199 | -0.109 | +0.308 (0.070) | TRACKS_ERROR |
| 298560 | path | 1 | +0.182 (0.028) | -0.070 (0.044) | +0.199 | -0.109 | +0.308 (0.070) | TRACKS_ERROR |
| 298560 | anchor | 5 | +0.167 (0.022) | -0.070 (0.041) | +0.182 | -0.103 | +0.285 (0.063) | TRACKS_ERROR |
| 298560 | path | 5 | +0.195 (0.020) | -0.010 (0.047) | +0.200 | -0.047 | +0.247 (0.070) | TRACKS_ERROR |
| 401936 | anchor | 1 | +0.103 (0.023) | -0.102 (0.039) | +0.140 | -0.139 | +0.278 (0.059) | TRACKS_ERROR |
| 401936 | path | 1 | +0.103 (0.023) | -0.102 (0.039) | +0.140 | -0.139 | +0.278 (0.059) | TRACKS_ERROR |
| 401936 | anchor | 5 | +0.115 (0.018) | -0.102 (0.037) | +0.142 | -0.131 | +0.273 (0.050) | TRACKS_ERROR |
| 401936 | path | 5 | +0.153 (0.020) | -0.051 (0.039) | +0.169 | -0.088 | +0.257 (0.056) | TRACKS_ERROR |
| 496336 | anchor | 1 | +0.025 (0.022) | -0.132 (0.032) | +0.064 | -0.144 | +0.208 (0.049) | TRACKS_ERROR |
| 496336 | path | 1 | +0.025 (0.022) | -0.132 (0.032) | +0.064 | -0.144 | +0.208 (0.049) | TRACKS_ERROR |
| 496336 | anchor | 5 | +0.029 (0.019) | -0.132 (0.033) | +0.063 | -0.143 | +0.206 (0.042) | TRACKS_ERROR |
| 496336 | path | 5 | +0.047 (0.020) | -0.097 (0.034) | +0.072 | -0.111 | +0.183 (0.045) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_cup_seed3

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100736 | anchor | 1 | +0.404 (0.021) | -0.092 (0.032) | +0.417 | -0.144 | +0.560 (0.031) | TRACKS_ERROR |
| 100736 | path | 1 | +0.404 (0.021) | -0.092 (0.032) | +0.417 | -0.144 | +0.560 (0.031) | TRACKS_ERROR |
| 100736 | anchor | 5 | +0.411 (0.020) | -0.092 (0.033) | +0.414 | -0.107 | +0.521 (0.033) | TRACKS_ERROR |
| 100736 | path | 5 | +0.449 (0.021) | -0.092 (0.027) | +0.453 | -0.111 | +0.564 (0.026) | TRACKS_ERROR |
| 199536 | anchor | 1 | +0.205 (0.023) | -0.105 (0.026) | +0.227 | -0.145 | +0.372 (0.035) | TRACKS_ERROR |
| 199536 | path | 1 | +0.205 (0.023) | -0.105 (0.026) | +0.227 | -0.145 | +0.372 (0.035) | TRACKS_ERROR |
| 199536 | anchor | 5 | +0.200 (0.021) | -0.105 (0.026) | +0.209 | -0.122 | +0.331 (0.031) | TRACKS_ERROR |
| 199536 | path | 5 | +0.223 (0.021) | -0.108 (0.023) | +0.233 | -0.127 | +0.360 (0.028) | TRACKS_ERROR |
| 298336 | anchor | 1 | +0.162 (0.021) | -0.128 (0.029) | +0.207 | -0.182 | +0.389 (0.037) | TRACKS_ERROR |
| 298336 | path | 1 | +0.162 (0.021) | -0.128 (0.029) | +0.207 | -0.182 | +0.389 (0.037) | TRACKS_ERROR |
| 298336 | anchor | 5 | +0.117 (0.021) | -0.128 (0.029) | +0.151 | -0.160 | +0.311 (0.036) | TRACKS_ERROR |
| 298336 | path | 5 | +0.143 (0.023) | -0.128 (0.032) | +0.178 | -0.166 | +0.344 (0.035) | TRACKS_ERROR |
| 402368 | anchor | 1 | +0.141 (0.022) | -0.038 (0.027) | +0.156 | -0.078 | +0.233 (0.038) | TRACKS_ERROR |
| 402368 | path | 1 | +0.141 (0.022) | -0.038 (0.027) | +0.156 | -0.078 | +0.233 (0.038) | TRACKS_ERROR |
| 402368 | anchor | 5 | +0.143 (0.025) | -0.038 (0.026) | +0.155 | -0.072 | +0.228 (0.042) | TRACKS_ERROR |
| 402368 | path | 5 | +0.078 (0.026) | -0.041 (0.034) | +0.089 | -0.060 | +0.149 (0.044) | TRACKS_ERROR |
| 496000 | anchor | 1 | +0.140 (0.026) | -0.046 (0.026) | +0.157 | -0.085 | +0.242 (0.038) | TRACKS_ERROR |
| 496000 | path | 1 | +0.140 (0.026) | -0.046 (0.026) | +0.157 | -0.085 | +0.242 (0.038) | TRACKS_ERROR |
| 496000 | anchor | 5 | +0.144 (0.024) | -0.046 (0.027) | +0.162 | -0.089 | +0.251 (0.041) | TRACKS_ERROR |
| 496000 | path | 5 | +0.121 (0.025) | -0.016 (0.029) | +0.130 | -0.051 | +0.180 (0.039) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_cup_seed4

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100736 | anchor | 1 | +0.399 (0.026) | -0.112 (0.034) | +0.418 | -0.177 | +0.595 (0.026) | TRACKS_ERROR |
| 100736 | path | 1 | +0.399 (0.026) | -0.112 (0.034) | +0.418 | -0.177 | +0.595 (0.026) | TRACKS_ERROR |
| 100736 | anchor | 5 | +0.393 (0.022) | -0.112 (0.034) | +0.409 | -0.166 | +0.575 (0.026) | TRACKS_ERROR |
| 100736 | path | 5 | +0.441 (0.022) | -0.115 (0.033) | +0.458 | -0.178 | +0.636 (0.029) | TRACKS_ERROR |
| 199632 | anchor | 1 | +0.148 (0.025) | -0.122 (0.038) | +0.193 | -0.175 | +0.368 (0.046) | TRACKS_ERROR |
| 199632 | path | 1 | +0.148 (0.025) | -0.122 (0.038) | +0.193 | -0.175 | +0.368 (0.046) | TRACKS_ERROR |
| 199632 | anchor | 5 | +0.132 (0.021) | -0.122 (0.036) | +0.174 | -0.166 | +0.340 (0.041) | TRACKS_ERROR |
| 199632 | path | 5 | +0.160 (0.025) | -0.113 (0.037) | +0.200 | -0.166 | +0.367 (0.044) | TRACKS_ERROR |
| 298688 | anchor | 1 | +0.205 (0.028) | -0.121 (0.034) | +0.249 | -0.188 | +0.437 (0.046) | TRACKS_ERROR |
| 298688 | path | 1 | +0.205 (0.028) | -0.121 (0.034) | +0.249 | -0.188 | +0.437 (0.046) | TRACKS_ERROR |
| 298688 | anchor | 5 | +0.210 (0.025) | -0.121 (0.036) | +0.249 | -0.182 | +0.431 (0.040) | TRACKS_ERROR |
| 298688 | path | 5 | +0.191 (0.026) | -0.105 (0.036) | +0.225 | -0.159 | +0.384 (0.046) | TRACKS_ERROR |
| 397712 | anchor | 1 | +0.102 (0.035) | -0.098 (0.034) | +0.136 | -0.132 | +0.268 (0.051) | TRACKS_ERROR |
| 397712 | path | 1 | +0.102 (0.035) | -0.098 (0.034) | +0.136 | -0.132 | +0.268 (0.051) | TRACKS_ERROR |
| 397712 | anchor | 5 | +0.094 (0.033) | -0.098 (0.035) | +0.123 | -0.125 | +0.248 (0.047) | TRACKS_ERROR |
| 397712 | path | 5 | +0.043 (0.030) | -0.068 (0.039) | +0.062 | -0.081 | +0.143 (0.047) | TRACKS_ERROR |
| 496768 | anchor | 1 | +0.184 (0.031) | -0.127 (0.030) | +0.231 | -0.189 | +0.420 (0.044) | TRACKS_ERROR |
| 496768 | path | 1 | +0.184 (0.031) | -0.127 (0.030) | +0.231 | -0.189 | +0.420 (0.044) | TRACKS_ERROR |
| 496768 | anchor | 5 | +0.199 (0.029) | -0.127 (0.031) | +0.241 | -0.187 | +0.428 (0.046) | TRACKS_ERROR |
| 496768 | path | 5 | +0.190 (0.030) | -0.083 (0.037) | +0.219 | -0.138 | +0.357 (0.050) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_cup_seed5

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100832 | anchor | 1 | +0.462 (0.023) | -0.160 (0.024) | +0.479 | -0.213 | +0.692 (0.031) | TRACKS_ERROR |
| 100832 | path | 1 | +0.462 (0.023) | -0.160 (0.024) | +0.479 | -0.213 | +0.692 (0.031) | TRACKS_ERROR |
| 100832 | anchor | 5 | +0.485 (0.016) | -0.160 (0.024) | +0.490 | -0.180 | +0.670 (0.025) | TRACKS_ERROR |
| 100832 | path | 5 | +0.500 (0.017) | -0.177 (0.024) | +0.506 | -0.201 | +0.707 (0.024) | TRACKS_ERROR |
| 199856 | anchor | 1 | +0.182 (0.024) | -0.176 (0.038) | +0.252 | -0.248 | +0.500 (0.049) | TRACKS_ERROR |
| 199856 | path | 1 | +0.182 (0.024) | -0.176 (0.038) | +0.252 | -0.248 | +0.500 (0.049) | TRACKS_ERROR |
| 199856 | anchor | 5 | +0.173 (0.022) | -0.176 (0.036) | +0.219 | -0.222 | +0.441 (0.046) | TRACKS_ERROR |
| 199856 | path | 5 | +0.178 (0.022) | -0.159 (0.037) | +0.221 | -0.206 | +0.427 (0.046) | TRACKS_ERROR |
| 298944 | anchor | 1 | +0.057 (0.024) | -0.123 (0.037) | +0.118 | -0.160 | +0.278 (0.051) | TRACKS_ERROR |
| 298944 | path | 1 | +0.057 (0.024) | -0.123 (0.037) | +0.118 | -0.160 | +0.278 (0.051) | TRACKS_ERROR |
| 298944 | anchor | 5 | +0.053 (0.021) | -0.123 (0.039) | +0.099 | -0.148 | +0.248 (0.050) | TRACKS_ERROR |
| 298944 | path | 5 | +0.076 (0.020) | -0.136 (0.043) | +0.128 | -0.170 | +0.298 (0.049) | TRACKS_ERROR |
| 397984 | anchor | 1 | +0.070 (0.023) | -0.063 (0.036) | +0.108 | -0.103 | +0.211 (0.051) | TRACKS_ERROR |
| 397984 | path | 1 | +0.070 (0.023) | -0.063 (0.036) | +0.108 | -0.103 | +0.211 (0.051) | TRACKS_ERROR |
| 397984 | anchor | 5 | +0.094 (0.027) | -0.063 (0.037) | +0.125 | -0.104 | +0.230 (0.048) | TRACKS_ERROR |
| 397984 | path | 5 | +0.093 (0.028) | -0.028 (0.042) | +0.110 | -0.066 | +0.176 (0.052) | TRACKS_ERROR |
| 497040 | anchor | 1 | +0.080 (0.025) | -0.047 (0.036) | +0.104 | -0.082 | +0.185 (0.053) | TRACKS_ERROR |
| 497040 | path | 1 | +0.080 (0.025) | -0.047 (0.036) | +0.104 | -0.082 | +0.185 (0.053) | TRACKS_ERROR |
| 497040 | anchor | 5 | +0.076 (0.026) | -0.047 (0.037) | +0.096 | -0.076 | +0.172 (0.053) | TRACKS_ERROR |
| 497040 | path | 5 | +0.059 (0.029) | -0.047 (0.040) | +0.078 | -0.070 | +0.148 (0.058) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/cup_p2e/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --dumps p2e_cup_seed1=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_cup_seed1/latent_uq/cup_v1 p2e_cup_seed2=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_cup_seed2/latent_uq/cup_v1 p2e_cup_seed3=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_cup_seed3/latent_uq/cup_v1 p2e_cup_seed4=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_cup_seed4/latent_uq/cup_v1 p2e_cup_seed5=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_cup_seed5/latent_uq/cup_v1 --horizons 1 5 --n_boot 300 --instrument density_residual --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/cup_p2e`
- git commit: `c92f984794e7f1b57db82ccfc3f8faae3c65df52`
