# Latent-UQ Stage 0 evidence memo — 2026-07-10

**Replication verdict (Biased Dreams, arXiv 2604.25416): REPLICATES: disagreement tracks training density, not held-out error (attractor bias present).**

## Protocol

- Dumps: `p2e_finger_seed1`, `p2e_finger_seed2`, `p2e_finger_seed3`, `p2e_finger_seed4`, `p2e_finger_seed5`
- Probe set(s): finger_v1_7a32ff08 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: p2e_finger_seed1

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100432 | anchor | 1 | +0.181 (0.033) | -0.046 (0.030) | +0.183 | -0.052 | +0.235 (0.039) | TRACKS_ERROR |
| 100432 | path | 1 | +0.181 (0.033) | -0.046 (0.030) | +0.183 | -0.052 | +0.235 (0.039) | TRACKS_ERROR |
| 100432 | anchor | 5 | +0.148 (0.027) | -0.046 (0.029) | +0.146 | -0.040 | +0.186 (0.039) | TRACKS_ERROR |
| 100432 | path | 5 | +0.221 (0.028) | -0.130 (0.026) | +0.218 | -0.124 | +0.341 (0.035) | TRACKS_ERROR |
| 198928 | anchor | 1 | +0.147 (0.029) | +0.180 (0.030) | +0.142 | +0.176 | -0.034 (0.035) | AMBIGUOUS |
| 198928 | path | 1 | +0.147 (0.029) | +0.180 (0.030) | +0.142 | +0.176 | -0.034 (0.035) | AMBIGUOUS |
| 198928 | anchor | 5 | +0.105 (0.026) | +0.180 (0.027) | +0.109 | +0.183 | -0.073 (0.035) | TRACKS_DENSITY |
| 198928 | path | 5 | +0.190 (0.026) | +0.086 (0.027) | +0.192 | +0.091 | +0.101 (0.032) | TRACKS_ERROR |
| 297600 | anchor | 1 | +0.118 (0.027) | +0.222 (0.026) | +0.104 | +0.215 | -0.112 (0.039) | TRACKS_DENSITY |
| 297600 | path | 1 | +0.118 (0.027) | +0.222 (0.026) | +0.104 | +0.215 | -0.112 (0.039) | TRACKS_DENSITY |
| 297600 | anchor | 5 | +0.092 (0.024) | +0.222 (0.025) | +0.090 | +0.222 | -0.131 (0.035) | TRACKS_DENSITY |
| 297600 | path | 5 | +0.162 (0.025) | +0.138 (0.027) | +0.161 | +0.137 | +0.024 (0.033) | AMBIGUOUS |
| 401536 | anchor | 1 | +0.140 (0.022) | +0.273 (0.027) | +0.116 | +0.262 | -0.146 (0.035) | TRACKS_DENSITY |
| 401536 | path | 1 | +0.140 (0.022) | +0.273 (0.027) | +0.116 | +0.262 | -0.146 (0.035) | TRACKS_DENSITY |
| 401536 | anchor | 5 | +0.105 (0.018) | +0.273 (0.028) | +0.092 | +0.269 | -0.177 (0.034) | TRACKS_DENSITY |
| 401536 | path | 5 | +0.171 (0.021) | +0.202 (0.030) | +0.162 | +0.195 | -0.032 (0.033) | AMBIGUOUS |
| 494928 | anchor | 1 | +0.138 (0.023) | +0.315 (0.024) | +0.110 | +0.304 | -0.194 (0.032) | TRACKS_DENSITY |
| 494928 | path | 1 | +0.138 (0.023) | +0.315 (0.024) | +0.110 | +0.304 | -0.194 (0.032) | TRACKS_DENSITY |
| 494928 | anchor | 5 | +0.103 (0.023) | +0.315 (0.023) | +0.086 | +0.310 | -0.224 (0.030) | TRACKS_DENSITY |
| 494928 | path | 5 | +0.147 (0.028) | +0.262 (0.025) | +0.134 | +0.254 | -0.121 (0.033) | TRACKS_DENSITY |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_finger_seed2

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100464 | anchor | 1 | +0.175 (0.033) | +0.040 (0.026) | +0.176 | +0.044 | +0.132 (0.034) | TRACKS_ERROR |
| 100464 | path | 1 | +0.175 (0.033) | +0.040 (0.026) | +0.176 | +0.044 | +0.132 (0.034) | TRACKS_ERROR |
| 100464 | anchor | 5 | +0.155 (0.028) | +0.040 (0.025) | +0.157 | +0.050 | +0.108 (0.039) | TRACKS_ERROR |
| 100464 | path | 5 | +0.227 (0.030) | -0.028 (0.020) | +0.226 | -0.015 | +0.242 (0.035) | TRACKS_ERROR |
| 198992 | anchor | 1 | +0.164 (0.030) | +0.094 (0.028) | +0.163 | +0.092 | +0.071 (0.037) | AMBIGUOUS |
| 198992 | path | 1 | +0.164 (0.030) | +0.094 (0.028) | +0.163 | +0.092 | +0.071 (0.037) | AMBIGUOUS |
| 198992 | anchor | 5 | +0.156 (0.025) | +0.094 (0.026) | +0.160 | +0.100 | +0.060 (0.035) | AMBIGUOUS |
| 198992 | path | 5 | +0.234 (0.028) | +0.013 (0.023) | +0.234 | +0.019 | +0.215 (0.033) | TRACKS_ERROR |
| 297536 | anchor | 1 | +0.139 (0.026) | +0.146 (0.028) | +0.140 | +0.146 | -0.007 (0.034) | AMBIGUOUS |
| 297536 | path | 1 | +0.139 (0.026) | +0.146 (0.028) | +0.140 | +0.146 | -0.007 (0.034) | AMBIGUOUS |
| 297536 | anchor | 5 | +0.124 (0.023) | +0.146 (0.027) | +0.130 | +0.151 | -0.021 (0.037) | AMBIGUOUS |
| 297536 | path | 5 | +0.188 (0.022) | +0.070 (0.027) | +0.191 | +0.078 | +0.113 (0.035) | TRACKS_ERROR |
| 401296 | anchor | 1 | +0.111 (0.024) | +0.198 (0.028) | +0.109 | +0.197 | -0.088 (0.038) | TRACKS_DENSITY |
| 401296 | path | 1 | +0.111 (0.024) | +0.198 (0.028) | +0.109 | +0.197 | -0.088 (0.038) | TRACKS_DENSITY |
| 401296 | anchor | 5 | +0.084 (0.023) | +0.198 (0.029) | +0.095 | +0.203 | -0.108 (0.038) | TRACKS_DENSITY |
| 401296 | path | 5 | +0.143 (0.024) | +0.134 (0.030) | +0.150 | +0.142 | +0.008 (0.036) | AMBIGUOUS |
| 499840 | anchor | 1 | +0.132 (0.025) | +0.190 (0.024) | +0.126 | +0.186 | -0.060 (0.031) | AMBIGUOUS |
| 499840 | path | 1 | +0.132 (0.025) | +0.190 (0.024) | +0.126 | +0.186 | -0.060 (0.031) | AMBIGUOUS |
| 499840 | anchor | 5 | +0.081 (0.022) | +0.190 (0.024) | +0.090 | +0.194 | -0.104 (0.031) | TRACKS_DENSITY |
| 499840 | path | 5 | +0.154 (0.021) | +0.111 (0.028) | +0.159 | +0.119 | +0.040 (0.032) | AMBIGUOUS |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_finger_seed3

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100864 | anchor | 1 | +0.176 (0.031) | -0.134 (0.032) | +0.183 | -0.143 | +0.326 (0.044) | TRACKS_ERROR |
| 100864 | path | 1 | +0.176 (0.031) | -0.134 (0.032) | +0.183 | -0.143 | +0.326 (0.044) | TRACKS_ERROR |
| 100864 | anchor | 5 | +0.173 (0.030) | -0.134 (0.031) | +0.168 | -0.128 | +0.297 (0.042) | TRACKS_ERROR |
| 100864 | path | 5 | +0.232 (0.031) | -0.210 (0.028) | +0.228 | -0.205 | +0.433 (0.038) | TRACKS_ERROR |
| 199776 | anchor | 1 | +0.128 (0.030) | +0.024 (0.030) | +0.127 | +0.016 | +0.111 (0.032) | TRACKS_ERROR |
| 199776 | path | 1 | +0.128 (0.030) | +0.024 (0.030) | +0.127 | +0.016 | +0.111 (0.032) | TRACKS_ERROR |
| 199776 | anchor | 5 | +0.121 (0.030) | +0.024 (0.028) | +0.121 | +0.023 | +0.098 (0.036) | TRACKS_ERROR |
| 199776 | path | 5 | +0.197 (0.030) | -0.045 (0.028) | +0.198 | -0.047 | +0.245 (0.031) | TRACKS_ERROR |
| 298688 | anchor | 1 | +0.141 (0.022) | +0.076 (0.030) | +0.137 | +0.069 | +0.068 (0.031) | TRACKS_ERROR |
| 298688 | path | 1 | +0.141 (0.022) | +0.076 (0.030) | +0.137 | +0.069 | +0.068 (0.031) | TRACKS_ERROR |
| 298688 | anchor | 5 | +0.107 (0.025) | +0.076 (0.029) | +0.107 | +0.076 | +0.030 (0.038) | AMBIGUOUS |
| 298688 | path | 5 | +0.167 (0.027) | +0.008 (0.031) | +0.167 | +0.008 | +0.160 (0.037) | TRACKS_ERROR |
| 402096 | anchor | 1 | +0.135 (0.024) | +0.127 (0.032) | +0.126 | +0.118 | +0.008 (0.033) | AMBIGUOUS |
| 402096 | path | 1 | +0.135 (0.024) | +0.127 (0.032) | +0.126 | +0.118 | +0.008 (0.033) | AMBIGUOUS |
| 402096 | anchor | 5 | +0.105 (0.024) | +0.127 (0.030) | +0.105 | +0.127 | -0.022 (0.036) | AMBIGUOUS |
| 402096 | path | 5 | +0.175 (0.024) | +0.069 (0.032) | +0.175 | +0.069 | +0.106 (0.039) | TRACKS_ERROR |
| 496512 | anchor | 1 | +0.159 (0.021) | +0.244 (0.025) | +0.137 | +0.231 | -0.094 (0.031) | TRACKS_DENSITY |
| 496512 | path | 1 | +0.159 (0.021) | +0.244 (0.025) | +0.137 | +0.231 | -0.094 (0.031) | TRACKS_DENSITY |
| 496512 | anchor | 5 | +0.105 (0.021) | +0.244 (0.023) | +0.099 | +0.242 | -0.143 (0.029) | TRACKS_DENSITY |
| 496512 | path | 5 | +0.178 (0.023) | +0.193 (0.027) | +0.174 | +0.190 | -0.016 (0.036) | AMBIGUOUS |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_finger_seed4

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100848 | anchor | 1 | +0.217 (0.028) | -0.041 (0.045) | +0.217 | -0.043 | +0.260 (0.046) | TRACKS_ERROR |
| 100848 | path | 1 | +0.217 (0.028) | -0.041 (0.045) | +0.217 | -0.043 | +0.260 (0.046) | TRACKS_ERROR |
| 100848 | anchor | 5 | +0.192 (0.024) | -0.041 (0.045) | +0.191 | -0.036 | +0.227 (0.052) | TRACKS_ERROR |
| 100848 | path | 5 | +0.277 (0.025) | -0.129 (0.043) | +0.275 | -0.124 | +0.400 (0.049) | TRACKS_ERROR |
| 199760 | anchor | 1 | +0.183 (0.028) | +0.134 (0.043) | +0.173 | +0.120 | +0.053 (0.050) | AMBIGUOUS |
| 199760 | path | 1 | +0.183 (0.028) | +0.134 (0.043) | +0.173 | +0.120 | +0.053 (0.050) | AMBIGUOUS |
| 199760 | anchor | 5 | +0.163 (0.030) | +0.134 (0.040) | +0.159 | +0.129 | +0.030 (0.049) | AMBIGUOUS |
| 199760 | path | 5 | +0.243 (0.030) | +0.042 (0.042) | +0.242 | +0.033 | +0.209 (0.048) | TRACKS_ERROR |
| 298656 | anchor | 1 | +0.178 (0.025) | +0.198 (0.031) | +0.167 | +0.188 | -0.021 (0.042) | AMBIGUOUS |
| 298656 | path | 1 | +0.178 (0.025) | +0.198 (0.031) | +0.167 | +0.188 | -0.021 (0.042) | AMBIGUOUS |
| 298656 | anchor | 5 | +0.156 (0.025) | +0.198 (0.030) | +0.151 | +0.193 | -0.043 (0.046) | AMBIGUOUS |
| 298656 | path | 5 | +0.251 (0.022) | +0.114 (0.036) | +0.248 | +0.107 | +0.141 (0.045) | TRACKS_ERROR |
| 401984 | anchor | 1 | +0.175 (0.023) | +0.185 (0.034) | +0.166 | +0.177 | -0.011 (0.043) | AMBIGUOUS |
| 401984 | path | 1 | +0.175 (0.023) | +0.185 (0.034) | +0.166 | +0.177 | -0.011 (0.043) | AMBIGUOUS |
| 401984 | anchor | 5 | +0.148 (0.024) | +0.185 (0.035) | +0.145 | +0.183 | -0.038 (0.045) | AMBIGUOUS |
| 401984 | path | 5 | +0.219 (0.022) | +0.120 (0.039) | +0.218 | +0.117 | +0.101 (0.048) | TRACKS_ERROR |
| 496352 | anchor | 1 | +0.169 (0.023) | +0.255 (0.030) | +0.151 | +0.244 | -0.093 (0.038) | TRACKS_DENSITY |
| 496352 | path | 1 | +0.169 (0.023) | +0.255 (0.030) | +0.151 | +0.244 | -0.093 (0.038) | TRACKS_DENSITY |
| 496352 | anchor | 5 | +0.156 (0.021) | +0.255 (0.029) | +0.145 | +0.249 | -0.104 (0.036) | TRACKS_DENSITY |
| 496352 | path | 5 | +0.233 (0.023) | +0.171 (0.035) | +0.226 | +0.162 | +0.065 (0.040) | AMBIGUOUS |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Cell: p2e_finger_seed5

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100896 | anchor | 1 | +0.197 (0.029) | -0.235 (0.037) | +0.187 | -0.226 | +0.413 (0.036) | TRACKS_ERROR |
| 100896 | path | 1 | +0.197 (0.029) | -0.235 (0.037) | +0.187 | -0.226 | +0.413 (0.036) | TRACKS_ERROR |
| 100896 | anchor | 5 | +0.213 (0.023) | -0.235 (0.035) | +0.190 | -0.215 | +0.405 (0.037) | TRACKS_ERROR |
| 100896 | path | 5 | +0.277 (0.023) | -0.290 (0.033) | +0.254 | -0.267 | +0.521 (0.035) | TRACKS_ERROR |
| 200000 | anchor | 1 | +0.156 (0.030) | -0.008 (0.030) | +0.155 | -0.004 | +0.160 (0.040) | TRACKS_ERROR |
| 200000 | path | 1 | +0.156 (0.030) | -0.008 (0.030) | +0.155 | -0.004 | +0.160 (0.040) | TRACKS_ERROR |
| 200000 | anchor | 5 | +0.154 (0.026) | -0.008 (0.030) | +0.154 | +0.003 | +0.151 (0.037) | TRACKS_ERROR |
| 200000 | path | 5 | +0.222 (0.026) | -0.090 (0.029) | +0.217 | -0.076 | +0.293 (0.038) | TRACKS_ERROR |
| 299136 | anchor | 1 | +0.136 (0.028) | +0.063 (0.033) | +0.135 | +0.062 | +0.073 (0.041) | AMBIGUOUS |
| 299136 | path | 1 | +0.136 (0.028) | +0.063 (0.033) | +0.135 | +0.062 | +0.073 (0.041) | AMBIGUOUS |
| 299136 | anchor | 5 | +0.136 (0.024) | +0.063 (0.029) | +0.138 | +0.067 | +0.071 (0.040) | AMBIGUOUS |
| 299136 | path | 5 | +0.228 (0.024) | -0.004 (0.029) | +0.228 | +0.001 | +0.227 (0.037) | TRACKS_ERROR |
| 398096 | anchor | 1 | +0.113 (0.025) | +0.159 (0.028) | +0.111 | +0.157 | -0.046 (0.034) | AMBIGUOUS |
| 398096 | path | 1 | +0.113 (0.025) | +0.159 (0.028) | +0.111 | +0.157 | -0.046 (0.034) | AMBIGUOUS |
| 398096 | anchor | 5 | +0.101 (0.024) | +0.159 (0.028) | +0.103 | +0.160 | -0.057 (0.036) | AMBIGUOUS |
| 398096 | path | 5 | +0.164 (0.022) | +0.085 (0.033) | +0.165 | +0.087 | +0.078 (0.036) | TRACKS_ERROR |
| 497296 | anchor | 1 | +0.115 (0.023) | +0.193 (0.024) | +0.111 | +0.191 | -0.080 (0.032) | TRACKS_DENSITY |
| 497296 | path | 1 | +0.115 (0.023) | +0.193 (0.024) | +0.111 | +0.191 | -0.080 (0.032) | TRACKS_DENSITY |
| 497296 | anchor | 5 | +0.094 (0.023) | +0.193 (0.025) | +0.101 | +0.197 | -0.096 (0.033) | TRACKS_DENSITY |
| 497296 | path | 5 | +0.167 (0.022) | +0.152 (0.028) | +0.174 | +0.158 | +0.015 (0.034) | AMBIGUOUS |

Final-checkpoint verdict (anchor, h=5): **TRACKS_DENSITY**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **FAIL**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/finger_p2e/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --cell finger_p2e --dumps p2e_finger_seed1=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_finger_seed1/latent_uq/finger_v1 p2e_finger_seed2=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_finger_seed2/latent_uq/finger_v1 p2e_finger_seed3=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_finger_seed3/latent_uq/finger_v1 p2e_finger_seed4=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_finger_seed4/latent_uq/finger_v1 p2e_finger_seed5=/scratch/midway3/rickybao/dreamerv3_runs/pretrain_p2e_finger_seed5/latent_uq/finger_v1 --horizons 1 5 --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage0_20260710_221232/finger_p2e`
- git commit: `d102ac4fd5cf27d13eae50171d4dacc56b18d79e`
