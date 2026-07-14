# Latent-UQ Stage 0 evidence memo — 2026-07-14

**Replication verdict (Biased Dreams, arXiv 2604.25416): DOES NOT REPLICATE: disagreement tracks held-out error over training density on this stack.**

## Protocol

- Dumps: `p2e_finger_seed1`, `p2e_finger_seed2`, `p2e_finger_seed3`, `p2e_finger_seed4`, `p2e_finger_seed5`
- Probe set(s): finger_v1_7a32ff08 (frozen, hash-verified at dump time)
- Horizons: [1, 5] (open-loop, decoder target space); headline read: anchor one-step disagreement vs horizon 5 error; addendum read: anchor one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4 item 5)
- Density proxy: mean kNN distance of anchor posterior means against training-buffer encodings (larger = sparser); bias signature = positive partials with disagreement
- SEs: stream-clustered bootstrap, n_boot=300; verdict threshold 2.0 SE on Delta = pcorr(D,E|rho) − pcorr(D,rho|E)
- Instrument: **density_residual** (out-of-fold isotonic residual on the density proxy, 5 stream-cluster folds — Stage-1A primary, PREREG_gate_d1_stage1a_20260714.md)

## Cell: p2e_finger_seed1

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100432 | anchor | 1 | +0.173 (0.033) | -0.102 (0.031) | +0.177 | -0.109 | +0.286 (0.040) | TRACKS_ERROR |
| 100432 | path | 1 | +0.173 (0.033) | -0.102 (0.031) | +0.177 | -0.109 | +0.286 (0.040) | TRACKS_ERROR |
| 100432 | anchor | 5 | +0.139 (0.027) | -0.102 (0.031) | +0.135 | -0.097 | +0.233 (0.040) | TRACKS_ERROR |
| 100432 | path | 5 | +0.216 (0.028) | -0.167 (0.027) | +0.212 | -0.161 | +0.373 (0.036) | TRACKS_ERROR |
| 198928 | anchor | 1 | +0.136 (0.029) | -0.058 (0.027) | +0.139 | -0.065 | +0.203 (0.038) | TRACKS_ERROR |
| 198928 | path | 1 | +0.136 (0.029) | -0.058 (0.027) | +0.139 | -0.065 | +0.203 (0.038) | TRACKS_ERROR |
| 198928 | anchor | 5 | +0.098 (0.025) | -0.058 (0.029) | +0.097 | -0.057 | +0.154 (0.036) | TRACKS_ERROR |
| 198928 | path | 5 | +0.186 (0.025) | -0.069 (0.025) | +0.185 | -0.067 | +0.252 (0.032) | TRACKS_ERROR |
| 297600 | anchor | 1 | +0.093 (0.027) | -0.078 (0.028) | +0.100 | -0.086 | +0.186 (0.041) | TRACKS_ERROR |
| 297600 | path | 1 | +0.093 (0.027) | -0.078 (0.028) | +0.100 | -0.086 | +0.186 (0.041) | TRACKS_ERROR |
| 297600 | anchor | 5 | +0.079 (0.024) | -0.078 (0.027) | +0.081 | -0.080 | +0.161 (0.039) | TRACKS_ERROR |
| 297600 | path | 5 | +0.157 (0.024) | -0.055 (0.027) | +0.159 | -0.059 | +0.218 (0.036) | TRACKS_ERROR |
| 401536 | anchor | 1 | +0.110 (0.025) | -0.049 (0.029) | +0.116 | -0.061 | +0.177 (0.035) | TRACKS_ERROR |
| 401536 | path | 1 | +0.110 (0.025) | -0.049 (0.029) | +0.116 | -0.061 | +0.177 (0.035) | TRACKS_ERROR |
| 401536 | anchor | 5 | +0.084 (0.020) | -0.049 (0.030) | +0.088 | -0.054 | +0.142 (0.036) | TRACKS_ERROR |
| 401536 | path | 5 | +0.163 (0.022) | -0.036 (0.030) | +0.166 | -0.047 | +0.213 (0.034) | TRACKS_ERROR |
| 494928 | anchor | 1 | +0.106 (0.027) | -0.062 (0.030) | +0.114 | -0.074 | +0.188 (0.036) | TRACKS_ERROR |
| 494928 | path | 1 | +0.106 (0.027) | -0.062 (0.030) | +0.114 | -0.074 | +0.188 (0.036) | TRACKS_ERROR |
| 494928 | anchor | 5 | +0.077 (0.022) | -0.062 (0.029) | +0.082 | -0.068 | +0.150 (0.036) | TRACKS_ERROR |
| 494928 | path | 5 | +0.136 (0.028) | -0.033 (0.027) | +0.138 | -0.044 | +0.182 (0.037) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_finger_seed2

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100464 | anchor | 1 | +0.179 (0.032) | -0.067 (0.029) | +0.178 | -0.065 | +0.243 (0.035) | TRACKS_ERROR |
| 100464 | path | 1 | +0.179 (0.032) | -0.067 (0.029) | +0.178 | -0.065 | +0.243 (0.035) | TRACKS_ERROR |
| 100464 | anchor | 5 | +0.160 (0.028) | -0.067 (0.027) | +0.157 | -0.059 | +0.216 (0.039) | TRACKS_ERROR |
| 100464 | path | 5 | +0.229 (0.029) | -0.074 (0.021) | +0.225 | -0.063 | +0.288 (0.035) | TRACKS_ERROR |
| 198992 | anchor | 1 | +0.159 (0.030) | -0.044 (0.033) | +0.160 | -0.048 | +0.209 (0.040) | TRACKS_ERROR |
| 198992 | path | 1 | +0.159 (0.030) | -0.044 (0.033) | +0.160 | -0.048 | +0.209 (0.040) | TRACKS_ERROR |
| 198992 | anchor | 5 | +0.157 (0.025) | -0.044 (0.031) | +0.156 | -0.041 | +0.197 (0.039) | TRACKS_ERROR |
| 198992 | path | 5 | +0.235 (0.028) | -0.057 (0.025) | +0.233 | -0.052 | +0.285 (0.035) | TRACKS_ERROR |
| 297536 | anchor | 1 | +0.135 (0.026) | -0.045 (0.028) | +0.135 | -0.047 | +0.182 (0.034) | TRACKS_ERROR |
| 297536 | path | 1 | +0.135 (0.026) | -0.045 (0.028) | +0.135 | -0.047 | +0.182 (0.034) | TRACKS_ERROR |
| 297536 | anchor | 5 | +0.125 (0.024) | -0.045 (0.027) | +0.123 | -0.042 | +0.165 (0.038) | TRACKS_ERROR |
| 297536 | path | 5 | +0.188 (0.022) | -0.040 (0.028) | +0.187 | -0.035 | +0.222 (0.037) | TRACKS_ERROR |
| 401296 | anchor | 1 | +0.103 (0.024) | -0.046 (0.032) | +0.104 | -0.049 | +0.153 (0.040) | TRACKS_ERROR |
| 401296 | path | 1 | +0.103 (0.024) | -0.046 (0.032) | +0.104 | -0.049 | +0.153 (0.040) | TRACKS_ERROR |
| 401296 | anchor | 5 | +0.089 (0.022) | -0.046 (0.034) | +0.087 | -0.042 | +0.129 (0.041) | TRACKS_ERROR |
| 401296 | path | 5 | +0.145 (0.023) | -0.041 (0.031) | +0.144 | -0.035 | +0.179 (0.037) | TRACKS_ERROR |
| 499840 | anchor | 1 | +0.119 (0.025) | -0.060 (0.028) | +0.122 | -0.066 | +0.188 (0.033) | TRACKS_ERROR |
| 499840 | path | 1 | +0.119 (0.025) | -0.060 (0.028) | +0.122 | -0.066 | +0.188 (0.033) | TRACKS_ERROR |
| 499840 | anchor | 5 | +0.084 (0.021) | -0.060 (0.029) | +0.081 | -0.057 | +0.139 (0.032) | TRACKS_ERROR |
| 499840 | path | 5 | +0.159 (0.021) | -0.050 (0.030) | +0.157 | -0.044 | +0.201 (0.033) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_finger_seed3

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100864 | anchor | 1 | +0.166 (0.031) | -0.200 (0.031) | +0.178 | -0.210 | +0.388 (0.043) | TRACKS_ERROR |
| 100864 | path | 1 | +0.166 (0.031) | -0.200 (0.031) | +0.178 | -0.210 | +0.388 (0.043) | TRACKS_ERROR |
| 100864 | anchor | 5 | +0.169 (0.029) | -0.200 (0.029) | +0.163 | -0.195 | +0.358 (0.040) | TRACKS_ERROR |
| 100864 | path | 5 | +0.235 (0.030) | -0.102 (0.031) | +0.232 | -0.094 | +0.326 (0.043) | TRACKS_ERROR |
| 199776 | anchor | 1 | +0.119 (0.030) | -0.092 (0.031) | +0.126 | -0.101 | +0.227 (0.034) | TRACKS_ERROR |
| 199776 | path | 1 | +0.119 (0.030) | -0.092 (0.031) | +0.126 | -0.101 | +0.227 (0.034) | TRACKS_ERROR |
| 199776 | anchor | 5 | +0.116 (0.029) | -0.092 (0.028) | +0.117 | -0.094 | +0.211 (0.036) | TRACKS_ERROR |
| 199776 | path | 5 | +0.193 (0.029) | -0.119 (0.027) | +0.195 | -0.123 | +0.318 (0.032) | TRACKS_ERROR |
| 298688 | anchor | 1 | +0.126 (0.022) | -0.067 (0.031) | +0.130 | -0.075 | +0.205 (0.033) | TRACKS_ERROR |
| 298688 | path | 1 | +0.126 (0.022) | -0.067 (0.031) | +0.130 | -0.075 | +0.205 (0.033) | TRACKS_ERROR |
| 298688 | anchor | 5 | +0.100 (0.024) | -0.067 (0.032) | +0.101 | -0.068 | +0.169 (0.042) | TRACKS_ERROR |
| 298688 | path | 5 | +0.160 (0.026) | -0.077 (0.033) | +0.161 | -0.079 | +0.239 (0.039) | TRACKS_ERROR |
| 402096 | anchor | 1 | +0.116 (0.023) | -0.068 (0.034) | +0.121 | -0.077 | +0.199 (0.035) | TRACKS_ERROR |
| 402096 | path | 1 | +0.116 (0.023) | -0.068 (0.034) | +0.121 | -0.077 | +0.199 (0.035) | TRACKS_ERROR |
| 402096 | anchor | 5 | +0.097 (0.023) | -0.068 (0.033) | +0.098 | -0.069 | +0.167 (0.037) | TRACKS_ERROR |
| 402096 | path | 5 | +0.168 (0.023) | -0.059 (0.030) | +0.169 | -0.061 | +0.230 (0.038) | TRACKS_ERROR |
| 496512 | anchor | 1 | +0.122 (0.023) | -0.068 (0.027) | +0.131 | -0.083 | +0.214 (0.031) | TRACKS_ERROR |
| 496512 | path | 1 | +0.122 (0.023) | -0.068 (0.027) | +0.131 | -0.083 | +0.214 (0.031) | TRACKS_ERROR |
| 496512 | anchor | 5 | +0.088 (0.022) | -0.068 (0.026) | +0.091 | -0.072 | +0.162 (0.031) | TRACKS_ERROR |
| 496512 | path | 5 | +0.158 (0.023) | -0.043 (0.028) | +0.159 | -0.049 | +0.208 (0.036) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_finger_seed4

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100848 | anchor | 1 | +0.213 (0.027) | -0.132 (0.044) | +0.215 | -0.136 | +0.351 (0.046) | TRACKS_ERROR |
| 100848 | path | 1 | +0.213 (0.027) | -0.132 (0.044) | +0.215 | -0.136 | +0.351 (0.046) | TRACKS_ERROR |
| 100848 | anchor | 5 | +0.187 (0.024) | -0.132 (0.044) | +0.184 | -0.128 | +0.312 (0.053) | TRACKS_ERROR |
| 100848 | path | 5 | +0.273 (0.025) | -0.193 (0.041) | +0.272 | -0.192 | +0.464 (0.049) | TRACKS_ERROR |
| 199760 | anchor | 1 | +0.160 (0.028) | -0.091 (0.046) | +0.170 | -0.108 | +0.278 (0.055) | TRACKS_ERROR |
| 199760 | path | 1 | +0.160 (0.028) | -0.091 (0.046) | +0.170 | -0.108 | +0.278 (0.055) | TRACKS_ERROR |
| 199760 | anchor | 5 | +0.145 (0.028) | -0.091 (0.043) | +0.150 | -0.099 | +0.249 (0.055) | TRACKS_ERROR |
| 199760 | path | 5 | +0.230 (0.028) | -0.086 (0.042) | +0.235 | -0.099 | +0.334 (0.051) | TRACKS_ERROR |
| 298656 | anchor | 1 | +0.157 (0.024) | -0.052 (0.037) | +0.161 | -0.065 | +0.226 (0.046) | TRACKS_ERROR |
| 298656 | path | 1 | +0.157 (0.024) | -0.052 (0.037) | +0.161 | -0.065 | +0.226 (0.046) | TRACKS_ERROR |
| 298656 | anchor | 5 | +0.137 (0.024) | -0.052 (0.034) | +0.140 | -0.059 | +0.199 (0.052) | TRACKS_ERROR |
| 298656 | path | 5 | +0.234 (0.023) | -0.058 (0.036) | +0.238 | -0.071 | +0.308 (0.047) | TRACKS_ERROR |
| 401984 | anchor | 1 | +0.147 (0.022) | -0.082 (0.039) | +0.153 | -0.092 | +0.246 (0.049) | TRACKS_ERROR |
| 401984 | path | 1 | +0.147 (0.022) | -0.082 (0.039) | +0.153 | -0.092 | +0.246 (0.049) | TRACKS_ERROR |
| 401984 | anchor | 5 | +0.120 (0.024) | -0.082 (0.038) | +0.123 | -0.086 | +0.209 (0.052) | TRACKS_ERROR |
| 401984 | path | 5 | +0.201 (0.022) | -0.057 (0.039) | +0.203 | -0.064 | +0.267 (0.051) | TRACKS_ERROR |
| 496352 | anchor | 1 | +0.130 (0.024) | -0.065 (0.032) | +0.137 | -0.078 | +0.216 (0.041) | TRACKS_ERROR |
| 496352 | path | 1 | +0.130 (0.024) | -0.065 (0.032) | +0.137 | -0.078 | +0.216 (0.041) | TRACKS_ERROR |
| 496352 | anchor | 5 | +0.127 (0.022) | -0.065 (0.031) | +0.132 | -0.074 | +0.206 (0.042) | TRACKS_ERROR |
| 496352 | path | 5 | +0.215 (0.026) | -0.050 (0.035) | +0.219 | -0.065 | +0.284 (0.043) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Cell: p2e_finger_seed5

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| 100896 | anchor | 1 | +0.185 (0.027) | -0.001 (0.033) | +0.185 | +0.012 | +0.173 (0.035) | TRACKS_ERROR |
| 100896 | path | 1 | +0.185 (0.027) | -0.001 (0.033) | +0.185 | +0.012 | +0.173 (0.035) | TRACKS_ERROR |
| 100896 | anchor | 5 | +0.184 (0.022) | -0.001 (0.032) | +0.185 | +0.023 | +0.162 (0.034) | TRACKS_ERROR |
| 100896 | path | 5 | +0.250 (0.021) | -0.022 (0.030) | +0.249 | +0.010 | +0.239 (0.032) | TRACKS_ERROR |
| 200000 | anchor | 1 | +0.163 (0.030) | -0.133 (0.032) | +0.161 | -0.131 | +0.292 (0.043) | TRACKS_ERROR |
| 200000 | path | 1 | +0.163 (0.030) | -0.133 (0.032) | +0.161 | -0.131 | +0.292 (0.043) | TRACKS_ERROR |
| 200000 | anchor | 5 | +0.163 (0.026) | -0.133 (0.032) | +0.155 | -0.123 | +0.278 (0.040) | TRACKS_ERROR |
| 200000 | path | 5 | +0.227 (0.026) | -0.181 (0.030) | +0.218 | -0.170 | +0.388 (0.039) | TRACKS_ERROR |
| 299136 | anchor | 1 | +0.133 (0.028) | -0.102 (0.041) | +0.135 | -0.105 | +0.241 (0.047) | TRACKS_ERROR |
| 299136 | path | 1 | +0.133 (0.028) | -0.102 (0.041) | +0.135 | -0.105 | +0.241 (0.047) | TRACKS_ERROR |
| 299136 | anchor | 5 | +0.135 (0.025) | -0.102 (0.037) | +0.133 | -0.100 | +0.233 (0.047) | TRACKS_ERROR |
| 299136 | path | 5 | +0.227 (0.025) | -0.116 (0.034) | +0.226 | -0.114 | +0.340 (0.043) | TRACKS_ERROR |
| 398096 | anchor | 1 | +0.111 (0.027) | -0.104 (0.034) | +0.114 | -0.107 | +0.221 (0.041) | TRACKS_ERROR |
| 398096 | path | 1 | +0.111 (0.027) | -0.104 (0.034) | +0.114 | -0.107 | +0.221 (0.041) | TRACKS_ERROR |
| 398096 | anchor | 5 | +0.100 (0.023) | -0.104 (0.034) | +0.101 | -0.104 | +0.205 (0.041) | TRACKS_ERROR |
| 398096 | path | 5 | +0.161 (0.023) | -0.084 (0.038) | +0.162 | -0.084 | +0.246 (0.040) | TRACKS_ERROR |
| 497296 | anchor | 1 | +0.109 (0.022) | -0.108 (0.031) | +0.114 | -0.113 | +0.226 (0.038) | TRACKS_ERROR |
| 497296 | path | 1 | +0.109 (0.022) | -0.108 (0.031) | +0.114 | -0.113 | +0.226 (0.038) | TRACKS_ERROR |
| 497296 | anchor | 5 | +0.095 (0.023) | -0.108 (0.033) | +0.093 | -0.106 | +0.199 (0.039) | TRACKS_ERROR |
| 497296 | path | 5 | +0.169 (0.023) | -0.070 (0.034) | +0.168 | -0.067 | +0.235 (0.040) | TRACKS_ERROR |

Final-checkpoint verdict (anchor, h=5): **TRACKS_ERROR**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint, anchor h=1; the App. A.7 gate criterion additionally applies the dose adjustment, evaluated in the D0 pipeline, not here): **PASS**

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: disagreement is explained by training density beyond what error explains. Route per the brief: Stage 1 fixes become the priority and the D0 ensemble read is suspect.
- Per-window reads are never reported; all claims above are aggregates with stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open questions); a TRACKS_DENSITY verdict should be sanity-checked against a second k before externalizing.

## Files

- analysis: `/home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/finger_p2e/analysis.json`
- command: `/home/rickybao/projects/dreamerv3/probing/latent_uq_analysis.py --dumps p2e_finger_seed1=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_finger_seed1/latent_uq/finger_v1 p2e_finger_seed2=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_finger_seed2/latent_uq/finger_v1 p2e_finger_seed3=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_finger_seed3/latent_uq/finger_v1 p2e_finger_seed4=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_finger_seed4/latent_uq/finger_v1 p2e_finger_seed5=/home/rickybao/projects/dreamerv3/local_results/evpi_stage0_20260710_221924/raw/pretrain_p2e_finger_seed5/latent_uq/finger_v1 --horizons 1 5 --n_boot 300 --instrument density_residual --output /home/rickybao/projects/dreamerv3/artifacts/latent_uq_stage1a_20260714/finger_p2e`
- git commit: `c92f984794e7f1b57db82ccfc3f8faae3c65df52`
