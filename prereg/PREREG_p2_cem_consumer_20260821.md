# PREREG — WCEM: CEM consumer on the frozen dv3 RSSM (21 Aug 2026)

**Status: FROZEN at user commit. ONE read execution. EXECUTION
SEQUENCED BEHIND the Wave-1 (first-update) read** — the slate's order
(calibrate → attribute → substrate, ThreeAxes §5): this cell's
interpretation depends on Wave 1's attribution branch, so it is
registered now and submitted only after that read lands. Fence: SEED
(the anti-harvest separate-paper seed, ThreeAxes item 10 — "the
sharpest generality test of anti-harvest, and it needs no new
training"). Reviewer: ONE Opus pass, user pre-approved, triggered at
build.

## Question

S3/S4 (TM2's planner anti-harvests; below its own prior mode) could be
a property of THAT planner or of MPC consumption through a learned
model per se. A CEM planner over the FROZEN dv3 RSSM — a family whose
actor consumer showed NO anti-harvest (P-W1b dv3 +0.024 ns) — separates
the two with no new training.

## Instruments (all built + smoked at registration)

- `dreamerv3/agent.py::_cem_signals` (additive; `d0.cem_iters 0`
  disables — default path unchanged; configs.yaml keys added, old saved
  configs backfill to 0 via the labeler's schema-drift path): CEM over
  H-step action sequences imagined through the prior, scored with the
  _d0_signals head convention generalized to H steps (w_1 = 1,
  w_{t+1} = w_t·disc·con_t; valens-mean bootstrap at H). Emits
  'd0/cem_act'. Sampling draws from nj.seed() ⇒ the n_actions counter
  marks give CEM the same CRN discipline as policy sampling
  (**verified**: `d0/wcem_smoke.py` PASS — bit-identical replay under a
  restored mark, divergence under a fresh mark, cem_iters=0 inert).
- **Pins** (set by `d0/sweep.load_config` under `--cem_consumer`,
  recorded in meta, refused otherwise by the reader): iters 4,
  samples 64, horizon 6, elites 8, std 0.5.
- **Mode gate (build-review finding 2)**: `_cem_signals` runs only
  under the static policy mode 'cemplan' — the CEM oracle's own calls
  and the labeled-state d0_eval; candidate/actor branches and
  bootstrap d0_evals pay nothing (verified in the smoke: eval emits no
  cem keys; the sampled action at a given mark is byte-identical
  across modes, so CRN is untouched).
- `d0/oracle_labels.py`: `CEMOracle` (executes the CEM plan; shares the
  agent's RNG counter) + `_wcem_label_state` (the W1 protocol + one CEM
  consumer branch per repeat under the SAME marks — the CEM/actor
  contrast is CRN-paired by construction — plus a candidate-0 CRN
  WITNESS re-run per repeat, bit-identity enforced labeler-side and
  reader-side, and the per-state competence diagnostics cem_score /
  q_best stored). selfcheck_wcem PASS (mode gate; witness;
  bit-identical reproduction; guards).

## Design

8 cells: cup/finger × e1/e4 × seeds 31–32 (dv3 R3 LATE, RCC-confirmed);
R=8, S=200, horizon 100, `--env_seed 20260824` (fresh); names
`wcem_{dom}_{dose}_seed{31,32}_late.npz`; version '_wcem'. Dup gates:
`wcemdup_finger_e1_seed31`, `wcemdup_cup_e4_seed32`.

Cluster-level smoke BEFORE the wave (ops): one pass on any R3 run with
`--states 5` — verifies the checkpoint's heads under cem and prices the
pass; smoke output deleted (unregistered names refuse anyway).

## Estimands and decision rules

Reader `analysis/wcem_read.py` FROZEN with this file (selfcheck PASS).
Cluster = cell (n=8), permutation-primary + BCa at α=.05.

- **PLANNER-COMPETENCE GATE (build-review finding 1, adjudicated
  first)**: pooled mean_s [cem_score − q_best] ≥ 0 — the CEM plan's own
  model-predicted H-step value against the best one-step candidate Q
  (same heads, same valnorm convention). A degenerate CEM (null plan
  earning ~0 return) would otherwise be READ AS the flagship
  anti-harvest branch — the construct-invalidity shape that struck
  TM2-R3. Gate failure ⇒ **NO-CALL-PLANNER-INCOMPETENT** (no
  adjudication; diagnostics reported). The gate is conservative
  AGAINST the flagship: a false NO-CALL costs the claim, never
  licenses it.
- **PRIMARY**: d_cell = mean_s mean_r [g_cem_rep − g_actor_rep]
  (CRN-paired).
  - **CEM-ANTI-HARVESTS** (fires negative): planning through the model
    hurts on dv3 too ⇒ anti-harvest generalizes across families and
    planners — the seed paper's flagship cell.
  - **CEM-OUTPERFORMS-ACTOR** (fires positive): anti-harvest is
    planner/family-specific ⇒ S3/S4 scope narrows to TM2's MPPI.
  - **CEM-ACTOR-INDISTINGUISHABLE**: no fire; the realized MDE80 is
    printed and carried.
- Secondary (descriptive): CEM harvest vs the candidate mean on ODD
  repeats (the exact P-W1b estimator slices — review finding 3).

## Gates (refusal = read not consumed)

Exactly the 8 cells + 2 dup passes; cem meta pins (all five) +
meta.w1.cem; env_seed 20260824 / repeats 8 / states 200 / horizon 100;
version '_wcem'; filename ↔ run_id identity; cem_act within the action
box; CRN witness bit-identity; finite arrays (incl. cem_act/cem_score/
q_best/r_real); no unregistered wcem* files; duplicate-null exact
zero; **sequencing gate in code** (review finding 4): the reader
refuses without `--after_wave1 <path to the executed Wave-1 read
json>`.

## Ops (AFTER the Wave-1 read)

R3 checkpoints → cluster smoke (--states 5, deleted) → 2 dup passes →
8 cells (`--w1_repeats 8 --cem_consumer --env_seed 20260824
--oracle_all`) → bundle + sha manifest → **[ME] ONE read** (with
--after_wave1). Cost (corrected per review finding 2): with the mode
gate, CEM runs on the labeled-state d0_eval + the CEM branch's own
follower calls only — ≈ 1/10 of branch rollouts pay CEM, each CEM call
≈ 12× a signals call, so the wave ≈ 10 W1-scale passes × (1 + ~1.2
CEM overhead share) — an UNVERIFIED planning estimate; ops re-derives
from the cluster smoke and records it in the bundle NOTES.
