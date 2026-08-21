# PREREG — Paper-2 WDC: diverse-candidate variant (21 Aug 2026)

**Status: FROZEN at user commit. ONE read execution.** Motivation:
`Rescue_Ideation_ThreeAxes_20260821.md` §5, the board's acknowledged
gap — "the flatness verdict is a property of (environment × trained
policy × candidate generator), not of the environment. Converged actors
sample near-duplicate candidates, so part of the measured flatness is
policy convergence, not substrate poverty... This is a real gap in the
board." No lens owned it; this registration does. Reviewer: ONE Opus
pass, user pre-approved, triggered at build.

## Design — 16 existing dv3 R3 LATE cells, no new training

- Cells: cup/finger × e1/e4 × **seeds 31–34** (16), the same dv3 R3
  LATE checkpoints as PREREG_w1_wave_20260809 (confirmed on RCC,
  21 Aug user statement). dv3-only: dv3 is the family carrying the
  flatness/null verdict this wave interrogates.
- Labeler: `d0/oracle_labels.py` NEW additive `--dcand_uniform 4` path
  (`_wdc_label_state`; rides the W1 protocol — same mark discipline,
  probe, actor branch, dup semantics; no executed path modified;
  selfcheck_wdc PASS). Per state, in addition to the M=8 policy
  candidates:
  - **U=4 uniform candidates** ~ U[−1,1]^A from the pinned per-state
    stream `default_rng([env_seed, ep, t])` — deterministic; the
    reader RECOMPUTES them and refuses on mismatch (provenance gate);
  - each evaluated under the SAME repeat marks (CRN across all
    branches within a repeat);
  - a **candidate-0 CRN witness** re-run per repeat: must reproduce
    g_all_rep[:, 0] bit-exactly (labeler fail-fast assert + reader
    refusal gate).
- Dials: R=8 repeats, S=200 states, horizon 100, label_every 25,
  M=8 (all reader-pinned; build review 5.3/6.1), `--env_seed 20260822`
  (fresh: ≠ 20260809 W1/W2, ≠ 20260821 Wave-1). Names
  `wdc_{dom}_{dose}_seed{31..34}_late.npz`.
- **Shared-draw disclosure (review 3.2)**: the uniform set is shared
  across cells by construction (`[env_seed, ep, t]` does not depend on
  the cell and the labeled (ep, t) grid is identical across cells) —
  CRN across cells, which removes candidate-draw noise from the paired
  contrast; the residual cross-cell dependence it induces is not
  modelled by the cluster bootstrap (negligible after averaging S=200
  states; disclosed).
- **Dup gates**: 2 passes (`--w1_dup_cand`, uniform set also collapsed
  to the plug-in candidate): `wdcdup_finger_e1_seed31_late.npz`,
  `wdcdup_cup_e4_seed34_late.npz` (full filenames — the reader refuses
  anything else) — every diverse estimand exactly zero, else
  INSTRUMENT-INVALID refusal.
- Cost (corrected, review 7.2): 14 branch-rollouts per repeat vs W1's
  9 (8 policy + actor + 4 uniform + 1 witness) ⇒ ≈1.56× a W1 pass;
  18 passes total (16 cells + 2 dups) — an UNVERIFIED planning
  estimate; ops re-derives from the first measured pass.

## Estimands and decision rules

Reader `analysis/wdc_read.py` FROZEN with this file (selfcheck PASS:
estimator recovers planted truth at 0/0.5/2.0; branch fixtures; 7
refusal legs; ONE-read guard). Cluster = cell (n=16),
permutation-primary + BCa at α=.05 via the frozen w1_read machinery.

- **CEILING** per candidate set: per state, the ddof=1
  variance-components estimator (the corrected form of ThreeAxes §2.2,
  unbiased at planted truth): S²(s) = Var_m[mean_r g] − (1/R)·mean_m
  Var_r[g]; cell ceiling = √max(mean_s S², 0), G units. Sets: **P** =
  the 8 policy candidates; **D** = P + the 4 uniform candidates.
- **NOISE FLOOR (review 5.1, registered)**: √max(·,0) gives the ceiling
  a positive floor (~0.09 pooled at the realized within-repeat
  dispersions — pre-freeze context derived by the reviewer from the
  archived W1 bundle, disclosed per §1.1 governance); the reader
  computes a matched per-cell null floor (candidate axis shuffled
  within each repeat, 20 pinned shuffles, rng 20260826) and the
  above-bar conjunct is adjudicated NET of it. The sign-flip perm_p is
  suppressed on the two ceiling blocks (structurally minimal on a
  non-negative quantity — review 5.2).
- **PRIMARY**: Δceiling = ceiling_D − ceiling_P, paired per cell.
- Secondary (descriptive): split-selected opportunity vs the
  CANDIDATE-0 baseline (identical in both sets, so it cancels exactly
  in the paired D−P; NOT the m_now baseline W1 uses — review 5.4),
  paired D−P.
- Branch map (BAR = 0.20, the corrected substrate-ceiling bar; the bar
  value is context from non-registered archived compute — ThreeAxes
  §1.1/§2.2 — pinned here pre-outcome):
  - **FLATNESS-IS-POLICY**: Δ fires positive AND pooled ceiling_D NET
    of its noise floor > BAR ⇒ the substrate holds resolvable decision value the
    policy generator hides; the flatness verdict is re-scoped to the
    generator in every draft, and the environments lens's
    "empty room" reading is retired.
  - **DIVERSITY-RAISES-SUBCEILING**: Δ fires positive, net ceiling_D ≤
    BAR ⇒ diversity helps but the substrate stays under the bar; both
    factors named.
  - **GENERATOR-INSENSITIVE**: Δ null ⇒ the substrate-poverty leg is
    strengthened (an informative null; carried symmetrically).
  - **CEILING-FALLS**: Δ fires negative — mechanism-anomalous;
    reported, licenses nothing.

## Gates (refusal = read not consumed)

Exactly the 16 registered cells + the 2 registered dup passes; meta
pins env_seed 20260822 / repeats 8 / dcand_uniform 4 / states 200 /
horizon 100 / label_every 25 / M 8; labeler_version endswith '_wdc';
filename ↔ run_id identity; CRN witness bit-identity; dcands
pinned-stream reproducibility; finite arrays (g, gd, gw, dcands,
cands); no unregistered wdc* files; dup passes exactly zero.

## Ops

dv3 R3 LATE checkpoints (RCC) → **one `--states 5 --w1_repeats 2`
smoke pass per domain first** (review 3.1: the witness `SystemExit`s a
whole 200-state pass on any CRN break — clear it cheaply; smoke
outputs deleted, unregistered names refuse anyway) → 2 dup passes →
16 cells
(`--w1_repeats 8 --dcand_uniform 4 --env_seed 20260822 --oracle_all`)
→ bundle + sha manifest → **[ME] ONE read**.
