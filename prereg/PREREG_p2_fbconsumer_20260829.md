# PREREG — Paper-2 FB-consumer wave (wfb; 29 Aug 2026)

**Status: FROZEN at user commit. ONE read execution.** Reviewer: ONE
Opus pass, user pre-approved, triggered at build. Motivation: Paper
2's completed story ("a replicated identified opportunity exists; no
tested consumer harvests it; deployment optimization through the
model's value surface converts to nothing") rests on TWO consumers
that share one value-surface class — TD-critics distilled along the
deployed policy's visitation (TM2's Q-ensemble; dv3's critic behind
the CEM). FB (forward-backward successor-measure factorization,
Paper 1's FB wave) supplies a structurally different surface: Q_z =
F(s,a,z)·z, learned WITHOUT bootstrapping off the deployed policy.
This wave asks the sharpest open scope question ON THE SUBSTRATE
CARRYING THE REPLICATED OPPORTUNITY (TM2 finger, S2 = +0.062 both
cohorts): **whose value surface can see the opportunity?** Both main
branches strengthen the paper: FB-sees ⇒ an existence proof + the
failure localizes to the critic pathway; FB-blind ⇒ the flagship
generalizes across value-surface classes.

## Design — the SELECTOR estimand first (level/selection decomposed)

The 22-Aug lesson (planner-landscape F1/F5) is baked in: a consumer's
ACTION-level contrast confounds selection with level. The PRIMARY here
is therefore **FB as a SELECTOR over the identical candidate set**:
score each of the M=8 policy candidates by FB's Q_z(s, cand_m), pick
the argmax, and evaluate the pick on the SAME ground-truth candidate
panel every other selector is judged on. Directly comparable, on the
same arrays, with: the plug-in m_now baseline, the native model's own
Q-selector (qfull is stored per pass), and the split-selected ceiling
(the S2 estimator's own form).

- Cells: **TM2 finger e1 × seeds 51–58 + 59–66 = 16 cells** (both
  checkpoint cohorts; LATE ckpts) + 2 dup gates
  (`wfbdup_finger_e1_seed51`, `wfbdup_finger_e1_seed59` — one per
  cohort). **e1-only, finger-only**: the FB fits are finger-q1 and
  the obs-order gate requires the exact 12-dim OBS_KEYS layout with
  no distractor keys (cup and e4 are out of scope, disclosed).
- Labeler: the ADDITIVE `--fb_ckpts/--fb_datas/--fb_seeds` path in
  probing/tdmpc2_oracle_labels.py (selfcheck PASS incl. the wfb leg:
  in-file CRN witness BITWISE, W1 fields unchanged). Per state it
  adds: `fb_q` (K, M) selector scores; `fb_acts` (K+1, A) the K FB
  policies' own actions + slot K = candidate 0 VERBATIM (the CRN
  witness); `g_fb_rep` (R, K+1) their returns under the SAME repeat
  marks; `obs_now` (12,) for the pre-named descriptive exhibit.
  Version stamps `_wfb` (no frozen reader ingests these files); names
  `wfb_finger_e1_seed{51..66}_late.npz`.
- **K=2 pinned FB fits** (lower-median published zeroshot per side —
  a pre-outcome rule over ALREADY-PUBLIC values, P-FB1 read 17 Aug;
  review m5 tie disclosure: s1 seeds 2 and 8 BOTH published 197.3 —
  the tie is broken by (value, run-name) sort order, selecting seed8;
  both values public, so the choice cannot be outcome-driven):
  - s1 (PRIMARY side, the FB escape side):
    `fbwm_finger_q1s1_seed8/fb_ckpt.pt`, sha256
    `8a5409df637bd177b86c6addca2681b0a0df57a8ddc0adca369d9e335c179285`
    (published zeroshot 197.3), z from `fb_data/finger_q1_side1.npz`,
    z_seed 8.
  - s0 (registered secondary — the support-poor side):
    `fbwm_finger_q1s0_seed1/fb_ckpt.pt`, sha256
    `183bf521969719f3efa92a88a0c718ed8f0ab15769eb8f550fe6e22cb5947ff7`
    (published zeroshot 98.8), z from `fb_data/finger_q1_side0.npz`,
    z_seed 1.
  z-inference = the fb_zeroshot registered protocol verbatim
  (N_INFER=5120, seeded, own-side export; frac_pos/z_norm recorded).
- **Cross-paper registered prediction (stated pre-outcome)**: per
  Paper 1's FB support-gate fact (escape carried by side1; frac_pos
  8/8 separation), the s1 selector is predicted to capture MORE of
  the opportunity than s0. Descriptive comparison, no α; the
  prediction is on file either way.
- Dials: `--w1_repeats 8 --oracle_all`, states 200, horizon 100,
  **fresh `--env_seed 20260830`** (≠ every prior wave). Marks fresh;
  nothing pools with any executed read.
- Concurrent execution regime permitted under the four standing
  conditions (uniform incl. dups; recorded in NOTES; dup gates pass
  under the production regime; no override).

## Estimands and decision rules (frozen; cluster = cell, n=16;
## permutation-primary + BCa, α=.05)

Per cell, all selector estimands use the SAME per-state quantities:
`pick_X = argmax_m score_X(m)`; value(X) = mean_s mean_r
[g_all_rep[s, r, pick_X(s)] − g_all_rep[s, r, m_now(s)]].
FB picks are G-independent (fb_q never sees the rollouts), so the
plain all-repeat mean is unbiased for FB and native-Q selectors; the
split-selected ceiling is computed in the S2 estimator's own
even-select/odd-evaluate form (selection-noise-corrected).

- **THE NATIVE-SELECTOR IDENTITY (review B1, registered as a framing
  fact, not an estimand)**: m_now IS plugin_choice(qfull) = the
  argmax of the Q-ensemble head MEAN — so value(native-mean-selector)
  ≡ 0 BY CONSTRUCTION. The deployed plug-in is already the native
  surface's best pick; that identity is precisely WHY the S2
  opportunity is "unharvested," and it is why this wave has ONE
  primary: only a FOREIGN surface can even attempt the question.
- **P-FB1 (the ONE PRIMARY)**: value(fb_s1) — the s1 FB-Q selector's
  captured opportunity. FIRES iff perm p < .05 AND BCa CI > 0;
  fires NEGATIVE symmetrically.
- Registered branch map:
  - **FB-SEES-OPPORTUNITY** (fires +): existence proof — the
    opportunity is selectable by a value surface at all — plus
    critic-pathway localization (the native surface definitionally
    cannot outperform its own pick).
  - **FB-ANTI-SELECTS** (fires −): the FB ranking anti-correlates
    with real value; surprising, reported symmetrically, licenses
    nothing.
  - **FB-BLIND** (no fire): the flagship generalizes across surface
    classes; realized MDE80 + the capture ratio against the
    replicated ceiling reported.
- SECONDARIES (descriptive, no α): value(fb_s0) + the s1>s0
  cross-paper prediction; the PESSIMISTIC native selector (argmax of
  the head MIN — TD-MPC2's own planning convention) as the
  ensemble-disagreement value over the deployed mean-argmax baseline;
  the split-selected ceiling on these cells + capture ratios; the FB
  ACTION branches [mean g_fb_rep(k) − mean_m g_all] — the
  level-confounded consumer form, disclosed as such (the F1/F5
  lesson), never verdict-bearing.
- **Fixed-effect scope disclosure (review M2)**: all 16 cells share
  the SAME two FB checkpoints and z vectors (cluster = cell resamples
  TM2 seeds only), and the published s1 zeroshot span is 0.2–394.3 —
  a firing P-FB1 licenses "THIS pinned s1 fit's surface sees it,"
  and every branch sentence is conditioned on the pinned fits; the
  FB-fit-population generalization would need its own wave.
- **Pre-named descriptive exhibit** (`analysis/fb_bdiv_exhibit.py`,
  no α, no verdict weight; review m12 — named for exactly what it
  computes): (1) VALUE-RANGE TRACKING — per state, the spread (sd
  over candidates) of the s1 FB-Q scores vs the spread of the
  ground-truth candidate values, Spearman per cell + pooled (the
  WDC-flatness bridge on this wave's own cells); (2) B-STATE
  GEOMETRY (requires the pinned s1 ckpt) — per cell, B(obs_now)
  embedding dispersion vs the cell's split-selected opportunity.
  Pre-named here so neither is a post-hoc search.

## Gates (refusal = read not consumed)

Reader `analysis/fbcons_read.py` (frozen with this file; selfcheck
PASS): exactly the 16 registered cells + 2 registered dup files, no
unregistered wfb* files; meta pins env_seed 20260830 / repeats 8 /
states 200 / horizon 100 / version `_wfb` / fb.k == 2 /
fb.obs_keys_ok TRUE; **fb record pins**: ckpt_sha256 EQUAL to the two
registered shas above, data_sha256 recorded, z_seed ∈ {8, 1},
n_infer == 5120, frac_pos_reward EQUAL to the published witnesses
(s1: 0.3263671875, s0: 0.0541015625 — review m4: equality proves z
was reproduced from the same export bytes and seed); run_id↔filename
identity; label_every == 25, actions == 8, task/dose ==
finger/e1 pinned in meta (review m9/m14);
finiteness on g_all_rep/g_fb_rep/fb_q/qfull/r_real; **in-file CRN
witness**: g_fb_rep[:, :, K] == g_all_rep[:, :, 0] BITWISE in every
file (end-to-end pairing proof); duplicate-null EXACT zero on the dup
passes (g_all branches + r_real constancy; fb_q constancy across the
duplicated candidates — identical candidates must receive identical
FB scores); ONE-read guard PATH-PINNED to
`artifacts/fbcons_read_20260829` (the wcem2 B1 lesson, applied from
birth). (An earlier draft promised a "STRICT modal n_ep" gate; no
episode-count quantity exists in this label family — struck per
review M1, replaced by the label_every/actions pins above.)

## Ops

Existence preconditions (attested before any pass): the 2 pinned
fb_ckpt.pt + the 2 fb_data exports on the venue with matching shas;
`CONTROLLABLE_AGENT_ROOT` importable inside the TM2 label env
(`python -c "from probing.fb_consumer import build_fb_consumer"` +
ONE 5-state `_smoke.npz` pass, deleted, before the spend — the
registered cheap-insurance smoke). The obs-order gate is the STATIC
startup assert in the labeler's fb block (OBS_ORDER == OBS_KEYS + no
extra keys — review m1), attested in meta as fb.obs_keys_ok.
**Registered commands (review M3 — every flag spelled; a dropped
flag is the standing hand-queued-cmds failure)**, with
FB="--fb_ckpts $FB1,$FB0 --fb_datas $D1,$D0 --fb_seeds 8,1
--fb_shas 8a5409df637bd177b86c6addca2681b0a0df57a8ddc0adca369d9e335c179285,183bf521969719f3efa92a88a0c718ed8f0ab15769eb8f550fe6e22cb5947ff7"
(the --fb_shas startup assert is review m3: a swapped ckpt order
costs 0 seconds, not 18 passes):
- dup gates (2): `python -m probing.tdmpc2_oracle_labels
  --run_logdir <cell run> --checkpoint <ckpt_late.pt> --states 200
  --horizon 100 --label_every 25 --actions 8 --seed 0 --ref_stride 5
  --oracle_all --w1_repeats 8 --w1_dup_cand --env_seed 20260830 $FB
  --output <labels>/wfbdup_finger_e1_seed{51,59}_late.npz`
- cells (16): identical WITHOUT `--w1_dup_cand`, outputs
  `wfb_finger_e1_seed{51..66}_late.npz`.
Then bundle `local_results/p2_wfb_<ts>` (labels + NOTES: regime +
producer commit sha + fb attestations + sha manifest; counts asserted
against the reader constants) → **[ME] ONE read** → the pre-named
exhibit (descriptive) → RECORD. Cost ≈ 18 W1-scale passes × ~1.33
(K+1 = 3 extra rollout branches per repeat: 72→96 rollouts/state —
review m6) — still cheap under the concurrent regime (UNVERIFIED;
ops re-derives from the smoke). Disclosure (review m7): building the
FB nets consumes global torch RNG before the TM2 agent constructs,
so a wfb pass visits different labeled states than a plain W1 pass
would — benign (fresh env_seed; nothing pools; every gate is
within-file), disclosed.
