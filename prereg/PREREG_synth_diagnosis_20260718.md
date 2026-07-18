# Synth Phase-B diagnosis registration — in-subspace check (frozen 2026-07-18; descriptive-only)

The Phase-B read (`artifacts/synth_phaseb_20260718/`, 18 Jul) landed on
the registered collapse-only branch, whose instruction is "diagnose
which DMC ingredient synth lacks before any Phase C". This note
registers the diagnosis instruments and their directional predictions
**before any diagnosis outcome exists**. Everything here is
DESCRIPTIVE: nothing alters the Phase-B read; the only decision this
gates is a RESOURCE one (whether Phase-B′ gets registered).

## Hypothesis under test

**In-subspace placement (theory's cup/E5 cell):** the synth obs
includes `to_target` — the reward-aligned coordinate as a directly
observed, high-variance, reconstruction-decoded key — so λ_j ≥ g_k and
every arm's predictive objective already retains the reward direction.
The spectral-competition theory then predicts no supervision
interaction, which is what Phase B observed. The engineered domain
landed in the wrong spectral cell by obs design, not by a failure of
the law.

## Instruments

1. **Data-level:** `probing/reward_direction_rank.py` on built buffer
   sides — rank, in the obs correlation spectrum, of the
   **cross-correlation direction ν = E[r·x]** (the theory's
   ν = aλ_je_j object). Instrument revision at this freeze
   (validation-discovered, before any cluster diagnosis outcome
   exists): the 17-Jul draft ranked the min-norm *regression*
   direction, which local validation showed falls into the correlation
   null space under exact obs collinearity (synth's
   `to_target = GOAL − position`; lstsq keeps near-zero singular
   values), spuriously reporting bottom-of-spectrum rank. ν has no
   inversion and no null-space pathology; the regression direction
   stays as a labeled secondary output. Selfcheck extended with an
   exact-collinearity case. The tool also inherits the 18-Jul
   `stamp_obs_keys` fix (dyn/* extras excluded); it has never run on
   real cluster data, so no reproduction concern. Targets: synth
   `axis1_synth/q1` side0+side1 AND finger `axis1_finger/q1`
   side0+side1 (the comparison population).
2. **Model-level:** `stratified_error` probe set `synth_v1` — fresh
   generator episodes (seed 7, 12 episodes/side [prov. — exact dials
   pinned in the FROZEN probe-set manifest], both occupancy levels as
   sources hi/lo, same density/predictability dials as the wave) —
   then the standard measure pass over all 48 existing synth WM
   checkpoints (`E4_DOMAINS=synth`, glob `ax1wm_synth_*`). No new
   training anywhere.

## Directional predictions (frozen before outcomes)

- **P-D1 (data-level placement):** the reward direction's rank is ≤ 3
  (of 14 obs dims) in BOTH synth sides, and > 3 (of 12 dims) in BOTH
  finger q1 sides — synth strictly lower than finger everywhere.
- **P-D2 (head-level, no side gating):** among task-arm fits, the
  s0-fit vs s1-fit difference in reward-head NLL (h0, in-regime
  stratum, mean over seeds) is SMALLER in absolute value than finger
  q1's E4 separation (0.242 − 0.191 = 0.051), and the task rew-NLL
  LEVEL is low on both sides (no membership gap for supervision to
  create).
- **P-D3 (trunk-level universal inclusion):** apt-arm decoder NLL on
  the `to_target` key (h0, in-regime, mean over seeds) is
  approximately the task-arm's — ratio in [0.8, 1.25] [prov.] — i.e.
  the reward-bearing observable is reconstructed arm-invariantly.

## Adjudication (resource gate only)

All three hold ⇒ in-subspace confirmed ⇒ **Phase-B′ authorized for
registration**: rerun the Phase-B 2×2 (task/apt) with `to_target`
hidden from the model via the `agent.model_obs` regex (fit AND adapt;
16–32 jobs), predicting the interaction appears. Phase-B′ and any
Phase C freeze their own files. Any prediction fails ⇒ no Phase-B′;
re-diagnose. If results are mixed, the failing instrument is reported
and the gate stays closed.

## Disclosure and ordering

Known at freeze: the full Phase-B read (this is a post-outcome
diagnosis of a registered branch), all study outcomes through 18 Jul.
**Also disclosed:** local validation of the revised rank instrument ran
on tiny fresh-seed generator buffers (statistically identical
machinery to the real pair) and previewed the SYNTH half of P-D1 at
debug scale — rank 2/14, both sides, variance ≈2.9 along ν. The synth
half of P-D1 is therefore an informed prediction; prospective content
lies in the full-scale synth buffers, the FINGER half (never measured
at any scale), and all of P-D2/P-D3. Unknown: no synth probe set
exists; no measure pass has touched any synth checkpoint; no finger
rank exists. Local validation used tiny fresh-seed debug fits (seed
5/7 generator, 30-update fits; excluded by rule).
Ordering: this file + `analysis/synth_diagnosis_read.py` are committed
before the cluster diagnosis runs; the probe set gets its FROZEN
marker before the first measure job.
