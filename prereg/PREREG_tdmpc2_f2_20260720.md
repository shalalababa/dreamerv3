# PREREG: TD-MPC2 F2 — E4 mechanism port (frozen 2026-07-20)

Authorized by Gate G-F2 (default GO; the F1 read is coherent:
`artifacts/tdmpc2_f1_20260719/` — primary +22.9 [−19.9, +68.9] with the
fresh batch sign-cohering, aware simple +35.5 CI>0). Per the plan, in
this attenuated-null cell **F2 is the discriminator** between two
accounts of the free arm: consistency-only training *sometimes* includes
the reward direction by subspace lottery (decoder-free limit,
Theory addendum Prop A2 α→0) vs *never* includes it. This file + both
scripts are committed **before any TD-MPC2 stratified-error outcome
exists**.

## Instruments (committed with this file)

- `probing/tdmpc2_stratified_error.py` (NEW; numpy core selfchecked —
  planted-direction recovery, permutation destruction, determinism,
  split/strata/ridge exactness). Measure passes over fit checkpoints,
  NO training.
- `analysis/tdmpc2_f2_read.py` (NEW; selfchecked: lottery /
  never-included / null / missing-cell branches).
- Probe set: the frozen `finger_v1` E4 probe set (sha-pinned; same
  strata as the DreamerV3 E4 ⇒ qualitative cross-family comparison of
  PATTERNS only — units differ (MSE vs NLL) and are never numerically
  pooled, per the frozen never-compare-numerically rule).

## Why a fresh probe (registered rationale)

The free arm trains with `reward_coef=0` — its native reward head stays
at init and cannot report what the trunk contains. The decisional
quantity is therefore **probe_mse**: a deterministic closed-form ridge
readout of the current-frame reward from the frozen latent
z_t = encode(obs_t) (the h=0 analog of Dreamer's reward-head NLL),
trained on even probe episodes, evaluated on odd episodes (registered
split; λ = 1e-3·n_train on standardized latents). Companion measures:
**cons_err** (one-step latent-consistency error at the target frame —
the d_errin analog; both arms train this loss) and **rew_head_mse**
(native head, descriptive; meaningful for aware only).

## Design: 64 measure passes (+1 no-outcome smoke)

All fit checkpoints: {aware, free} × {s0, s1} × seeds 1–16. Cluster,
tdmpc2 conda env, 1 GPU, minutes per pass.

```bash
# 0) API smoke FIRST — random-init model, no checkpoint touched, no
#    outcome revealed; if it fails, the instrument fix is committed as a
#    dated amendment BEFORE any real measure runs:
python -m probing.tdmpc2_stratified_error measure \
  --run $RUNROOT/tm2wm_finger_awareq1s0_seed1 --smoke_random_init \
  --probeset $RUNROOT/e4_probesets/finger_v1 \
  --task dmc_finger_turn_hard --output $RUNROOT/tm2_e4/smoke.json
# 1) the 64 passes:
for arm in aware free; do for side in 0 1; do for k in $(seq 1 16); do
  run=tm2wm_finger_${arm}q1s${side}_seed${k}
  python -m probing.tdmpc2_stratified_error measure \
    --run $RUNROOT/$run --probeset $RUNROOT/e4_probesets/finger_v1 \
    --task dmc_finger_turn_hard \
    --output $RUNROOT/tm2_e4/tm2e4_${run}.json
done; done; done
# 2) collate (smoke.json cannot match the glob; the reader also
#    asserts smoke=0 on every consumed row):
python -m probing.tdmpc2_stratified_error collate \
  --inputs "$RUNROOT/tm2_e4/tm2e4_tm2wm_*.json" \
  --output $RUNROOT/tm2_e4/tm2_e4.csv
```

## Registered read (frozen `analysis/tdmpc2_f2_read.py`)

- **PRIMARY P-F2a (reward-decodability separation):** per (side, seed)
  pair D = ln(free probe_mse_in) − ln(aware probe_mse_in); mean over
  the 32 pairs; cluster bootstrap over seeds (16 clusters) B=10,000
  percentile CI `default_rng(0)`; **FIRES iff CI > 0**. Decision on the
  CI alone.
- **Descriptors (registered, never decisional):** P-F2b consistency
  ln-ratio with CI, invariant-like iff |mean| < 0.5 (the d_errin
  analog); lottery_fraction = fraction of free runs at/below the aware
  runs' 90th-percentile ln probe_mse_in; dispersion ratio
  sd_free/sd_aware; aware-arm rew_head_mse.
- **Interpretive map (frozen):**
  - FIRES ∧ lottery_fraction ≥ 0.15 ⇒ **lottery-with-partial-inclusion**
    (Prop A2's account: membership uncontested but not guaranteed) —
    the paper's family-boundary panel gains the mechanism signature
    with attenuated behavioral consequence.
  - FIRES ∧ lottery_fraction < 0.15 ⇒ **never-included-like** — the
    free arm's +12.5 ns benefit is not reward-representational;
    weakens Prop A2's relevance (registered honestly).
  - Does not fire ⇒ **no representational separation** — consistency
    alone includes the reward direction at this task scale; the F1
    attenuated null is then a representation-equivalence result, and
    the family boundary is behavioral, not representational.
- G-F3 (third family) remains DEFAULT NO-GO regardless; it requires the
  mechanism dissociation AND a named paper-blocking ambiguity.

## Disclosure and ordering

Known at freeze: all outcomes through 20 Jul (F1 read incl. cell means;
Dreamer E4 patterns; theory addendum). Unknown: every TD-MPC2
stratified-error quantity — no probe, consistency, or head measure has
ever been computed on any TD-MPC2 fit. The measure's torch/model path
mirrors the fit script's API and is validated only by the registered
no-outcome smoke; the numpy core is selfchecked locally. Ordering: this
file + both scripts committed BEFORE the smoke or any measure runs.
