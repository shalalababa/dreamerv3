# PREREG: R3 candidate-count doubling test — M=8 → M=16 (frozen 2026-07-29)

Closes the residue of 11-Jul editorial objection 6 (the
continuous-action candidate-set approximation was "an unregistered
approximation — freeze and ablate"). Theory T4
(`EVPI_theory_addendum_20260712.tex` §5, Prop biassign) makes a signed
prediction: the M-candidate opportunity is a conservative,
monotonically improving lower estimate of the true opportunity
(E[opp_M] increases in M, deficit Θ(M^{−2/d})), so candidate-set error
ATTENUATES the R3 contrasts and cannot manufacture them. This wave
measures the 8→16 increment on the SAME checkpoints. This file +
`analysis/r3_doubling_read.py` (selfcheck PASS) are committed BEFORE
any M=16 label pass exists.

## Design: 1 smoke + 32 late-cell M=16 label passes

- Cells: the 32 LATE cells of the executed R3 wave (cup/finger ×
  e1/e4 × seeds 31–38), final checkpoints, `--actions 16`; every other
  dial identical (states 200, horizon 100, label_every 25, rollouts
  16, labeler seed 0, ref_stride 5, `--oracle_all`, labeler
  `d1fix_20260724`). Late-only is registered: late is the
  deployment-relevant maturity and halves the compute; the maturity
  axis is not re-tested here.
- **Pairing is CELL-level by design**: `--actions` configures the
  agent-side candidate sampler, which shares the agent RNG stream, so
  the base trajectory diverges from the M=8 run after the first
  labeled state — state sets are NOT paired across M. Comparisons are
  per run cell (same checkpoint, same state-sampling process);
  per-state pairing is never computed. This is disclosed, not a
  deviation.
- M=8 side = the ORIGINAL committed R3 late labels (bundle
  `local_results/r3_competence_20260729_105146/labels/` or the RCC
  labels dir — bitwise-identical, either is valid); nothing about the
  executed R3 read is recomputed beyond per-cell `cell_stats`.
- **SMOKE GATE (registered, machine-checked):** `--actions 16` has
  never run; one 5-state smoke on the cup e4 seed-31 late checkpoint
  (`r3dbl_cup_e4_seed31_late_m16_smoke.npz`) must exist; the reader
  asserts it.
- Precondition: the 32 run dirs with final `ckpt` must exist wherever
  the passes run (`ls $R3_ROOT/r3_{cup,finger}_{e1,e4}_seed3?/ckpt/STEP`
  → 32 lines; RCC scratch holds them per the R3 label meta).

Submit sequence (any box holding the run dirs; activated env):

```
mkdir -p $R3_ROOT/r3dbl_labels
# smoke first (inspect stdout before the full passes):
python -m d0.oracle_labels --run_logdir $R3_ROOT/r3_cup_e4_seed31 \
  --output $R3_ROOT/r3dbl_labels/r3dbl_cup_e4_seed31_late_m16_smoke.npz \
  --states 5 --horizon 100 --label_every 25 --actions 16 --rollouts 16 \
  --seed 0 --ref_stride 5 --oracle_all
# full passes:
for dom in cup finger; do for dose in e1 e4; do for seed in 31 32 33 34 35 36 37 38; do
  python -m d0.oracle_labels --run_logdir $R3_ROOT/r3_${dom}_${dose}_seed${seed} \
    --output $R3_ROOT/r3dbl_labels/r3dbl_${dom}_${dose}_seed${seed}_late_m16.npz \
    --states 200 --horizon 100 --label_every 25 --actions 16 --rollouts 16 \
    --seed 0 --ref_stride 5 --oracle_all
done; done; done
```

## Registered read (frozen `analysis/r3_doubling_read.py`, ONE execution)

Cluster = run cell (32 clusters), percentile bootstrap B=10K rng 0.

- **P-DBLa (monotone consistency)**: pooled paired [opp16 − opp8] CI
  entirely < 0 ⇒ **ANOMALY** — contradicts the T4 direction; audit the
  M=16 pass before ANY use. Otherwise consistent.
- **P-DBLb (materiality; adjudicated only on the consistent branch —
  an ANOMALY outcome never carries an immaterial flag)**: CI upper
  bound < **0.6465** (= 50% of the committed R3 pooled gap point
  1.29304688 from `artifacts/r3_competence_20260729/r3.json`,
  truncated to 4 dp; truncation direction is conservative — a smaller
  threshold makes IMMATERIAL harder to declare) ⇒ **TRUNCATION
  IMMATERIAL** — the 8-candidate estimate is an adequate conservative
  lower bound; objection-6 residue closed. Else ⇒ **MATERIAL
  TRUNCATION DISCLOSED** — the paper quantifies the truncation; the
  conservative direction (opportunity = lower bound) is unaffected
  either way.
- **Dial-identity guard (machine-checked)**: the reader asserts, from
  every file's meta on BOTH sides, the registered dials (states 200,
  horizon 100, label_every 25, labeler seed 0, rollouts 16, ref_stride
  5, mass_scale 1.0, no behavior checkpoint), actions == the side's M,
  train_seed == the filename seed, and that the checkpoint is the LATE
  ckpt of the named cell — a mis-dialed manual retry of one pass trips
  instead of silently biasing the paired primary.
- Secondaries (descriptive): gap16 pooled CI (re-fire check at M=16),
  per-domain increments, [ach16 − ach8] (expected ≈ 0: the realized
  choice m_real and the baseline m_now both widen with M and their
  selection gains approximately cancel under mode inclusion — the
  Prop-biassign argument; descriptive only), opp8 and opp16 pooled
  levels.

## Consequence map (frozen)

- **IMMATERIAL** ⇒ the paper's §2/§7 carries one sentence: "doubling
  the candidate set moves pooled opportunity by less than half the
  gap" — objection 6 closed empirically on top of T4's signed-bias
  argument.
- **MATERIAL** ⇒ the paper reports the measured truncation and frames
  M=8 opportunity as an explicit lower bound with the M=16 increment
  quantified; the R3 primaries are NOT re-adjudicated (they were
  registered at M=8 and the bias direction is conservative).
- **ANOMALY** ⇒ M=16 labels quarantined pending an instrument audit
  (registered: an entirely-negative increment is evidence of a defect,
  not of theory failure, given T4's direction); no paper use before
  the audit resolves.
- Regardless: the executed R3 read stands as registered.

## Costs and disclosure

~32 passes at ~3–4 h each (oracle_all doubles the per-state G-rollout
work vs M=8) ≈ 4–5 days on one GPU; runs concurrently with or after
the reacher wave. Known at freeze: the full R3 record incl. all M=8
opportunity values; T4's direction and rate. Unknown: every M=16
quantity — `--actions 16` has never been run anywhere. Ordering: this
file + reader committed BEFORE the smoke.
