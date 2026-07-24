# Gate D1 read — FAILS 0/4 cells; 24+24 stays frozen (2026-07-18)

> **INSTRUMENT-INVALIDATED (2026-07-24).** The shared D1 real-operation
> labeler contained two confirmed implementation defects (stale-carry
> candidate branches: obs_t never assimilated, prevact = a_{t-1} not the
> candidate; and policy-RNG not common across branches), so delta_real /
> G labels realize an unintended estimand. The read below remains a
> faithful record of the registered procedure ON THOSE LABELS, but its
> decision-value conclusions cannot be interpreted as established.
> Details: analysis/DEVIATIONS.md 2026-07-24 deviation entry; repair +
> relabel registration: prereg/PREREG_d1_relabel_20260724.md.


Snapshot: `local_results/d1_oracle_labels_20260718_103952/` — Stage-1B
oracle labels from the 6 D1-pilot runs (cup/finger × seeds 1–3, E1
dose-zero), 200 labeled states each, `d0/oracle_labels.py` at dials
horizon 100 / label_every 25 / actions 8 / rollouts 16 (recorded in
each npz meta; the planned dial-pinning amendment was not filed before
the sweep — disclosed, immaterial to the verdict below).

Read: `analysis/gate_d1_read.py` implementing the three criteria of
`prereg/PREREG_gate_d1_stage1a_20260714.md` §B (thresholds 0.3 / 2× /
net>0, run-clustered bootstrap B=10K seed 0, cross-fitting by run —
all frozen 14 Jul). Mechanical choices made at read time and disclosed
in the script header: ÊVSI = OLS on the five d0 signal features
(uq/aflip/evpi_plugin/evpi_split/gap from stored qfull),
leave-one-run-out within domain; criterion-3 price evaluated at 0 with
a price sweep as sensitivity. Selfcheck PASS (planted signal passes,
null fails).

## Integrity

- **CRN pairing exact**: Δ_o = 0 identically wherever m_o = m_now, all
  1200 states × both ops — restore determinism held through the sweep.
- Decision-change rates: real 0.83–0.86, imag 0.65–0.81 — the
  operations genuinely move decisions; the gate question is whether
  they move them *well* and *predictably*.
- Binding condition (addendum §4): features from stored qfull (run's
  own d0 path); Δ from separate CRN env rollouts — no shared
  RNG/rollouts.

## Registered criteria — all fail, every domain × op cell

| cell | Spearman(ÊVSI,Δ) [CI] | C1 (≥.3, CI>0) | lift [CI] | C2 (≥2×, CI>1) | net gain @p0 [CI] | C3 | gate |
|---|---|---|---|---|---|---|---|
| cup_real | −0.064 [−0.081,−0.033] | fail | −1.50 | fail | −0.088 [−0.235,−0.010] | fail | **FAIL** |
| cup_imag | −0.049 [−0.110,+0.047] | fail | n/a (pop≤0) | fail | −0.080 [−0.285,+0.010] | fail | **FAIL** |
| finger_real | −0.118 [−0.156,−0.046] | fail | −2.14 | fail | −0.433 [−0.605,−0.315] | fail | **FAIL** |
| finger_imag | −0.133 [−0.210,−0.056] | fail | n/a (pop≤0) | fail | −0.165 [−0.665,−0.090] | fail | **FAIL** |

The fail is robust to the mechanical choices: the best RAW single
feature anywhere is evpi_plugin at +0.085 (finger_imag) — a third of
the registered bar with no fitting at all; the price sweep
(0.0005–0.05 per real env step) never rescues criterion 3. The
*significantly negative* cross-fitted correlations (3/4 cells) exceed
what the ≈0 raw features explain and likely reflect between-run
heterogeneity inverting under leave-one-run-out at n=2 training runs —
reported descriptively; the decision needs only "not PASS."

## Registered consequence

**Gate D1 does not pass ⇒ the 24+24 D0 runs STAY FROZEN** (the
six-cell battery was necessary-but-not-sufficient; this was the
sufficiency test). Per the 12-Jul revision fork (owner RB), this is
strong evidence for **Route B** (merge into the Latent-UQ paper): the
package now holds attractor replication + phase change + cross-policy
boundary + passing density-residual fix (Stage-1A) + **a registered
failing gate — even against oracle ground-truth Δ labels, plug-in
Q-ensemble EVSI signals cannot rank per-state value of computation at
converged checkpoints** (calibration ≈0-to-negative; signal-gated
compute allocation loses to "always" and "never").

**One registered Route-A move remains:** the Stage-1A PASSING
instrument (density-residualized disagreement) was never tested here —
`oracle_labels.py` does not emit udyn or a latent/kNN-density proxy at
labeled states. A labeler extension + 6-job relabel would test the
actual Stage-1A asset as the ÊVSI feature. Prior honest effect sizes
(residual error-corr +0.03..+0.23 at h=5) make reaching 0.3
calibration against Δ look unlikely, but it is the asset's real test
and the only registered path to unfreezing. User's call.

## Descriptive findings worth keeping (either route)

- **Real-vs-imagined purchase asymmetry at matched candidate budget:**
  pooled mean Δ_real = +0.083 (cup) / +0.222 (finger) vs Δ_imag =
  −0.003 / −0.033. Real env probes improve decisions on average;
  additional imagination at the same state does not — and in finger
  "always buy real" survives net cost up to ~0.005 return/env-step
  (+0.18 net) while any signal-gated policy is negative. Information
  has value here; these signals just can't tell you *where*.
- Coheres with the Stage-0 attractor mechanism: at converged
  checkpoints, ensemble signals track density, not decision-relevant
  error — now shown to extend from error-tracking to realized
  value-of-computation.

## Provenance

- `gate_d1_read.json` — full output (per-cell criteria, per-feature
  Spearmans, price sweep, integrity per run).
- Read command: `python -m analysis.gate_d1_read --labels
  <snapshot>/runroot_light/d1_labels --output <dir>`.
