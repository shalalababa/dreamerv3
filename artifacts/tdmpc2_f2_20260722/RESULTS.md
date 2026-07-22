# TD-MPC2 F2 — E4 mechanism port: READ (2026-07-22)

Registered read per `prereg/PREREG_tdmpc2_f2_20260720.md` (frozen +
committed before any TD-MPC2 stratified-error outcome existed). Measure
passes over all 64 fit checkpoints ({aware, free} × {s0, s1} × seeds
1–16); NO training. Read = frozen `analysis/tdmpc2_f2_read.py`.

## Integrity

- Cluster-run code diffed against repo frozen copies:
  `tdmpc2_stratified_error.py`, `tdmpc2_f2_read.py`, prereg — all
  bit-identical.
- Grid: 64/64 rows, 16 per cell, no duplicates; `smoke=0` on every
  consumed row; single probeset sha `a78fe58c…` (frozen `finger_v1`);
  `head_trained` = 1 for all aware, 0 for all free rows (as designed).
- Registered no-outcome smoke: 4 invocations found
  (`smoke_smoke{0..3}.json`), all `smoke: true` (random-init model, no
  checkpoint loaded) ⇒ no outcome revealed pre-read; the collate glob
  excludes them and the reader asserts smoke=0. No instrument amendment
  was needed (code ran unmodified).
- Canonical json = local frozen read (`tdmpc2_f2.json` here). Cluster
  json agrees on every decision field; four float fields differ in the
  last 1–2 ulp (≤2e-16 relative; platform BLAS reduction order), so not
  bit-identical — recorded here, immaterial to every registered
  quantity at reported precision.

## Registered read

| Quantity | Value | Registered rule | Outcome |
|---|---|---|---|
| **P-F2a** ln(free/aware) probe_mse_in, 32 pairs, seed-clustered CI | **+1.717 [+1.650, +1.775]** | FIRES iff CI > 0 | **FIRES** |
| lottery_fraction (free runs ≤ aware p90 ln probe_mse_in) | **0.000** | ≥ 0.15 ⇒ lottery branch | below bar |
| **Interpretive map (frozen)** | fires ∧ fraction < 0.15 | | **NEVER-INCLUDED-LIKE** |
| P-F2b cons_err ln-ratio (descriptor) | −0.608 [−0.679, −0.529] | invariant-like iff \|mean\| < 0.5 | **not invariant-like** |
| dispersion ratio sd_free/sd_aware (descriptor) | 0.82 | — | free arm tight, not heterogeneous |
| rew_head_mse_in, aware (descriptor) | 0.065 | — | trained head functional |

Cell descriptives (probe_mse_in mean ± sd; probe R² on held-out
episodes):

| cell | probe_mse_in | mean ln | probe R² | cons_err_in | rew_head_mse_in |
|---|---|---|---|---|---|
| aware s0 | 0.0757 ± 0.0205 | −2.61 | **+0.63** | 0.0002 | 0.114 |
| aware s1 | 0.0631 ± 0.0128 | −2.78 | **+0.76** | 0.0002 | 0.017 |
| free s0 | 0.4434 ± 0.0304 | −0.82 | **−0.24** | 0.0001 | 1.000 (untrained) |
| free s1 | 0.3209 ± 0.0428 | −1.15 | **−0.18** | 0.0001 | 1.000 (untrained) |

Separation is complete: aware ln range [−3.25, −2.08] vs free
[−1.61, −0.68] — **zero overlap** (gap +0.47 ln units; geometric-mean
ratio 5.6×). Hence lottery_fraction exactly 0: not one of 32 free fits
reaches the aware band.

## Reading (per the frozen interpretive map)

**Never-included-like.** Consistency-only training at this task scale
does NOT include the reward direction in the trunk — the ridge probe on
free latents has *negative* held-out R² (worse than predicting the
mean), across all 32 free fits, both sides. This is the frozen map's
second branch, registered in advance with its consequence:

- The free arm's F1 behavioral +12.5 [−8.0, +35.9] (ns) is **not
  reward-representational** — coherent with it being noise.
- **Prop A2 (theory addendum α→0 subspace-lottery account) is
  WEAKENED** — the registered honest outcome. The decoder-free limit's
  attenuation is not a lottery over inclusion; it is deterministic
  exclusion without reward-legible gradients.
- The TD-MPC2 family boundary is **representational, and sharper than
  the behavioral F1 null suggested**: reward/value gradients (aware)
  reliably include the reward direction (R² +.6–.8), their absence
  reliably excludes it. Cross-family, this RHYMES with the Dreamer
  mechanism panel (rgo low rew-NLL vs sgb/vgo high): in both families,
  inclusion tracks reward-legible gradients — supporting the headline
  "support must be legible to the objective" — while the *behavioral*
  interaction is family-dependent (large in Dreamer, attenuated in
  TD-MPC2).

Registered descriptors, honestly reported:

- **P-F2b NOT invariant-like** (−0.61, just outside the ±0.5 band):
  unlike Dreamer's arm-invariant d_errin, the free arm is ~1.8× BETTER
  at one-step consistency — mechanism-coherent (consistency is its only
  trunk objective; aware's reward/value gradients compete for trunk
  capacity), and itself spectral-competition-flavored, but this is a
  registered qualitative difference from the Dreamer pattern, not a
  confirmation.
- Aware s1 probe/head errors < s0 (0.063 vs 0.076; head 0.017 vs
  0.114): occupancy exposure leaks reward structure — same direction as
  the stamping true-label observation (descriptive only).

## Gates

- **G-F3 (third family): stays NO-GO.** Condition (i) mechanism
  dissociation is now met, but condition (ii) — a named paper-blocking
  ambiguity — is not: never-included *resolves* the F1 ambiguity rather
  than creating one. No further TD-MPC2 arms planned; the family-
  boundary panel (F1 behavior + F2 mechanism) is paper-ready.

## Provenance

Cluster results `local_results/tdmpc2_f2_20260722_001200/` (64 measure
jsons + collate csv under `runroot_light/tm2_e4/`, cluster read json
under `analysis/`, code snapshot under `code/`). Files here:
`tdmpc2_f2.json` (canonical local read), `tm2_e4.csv` (collate).
