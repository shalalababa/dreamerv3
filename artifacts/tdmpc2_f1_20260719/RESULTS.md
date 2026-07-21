# TD-MPC2 F1 (seeds 9–16 pooled) — primary null at doubled power; P-C1 PARTIAL; G-F2 GO (2026-07-19)

Snapshot: `local_results/tdmpc2_f1_20260719_202553/` (32 new fit+adapt
jobs, seeds 9–16 × {aware, free} × {s0, s1}, finger q1). Read exactly as
registered in `prereg/PREREG_tdmpc2_amend1_20260718.md` with the
pre-frozen `analysis/tdmpc2_amend_read.py`.

## Pipeline integrity

- Local canonical AUC recompute vs cluster csv: 64/64 snapshot rows
  bit-identical (seeds 1–16 both arms); fresh 32 rows archived here.
- Seeds 1–8 taken from the authoritative frozen
  `artifacts/tdmpc2_goodhart_d1pilot_20260717/auc.csv`; in-script
  overlap bit-check passed on all 32 overlap rows.
- Registered config audit: **32/32 seed-9–16 runs pass, 0 problems**
  (aware coefs 0.1/0.1, free 0.0/0.0, consistency 20; TM2_FIT_DONE /
  ADAPT_DONE markers).
- Local read json bit-identical to the cluster-side json (incl. audit).

## Registered read (B=10K cluster bootstrap seed 0; decision on CI alone)

| quantity | mean | 95% CI | notes |
|---|---|---|---|
| **PRIMARY [B_aware − B_free] pooled 1–16** | **+22.9** | **[−19.9, +68.9]** | **does NOT fire** (pos 11/16, perm p=.34, d_z=.25) |
| fresh batch 9–16 | +36.4 | [−32.9, +110.3] | same sign as old batch — coherent |
| old batch 1–8 (frozen record) | +9.5 | [−43.2, +51.4] | unchanged |
| SEC B_aware pooled | +35.5 | [+2.8, +71.2] | CI > 0 — aware simple effect confirmed at n=16 |
| SEC B_free pooled | +12.5 | [−8.0, +35.9] | spans 0, point ≥ +10 |

Cell means: aware s0 222±45 → s1 257±52; free s0 88±27 → s1 101±39
(aware LEVEL ≈ 2.5× free on both sides, matching the n=8 pattern).

**Consequence (registered): the family-scope limitation STANDS at
doubled power.** The paper headline stays restricted to
reconstruction-based world models; the aware-arm simple effect (+35.5,
CI > 0) is the salvage — occupancy interacts with reward supervision
inside TD-MPC2 too, but the aware−free contrast does not separate at
n=16.

## P-C1 adjudication (scale-free forms, third theory prediction)

- **C1a (share < 0.6, directional): FAIL** — share = 22.9/35.5 =
  **0.646**, narrowly above the registered convention (DreamerV3 analog
  ≈ 1.04; disclosed n=8 value 0.27 did not persist).
- **C1b (free-arm CI > 0, the sharp one): UNRESOLVED** — CI spans 0
  with point +12.5 ≥ +10 (registered trichotomy middle branch; not the
  refuting branch).
- **C1c (aware CI > 0): PASS.**

**⇒ P-C1 = PARTIAL** (LANDS required a ∧ b-lands ∧ c; REFUTED required
b-fails). Ledger: P-A1 landed, P-B1 landed, P-B4 refuted (Option C
corrective, same day), P-C1 partial. The decoder-free limit's
qualitative signature (free arm not tightly null, unlike DreamerV3 apt
−3.7; strong attenuation of the contrast vs Dreamer's share ≈ 1.04) is
directionally present but neither registered sharp form resolved.

## Gate G-F2 (non-inferential, registered default GO)

Fresh batch sign-coheres with the old batch and the audit is clean ⇒
**not incoherent ⇒ F2 (E4-style mechanism port over TD-MPC2 fits)
proceeds** under the plan's default GO. In the plan's cell taxonomy this
read is "attenuated-null / stable-but-ambiguous", so F2 is reframed as
the discriminator: whether aware/free reward-NLL levels separate the way
rgo/sgb did in DreamerV3 decides between lottery-inclusion and
never-included accounts of the free arm. G-F3 (third family) remains
default NO-GO (needs F2 mechanism dissociation AND an external trigger).

## Provenance

- `auc.csv` — canonical local recompute, fresh 32 rows (bit-identical).
- `tdmpc2_amend_read.json` — full read output incl. audit (local;
  verified identical to the cluster-side json).
- Read command: `python -m analysis.tdmpc2_amend_read --auc <cluster
  csv> --runroot <runroot> --output <dir>`.
