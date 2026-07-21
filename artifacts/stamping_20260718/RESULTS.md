# Stamping read — PRIMARY does NOT fire; P-B1 strong form STANDS (2026-07-18)

Snapshot: `local_results/stamping_20260718_105655/` (48 fit+adapt runs,
arms srd0/srd1/sid × sides × seeds 1–8, finger q1). Read exactly as
registered in `prereg/PREREG_stamping_20260717.md` with the pre-frozen
`analysis/stamping_read.py` (committed before any stamped outcome
existed). The E4-style mechanism panel (stamp-NLL vs true-NLL,
`--reward_override`) was **still running** at this read — descriptive
only, never decision-bearing; will be appended when it lands.

## Pipeline integrity

- Local canonical AUC recompute (untouched `analysis/adaptation_auc.py`)
  vs cluster-side `auc/auc.csv`: **48/48 rows bit-identical, 0 field
  diffs**; 48/48 pass QC, 0 exclusions. Canonical csv archived here.
- Registered audit (in-script): **48/48 pass, 0 problems** — saved WM
  config flags = full-arm row (`expl.mode task, reward_grad T,
  repval_loss T, repval_grad T`), `Static replay: .../q1_<code>/side<s>`
  in retained fit stdout, ADAPT_DONE, QC.

## Registered read (B=10K cluster bootstrap seed 0; decision on CI alone)

B_arm(k) = AUC100k(s1) − AUC100k(s0); B_srd(k) = mean(B_srd0(k), B_srd1(k)).

| contrast | mean | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY [B_srd − B_sid]** | **+8.90** | **[−11.49, +31.85]** | **CI includes 0 ⇒ P-B1 strong form STANDS** |
| S1 B_srd − B_apt (hist) | +14.43 | [−8.88, +40.11] | ns — stamping does not demonstrably beat "nothing" |
| S2 B_srd − B_sh (hist) | −14.95 | [−48.72, +21.23] | ns |
| S3 B_srd0 − B_srd1 | +0.85 | [−19.90, +27.05] | ≈0 — no function-draw luck; pooled reading valid |

Sensitivity (robustness only), PRIMARY: paired t p=0.49 (CI [−19.8,
+37.6]), exact sign-flip perm p=0.500, Wilcoxon p=0.84, d_z=0.26, LOO
range [+0.05, +15.4] — the null is not seed-driven.

Arm benefit means (descriptive): srd0 +16.9, srd1 +16.0, srd pooled
+16.5, sid +7.6; historical apt +2.0, sh +31.4. All stamped arms sit
far below rgo pooled +56.7 and full ~+98.7.

## Theory adjudication — P-B1 LANDS (second registered prediction)

`PREREG_theory_predictions_20260717.md` P-B1 strong form: frozen-random
stamping ⇒ approximately null transfer, **band = within the apt null
CIs**. Observed: srd pooled benefit +16.5 lies inside the P0 apt null
CI [−22.0, +24.2]; primary decision CI includes 0. Both the decision
rule and the registered band are satisfied. **P-B1 = second adjudicated
theory prediction (after P-A1), landed.**

## Interpretation (registered branch)

Learnable-but-unaligned scalar supervision through the identical
objective path (same flags, same reward head, same trunk gradients as
the confirmed rgo carrier) does **not** transfer. Combined with
Amendment 1 (true-reward rgo +51.5*) and S2/S3 shuffle/relocate
collapse, the licensed claim sharpens: it is not gradient flow through
a reward head per se — the supervised scalar must be **task-aligned and
frame-bound**. Paper-1 discussion gains "not any scalar — aligned
scalars"; differentiates from generic objective-shaping/subspace
expansion (and from UNREAL/RaMP per the registered framing).

Pending (descriptive): E4 stamp-NLL panel. The theory's signature
outcome is **inclusion-without-transfer** — low NLL against the STAMPED
labels (the stamp direction was learned into the trunk) alongside this
null primary. That panel decides nothing but determines whether the
null is "included-but-useless" (theory's account) or "never included"
(competing account: stamps too easy, learned in the head alone).

## Post-read caveat (added 2026-07-18, before the E4 panel ran)

Discovered while building the E4 override: the stamp functions g₀/g₁
consumed the chunks' stored `dyn/deter`/`dyn/stoch` source-agent
latents in addition to proprio (~640 of ~654 columns) — see the 18-Jul
DEVIATIONS entry. The PRIMARY and all numbers above are unaffected
(labels deterministic, marginal-matched, frame-bound, task-unaligned),
but the prereg's "trivially predictable at h=0 from obs" scope
statement is weakened: whether the stamp was *learnable by the fitted
trunk* is now exactly what the pending stamp-NLL panel adjudicates.
Low stamp-NLL ⇒ "learnable-but-unaligned scalars do not transfer"
stands in full; high stamp-NLL ⇒ the claim narrows to "unaligned
scalars of this class". The panel runs with
`stamp-probeset --latent_replay <pilot replays>` (stepid-join recovery
of the latent columns for probe frames).

## E4 override panel (landed 2026-07-18 pm; descriptive-only)

Snapshot `local_results/x0_stamp_synthdiag_20260718_175314/stamping_e4_override/`:
32 override-measure runs (srd0/srd1 × sides × 8 seeds), each srd fit
scored against ITS OWN stamped labels on the frozen `finger_v1` probe
set, dyn/* stamp inputs recovered by stepid join over the three pilot
replays (recovery recorded in every override meta; probeset sha
matches the frozen record).

Stamp-NLL (h0, seed means):

| cell | all | in-regime | out-regime |
|---|---|---|---|
| srd0 s0 | 0.244 | 1.176 | 0.151 |
| srd0 s1 | 1.489 | 2.504 | 1.387 |
| srd1 s0 | 1.374 | 0.501 | 1.461 |
| srd1 s1 | 1.300 | 0.952 | 1.335 |

Reference band (P3 E4, true labels, in-regime h0): included = full
1.02 / rgo 1.21; not-included = sgb 2.30 / vgo 2.31. Three of four
cells sit at-or-below the included band on all-frames NLL; patterns
are heterogeneous by function draw and stratum (srd0-s1 in-regime
2.50 ≈ sgb-level; srd0-s0 is base-rate-easy at its low stamp
density). The true-label pass on srd fits (panel (b)) has not run.

**Reading: the stamps were substantially LEARNED — on held-out probe
frames — and transfer nulled anyway. That is the theory's
inclusion-without-transfer signature, resolving the 18-Jul caveat in
the favorable direction: "learnable-but-unaligned scalars do not
transfer" stands in its full form (with the function-draw
heterogeneity noted). P-B1's adjudication is unchanged and now
mechanism-backed.** (The dyn/*-input deviation stands documented; the
learned stamp is a function of obs+source-latents, which the trunk's
history summary evidently suffices to predict.)

## True-label E4 panel — panel (b) (landed 2026-07-19; descriptive-only)

Snapshot `local_results/stamping_true_e4_20260719_122116/`
(`e4_stamping_true_label_finger_v1.csv`, archived here): the same 32
srd fits scored against the TRUE task reward on the frozen `finger_v1`
probe set (128 measures = 32 runs × horizons {0,1,5,20}; all runs
`expl.mode task`, reward_aware=1).

True-reward NLL (h0, seed means; reference band from P3 E4 true labels,
in-regime h0: included = full 1.02 / rgo 1.21, not-included = sgb 2.30):

| cell | all | in-regime | out-regime | total-NLL in/out | err_diff |
|---|---|---|---|---|---|
| srd0 s0 | 1.42 | **15.09** | 0.05 | 0.26 / 0.20 | −0.064 |
| srd0 s1 | 1.42 | **4.10** | 1.15 | 0.14 / 0.14 | +0.007 |
| srd1 s0 | 1.02 | **10.77** | 0.04 | 0.26 / 0.18 | −0.076 |
| srd1 s1 | 1.36 | **3.92** | 1.11 | 0.13 / 0.14 | +0.011 |

**Reading: the triangulation closes.** The srd reward heads are far
above even the not-included band on true reward in-regime (3.9–15.1 vs
sgb's 2.30) — they know the stamp (override panel: stamp-NLL 0.24–1.49)
and demonstrably do NOT know the true task reward. So the stamped arms'
supervision was genuinely task-unaligned (no accidental alignment
leakage through the dyn/*-input deviation), and the null primary is a
clean test of "learnable-but-unaligned scalar supervision". Together
with the override panel this is the full **included-but-useless**
signature from both directions; P-B1's mechanism backing is complete.
Secondary observations (descriptive): true-reward in-regime NLL is
~3× lower on s1 fits (3.9–4.1 vs 10.8–15.1) — high-occupancy exposure
leaks some true-reward structure even into stamp-trained heads — and
err_diff (total-NLL out−in) stays in the tiny arm-invariant band
(−0.08..+0.01), consistent with the P3 E4 d_errin invariance.

## Provenance

- `auc.csv` — canonical local recompute (bit-identical to cluster).
- `stamping_read.json` — full frozen-read output (deltas, CIs,
  sensitivity, audit).
- `e4_stamping_true_label_finger_v1.csv` — true-label E4 measures
  (panel (b) source).
- Read command: `python -m analysis.stamping_read --auc auc.csv
  --runroot <snapshot>/runroot_light --output <dir>`.
