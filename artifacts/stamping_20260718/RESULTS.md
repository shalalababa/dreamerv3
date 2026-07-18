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

## Provenance

- `auc.csv` — canonical local recompute (bit-identical to cluster).
- `stamping_read.json` — full frozen-read output (deltas, CIs,
  sensitivity, audit).
- Read command: `python -m analysis.stamping_read --auc auc.csv
  --runroot <snapshot>/runroot_light --output <dir>`.
