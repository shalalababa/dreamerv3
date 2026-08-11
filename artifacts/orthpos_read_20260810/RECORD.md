# Orthogonal-test positive control read (#18 turn_easy leg) — 2026-08-10

Registration: `prereg/PREREG_orthogonal_poscontrol_20260808.md` (frozen
1a36a65a, 08-08). Reader: `analysis/review_waves_read.py orthpos`
(selfcheck PASS pre-execution). Bundle `orthpos_20260809_212632`
(sha-verified; 16 ax1ter adapts + 5 rndte floor runs). ONE execution.

## Verdict: **P-OP1 FIRES — positive control passes; U3 strengthened**

- rgo − sgb LEVEL on turn_easy, per-(seed,side), n=8 pairs:
  **+112.4** [+53.4, +194.0], exact sign-flip perm p = .0078
  (8/8 pairs positive).
- Pooled turn_easy level 253.9 vs the MEASURED turn_easy random-policy
  floor band [127.7, 211.2] (5 rndte runs) — comfortably above floor.

Registered consequence: trunk differences ARE detectable on an adjacent
objective (same reward channel, wider target), so the U3/orthogonal spin
null (+1.0) is informative about ORTHOGONALITY, not a floor artifact.
The objective-specificity claim gains its missing positive control:
detectability gradient = identical objective (+45.0*) → adjacent
same-channel (+112.4* levels at 4-seed adapt scale) → orthogonal
channel (+1.0 ns).
