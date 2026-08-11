# Random-init WM control read (#15) — 2026-08-10

Registration: `prereg/PREREG_randominit_control_20260808.md` (frozen
1a36a65a, 08-08). Reader: `analysis/review_waves_read.py randominit`
(selfcheck PASS pre-execution). Bundle `randominit_20260809_213604`
(sha-verified; 8 `rif` adapts). Baselines merged per the registered
max-n_ep policy from volume_repl (apt `ax1fq1s*`, n=32) + U1 (scratch).
ONE execution.

## Verdict: **P-RI1 — reward-free frozen ≈ untrained: the null family upgrades to the no-learning form**

- `rif` (random-init frozen trunk) mean 70.67 vs pooled apt-frozen 81.30:
  diff **−10.64**, exact perm p = .129 (ns) AND |Δ| = 10.6 ≤ 15.1 (the
  registered one-floor-sd equivalence bar) ⇒ the registered upgrade
  fires: apt's frozen transfer is **indistinguishable from an untrained
  network**.
- Disclosure: the report-only unpaired bootstrap CI [−21.2, −0.40] sits
  just below 0 (apt marginally above untrained if anything); the
  registered decision rule is permutation + |Δ| bar, both met; Welch
  p=.079 consistent.
- **P-RI2**: rif 70.67 inside the random-policy floor band [53.3, 113.7].
- scratch anchor 147.7 (unfrozen counterpart, reused as registered).

Registered consequence: "reward-free transfer is null" sharpens to
"frozen reward-free features perform like random features" — the §19
floor-censoring asymmetry is discharged with data on the proprio side.
No headline change (as frozen); wording upgrade only.
