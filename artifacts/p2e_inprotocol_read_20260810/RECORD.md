# Plan2Explore in-protocol read (#20 external method) — 2026-08-10

Registration: `prereg/PREREG_inprotocol_external_20260808.md` (frozen
1a36a65a, 08-08). Reader: `analysis/review_waves_read.py external`
(selfcheck PASS pre-execution). Bundle `p2e_inprotocol_20260809_213605`
(sha-verified; 5 p2eof + 5 p2eou adapts on the online-P2E donors).
Baselines merged per registered policy (volume_repl apt n=32 + U1
scratch n=8). ONE execution. n=5 power limit disclosed as registered.

## Verdicts

- **P-X1 (frozen, p2eof vs apt): NULL** — 90.50 vs 81.30, diff +9.20
  [−0.8, +18.7], perm p = .282. The canonical published reward-free
  method's own ONLINE representation is also frozen-null in-protocol ⇒
  **the apt arm is a fair family representative**; limitation-item 7
  converts from rebuttal to result.
- **P-X2 (unfrozen, p2eou vs scratch): BELOW SCRATCH** — 82.80 vs
  147.68, diff **−64.9** [−92.6, −39.2], perm p = .0032 ⇒ **the
  negative-transfer claim (C-R2) extends to Plan2Explore**: initializing
  from the canonical external reward-free representation actively hurts
  relative to training from scratch, matching the apt-unfrozen 0.54×/0.61×
  pattern.
- Descriptive: p2eof 90.5 inside the random floor band [53.3, 113.7];
  p2eou 82.8 also inside.

Together with #15 (random-init) this closes the two strongest external
attacks on the reward-free null: it is not protocol floor (orthpos
control), not our-method-specific (P-X1), not fixable by unfreezing
(P-X2 replicates the negative transfer on the external method).
