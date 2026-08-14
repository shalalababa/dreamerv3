# Adaptation-horizon wave (late-crossing test, 4× budget) — ONE registered read — 2026-08-13

- Registration: `prereg/PREREG_horizon_20260811.md` (post-review
  REVISED form; sha256
  93613d3b85e4e333a9bfa2dd221f819c43c91c50f8c8680c87f82bb063665247;
  registered 93d1fab3, revised form freeze-committed 8cd62845 on
  2026-08-12 — before any hz/hscratch run was submitted).
- Bundle: `local_results/horizon_20260813_122500` — MANIFEST.sha256
  verified (72 entries, all OK; byte-identical to committed
  `manifests/horizon_20260813_122500.sha256`) before any file was
  opened. Exactly the 24 registered run dirs (8 hzf + 8 hzt + 8
  hscratch), each with scores.jsonl + config.yaml.
- Reader: `analysis/horizon_read.py` (sha256 d7b67e5e…54e41f2),
  selfcheck PASS re-run immediately before execution. THIS is the
  wave's single read. All reader gates passed: config provenance
  (from_checkpoint WM identity + regex + frozen_wm False; scratch
  from_checkpoint ''), completeness 24/24, strictly-increasing steps,
  `excluded_short == []` (all runs max_step 496496 ≥ 4.9e5, 496
  episodes each — modal-uniform).

## Verdict (registered rules, quoted from read.json)

**"INSTRUMENT-SUSPECT (positive control failed) |
APT-INIT-BELOW-SCRATCH-AT-4X (also bounded: no meaningful late
advantage, CI upper < delta)"**

- **P-H3 positive control FAILED** (registered flag condition): hzt −
  hscratch late-window = **−19.9** [−41.9, +3.4], perm p = .136 — the
  task-unfrozen arm does NOT beat scratch in [4.0e5, 5.0e5]. Per the
  frozen rule the entire read carries **INSTRUMENT-SUSPECT**; branches
  are still reported under the flag.
- **P-H1 (under the flag): APT-INIT-BELOW-SCRATCH-AT-4X** — hzf −
  hscratch = **−89.9** [−147.3, −40.3], perm p = .0057 (n=8v8;
  δ = 0.15 × scratch-late = 140.0; CI upper −40.3 < δ ⇒ also
  BOUNDED). Strict late-window inferiority of the apt-unfrozen arm;
  reported as its own branch, NOT as "≈ untrained".
- **P-H2 (descriptive, gated runs)**: first sustained-positive bin of
  (hzf − hscratch): **none** — hzf never crosses scratch in any 5e4
  bin over the whole 5e5 horizon. Early-window ratio
  hzf/scratch = **0.323** (directional, non-matched vs the U-wave
  0.5–0.6× AUC anchors). Bin curves: hzt is the EARLY winner (820 by
  the 1e5 bin vs scratch 467) and is caught by scratch from ~2.5e5 on;
  scratch plateaus ~925–935; hzf climbs monotonically to ~892 in the
  final bin without reaching either.
- Late-window arm means: hzf 843.4 / hzt 913.4 / scratch 933.3.

## What this licenses (per the frozen texts)

- The **late-crossing attack receives no support**: "the reward-free
  payoff arrives later" is answered — at 4× the registered budget the
  apt-unfrozen arm is strictly BELOW scratch in the late window and
  never crosses it in any bin. No headline wording change; the
  existing budget qualifier stays as-is.
- Because P-H3 failed, the read is **INSTRUMENT-SUSPECT** and the
  quotable form is correspondingly limited: the late-window contrast
  at 5e5 lacks dynamic range for ANY initialization advantage (scratch
  reaches ceiling ~933), so this wave cannot be cited as a calibrated
  bound on transfer magnitude — only the sign/ordering facts above.

## Post-read notes (labeled, non-registered)

- The P-H3 "failure" is mechanism-coherent rather than anomalous:
  finger turn_hard is learnable from scratch to ~930 within 5e5 steps,
  so ALL arms converge and the transfer advantage is intrinsically
  EARLY (hzt's bin curve: +350 over scratch at 1e5, gone by 2.5e5).
  This is the same shape as the frozen U-wave facts — pretraining
  buys speed, not asymptote — now visible on the task-arm too.
- hzf still rising at 4.5e5 (892 last bin) suggests the apt arm would
  eventually reach ceiling as well; its LATE deficit is a slower
  transient, not an asymptotic gap claim (any such claim would need
  its own registration at a longer horizon — not planned; capacity/
  budget axes are capped by standing resource decisions).
