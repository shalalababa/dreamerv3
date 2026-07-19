# Synth diagnosis read — gate CLOSED: in-subspace refuted at the model level (2026-07-18)

Snapshot: `local_results/x0_stamp_synthdiag_20260718_175314/synth_diagnosis/`.
Read per `prereg/PREREG_synth_diagnosis_20260718.md` (descriptive-only;
gates Phase-B′). The user's cluster read included two seed-99 smoke
fits swept up by the `ax1wm_synth_*` glob (n=9/side); the canonical
numbers here are the local recompute on the registered 48 seed-1–8
summaries (smoke excluded by rule; verdicts identical, numbers
slightly sharper). `synth_diagnosis.json` here.

| prediction | result | verdict |
|---|---|---|
| P-D1 rank: synth ≤3 both, finger >3 both, synth<finger | synth [3, 2] vs finger [6, 5] | **PASS** |
| P-D2 task rew-NLL side separation < 0.051 | s0-fits 4.61 vs s1-fits 0.67 — separation **3.93** (~77× finger's) | **FAIL** |
| P-D3 apt/task to_target-NLL ratio ∈ [0.8, 1.25] | side0 **3.74**, side1 0.63 | **FAIL** |

**Gate CLOSED — no Phase-B′.** The registered adjudication: any
prediction fails ⇒ re-diagnose.

## What this actually shows

The in-subspace hypothesis survives at the DATA level (P-D1: the synth
reward direction is top-of-spectrum, rank 2–3 of 14; finger's is
deeper, 5–6 of 12) but is refuted at the MODEL level:

- On the LO-occ synth buffers, the task-arm reward head largely fails
  to learn the reward (NLL 4.61 vs 0.67 on hi) — a **massive
  occupancy-gated learnability gap**, the opposite of "nothing left to
  rescue".
- Reconstruction of `to_target` itself is arm-dependent on the lo side
  (apt 3.74× worse than task) and slightly arm-reversed on the hi side
  (0.63) — supervision measurably shapes even the directly-observed
  reward coordinate when regime data is scarce.

So synth exhibits real occupancy- and supervision-gated
representation structure — exactly the ingredients of the DMC
interaction — **yet the behavioral interaction was null** (Phase B:
+44.2 [−88.0, +178.4]). The bottleneck implied is DOWNSTREAM of the
representation: the frozen-readout adaptation stage. Consistent with
this: the enormous per-seed adapt variance in Phase B (task deltas
−256..+375) and the point-mass task being controllable from mediocre
trunks. New leading hypothesis (post-hoc, unregistered):
**adapt-stage variance/exploration dominates synth transfer, washing
out a real representation-level effect.** Candidate re-diagnosis
moves (user's call, none authorized by the closed gate): more adapt
seeds per fit (variance decomposition fit-vs-adapt), longer adapt
window, or a dense-reward readout probe of trunk quality that bypasses
the sparse-reward adaptation lottery.

Theory note: P-D1's pass + P-D2/P-D3's fail jointly refute the simple
static-spectrum reading for sequence models — the RSSM's allocation
tracked occupancy and supervision even for a top-of-spectrum
direction. Worth a line in the theory's honesty box; no registered
prediction was at stake (the note's predictions were the gate's, and
the gate closed as registered).

## Provenance

- `synth_diagnosis.json` — canonical recompute (48 registered
  summaries; seed-99 exclusion).
- Inputs in the snapshot: `axis1_{synth,finger}/rank_s{0,1}.json`
  (ν-direction instrument), `e4_outputs/ax1wm_synth_*/e4_synth_v1/`
  (48 measure outputs), `e4_probesets/synth_v1` (FROZEN).
- Cluster-side read (n=9 contamination, verdicts identical) at
  `analysis/analysis_synth_diagnosis_20260718_174920/` in the snapshot.
