# PREREG — Track A2: SCARECROW (structured noise as a region-fencing
tool), 22 Aug 2026
(review #34 adjudicated same day: 3B/12M/15m ALL applied — including
the B1 frozen-reward-head catch that invalidated the first smoke's
own evidence)

**Program:** Track A of `Plan_FollowupPrograms_20260821.md`; the A2
leg, GATED on A1 — **gate satisfied 22 Aug** (A1 read:
AVOIDANCE-CONFIRMED, Δ=−0.0598, p=.0143 = the exact floor;
review #34 verified the gate-as-fired matches the gate-as-registered).
User GO 22 Aug ("GO for A2, reviewers pre-approved").

**Question:** A1 established that spatially structured fictitious
noise REPELS a disagreement-driven explorer. A2 registers the TOOL
reading: can a practitioner FENCE a region by planting a scarecrow,
and how does that compare to the standard lever, a reward penalty?
Pre-declared stance (plan, pre-outcome): P(scarecrow beats the
penalty) ≈ 0.10 — **the paper's spine is the mechanism; the
algorithm is the demo.** The scarecrow's pitch is DEPLOYABILITY: it
acts purely through the observation channel and needs no access to
the agent's objective — applying the penalty baseline to a
reward-free explorer required building objective-surgery machinery
(`penalty_mix`) that did not previously exist.

## 1. Arms (20 runs, seeds 76–95, ONE wave, ONE INSTANCE
(non-negotiable — 1-Aug cross-invocation rule; every contrast is a
cross-arm permutation), 5e5 steps, Stage-1 substrate; region = the
pinned se_m3_read gate, position[0] > −0.13009691, throughout)

| arm | RUN_IDs | seeds | env (beyond Stage-1 defaults; every arm passes the FULL variable set incl. inert values — A1 rule) |
|---|---|---|---|
| sc_ctrl | `sc_ctrl_s76..79` | 76–79 | `DISTRACTOR_SCALE=0.35891393` (A1-flat VERBATIM) |
| sc_scare | `sc_scare_s80..83` | 80–83 | A1-hetero mod quadruple VERBATIM (`MOD_KEY=position INDEX=0 LO=-0.32203162 HI=0.19711795`), scale 1.0 |
| sc_scare2 | `sc_scare2_s84..87` | 84–87 | same quadruple + `DISTRACTOR_SCALE=2.0` |
| sc_pen1 | `sc_pen1_s88..91` | 88–91 | `PENALTY_MIX=True PENALTY_SCALE=0.5` + penalty gate triple = the pinned region; distractor = ctrl CONFIGURATION |
| sc_pen2 | `sc_pen2_s92..95` | 92–95 | as pen1 with `PENALTY_SCALE=2.0` |

Penalty arms carry the ctrl observation CONFIGURATION (flat
0.35891393, no modulation) at **matched observation statistics —
independent streams, different seeds** (#34 M9: there is no common
random numbers anywhere in this design).

**λ calibration (#34 M5, corrected units):** the Stage-1 probes'
`pse2.intrinsic_mean` is a HORIZON-15 SUM, measured 0.00188–0.00255
across the 8 Stage-1 seeds ⇒ ≈1.5e-4/step ⇒ steady-state imagined
intrinsic ≈ disag_scale × 1.5e-4 ≈ **0.15/step**. λ=0.5 is therefore
≈3.4× and λ=2.0 ≈13× the steady intrinsic — **both arms dominate at
steady state**; early in training the intrinsic is orders larger
(smoke: raw 0.287 ⇒ ×1000 = 287), so the penalty's influence GROWS
over the run. The registered bracket is {moderately-dominant,
strongly-dominant}, and the read carries a REGISTERED
realized-ratio row per penalty run (first/last `train/loss/rew`,
`train/expl/penalty_rew`, `train/expl/intr_rew_raw`) — without it
neither a null nor a fire on S2/S3 is interpretable.

**New machinery (built + selfchecked BEFORE this compute; all
BEHAVIOURALLY inert at defaults (#34 m14: the config dump gains the
new keys; every existing arm's behavior is unchanged — verified by
the Stage-1-argv compose assert and the scale-0 wrapper skip)):**
- `embodied/envs/regionpenalty.py`: REPLACES reward with
  −scale·1[region] (replacement pinned — the task reward must not
  leak; consequence, REGISTERED SCOPE LIMIT (#34 m3): the penalty
  arms' `scores.jsonl` is the penalty return, NOT cross-arm
  comparable, so A2 has no task-return neutrality leg for those
  arms). Gate-triple convention shared with distractor/planted;
  deterministic; estimand = penalized-for-the-state-entered (the
  post-step obs gates the reward written into the same replay row —
  #34 verified alignment with the reader's occupancy). λ must be
  float32-exact (#34 m5; the grid {0.5, 2.0} is).
- `agent.expl.penalty_mix` (default False): p2e explorer that also
  trains the reward head on the penalty and adds its prediction to
  the imagined intrinsic (`sg(disag×scale) + rew.pred()`; #34
  verified the composition is gradient-safe — imag_loss sg's the
  return at every consumer). **Review #34 B1: the head was NOT an
  optimizer target under p2e (nj.grad selects by module path), so
  the first smoke's `loss/rew = 5.5413 = ln(255)` was the exact
  signature of a FROZEN head and its constant `penalty_rew =
  −0.16448` the frozen TwoHot output. FIXED: `self.rew` joins
  `self.modules` under penalty_mix, plus a construction-time
  tripwire `assert (rew in modules) == (not reward_free)`.
  POST-FIX EVIDENCE (freeze note): short smoke `loss/rew` strictly
  decreasing (5.5413 → 5.5381 over 125 CPU grad steps), long smoke
  **5.541 → 2.2335 over ~3000 grad steps (< 5.0 bar cleared
  decisively)** with `penalty_rew` varying and moving toward the
  region penalty (last −0.127).** Disclosures: `expl/intr_rew` is not logged on
  penalty arms (reward_free-gated metric — monitoring asymmetry
  only, #34 m1); `reward_grad`/`repval_grad` mean the penalty arms'
  WM trunk is additionally shaped by the rew/repval losses —
  inherent to objective surgery, a second registered difference
  from ctrl/scare beyond the reward channel (#34 m12).
- Producer overrides `PENALTY_*` + BIND-CHECK wants **including the
  region-defining variables** (gate_index, gate_threshold, and the
  DISTRACTOR_MOD_* trio — #34 M8).

## 2. Registered read (frozen reader `uncfield/se_scarecrow_read.py`,
review-#34 form, selfcheck PASS; ONE execution)

- **Validity per run:** Stage-1 pin block + per-arm pins with STRICT
  key presence (#34 M7 — a pre-freeze checkout cannot pass by
  omission); se_probe validity + final-ckpt gate + the
  NEWEST-ckpt-dir cross-check (#34 M15: `ckpt/latest` itself can be
  a frozen stale pointer — the A1 append-verify incident);
  occupancy over the entire replay, MIN_STEPS 1e5; ONE replay scan
  per run (#34 m9); penalty arms: delivery gate (float32-exact −λ
  in-region / 0 out) AND the TRAINED-HEAD gate (#34 M6/B1: last
  `train/loss/rew` < 0.9·ln(255) AND below the first — presence
  alone is the frozen-head bug's own signature) + the realized-ratio
  row. Defective runs → excluded_runs, read persists; missing seeds
  recorded explicitly (#34 m13).
- **DEGENERACY, PER CONTRAST (#34 M4):** per-run collapse flags vs
  the **PINNED Stage-1 reference (#34 M14):
  `local_results/uncfield_se_stage1_20260813_193324/runroot/
  se_cheetah_seed10..17`, ref n == 8 hard-asserted** (measured
  intrinsic_mean 0.00188–0.00255); a collapsed run voids ITS
  contrast only (a λ=2.0 penalty crushing exploration is an
  expected S2 finding, not a read-killer); per-arm collapse rows
  always emitted.
- **PRIMARY:** occupancy(sc_scare) < occupancy(sc_ctrl), exact
  C(8,4)=70 one-sided permutation, fires iff p ≤ .05 (floor .0143).
  **FULL 4+4 ONLY (#34 B3): any missing/invalid/collapsed run in
  either primary arm ⇒ NOT-ADJUDICABLE** — review #34 demonstrated
  the drop-and-re-enumerate fallback lets an excluded low-occupancy
  control MANUFACTURE a fire; exclusion must only ever cost
  adjudicability. The partial contrast is reported descriptively.
- **REGISTERED SECONDARIES** (each computed iff its own arms are
  whole — #34 M4; raw per-arm occupancy/coverage/θ₁ rows are emitted
  UNCONDITIONALLY on every path (#34 m4 wording)): S1 scare2-vs-ctrl
  perm + scare2−scare delta; S2 pen1/pen2-vs-ctrl perms; S3
  scare-vs-pen2 DESCRIPTIVE (pre-declared P(win)≈0.10, no fire,
  scores not cross-arm comparable); **S4 off-region coverage
  EXCLUDING the gated position dim (#34 M10 — conditioning on a
  coordinate and then measuring that coordinate's truncated spread
  is mechanically coupled to the fence; the gated dim's off-region
  std is its own row)** + the unrestricted proxy (std floor 1e-6,
  per-dim min recorded — #34 m10); S5 θ₁ rows per arm.
- **Outcome map:** SCARECROW-FENCES / TOOL-NOT-CONFIRMED (an
  informative fragility boundary) / NOT-ADJUDICABLE. Penalty rows
  never rescue or veto the cell.

## 3. Execution

`ops/waves/a2_scarecrow/spec.yaml` (review-#34 form): ONE instance,
lanes 0–3, ~35 h drain, arms balanced per lane by stage structure;
`done_when` regexes newline-tolerant (`\s+` — #34 B2: the config
emitter wraps flow maps and the literal patterns matched NOTHING on
4 of 5 stages); full inert variable set on every cmd; `bundle:`
block present (#34 M12). **DISK (#34 M13): 20 × 4.6 G ≈ 92 G does
not fit an instance — per-run pulls to RCC as runs finish (verify,
then delete instance-side; the RCC copy is the archive), never an
end-of-wave drain.** Stale-checkout defenses, in order: wavegen
var-read preflight, unknown-flag startup death on old checkouts,
the producer BIND-CHECK, the reader's strict pins (#34 m11 wording).
Probes: `scripts/se_scarecrow_probe.sbatch`, ONE CPU job,
ckpt-identity skip rule + the latest-vs-newest stale-pointer refusal
(#34 M15). Then bundle per results-sync v2 WITH replay/ → verify →
ONE read.

## 4. Fences & placement

Track-A paper (A1 mechanism + A2 tool demo); cites Paper 5's arXiv.
`penalty_mix` is instrument, not contribution. No Paper-5 content
depends on this wave.

Freeze = review #34 adjudicated (this file) + commit of: this file,
`embodied/envs/regionpenalty.py`, the `main.py` / `configs.yaml` /
`agent.py` / `scripts/uncfield_se.sbatch` deltas,
`uncfield/se_scarecrow_read.py`, `scripts/se_scarecrow_probe.sbatch`,
`ops/waves/a2_scarecrow/spec.yaml` — then the 20 submissions.
