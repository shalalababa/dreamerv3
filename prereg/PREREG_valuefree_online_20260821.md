# PREREG — value-free pretraining vs online task pretraining
# (algorithm slate; 21 Aug 2026, v3 — v1 and v2 both returned
# NOT-FREEZE-READY (reviews #1 and #2); this file supersedes both in
# full, pre-freeze)

**Status: FROZEN at user commit. ONE read execution.** Reviewers: v1
Opus pass (protocol mismatch, empty equivalence branch, unreachable
anchor, discretionary side pin, 6 mechanical blockers) + v2 Opus delta
pass (D1 the arm trained a value head; D2 self-referential band; D3
unexecutable adapt plan; 8 MAJOR) + ONE v3 delta confirm pass — all
user pre-approved.

## Question (restated per D1)

Does the VALUE-FREE recipe — reward-free explorer collection + offline
reward-grounded WM fit in which **no value-derived gradient shapes the
pretrained trunk** (rgo: repval_grad False; a vestigial value head is
trained on stop-gradient features and discarded at transfer) and **no
online task interaction occurs at pretraining** — match a STANDARD
online, critic-driven, task-trained Dreamer trunk? Licensed by the
carrier result (reward-gradient path = sufficient trunk carrier; heads
never transfer) as the item-2 open question.

## Budget matching (v1 adjudication, unchanged)

**UPDATE-MATCHED: 500000 gradient updates both arms** (the house
convention; every env-step variant is confounded — ops inventory
21 Aug). Residual asymmetry registered with an ASYMMETRIC consequence:
online sees ~500k unique task-directed frames vs the q1 trunk's 200200
curated explorer frames ⇒ recipe match/win is a-fortiori; an online
win is attributable to {onlineness, task-directedness, 2.5× frames}
and licenses ONLY "no match at these budgets". pilot_goal_* excluded
(seeds, 5.1× budget, code-generation split).

## v3 design (finger, RGO arm; fixes delta-review D1–D19)

**Domain: finger_turn_hard** — chosen by a zero-compute substrate scan:
the unfrozen task trunk's value-over-scratch GROWS through the window
(auc50k G≈+244 → auc100k G≈+341 for the FULL task fit — used as PROXY
planning only, see below).

**The recipe arm is the RGO fits (fixes D1 — the core claim's arm).**
The full task fits train a value head whose repval gradient shapes the
trunk, so they cannot carry a "value-free" claim. The rgo fits
(`--agent.repval_grad False`) block the value-gradient path into the
trunk entirely; a vestigial value head is trained on stop-gradient
features and DISCARDED at transfer (adapt regex `^(enc|dyn|dec)/`) —
both facts disclosed in every use of the claim. The reader WITNESSES
the property: cfg_pins gate `agent.repval_grad == False`,
`agent.reward_grad == True` on all 16 recipe fits. Arm source: the
CARRIER wave's fresh fits `ax1wm_finger_rgoq1s{1,0}_seed{17..24}` —
one homogeneous batch, configs already verified against the
Amendment-1 references (amend2 schema-normalized match TRUE).

**Outcome-blindness (fixes D8, honestly)**: rgo-unfrozen adapts have
NEVER been run — the recipe arm's outcomes are genuinely unseen. The
disclosed full-task-fit values (uzt_s1 489.2 sd 124.9 / uzt_s0 333.5
sd 110.7 / scratch 147.7 sd 33.2, auc100k, seeds 1–8) are a PROXY for
planning only; rgo-unfrozen levels may differ, and the anchor gate
adjudicates feasibility at read time.

**ONE unfrozen protocol; ALL 32 adapts fresh in ONE batch.** Side s1
primary is pinned by the house headline-side convention (side1 is the
reported side in vgo_extended / scaling / FB; it is the
high-rewarded-regime-occupancy side) — fixed pre-outcome for the
never-run rgo-unfrozen arm (D12); s0 is a registered descriptive
secondary. **Buffer provenance disclosed (D13)**: `axis1_finger/q1/
side1` is a 20-source mixture including 4/200 episodes (~2% of
frames) from an online goal policy; side0 is single-source p2e3
(source_l1 = 0.9). The recipe's "reward-free collection" is therefore
~98% strict on s1 — carried verbatim wherever the recipe is described.

| arm | mode | source | adapt run_id |
|---|---|---|---|
| online | `ontask` | 8 FRESH runs `ontask_finger_seed{17..24}` | `adapt_ontask_finger_seed{k}_ckpt500000` |
| recipe s1 (PRIMARY) | `q1uzs1` | `ax1wm_finger_rgoq1s1_seed{17..24}` | `adapt_q1uzs1_finger_seed{k}_ckpt500000` |
| recipe s0 (secondary) | `q1uzs0` | `ax1wm_finger_rgoq1s0_seed{17..24}` | `adapt_q1uzs0_finger_seed{k}_ckpt500000` |
| scratch | `scratchvf` | random init | `adapt_scratchvf_finger_seed{k}_ckpt0` |

## Registered commands (fixes D3 — literal, never re-derived)

- Online runs (walltime override registered — pilot.sbatch ships 3 h):
  `for k in $(seq 17 24); do RUN_ID=ontask_finger_seed$k
  TASK=dmc_finger_turn_hard MODE=goal SEED=$k STEPS=5e5
  sbatch --time=10:00:00 scripts/pilot.sbatch; done`
- ontask adapts (DIRECT main.py — **NEVER scripts/axis1.sbatch for
  ontask**: its fit-skip guard reads 0 from a suffix-less online ckpt
  and would launch a 500k offline fit INTO the online run dir):
  `python dreamerv3/main.py --logdir
  $RUNROOT/adapt_ontask_finger_seed$k\_ckpt500000 --configs
  dmc_proprio unfrozen_readout --task dmc_finger_turn_hard --seed $k
  --run.from_checkpoint $RUNROOT/ontask_finger_seed$k/ckpt/<latest
  stamp> --run.steps 1.25e5 --env.dmc.render False`
- q1uzs{1,0} adapts (the U1 axis1 line, arm swapped to rgo):
  `RUN_ID=adapt_q1uzs<sd>_finger_seed$k\_ckpt500000
  WM_RUN=ax1wm_finger_rgoq1s<sd>_seed$k TASK=dmc_finger_turn_hard
  SEED=$k REPLAY=$RUNROOT/axis1_finger/q1/side<sd> UPDATES=500000
  STEPS=1.25e5 AXIS1_EXPL_MODE=task AXIS1_ARM=rgo
  AXIS1_ADAPT_CONFIG=unfrozen_readout bash scripts/axis1.sbatch`
  (the fit exists ⇒ stage 1 idempotently skips)
- scratchvf adapts: the ontask-adapt command WITHOUT
  `--run.from_checkpoint`, logdir `adapt_scratchvf_finger_seed$k\_ckpt0`.
- **Existence precondition (D7, transcribed from U1)**: before ANY
  adapt submission, verify all 16 rgo fits hold a done-ckpt at 500000
  and `axis1_finger/q1/manifest.json` exists — axis1's fit-skip guard
  silently REFITS a missing WM, and `ax1wm_finger_rgoq1s*` has no
  archive fallback yet (archive it first per never-delete).

## Window ladder + estimands (unchanged from v2 except the band)

PRIMARY auc100k; secondaries auc50k/auc125k with the pre-stated
ONLINE-SUPERIOR override (downgrades a match-grade license to
window-scoped); RECIPE-SUPERIOR-at-secondary decorates only a
match-grade/recipe primary (D11); any cross-window sign disagreement
is recorded symmetrically as DIVERGENT. Scope sentences: 500k-update
pretraining; reconstruction-based WMs.

d = mean(online) − mean(q1s1), n=8/arm, two_sample at α=.05 with CI
conjunction. **The equivalence yardstick is the RECIPE ARM's value
over scratch (fixes D2): G* = mean(q1s1*) − mean(scratch*)** — it does
not contain the online arm, so a stronger online arm can never widen
its own tolerance; the anchor gate is likewise q1s1-vs-scratch.
MATCHED: joint seed-bootstrap (B=200000 pinned — D15) P(G* > 0 AND
|d*| ≤ 0.5·G*) ≥ 0.95; RECIPE-NON-INFERIOR: one-sided analogue;
percentile joint bootstrap disclosed as not BCa-corrected (D16).
NO-CALL-ANCHOR suppresses EVERYTHING beyond point estimates — all
windows' CIs, certificates, and the ladder fire pattern (D4).

## Gates (refusal = read not consumed)

Witness (`build_rescue_bundle --wave valuefree --extra_fits <16 rgo
fit names>`): online counters = realized metrics.jsonl
`train/opt/updates` in [485000, 515000] with counters_source
'metrics_updates' (D14: replay/updates is never consulted); env_steps
in [480000, 520000] (two-sided — D6); cfg_pins gates (online: task /
train_ratio 1024 / steps 5e5; recipe fits: task + reward_grad True +
repval_grad False — the value-free witness) (D10); **adapt→checkpoint
linkage: every arm adapt's saved `run.from_checkpoint` must reference
its registered fit and scratchvf must load nothing;
`_meta.adapt_ckpt_ok` gated (D5)**; 16 rgo fit counters == 500000
exact; config identity per group; milestones (arms ckpt500000 /
scratch ckpt0); STRICT modal n_ep at every ladder window; qc;
ONE-read guard. Update-match tolerance disclosed: the online arm's
realized updates match 500000 to ±3% (D18).

## Ops

Registered commands above, in order: existence precondition → 8 online
runs (~50 GPU-h, walltime 10 h) → ALL 32 adapts in ONE batch (~24
GPU-h) → `build_rescue_bundle --wave valuefree --extra_fits
ax1wm_finger_rgoq1s1_seed17,...,ax1wm_finger_rgoq1s0_seed24` → bundle
+ sha manifest → **[ME] ONE read**. Total ≈ 75 GPU-h (UNVERIFIED;
re-derived from the first measured run).

**Venue pin (review-3 minor)**: every stage of this wave — the 8
online runs, the 16 rgo fits (pulled from the carrier runroot), all
32 adapts, and the bundle build — lives under ONE $RUNROOT on ONE
venue before bundling; if any stage runs elsewhere, FULL-pull first
(the 17-Aug bundle-after-pull rule), then bundle. Disclosure: the
ontask adapts load the latest online checkpoint, which trails the run
end by ≤ save_every (~0.3% of updates); direction is against the
a-fortiori arm and negligible (recorded, review-3 minor).
