# Scaling Option B read + 12m E4 pass — interaction SCALE-ROBUST at 12× (2026-07-26)

Snapshots: `local_results/scaling_option_b_20260726_130956/` (32 new
12m jobs: {task, apt} × {s0, s1} × seeds 1–8, size12m fit+adapt on the
`q1_ctx12` resized pair; 1m strata = the committed historical cells)
and `local_results/scaling_b_e4_20260726_152307/` (E4 pass over all 32
12m fits, frozen `finger_v1` probe set, glob `ax1wm_finger_*s12*`).
Read exactly as registered (`prereg/PREREG_scaling_pilot_20260717.md`)
with the pre-frozen `analysis/scaling_read.py` (selfcheck PASS).
Freeze-ordering honored: `PREREG_compcapacity_theory_20260724.md` was
committed (746d0b13) before this read executed.

## Pipeline integrity

- exclusions.csv EMPTY — all 32 cells complete (the 4 previously
  in-flight runs finished as the owner stated).
- Local read IDENTICAL to the cluster-side json on every decisional
  quantity (per-seed deltas, means, bootstrap CIs, perm p, pos
  counts); only ULP-level float noise in the scipy t-interval
  *sensitivity* sub-blocks + the csv path string differ.
- Size audit: E4 log shows every s12 fit loading at deter 2048 /
  stoch 32×16 / 9.73M params (vs 0.63M at 1m) — the 12m cells are
  genuinely size12m (a size mismatch would have failed checkpoint
  loading by design).
- Buffer protocol: 14/14 retained adapt logs show
  `Static replay: .../q1_ctx12/side<s>` (registered ctx-resize
  protocol; the 15th file is a worker wrapper log).
- 1m strata bit-check against the original paired JSONs passed inside
  the frozen read.
- Registration housekeeping disclosure: the prereg's FILL slots
  (upd/s, bundle time, rank prior) were never filled in. The update
  count is verifiable as the registered 500000 (submission env +
  configs); the reward-direction rank prior was stated-prior-only
  with no decision role and was simply never recorded. Neither
  affects any registered statistic.

## Registered read (seed-paired, cluster bootstrap B=10K rng 0)

| endpoint | value | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY: [B_task@12m − B_apt@12m] − [B_task@1m − B_apt@1m] (AUC100k)** | **−30.8** | **[−154.5, +83.4]** | **CI includes 0** (pos 4/8, perm p=.65) |
| S1: same three-way, final10 | −51.4 | [−248.8, +186.0] | directional only |
| S1d: final10 − AUC100k (P-B5 dissociation) | −20.6 | [−241.8, +231.8] | sign-consistent, uninformative |
| S2: B_apt@12m (P-B6-lite) | +12.7 | [−16.3, +42.5] | **includes 0 — as predicted** |
| interaction @12m (descriptive) | **+126.5** | [+63.0, +176.3] | fires, 7/8 seeds |
| interaction @1m (descriptive, optimistic-draw stratum) | +157.3 | [+76.5, +240.7] | 8/8 seeds |

Registered branch: **capacity does not substitute at 12× —
compression account unchallenged; the interaction is SCALE-ROBUST
(Paper-1 strengthens); the 9b ladder needs a wider size range to find
any crossing.**

Robustness variant (registered "reported alongside", computed
descriptively here — see DEVIATIONS note on the reader gap): replacing
the optimistic 1m stratum with the honest W0 pooled interaction
(+98.7) gives three-way **+27.8 [−35.7, +77.6]** — still includes 0,
same branch; under the honest baseline the 12m supervision effect is
if anything numerically LARGER. Cell means (AUC100k, s0/s1): task@12m
97.0/236.1, apt@12m 87.5/100.1, task@1m 108.6/267.9, apt@1m 78.9/80.9
— levels are stratum-internal (ctx12 protocol caveat) and modest;
the contrasts carry the inference.

## 12m E4 pass (registered descriptive membership readout)

32/32 fits scored ("measured 32 new, 0 skipped"), frozen finger_v1
probe set. Task@12m in-regime reward-NLL at h0: **hi side 1.304
(0.94–1.59) — inside the 1m membership band** (full 1.02 / rgo 1.21);
lo side 5.77 (2.0–9.2) — beyond the 1m non-membership band.
Membership at 12m mirrors the 1m occupancy pattern exactly. d_errin ≈
0 both arms (arm-invariance replicates again at 12×). Disclosure: the
apt arm's reward head trains on the APT intrinsic reward, so the
frozen machinery produces NO apt true-reward NLL (reward_aware gate —
absent-by-design); the prereg's "does apt@12m rew-NLL move toward
task@12m" is unevaluable in that literal form, and the substitution
question is instead answered by the AUC contrasts (no substitution).

## Theory adjudications (registered forms)

- **P-E1 (ordered un-nulling / forbidden pattern): HOLDS.** The
  forbidden pattern (apt significantly above its 1m floor while
  task-lo floored) is NOT observed: S2 includes 0, no cell un-nulls,
  the interaction persists ⇒ 12m is still regime A. Registered
  consequence: **the 25m point targets the A→B transition (task-lo
  rise)** — this parameterizes Paper 3's prereg.
- **P-E2 (task-lo moves first): premise not triggered** (no cell
  moved off the 1m pattern); consistent, idle.
- **P-E3 (metric split): premise not engaged** — membership
  substitution was not reached anywhere (AUC gap persists; task-arm
  E4 mirrors 1m). The NOT-predicted pattern ("AUC gap gone while E4
  levels separated") did not occur.
- **P-B5 (representation-not-compression): directionally consistent,
  underpowered.** S1d is sign-consistent (−20.6) with a huge CI; the
  E4 leg supports the representation half (membership present at 12m
  hi with no compression gain). No fire either way at this n.
- **P-B6-lite: CONFIRMED as predicted** — reward-free transfer stays
  null at 12m (+12.7 [−16.3, +42.5]); capacity does not substitute
  for absent support. This is the FOURTH reward-free finger null,
  now also at 12× capacity.

## Consequences

- **Paper 1**: the interaction is scale-robust at 12× — the strongest
  possible outcome for the headline (the effect is not a small-model
  artifact). The scale-robustness sentence joins the band-complete
  claim set.
- **Paper 3**: wakes up — its prereg freezes next with the registered
  25m buy rule targeting the A→B transition (task-lo rise), per the
  consequence map; the volume-anomaly replication (P-E4a) is now
  unblocked and needs its own registration at submission.
- 9a/9b recipes: unaffected (9b crossing not found at 12×; wider
  ladder is a Paper-3 question).

## Provenance

- `scaling_read.json` — full read output (deltas per seed, all
  endpoints, sensitivity suite, registered branch).
- `auc.csv` — canonical AUC rows (12m modes + historical 1m strata).
- `e4_finger_v1_scaling_b.csv` — 12m E4 pass (128 rows, 32 runs).
- Read command: `python -m analysis.scaling_read --auc <csv>
  --output <dir>`.
