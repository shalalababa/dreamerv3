# PREREG: third-domain wave — sparse presence (P-D1) + dense attenuation (P-D2) — 2026-08-12

User GO 2026-08-12 ("proceed with all paper 1 GO remaining stuff"; the
slate decided 11 Aug: "let's run the late-crossing/horizon and third
domain"). This wave grades the two registered domain predictions of
`PREREG_theory_R1_gating_20260811` Part A:

- **P-D1 (sparse third domain)**: the task×transfer (arm×occupancy)
  interaction is PRESENT (CI > 0). Domain named here per the frozen
  text ("reacher_hard or acrobot-sparse; named at wave registration"):
  **dmc_reacher_hard** — a frozen regime spec already exists
  (`probing/regimes.py` 'reacher', to_target_norm < 0.025, verified vs
  the loaded MuJoCo model), and Paper-2's gate0 record
  (`artifacts/gate0_20260702/gate0_reacher/`) documents the regime.
- **P-D2 (dense domain, discriminating)**: in a dense-reward domain the
  interaction is ATTENUATED relative to finger. Instantiated as
  **dmc_walker_walk** (walker-class dense shaped locomotion). Honest
  deviation note: R1's parenthetical example was "walker_run-class";
  walker_walk is chosen PRE-OUTCOME because (a) it is the same dense
  shaped-velocity reward family, (b) the CONTROL-domain collector
  pretrains `pretrain_{p2e,apt,random}_walker_seed{1..5}` already exist
  (walker_walk was the study's control domain), so the collector pool
  is matched-in-kind to finger's at zero new compute, and (c) the
  `walker` short-name ↔ `dmc_walker_walk` mapping is already the
  codebase's reverse-lookup convention (`task_of`, submit_all.sh) — a
  walker_run leg would collide with it. The attenuation comparison is
  scale-normalized (d_z), per `PREREG_theory_R1_gating_amend2_20260812`.

## Honesty block

Known at freeze: all executed reads through 2026-08-12, in particular
the finger anchors this wave consumes: W0 n=16 interaction **+98.72
[+45.81, +158.83], d_z = 0.8284213656547132**
(`artifacts/w0_n16_interaction_20260713/n16_interaction_read.json`,
`primary_n16_interaction.sensitivity.d_z` — the file carries three
sensitivity blocks; this is the primary's) and the dv3 finger task−apt
level contrast +108.34 (computed 12 Aug from the committed
`artifacts/p0_axis1_corrective_20260713` csvs, seeds 1–8 pooled sides,
qc-passed). Unknown: every reacher/walker quantity — no axis1 run,
buffer, index, or pair exists for either domain.

**Registered instrument note (walker regime, review A5 — measured, not
asserted):** the proprio obs carries qvel only; the regime scalar is
qvel[1] (rootx slide velocity, layout verified against the loaded
model 12 Aug), NOT the reward's torso_subtreelinvel sensor (absent
from obs). Measured on RANDOM rollouts, TWICE independently (builder: 8 eps ×
400 steps; reviewer + verifier: 6 × 1000 steps at two seeds, 12 Aug):
occ(qvel[1] > 1.0) ≈ 0.045–0.047 vs the sensor's plateau occ ≈
0.006–0.007; pearson r ≈ 0.51–0.54; precision P(sensor > 1 |
qvel[1] > 1) ≈ 0.04–0.07, recall ≈ 0.36–0.44 — under random flailing
the proxy over-counts ≈ 6.6–8.2× (verification N5: quoted as ranges,
not the optimistic draw). The proxy is therefore honest only as a
"sustained-locomotion-enriched" occupancy, not the reward plateau
itself; the expectation (untested pre-goal-data) is that on
goal-policy episodes the two align far better. **Registered check:**
after the goal pilots land and BEFORE any fit is submitted, the
per-episode occupancy distribution by source (from `episodes.json`) is
recorded in PAIR.md; if goal-source episodes do NOT separate from
explorer episodes in qvel[1] occupancy (goal mean ≤ 2× explorer mean),
the walker leg is DOMAIN-BLOCKED — the proxy failed its only
discriminating job. Random-rollout occupancy at the 1.0 threshold
measured 0.002–0.07 (small in-regime corner, contract satisfied).

## Design — per domain: 2 arms × 2 sides × 16 seeds = 64 fit+adapt jobs

Protocol = the finger W0 protocol verbatim (dmc_proprio, size1m, 500k
offline fit updates, frozen-readout adapt 1.25e5 steps, auc100k).

### Stage 0 — collectors

- **Review A1 (BLOCKING, fixed here):** `DECOUPLERS=""` would NOT
  empty the decoupler list — submit_all.sh's `${DECOUPLERS:-…}` fires
  on empty-as-well-as-unset and would have submitted cup+finger
  pretrains too (and under FORCE=1, over frozen collector substrate).
  The registered value is a SINGLE SPACE (`DECOUPLERS=" "` — non-empty,
  so the default does not fire; `read -ra` yields an empty array).
  Requires bash ≥ 4.4 for empty-array expansion under `set -u`
  (**preflight: `bash --version` on the login node, recorded in
  PAIR.md**), and every submission below runs FIRST with `DRYRUN=1`
  and the printed job list is checked to contain ONLY the intended
  domain before the real invocation.
- Reacher (fresh):
  `CONTROL=dmc_reacher_hard DECOUPLERS=" " SEEDS="1 2 3 4 5" STEPS=5e5 ./scripts/submit_all.sh pretrain-bundles`
  (15 runs: p2e/apt/random × seeds 1–5).
- Walker: the existing CONTROL pretrains are the pool. **Preflight
  (registered):**
  `ls $RUNROOT/pretrain_{p2e,apt,random}_walker_seed{1,2,3,4,5}/replay/*.npz | head`
  — every one of the 15 replay dirs must be non-empty — PLUS the key
  check (review A10):
  `python -c "import numpy as np,glob,sys; f=sorted(glob.glob('$RUNROOT/pretrain_p2e_walker_seed1/replay/*.npz'))[0]; z=np.load(f); assert {'height','orientations','velocity'} <= set(z.keys()), sorted(z.keys()); assert z['velocity'].shape[1]==9, z['velocity'].shape; print('walker keys OK')"`.
  Any missing source is resubmitted via
  `CONTROL=dmc_walker_walk DECOUPLERS=" " SEEDS="<missing>" STEPS=5e5 FORCE=1 ./scripts/submit_all.sh pretrain-bundles`
  (DRYRUN-checked first; FORCE=1 is now scoped to walker only) and the
  substitution disclosed in PAIR.md (fresh episodes are a valid pool;
  the index records realized sources).
- Goal collectors (both domains; `pilots` stage gained PILOT_SEEDS /
  PILOT_MODES in this freeze-commit, default behavior unchanged):
  `PILOT_MODES="goal" PILOT_SEEDS="1 2 3" STEPS=1e5 ./scripts/submit_all.sh pilots dmc_reacher_hard dmc_walker_walk`

### Stage 1 — index → search → build (login node, per domain)

```bash
python -m probing.build_controlled_replay index \
  --replay p2e1=$RUNROOT/pretrain_p2e_<dom>_seed1/replay p2e2=... p2e3=... p2e4=... p2e5=... \
          apt1=$RUNROOT/pretrain_apt_<dom>_seed1/replay apt2=... apt3=... apt4=... apt5=... \
          rnd1=$RUNROOT/pretrain_random_<dom>_seed1/replay rnd2=... rnd3=... rnd4=... rnd5=... \
          goal1=$RUNROOT/pilot_goal_<dom>_seed1/replay goal2=... goal3=... \
  --task <TASK> --output $RUNROOT/axis1_<dom>/episodes.json
python -m probing.build_controlled_replay search \
  --index $RUNROOT/axis1_<dom>/episodes.json \
  --ref_replay $RUNROOT/pilot_goal_<dom>_seed1/replay \
  --n_episodes <K> --output $RUNROOT/axis1_<dom>/pairs.json
python -m probing.build_controlled_replay build \
  --index $RUNROOT/axis1_<dom>/episodes.json \
  --pairs $RUNROOT/axis1_<dom>/pairs.json \
  --which q1 --output_root $RUNROOT/axis1_<dom>/q1
```

with `<dom>/<TASK>` ∈ {reacher/dmc_reacher_hard, walker/dmc_walker_walk}.

- **Index-first decision point (review A6):** `index` runs FIRST as a
  standalone cheap step; the per-source per-episode occupancy
  distribution is read out of `episodes.json` and recorded in PAIR.md
  BEFORE choosing K and before ANY further spend on that domain. If no
  source shows a hi-occupancy episode population capable of meeting the
  side1 bar (feasibility read: max source-mean episode occupancy across
  sources < 0.15), the domain leg is **DOMAIN-BLOCKED at the index** —
  disclosed, before fits. Known structural risk (review A6): `search`
  draws candidates within (source, occ_bin) strata and its occ_bin
  splits at 0.5, so candidate occupancy concentrates near source means;
  a 0.15 side1 needs a genuinely hi-occ source (the goal pilots), not
  tail luck.
- **Side-order normalization (review A2 — BLOCKING-class precedent,
  DEVIATIONS.md Option-C side inversion):** `search` does NOT enforce
  side0 = low occupancy. Registered normalization between `search` and
  `build`, run verbatim:

```bash
python - "$RUNROOT/axis1_<dom>/pairs.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
q1 = d['pairs']['q1']
assert q1 is not None, 'no q1 pair'
if q1['sides'][0]['occ'] > q1['sides'][1]['occ']:
    q1['sides'] = [q1['sides'][1], q1['sides'][0]]
    json.dump(d, open(p, 'w'), indent=1)
    print('SIDES SWAPPED (record in PAIR.md)')
else:
    print('side order OK (side0 = low occ)')
PY
```

- **K ladder (registered, pixel-repl precedent):** n_episodes 200, else
  160, else 120 — first K whose search returns a q1 pair AND whose
  built manifest passes the quality gate. **Quality gate (buffer
  properties, decidable pre-outcome):** side1 `occ_recomputed` ≥ 0.15,
  side0 `occ_recomputed` ≤ 0.05, side1/side0 occupancy ratio ≥ 3
  (division guarded: side0 = 0 passes the ratio clause), search-native
  dcov/docc constraints already enforced by `search`. If K = 120 fails,
  that domain leg is **DOMAIN-BLOCKED** — reported, no fits submitted,
  no threshold loosening. (`search-matched` is the registered fallback
  DESIGN if the plain search proves structurally unable to reach the
  bars while the index shows feasible mass — that switch is a dated
  amendment, disclosed as a protocol-identity trade-off, never a
  silent substitution.)
- **Independent occupancy recompute (review A4; literal command per
  verification N4):** `occ_recomputed` in the manifest is
  arithmetically the search's own numbers (and walker is the first
  `direction='above'` regime). Before any fit submission:
  `python -m analysis.buffer_battery report --buffer s0=$RUNROOT/axis1_<dom>/q1/side0 --buffer s1=$RUNROOT/axis1_<dom>/q1/side1 --task <TASK> --output $RUNROOT/axis1_<dom>/battery.json`
  — the comparable field is `buffers.<label>.occ_frame`; both numbers
  land in PAIR.md; |Δ| > 0.01 on any side REFUSES the pair.
- **Fill record:** realized index composition + per-source occupancy
  readout, chosen K, pair stats, side-swap note, independent-recompute
  numbers, manifest confound deltas, holdout note (review A8: `search`
  runs WITHOUT --holdout here — no probeset consumer exists in this
  wave; stated so the omission is visible) →
  `artifacts/domains_pair_<date>/PAIR.md` BEFORE the first fit
  submission. No re-search after any fit starts.

### Stage 2 — fits + adapts (per domain)

```bash
AXIS1_EXPL_MODE=task AXIS1_DOMAINS=<dom> AXIS1_QUADS=q1 \
  AXIS1_SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" ./scripts/submit_all.sh axis1-bundles
AXIS1_EXPL_MODE=apt  AXIS1_DOMAINS=<dom> AXIS1_QUADS=q1 \
  AXIS1_SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" ./scripts/submit_all.sh axis1-bundles
```

Names (frozen registry conventions): WMs
`ax1wm_<dom>_q1s<side>_seed<s>` (task) / `ax1wm_<dom>_fq1s<side>_seed<s>`
(apt); adapts `adapt_ax1q1s<side>_<dom>_seed<s>_ckpt500000` /
`adapt_ax1fq1s<side>_<dom>_seed<s>_ckpt500000`. No name collisions
exist (no prior axis1 runs in either domain).

### Stage 3 — collate + witness + bundle (per domain)

- AUC collate: `python -m analysis.adaptation_auc --runroot $RUNROOT --output analysis_out_<dom>/`
  (the reader filters rows by domain + mode regex; extra rows are
  ignored, missing rows REFUSE).
- **Dual witness (standing rule), literal producer:**

```bash
# DOM = the leg being bundled (verification N1: the legs read
# independently — the producer must not require the OTHER domain's
# fits to exist)
cd $RUNROOT && DOM=<dom> python - <<'PY'
import glob, json, os, re
out_c, out_s = {}, {}
dom = os.environ['DOM']
pat = re.compile(rf'^ax1wm_{dom}_f?q1s[01]_seed(1[0-6]|[1-9])$')
for d in sorted(glob.glob('ax1wm_*')):
  if not pat.match(os.path.basename(d)): continue
  kv = dict(l.split('=', 1) for l in open(os.path.join(d, 'OFFLINE_FIT_PROGRESS')).read().splitlines() if '=' in l)
  out_c[os.path.basename(d)] = dict(update=int(kv['update'].split('/')[0]) if '/' in kv['update'] else int(kv['update']), total=int(kv.get('total_updates', kv['update'].split('/')[-1])))
  steps = []
  for c in glob.glob(os.path.join(d, 'ckpt', '*')):
    m = re.search(r'-(\d{12})$', c)
    if m and os.path.exists(os.path.join(c, 'done')): steps.append(int(m.group(1)))
  out_s[os.path.basename(d)] = max(steps) if steps else None
json.dump(out_c, open(f'domains_fit_counters_{dom}.json', 'w'), indent=1)
json.dump(out_s, open(f'domains_ckpt_steps_{dom}.json', 'w'), indent=1)
assert len(out_c) == 64, ('expected 64 fits', len(out_c))
print(len(out_c), 'fits witnessed')
PY
```

(the per-domain json names feed the reader's `--fit_counters` /
`--ckpt_steps` inputs for that leg)

- Bundle per domain: `auc.csv` + the 128 per-run `config.yaml`s
  (64 fits + 64 adapts, under `runroot_light/<run_id>/config.yaml`) +
  `domains_fit_counters_<dom>.json` + `domains_ckpt_steps_<dom>.json` +
  `q1_manifest.json` (the copy of `axis1_<dom>/q1/manifest.json` — the
  reader's `--manifest` input, review A12) + PAIR.md; rsync + sha256
  manifest per results-sync policy v2. Operational notes: submission
  commands are RE-RUN until drained (MAX_JOBS default 12 caps each
  invocation — review X3); the shared statistical core (`one_sample` in
  `analysis/domains_read.py`) is ALSO imported by the tm2-bridge and
  lewm frozen readers — any change requires dated amendments to all
  three registrations (review B10).

## Registered decision rules (frozen reader `analysis/domains_read.py`)

**TWO registered read executions, one per domain leg** (`--domain
reacher`, `--domain walker`), each executed ONCE, each as soon as its
own bundle lands (the legs share no estimand; neither waits for the
other). Gates per leg, fail-closed:

- Exact inventory: all 64 (arm × side × seed) auc rows present, no
  duplicates, `qc_pass` true, `n_ep_100k` STRICTLY equal to the modal
  count (realized-training rule, strict form — review X1/B4: sub-modal
  = truncation, super-modal = the scores.jsonl append-on-forced-restart
  duplication signature; benign variation goes through a dated
  amendment).
- Dual witness: every one of the 64 WMs has `update == total` fit
  counters AND max done-ckpt step == 500000.
- Config gates per run: fit `agent.expl.mode` == arm and task == the
  domain task; adapt `run.from_checkpoint` names the seed/side-matched
  WM, adapt `agent.frozen_wm` is not False, adapt task matches.
- Buffer manifest gate: the quality-gate bars re-checked from
  `manifest.json` (side1 occ ≥ 0.15, side0 ≤ 0.05, ratio ≥ 3).

Estimand (the W0 estimand verbatim): per-seed interaction delta
`[task(s1) − task(s0)] − [apt(s1) − apt(s0)]` on auc100k, n = 16.
Statistics: mean, one-sample BCa 95% CI (jackknife acceleration),
EXACT sign-flip permutation p (all 2^16 flips enumerated). Permutation
is primary, BCa the interval (standing rule).

- **P-D1 (reacher)**: CI > 0 ∧ perm p < .05 ⇒ **PRESENT** (the
  interaction is a three-domain fact for Paper 1's own protocol). CI <
  0 ∧ p < .05 ⇒ **REVERSED** (R1-violating; reported). Else ⇒
  **ABSENT-OR-UNDERPOWERED** with the registered MDE note (paired
  n=16, α=.05: power ≈ .87 at the finger d_z 0.828; MDE₈₀ d_z ≈ 0.749,
  reported also in realized score units 0.749 × sd(deltas)).
- **P-D2 (walker)**: normalized per
  `PREREG_theory_R1_gating_amend2_20260812`: d_z(walker) =
  mean/sd(ddof=1) of the 16 deltas vs the pinned finger anchor
  **FINGER_DZ = 0.8284213656547132**. Branches (review A3: a
  significant NEGATIVE interaction is a reversal, never attenuation):
  CI < 0 ∧ p < .05 ⇒ **REVERSED** (R1-violating; reported); else d_z
  point < FINGER_DZ ⇒ **ATTENUATED-CONSISTENT** (directional, as
  frozen); else **NOT-ATTENUATED** (R1-inconsistent; reported). The
  walker CI/perm are reported descriptively; if walker CI > 0 ∧
  p < .05 the read ALSO records BREADTH-PRESENT (win-win disclosed at
  GO: attenuation-with-presence is mechanism-consistent breadth).
- Secondaries (descriptive, never decisional): per-cell means, arm
  level contrast (task − apt pooled), per-side simple effects, the
  walker/finger and reacher/finger raw-scale ratios (labeled
  scale-confounded).

No cross-domain pooled estimand exists; no dose wording; no capacity
wording. R1 Part-A grading of P-D1/P-D2 happens in each leg's RECORD
against the frozen texts.

## Power + risk disclosures

- n=16 per cell-pair matches W0. Power ≈ .87 (exact .872) if the
  new-domain effect equals finger's d_z; ≈ .34 (exact noncentral-t
  .342) at half that (disclosed — an ABSENT-OR-UNDERPOWERED reacher
  leg licenses no absence claim).
- Reacher sparse-floor risk: R3 realized floors 1/32 (3.1%) at 1e5
  steps — low; the adapt qc gate (≥20 episodes) is the floor guard.
- Walker attenuation-by-ceiling risk (registered interpretation
  guard): if BOTH walker arms adapt near env-max, ATTENUATED-CONSISTENT
  is also a ceiling fact; the reader reports max cell mean / 1000 so
  the RECORD can flag ceiling compression; the branch verdict is
  unchanged (attenuation-by-ceiling IS R1's predicted mechanism — dense
  reward → legibility bottleneck small → transfer differences small —
  but the RECORD must not oversell d_z attenuation as mechanism-specific).

## Discipline

This file + `analysis/domains_read.py` (selfcheck PASS) +
`PREREG_theory_R1_gating_amend2_20260812.md` + the plumbing edits
(regimes.py walker spec; submit_all.sh domain cases + PILOT_SEEDS;
runroot_cleanup KEEPs) are freeze-committed BEFORE any collector,
buffer, fit, or adapt exists in either domain. ONE execution per
domain leg. Reviewer: 2026-08-12 build batch (Opus 5, sequential
build→verification per the 12-Aug standing policy).
