# PREREG: s×w_r wave — design, cells, decision rules — 2026-08-09

Implements `PREREG_swave_theory_correction_20260808.md` (committed in
`1a36a65a`; constraints 1–6 inherited unweakened: s×w_r
pairing mandatory, own-label scoring mandatory, buffer identity via the
`scale` kind, instrument form `ln(1+s)²·f` with relative response at
s=10 = (ln11/ln2)² ≈ 11.97×, frozen dose crossings). Frozen BEFORE any
scaled buffer, w_r config, or fit exists. Instruments: `relabel_replay`
`scale` kind + `transform-probeset` (built + selfchecked 08-08,
batch-reviewed); `AXIS1_WR` sbatch plumbing (added 08-09, additive,
empty-default byte-identical, task-mode-only guard).

## Design

**Grid (all on the lo side, f = 0.054 — `axis1_finger/q1/side0`, never
lost, on RCC): FULL 3×3, arm rgo, 8 seeds/cell, IN-WAVE baseline.**

- s ∈ {1, 2.45, 4.46} (indices 1,2,3) × w_r ∈ {1, 10, 100} (indices
  1,2,3). The (1,1) cell is REFIT fresh (in-wave baseline: every
  primary is within-wave, same protocol, immune to archived-anchor and
  destroyed-fit complications; the archived rgo q1s0 values — n=8 AUC
  97.0 sd 35.0, pooled n=16 99.5 sd 47.4 — are DESCRIPTIVE anchors
  only).
- **vgo repair leg (constraint 5): two cells** — vgo at (s=1, w=1) and
  (s=4.46, w=1), 8 seeds each (in-wave vgo baseline for the same
  reason; archived vgo n=8 AUC 82.4 sd 25.3 descriptive). The s-axis is
  the value-path lever (`valnorm.impl: none` ⇒ scale-live); w_r does
  not touch the value path, so no vgo w_r cells.
- Totals: 9 rgo + 2 vgo = 11 cells × 8 = **88 fits + 88 frozen-readout
  adapts** (500K updates, `dmc_proprio`, standard Axis-1 protocol) +
  E4 h0 measure passes.

**Run naming** (glob-checked CLEAN vs `runroot_cleanup.sh`, RUN_RE
parses): WM `ax1wm_finger_sw{r|v}<si><wi>q1s0_seed<k>`, adapt
`adapt_ax1sw{r|v}<si><wi>q1s0_finger_seed<k>_ckpt500000` (r = rgo,
v = vgo; si/wi = the 1-based dose indices above). KEEP_PENDING entries
added in the same freeze-commit.

**Buffers** (registered commands; materialize on cluster BEFORE fits;
the transform refuses to overwrite):
```
python -m probing.relabel_replay transform \
  --input $RUNROOT/axis1_finger/q1 --task dmc_finger_turn_hard \
  --kind scale --scale 2.45 --seed 0 \
  --output $RUNROOT/axis1_finger/q1_s245
python -m probing.relabel_replay transform ... --scale 4.46 \
  --output $RUNROOT/axis1_finger/q1_s446
```
Fits point REPLAY at `<pair>/side0`. s=1 cells use the ORIGINAL
`axis1_finger/q1/side0` (no transform — byte-identity with every prior
fit of this buffer). w_r enters ONLY via `AXIS1_WR` ∈ {10, 100} on the
same buffers (constraint 3: no relabeling on the w_r axis).

**Own-label probesets** (constraint 2; registered commands):
```
python -m probing.relabel_replay transform-probeset \
  --probeset $RUNROOT/e4_probesets/finger_v1 --kind scale --scale 2.45 \
  --seed 0 --output $RUNROOT/e4_probesets/finger_v1_scale245.npz
... --scale 4.46 ... finger_v1_scale446.npz
```
**E4 passes + collates — EXACT registered commands (batch-review F5;
default tool naming, no --output override):**
```
# per s=1 run (swr11/swr12/swr13/swv11): true labels
python -m probing.stratified_error measure \
  --probeset $RUNROOT/e4_probesets/finger_v1 --run_logdir $RUNROOT/<run>
#   -> <run>/e4_finger_v1/summary.json
# per s=2.45 run (swr21/swr22/swr23):
... measure --probeset .../finger_v1 \
  --reward_override $RUNROOT/e4_probesets/finger_v1_scale245.npz ...
#   -> <run>/e4_finger_v1_ov-finger_v1_scale245/summary.json
# per s=4.46 run (swr31/swr32/swr33/swv31):
... --reward_override .../finger_v1_scale446.npz
#   -> <run>/e4_finger_v1_ov-finger_v1_scale446/summary.json

# collates (three NEW csvs, never merged with archived summaries):
python -m probing.stratified_error collate --runroot $RUNROOT \
  --glob 'ax1wm_finger_sw?1?q1s0_seed*' --probeset_id finger_v1 \
  --output swave_e4_true.csv
... --glob 'ax1wm_finger_swr2?q1s0_seed*' \
  --probeset_id finger_v1_ov-finger_v1_scale245 --output swave_e4_s245.csv
... --glob 'ax1wm_finger_sw?3?q1s0_seed*' \
  --probeset_id finger_v1_ov-finger_v1_scale446 --output swave_e4_s446.csv
```
**Label-provenance machine checks (batch-review F4):** the collate now
carries additive `override_stem/kind/scale` columns from each
summary.json, and the reader REFUSES any row whose provenance does not
match its csv's registered scope (true ⇒ no override; s245 ⇒
kind=scale scale=2.45; s446 ⇒ 4.46). Additionally a REGISTERED
config-audit sweep runs before the read (fail-closed, log persisted):
```
python - <<'EOF'
import re, sys, yaml, os
bad = []
W = {'1': 1.0, '2': 10.0, '3': 100.0}
for d in sorted(os.listdir('.')):
  m = re.match(r'^ax1wm_finger_sw[rv]([123])([123])q1s0_seed[1-8]$', d)
  if not m:
    continue
  cfg = yaml.safe_load(open(os.path.join(d, 'config.yaml')))
  wr = float(cfg['agent']['loss_scales']['rew'])
  if wr != W[m.group(2)]:
    bad.append((d, wr))
print('checked; bad:', bad)
sys.exit(1 if bad else 0)
EOF
```

## Frozen decision rules

Statistics: per-cell seed means; UNPAIRED two-sample BCa 95% (B=10K,
rng 0) + two-sided full-rerandomization permutation p (batch-review F3:
the earlier "paired-by-seed-index"/"sign-flip" wording is struck —
seeds are independent draws across cells, nothing is paired); reader
verifies fit counters (OFFLINE_FIT_PROGRESS update == total; 08-07
standing rule), n_ep_100k vs modal, domain-filters every csv (B1
lesson), and pins dials. **Seeds pinned: 1–8 per cell (M6).**

**TWOHOT-FLOOR CORRECTION (batch-review F1 — BLOCKING fix, registered):**
the symexp_twohot achievable NLL per rewarded frame is a deterministic
function of where symlog(s) falls between bins (spacing 20/127):
floor(s=1) = **0.6735**, floor(s=2.45) = **0.3981**, floor(s=4.46) =
**0.5286** nats (entropy of the two-hot weights; zero labels sit
exactly on a bin, floor 0). Cross-scale own-label contrasts therefore
carry a seed-variance-free offset in the fire direction. EVERY NLL
quantity in this read is floor-CORRECTED per run:
`rew_nll_in − floor(s_cell) × n_in_rew/(n_in_rew + n_in_norew)`
(implemented in the reader; selfcheck carries a floor-kill leg proving
a floor-only difference no-fires corrected while the naive form fires).

- **P-SW1 (s-axis membership, own-label):** floor-corrected in-regime
  rew-NLL (h0, own-label) contrast [s=4.46,w=1] − [s=1,w=1], unpaired
  two-sample. FIRES (deepens) iff BCa CI entirely < 0 ∧ perm p < .05.
  Secondary shape read (descriptive, no fire): the three s-cells'
  FLOOR-CORRECTED NLL declines against ln(1+s)² vs s² regressors —
  which functional form tracks (with 3 points this is stated as a
  comparison, never a test; uncorrected, the floors are non-monotone
  in s and would corrupt the comparison — F1).
  **Constraint-6 bullet-1 reconciliation (batch-review F7, registered):
  the committed outcome "instrument form confirmed" is reachable in
  this wave only QUALITATIVELY (the descriptive shape comparison +
  P-SW1/P-SW2 both-fire pattern); no registered test can confirm a
  functional form from 3 points, and this cap is a disclosure, not a
  weakening — the constraint's adjudication-uses-instrument-form rule
  is honored by running every NLL rule on the corrected scale.**
- **P-SW2 (w_r-axis membership):** same contrast [s=1,w=100] −
  [s=1,w=1] on TRUE-label rew-NLL. Same fire rule.
- **P-SW3 (behavioral response):** AUC100k contrast of EVERY non-(1,1)
  rgo cell vs the in-wave (1,1) cell (8 contrasts, BH-corrected across
  the 8 — the 08-08 multiplicity lesson applied prospectively). Any
  BH-surviving CI>0 (or CI<0) cell = behavioral response registered.
- **P-SW4 (dissociation map, constraint 6 outcome map; registered
  branch PRECEDENCE: the specific s-inert-while-w_r-moves branch
  outranks the generic membership branch — it is the sharper mechanism
  claim):**
  - membership moves (P-SW1 or P-SW2 fires) ∧ no behavioral response ⇒
    another inclusion≠usefulness point, on a new axis (feeds the
    Paper-1 dual-lead mechanism section).
  - behavioral response ∧ no membership movement ⇒ registered anomaly
    (fit-pressure/energy affects behavior without the membership form)
    — disclosed, no headline.
  - s inert everywhere while w_r moves ⇒ label ENERGY is not the
    mechanism; the count of rewarded frames is (first positive account
    of the volume anomaly; feeds Paper-3 D3).
  - both axes inert everywhere ⇒ the corrected a²-term is refuted in
    the tested range (registered honest theory negative).
  - membership moves ∧ behavioral response (batch-review F2 — the
    previously unmapped branch, registered verbatim as implemented):
    verdict `BOTH-MOVE: dose-responsive axis, full map in contrasts` —
    the per-cell contrast table is then the result; no single-sentence
    headline is licensed beyond "the axis is dose-responsive on both
    measures".
- **P-SW5 (vgo repair leg):** vgo FLOOR-CORRECTED own-label in-regime
  rew-NLL [s=4.46] − [s=1], fire iff CI < 0 ∧ p < .05 ⇒ P-B2 converts
  from "inert" to "below threshold" (theory repair); null ⇒ P-B2's
  attenuation form stands. **Scope (M5): P-SW5 adjudicates P-B2 ONLY —
  it participates in no P-SW4 verdict branch; a SW5 fire alongside
  BOTH-AXES-INERT is a vgo-specific theory repair, not a contradiction
  of the rgo-axis negative.**

Power (honest; M2-corrected): per-cell n=8, within-wave two-sample
with sd ≈ 35–47.4 (archived rgo anchors) ⇒ behavioral MDE(80%) =
2.80·sd·√(2/8) ≈ **49–66 AUC** — adequate for rgo-lift-scale effects
(+65 archived) only at the low end of the sd range, underpowered below
~50 and disclosed as such; NLL contrasts historically resolve at n=8
(arm separations ≳1 nat vs seed sd ≲0.4).

Known at freeze: every archived rgo/sgb/vgo value (P3 + Amendment-1
reads), the 08-08/08-09 review-resolution record, the theory-correction
constants. Unknown: every s≠1 or w≠1 quantity anywhere (no scaled
buffer or w_r fit has ever existed — s ≡ 1, w_r ≡ 1 in every prior
run).

Reader: `analysis/swave_read.py` — BUILT + selfchecked 2026-08-09
(guards: finger-only domain filter, milestone pin, duplicate fatal,
qc/sub-modal excluded+disclosed, exactly-8-rows-per-cell pre-QC,
cell refusal below n=6, E4 own/true-label SCOPE enforcement per csv,
fit-counter refusal per the 08-07 standing rule; selfcheck legs: null
no-fire, planted SW1/SW2/SW3-BH-exact/SW5 fires, all verdict branches,
cup-contamination inert, duplicate/scope/counter/row-count trips).
Implements the rules above verbatim; ONE execution. Reviewer: ONE
batch reviewer over {this file, the sbatch plumbing diff, the reader}
(user-approved 2026-08-09).

**Fit-counter producer (registered; the reader's `--fit_counters`
input):** run at collate time on the runroot —
```
python - <<'EOF'
import json, os, re
out = {}
for d in sorted(os.listdir('.')):
  if not re.match(r'^ax1wm_finger_sw[rv][123][123]q1s0_seed\d+$', d):
    continue
  kv = dict(l.split('=', 1) for l in
            open(os.path.join(d, 'OFFLINE_FIT_PROGRESS'))
            .read().strip().split('\n'))
  out[d] = dict(update=int(kv['update']),
                total=int(kv['total_updates']))
json.dump(out, open('swave_fit_counters.json', 'w'), indent=1)
print(len(out), '-> swave_fit_counters.json')
EOF
```
(A missing OFFLINE_FIT_PROGRESS file crashes the producer — fail-closed
by construction.)

Ordering: [YOU] freeze-commit this file (+ cleanup KEEP_PENDING +
KEEP-precedence patch + sbatch diff + reader + collate columns +
override-overwrite refusal) → cluster: materialize the two scaled
buffers + two probeset overrides (record shas) → smoke (1 fit-cell,
swr21 seed 99, + per-run config audit: loss_scales.rew echoed) →
**REMOVE/EXCLUDE the seed-99 smoke run before any collate (M4: a
seed-99 adapt row trips the exactly-8 guard — fail-closed but stalls;
the counter producer's seed\\d+ regex would also pick it up)** → 88
fits + 88 adapts → E4 passes per the registered commands → three
collates + fit-counter producer + config-audit sweep → ONE read via
`analysis/swave_read.py`.
