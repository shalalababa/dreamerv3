# AMENDMENT 1 to PREREG_dose_task_20260810 (#21) — power extension, seeds 9–16 — 2026-08-11

User GO 2026-08-11 after the ONE registered read returned
**GRID-UNINFORMATIVE as a power event** (`artifacts/dose_task_read_20260811/`:
G1 +157.6 [+25.8, +309.5], perm p = .0663 fails the p-conjunct at
n=8/8; interior level2 missed BH by .0008; step non-decisive;
instrument fully clean — occupancy gates, dual witness, zero
exclusions). Precedent: P3 Amendment 1 (n=8→16 pooled re-test).

## Value-awareness (disclosed)

Wave-1 outcomes are fully known at this freeze (G1 +157.6 p=.0663;
cell means 146.5/282.5/273.1/304.1; shape TIE with step ahead
non-decisively). The pooled test below re-uses those 32 cells — that
is WHY the corrected threshold is registered. Unknown: every seeds-9–16
quantity.

## Design — 32 new fits + 32 new adapts (total pool 64 + 64)

- Same four level buffers (`$RUNROOT/axis1_finger/dose/level{0..3}`,
  same `dose.json` manifest, sha re-verified at read), same code path
  as the executed wave (no smoke — the identical pipeline ran 64/64
  clean 08-11; disclosed). All #21 guards apply unchanged (task-mode
  env-leakage refusal, td naming, KEEP globs already match `seed*`).
- Registered submission (the ONLY change vs the frozen command is
  AXIS1_SEEDS):
  `AXIS1_DOSE_TASK=1 AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger AXIS1_SEEDS="9 10 11 12 13 14 15 16" AXIS1_LEVELS="0 1 2 3" ./scripts/submit_all.sh axis1-dose-bundles`
  → 32 fits (`ax1wm_finger_td{l}_seed{9..16}`) → 32 adapts.
- E4 + collates + witnesses: the registered #21 producer commands
  re-run over ALL 64 fits/adapts (adaptation_auc collate, E4
  measure/collate, fit-counter awk json, ckpt-steps one-liner json) →
  one pooled bundle.

## Registered pre-read gates (batch review findings 3, 4, 22)

The frozen reader's witness loops cover seeds 1–8 only
(`SEEDS_PER_CELL = 8`), so the 32 NEW fits would enter unwitnessed —
therefore a REGISTERED PRE-READ GATE, run before the ONE execution,
output committed into the artifact, the read REFUSED on any failure
(the frozen reader stays untouched):

```
python - <<'PY'
import json
c = json.load(open('inputs/fit_counters.json'))
k = json.load(open('inputs/ckpt_steps.json'))
names = [f'ax1wm_finger_td{l}_seed{s}' for l in range(4)
         for s in range(1, 17)]
for n in names:
    assert n in c and int(c[n]['update']) == int(c[n]['total']), (n, c.get(n))
    assert n in k and int(k[n]) == 500000, (n, k.get(n))
print('64/64 dual witness OK')
PY
```

- **Pooled-integrity gate (finding 4)**: the pooled read is VOID
  unless its `read.json` shows `excluded_submodal == []`, `modal_n_ep
  == 96` (look-1's modal, pinned), and `n_per_cell == {16,16,16,16}`.
  The reader's tie-break (`bincount.argmax` → smallest) could
  otherwise silently reduce the pooled read to look-1 rows. VOID ⇒
  REFUSE + investigate + a dated Amendment 2 BEFORE any re-invoke; no
  silent re-run.
- **E4 row gate (finding 22; literal form per reviewer-2 M6 — the
  seed99 contamination is confirmed real: KEEP_PENDING
  `ax1wm_*_td[0-3]_seed*` outranks the seed99 DELETE glob, so the
  smoke fit persists and re-collates)**: run from the bundle dir,
  pre/post sha256 recorded in the artifact:

```
python - <<'PY'
import csv, hashlib
names = {f'ax1wm_finger_td{l}_seed{s}' for l in range(4)
         for s in range(1, 17)}
rows = list(csv.DictReader(open('inputs/e4_dose.csv')))
keep = [r for r in rows if any(n in r[k] for n in names
                               for k in ('run', 'run_id', 'checkpoint')
                               if k in r)]
dropped = [r for r in rows if r not in keep]
kept_ids = {n for n in names for r in keep
            if any(n in str(v) for v in r.values())}
assert kept_ids == names, ('missing fits', sorted(names - kept_ids))
with open('inputs/e4_dose_filtered.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, rows[0].keys())
    w.writeheader()
    w.writerows(keep)
print('dropped rows:', len(dropped),
      [str(r)[:80] for r in dropped])
print('sha_pre ', hashlib.sha256(open('inputs/e4_dose.csv','rb')
                                 .read()).hexdigest())
print('sha_post', hashlib.sha256(open('inputs/e4_dose_filtered.csv',
                                      'rb').read()).hexdigest())
PY
```

  (exact column name carrying the run identity conformed at run time
  and recorded; the read consumes the FILTERED csv; the drop list is
  committed with the artifact.)
- **Ckpt-steps producer json form (reviewer-2 m9)**: #21's registered
  one-liner emits space-separated text; the amendment registers the
  conversion the reader needs:
  `python -c "import json,sys; print(json.dumps(dict((l.split()[0], int(l.split()[1])) for l in open(sys.argv[1]) if l.strip())))" ckpt_steps.txt > inputs/ckpt_steps.json`

## Registered read — frozen reader VERBATIM, corrected thresholds on top

- Reader: `analysis/dose_task_read.py` UNTOUCHED (bitwise
  0251271c…d9d7; per-level gate ≥6 rows so n=16 loads as-is; verified
  in-review at n=16: cells 16/16/16/16, gates green).
  Never-edit-executed-readers rule holds.
- **ONE adjudicating execution** on the pooled 64-row csv (+ pooled
  e4/witness inputs, post-gates). **α accounting stated honestly
  (finding 13)**: look 1 was adjudicated at α=.05 and FAILED; this
  pooled look is graded at the Pocock K=2 boundary **α=.0294**. The
  realized family-wise error of the two-look procedure is ≤ .0794
  (union bound), ≈ .06 under the actual overlap correlation —
  disclosed; conservative relative to no correction, NOT a clean
  overall-.05 design. **The permutation p at .0294 is the GOVERNING
  leg** (permutation-primary house rule); the reader's 95% BCa CI is
  retained as a direction check (CI>0 required). The reader's internal
  `gate_pass`, `bh_survivors`, `activation`, and `verdict` strings
  are computed at .05 and are recorded but NON-GOVERNING at this look.
- **P-DR1 re-derivation (explicit; finding 13)**: interior contrasts =
  the reader's `p_dr1.contrasts` perm_p values for levels 1 and 2
  (m=2); BH thresholds .0147 (rank 1) and .0294 (rank 2) applied to
  those p's, conjoined with the corresponding reader CIs excluding 0.
  Activation labels re-derived from the surviving set per the original
  #21 map.
- **P-DR2 (finding 14): DEMOTED to descriptive-with-disclosure at
  this look.** Its decisiveness rule is a per-look nominal-5%
  false-decisive calibration now evaluated twice on overlapping data
  (realized ≤10%) with look-1's direction known (step ahead,
  non-decisive); the reader does not persist its null draws, so no
  stricter crit can be recomputed. The pooled shape output is
  reported verbatim as DESCRIPTIVE; **no confirmatory shape wording
  is licensed by this amendment**; a future confirmatory shape claim
  requires a fresh single-look registration.
- **Fresh-only replication check (finding 12 — replaces the draft's
  second reader invocation, which is unexecutable: the frozen witness
  loops refuse a seeds-9–16-only json)**: a registered supplementary
  computation, no reader invocation —

```
cd $REPO && BUNDLE=<abs path to the pooled bundle> python - <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from analysis.dose_task_read import load_auc, two_sample
cells = load_auc(os.environ['BUNDLE'] + '/inputs/auc_pooled.csv')
lv = lambda l: [float(r['auc100k']) for r in cells[str(l)]
                if 9 <= int(r['seed']) <= 16]
print(json.dumps(dict(fresh_g1=two_sample(lv(3), lv(0)),
                      n=[len(lv(3)), len(lv(0))])))
PY
```

  (Reviewer-2 M4 form: runs from the repo root with an absolute
  bundle path, and REUSES the frozen `load_auc` so the fresh subset
  inherits ALL of the reader's row filters — domain, milestone,
  qc_pass, duplicate gates — by construction; the exact `cells`
  row-shape access is conformed at run time and recorded.
  REPLICATION-DESCRIPTIVE — sign/magnitude reported, no branch
  wording, pooled witness covers these fits.)
- **Verdict map**: G1-pooled (perm p < .0294 ∧ CI > 0) passes ⇒ the
  wave regains decision weight — P-DR1 adjudicates under the
  corrected thresholds; P-DR2 stays descriptive per above. G1-pooled
  fails ⇒ **DOSE-GRID-RETIRED**: the grid cannot reproduce the
  occupancy effect at the extended power; no further extension (LAST
  look — registered).
- **Power basis (finding 15; supersedes the reader's stale look-1
  `mde_note` string, which will reappear verbatim in the pooled
  read.json)**: on the registered pooled-sd basis (per-cell sd ≈ 77),
  n=16v16 gives se ≈ 27.3; G1 power at α=.0294 ≈ **0.93 at a true
  +100** and ≈ **0.64 at +69**; MDE(80% power) ≈ **83**. "Registered
  power ×2" in the draft was loose — n doubles, se shrinks by 1/√2.
- **R1 regrade (registered now, pre-outcome)**: R1's A1 receives ONE
  regrade at the pooled read — A1′ = G1-pooled at the corrected
  alpha. The executed n=8 miss stays on the R1 scoreboard as look 1;
  A2 (monotonicity, via P-DR1) is graded iff the wave regains
  decision weight; A3/A4 (shape) remain UNADJUDICATED at this look
  per the P-DR2 demotion.

Reviewer: 2026-08-11 batch reviewer (ONE, Opus 5 — standing
auto-approval); findings 3/4/12/13/14/15/22 applied in this revised
form. ONE adjudicating execution.
