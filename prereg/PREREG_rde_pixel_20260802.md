# PREREG — recon-detached pixel arm (rde; frozen 2026-08-02)

The second and complementary lever test of the pixel swamping account,
registered after the pe read (NO-RELIEF, `artifacts/pe_pixel_read_20260802/`)
and GO'd by the user conditional on story value ("proceed if the new
registration can make the story clearer" — both live branches below end
the pixel arc decisively). This file + the frozen reader
`analysis/rde_pixel_read.py` (selfcheck PASS) + the code additions listed
below are committed BEFORE any rde run exists.

## Question and relationship to the pe arm

The swamping account says reconstruction's fit-time gradient bid for
trunk capacity is what excludes reward structure at pixel (P-SW1: task-arm
in-regime rew-NLL 23.34 [19.41, 27.29] ≫ the 2.0-nat bar). Its lever has
two pull directions:

- **pe (done, NO-RELIEF):** give the trunk competition-free FEATURES —
  frozen encoder pretrained by reconstruction+dynamics. Reward-NLL did
  not move (23.03 ≈ 23.34). Reading ambiguity disclosed in the pe
  record: recon-selected features may simply not be reward-legible (the
  paper's own headline expects that), so the pe null under-tests the
  competition claim.
- **rde (this wave):** keep the trunk TRAINABLE and remove the
  competition itself — `agent.recon_grad False` detaches the decoder's
  gradient from enc+dyn (decoder still trains, on stop-gradient latents;
  everything else identical to the committed X2 task cells). Reward,
  value, continuation, and dynamics losses shape the trunk unopposed.
  This is the competition claim tested directly, and it ties the pixel
  arc to the paper's central manipulation (gradient composition; the rgo
  arm's sufficiency at proprio).

Design-rationale disclosure: a task-trained-donor variant (freeze the X2
task encoder, refit heads) was considered and REJECTED as near-tautological
— P-SW1 already measured that trunk's reward illegibility; refitting heads
on it re-measures a known fact.

## Registered predictions and decision rules

- **P-RD1 (representation, primary):** the rde fits' in-regime rew-NLL
  (h=0, sides pooled, seed-clustered bootstrap B=10K rng 0 — the P-SW1
  estimator). Graded, frozen; the fire branch has THREE conjuncts:
  1. CI entirely < 2.0 (the swamping bar), AND
  2. **CI entirely < the trivial constant-predictor floor
     0.671345852414428** — the floor is DECISIONAL here (unlike pe): NLL
     in [floor, 2.0) means the head is at-or-worse-than the best
     constant prediction on in-regime frames, which is the marginal-head
     masquerade, not inclusion. Sub-bar-but-not-sub-floor ⇒ registered
     outcome **NOT-BETTER-THAN-CONSTANT** (no inclusion claim; the
     witness panel discriminates collapsed-features vs
     alive-features-with-marginal-head descriptively), AND
  3. latent-alive witness passes on ALL 16 fits — a numeric pass of
     (1)+(2) with any witness failure ⇒ **DEGENERATE-MASQUERADE**.
  All three ⇒ **INCLUSION-RESTORED**. Otherwise the P-PE1-comparable
  ladder: CI entirely below the committed baseline's lower bound
  19.409267325402837 ⇒ **PARTIAL-RELIEF**; else **NO-RELIEF**. (Pins are
  the full-precision committed values — baseline 23.338872993169502
  [19.409267325402837, 27.285688932142705]; membership-band descriptor
  CI ≤ 1.5 reported, never decisional. The extra fire conjuncts only
  make the strongest verdict harder; PARTIAL/NO-RELIEF are unchanged
  from the P-PE1 estimator, preserving pe comparability there. The
  floor conjunct is conservative and uncalibrated from above — no
  healthy pixel arm exists to say how far below the floor a genuinely
  included head lands; a genuine-but-weak inclusion straddling the
  floor reads NOT-BETTER-THAN-CONSTANT, a registered miss direction.)
- **P-RD2 (behavior):** frozen-readout adaptation lifts off the committed
  X2 floor: per-seed paired [rde − X2-task] level contrast, mean over the
  two side cells, vs `artifacts/pixel_x2_20260724/auc.csv` (the U4 relief
  estimator, identical to P-PE2); CI entirely > 0 ⇒ LIFTS.
- **P-RD3 (conditional secondary, adjudicated only if P-RD2 fires):**
  per-seed [s1 − s0] within the rde arm, CI > 0 fires. Else premise-idle.

Registered verdict map (in the frozen reader, verbatim):
INCLUSION-RESTORED + LIFTS ⇒ **COMPETITION-CONFIRMED** (the pixel
boundary is a causal consequence of objective composition — strongest
form of the legibility headline); INCLUSION-RESTORED without lift ⇒
**INCLUDED-BUT-USELESS AT PIXEL** (stamping-pattern analogue);
NOT-BETTER-THAN-CONSTANT / DEGENERATE-MASQUERADE ⇒ no inclusion claim,
wave uninformative on P-RD1 (disclosed); PARTIAL-RELIEF ⇒ directional
support only, no headline change; NO-RELIEF ⇒ **with pe, both pull
directions of the lever have failed: the causal competition account is
RETIRED and the pixel boundary stands as a modality-level scope fact**
(a clean, evidence-backed ending — the paper text retreats from
"swamping explains the boundary" to "P-SW1 measured fact + both causal
levers refuted"). P-SW1 itself is NOT re-adjudicated under any branch.

Registered disclosure clauses (appended to the verdict text wherever
they apply; none changes a decision, and no cell is silently absorbed
into a narrative it does not support):
- **COLLAPSE-QUALIFIED** — any witness-failing fit under PARTIAL-RELIEF
  or NO-RELIEF (signal-poverty vs optimization-collapse not separable
  for those cells).
- **P-RD2 NEGATIVE** — lift CI entirely < 0 (the rde arm significantly
  BELOW the X2 floor): outside registered accounts, reported as-is.
  This outcome is anticipated by the folklore prior disclosed under
  Power and has no registered mechanistic reading.
- **NO-RELIEF (representation) WITH BEHAVIORAL LIFT** — P-RD2 fires
  while P-RD1 stays in the swamping regime: transfer without measurable
  inclusion is outside registered accounts; the retirement conclusion
  applies to the representation claim only.

### Latent-alive witness (anti-masquerade guard — the new instrument piece)

Known trap, registered up front: with a trainable trunk and no
reconstruction anchor, enc+dyn can collapse toward constant features. A
reward head on constant features learns the stratum-marginal prediction,
whose in-regime NLL can sit BELOW the 2.0 bar — a false
INCLUSION-RESTORED. (The pe arm was immune: its frozen recon-trained
donors guaranteed non-constant features. The committed swamping/X2 arms
were immune for the same reason.)

Witness: `deter_std` — across-state std (per-dim, dim-averaged) of the
h=0 posterior deter features over the frozen probeset, computed inside
`probing/stratified_error.py` measure (additive; per-batch centered
moments combined by the unit-tested `_combine_deter_moments` — Chan
combination in float64, cancellation-free at any mean/std ratio) and
collated as a new csv column. Threshold: **0.10 × median deter_std of the
fpxpx calibration pass** (the 16 reward-free pixel fits — recon-trained,
guaranteed-alive reference, seed/side-matched, verified extant by the pe
integrity sweep on 2 Aug). Collapse is orders of magnitude; 0.10 is
generous to genuinely-alive-but-recon-poor latents. Rules, frozen in the
reader: INCLUSION-RESTORED requires ALL 16 rde fits ≥ threshold; numeric
fire + any failure ⇒ DEGENERATE-MASQUERADE. Under PARTIAL-RELIEF and
NO-RELIEF the witness is a registered DISCLOSURE only (verdict text
gains "COLLAPSE-QUALIFIED" for failing cells — signal-poverty vs
optimization-collapse not separable there) and never changes the
decision. Calibration median ≤ 1e-4 ⇒ the witness cannot discriminate
and the read REFUSES (instrument refusal, not a verdict).

Registered witness limitations (disclosed at freeze): (a) the witness
covers the deter path only — stoch is a sampled categorical whose
variance stays nonzero under many collapse modes, making it a weaker
signal; a deter-dead/stoch-alive trunk reads DEGENERATE-MASQUERADE,
which is the conservative direction for the fire branch. (b) The 0.10×
margin is calibrated against RECON-TRAINED reference trunks; no
recon-free reference exists pre-wave, so a genuinely-alive
recon-detached trunk operating at >10× smaller deter scale would read
DEGENERATE-MASQUERADE. Both limitations can only DOWNGRADE a fire —
never create one — and the full per-fit deter_std + calibration values
are disclosed in the read json under every branch.

## Design: 16 fits + 16 adapts + calibration pass + E4 pass

- **Fits** `ax1wm_finger_rdepxq1ms<side>_seed<f>`, side ∈ {0,1} × seeds
  1–8: `pixel_wm`, task-mode, `AXIS1_ARM=rde` (⇒ `--agent.recon_grad
  False`; everything else identical to the committed X2 task cells:
  same buffers `axis1_finger/pxq1m/side<side>`, UPDATES 500000, default
  reward_grad/repval_grad True, frozen_enc False, NO donors — this wave
  is donor-free; the fpxpx donors are needed only as the calibration
  reference).
- **Adapts:** 16 standard frozen-readout adapts
  `adapt_ax1rdepxq1ms<side>_finger_seed<f>_ckpt500000` (modes
  `ax1rdepxq1ms0/1` — collision-free: no frozen reader or cleanup glob
  matches them; verified against runroot_cleanup.sh and all committed
  reader mode pins).
- **Calibration pass (BEFORE the read; independent of rde outcomes):**
  re-measure the 16 fpxpx fits with the amended `stratified_error`,
  writing to per-run `e4_fingerpx_v1cal/` via explicit `--output` so the
  swamping wave's existing `e4_fingerpx_v1/` summaries are NEVER
  touched; collate to `$RUNROOT/e4_fingerpx_v1cal_fpxpx.csv`. The
  calibration values are properties of the committed 07-01 fits — no
  outcome of this wave leaks into the threshold.
- **E4 pass:** frozen machinery, `PROBESET fingerpx_v1`, `GLOB
  'ax1wm_finger_rdepxq1m*'`, default horizons; **COLLATE overridden to
  `$RUNROOT/e4_fingerpx_v1_rde.csv`** (committed swamping csv path never
  rewritten; rde run dirs are fresh so per-run summaries collide with
  nothing).

### Code registered with this wave (committed at this freeze)

1. `dreamerv3/configs.yaml`: `agent.recon_grad: True` default.
2. `dreamerv3/agent.py`: decoder input becomes
   `sg(repfeat, skip=self.config.recon_grad)` — the exact `reward_grad`
   sg idiom, one line + comment; decoder still trains (a live probe of
   latent informativeness). The only other `self.dec` call sites are
   inference/report paths (`training=False`, no loss).
3. `scripts/axis1.sbatch`: `AXIS1_ARM=rde` case ⇒ `--agent.recon_grad
   False` (flags derived from the registered arm token, as for
   sgb/rgo/vgo; the saved config.yaml records recon_grad for the audit).
4. `probing/stratified_error.py`: additive `deter_std` (measure-side
   per-batch CENTERED moments of h=0 posterior deter, finalized by the
   module-level `_combine_deter_moments` — unit-tested in the committed
   selfcheck with uneven batches, a 1e3 offset, and a constant input;
   summary field + collate column written only when present, so old
   bundles collate unchanged). Cross-wave disclosure: the pending f_R
   wave's E4 pass will also run this amended file; the change touches
   no existing computation or column (verified by inspection + the
   harness below), and every pending reader consumes csv columns by
   name (verified: highfr_read, pixel_swamping_read, pe/rde readers).
5. `scripts/runroot_cleanup.sh`: (a) fpxpx KEEP-PENDING hold
   transferred from the pe wave (read verified 2 Aug) to this wave's
   calibration pass; (b) **the DELETE-SAFE robustness-battery globs
   `adapt_ax1rd*` / `ax1wm_finger_rd*` are digit-anchored to
   `…rd[0-9]*`** — the bare forms matched every rde run and would have
   deleted the whole wave on any cleanup (reviewer BLOCKING B1;
   narrowed globs verified to still match the rediag runs
   `ax1rd<k>q1s<side>` and nothing of this wave).
6. `analysis/rde_pixel_read.py`: frozen reader (selfcheck PASS —
   confirmed / masquerade / borderline / not-better-than-constant /
   skewed-calibration / included-useless / span0-lift / negative-lift /
   partial(+straddle+collapse-qualified) /
   no-relief(+band+lift+collapse-qualified) branches + RD3 legs + 10
   guard trips; six decision-rule mutants — inverted fire bound,
   partial-bound-vs-point, lift-point-vs-CI, calibration mean-vs-median,
   floor conjunct disabled, alive rule inverted — verified CAUGHT on
   mirror copies, clean copy PASS).

### Local verification at freeze (CPU, tiny pixel config; archived in
`artifacts/rde_pixel_reg_20260802/` with stdout — harness rev 2, all
review hardenings applied)

Real `Agent.loss` traced through `nj.pure` under **training=True** with
synthetic pixel spaces; ALL EIGHT loss components' trunk grad norms
compared across both flag settings:
- recon_grad True: image-loss grads enc 114.14 / dyn 7528.4 / dec 3000.68.
- recon_grad False: image→enc **0.0 exact**, image→dyn **0.0 exact**,
  image→dec 3000.68 (unchanged — decoder trains).
- Every non-image component (con, dyn, rep, rew, repval, policy, value)
  **bit-identical** across settings; rew (enc 14.19 / dyn 3220.85) and
  repval (enc 8.44 / dyn 2291.07) trunk paths verified OPEN.
- deter_std: the SHIPPED `_combine_deter_moments` ≡ direct np.std to
  1e-9 relative under uneven batches + a 1e3 offset; constant input → 0.
- Harness note: rew/value/policy outscales overridden 0→1 locally
  because zero-initialized output kernels block feature gradients AT
  INIT (W^T·δ = 0), which would make the comparison vacuous for those
  paths; real runs have nonzero kernels after the first update.

### Smoke gate (machine-checked; BEFORE the 16 fits)

One short rde job named `ax1wm_finger_rdesmoke_seed99` (never matched by
the E4 glob `ax1wm_finger_rdepxq1m*`; seed-99 cleanup globs cover it),
submitted THROUGH `axis1.sbatch` so the smoke exercises the exact
production path — the `AXIS1_ARM=rde` case derivation and `--export`
propagation included (a dropped arm token would otherwise leave all 16
fits at recon_grad True and pass a direct-offline_fit smoke):

```
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
  --gres=$SLURM_GRES --time=4:00:00 \
  --job-name=adapt_ax1rdesmoke_finger_seed99_ckpt2000 \
  --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1rdesmoke_finger_seed99_ckpt2000,WM_RUN=ax1wm_finger_rdesmoke_seed99,TASK=dmc_finger_turn_hard,SEED=99,REPLAY=$RUNROOT/axis1_finger/pxq1m/side0,UPDATES=2000,STEPS=1000,AXIS1_EXPL_MODE=task,AXIS1_BASE_CONFIG=pixel_wm,AXIS1_ARM=rde \
  $REPO/scripts/axis1.sbatch
# after the job: machine-checked gate (GPU node for the measure step)
SMOKE=$RUNROOT/ax1wm_finger_rdesmoke_seed99
python - "$SMOKE/config.yaml" <<'EOF'
import sys, ruamel.yaml as y
a = y.YAML(typ='safe').load(open(sys.argv[1]))['agent']
assert a['recon_grad'] is False, a.get('recon_grad')
assert a['reward_grad'] is True and a['repval_grad'] is True
print('SMOKE config: recon_grad False, head grads True')
EOF
python -m probing.stratified_error measure \
  --probeset $RUNROOT/e4_probesets/fingerpx_v1 --run_logdir $SMOKE \
  --horizons 1 --output $SMOKE/e4_smokecal
python - "$SMOKE/e4_smokecal/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
assert s.get('deter_std') is not None and s['deter_std'] > 0, s.get('deter_std')
print('SMOKE deter_std =', s['deter_std'])
EOF
```

GATE: job completes rc=0 AND the config assert passes (recon_grad False
via the ARM path — machine-checked, not eyeballed) AND the smoke E4
summary carries a positive `deter_std`. The 16 full fits are submitted
only after the gate passes. (No PARTIAL_INIT line — this wave loads no
checkpoint.)

### Submit (clean shell; after the smoke gate)

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=36:00:00 \
    --job-name=adapt_ax1rdepxq1ms${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1rdepxq1ms${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_rdepxq1ms${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/pxq1m/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_BASE_CONFIG=pixel_wm,AXIS1_ARM=rde \
    $REPO/scripts/axis1.sbatch
done; done
# Calibration pass (GPU node; any time after the freeze commit, before
# the read; per-run --output keeps the swamping summaries untouched;
# --horizons 1 because deter_std is h=0-only — no need to duplicate the
# swamping wave's full-horizon errors on scratch):
{ for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
    D=$RUNROOT/ax1wm_finger_fpxpxq1ms${side}_seed${f}
    [ -f "$D/e4_fingerpx_v1cal/summary.json" ] && { echo "SKIP $D"; continue; }
    python -m probing.stratified_error measure \
      --probeset $RUNROOT/e4_probesets/fingerpx_v1 --run_logdir "$D" \
      --horizons 1 --output "$D/e4_fingerpx_v1cal"
  done; done; } 2>&1 | tee $RUNROOT/rde_calib_sweep.log
python -m probing.stratified_error collate \
  --runroot $RUNROOT --glob 'ax1wm_finger_fpxpxq1m*' \
  --probeset_id fingerpx_v1cal --output $RUNROOT/e4_fingerpx_v1cal_fpxpx.csv
# Config-audit integrity sweep (after all 16 fits; persisted log ships
# with the bundle as the registered integrity evidence). FAIL-CLOSED:
# a missing file, missing key, or parse error prints QUARANTINE — it
# never silently passes — and the PASS count is asserted, not eyeballed.
{ for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
    C=$RUNROOT/ax1wm_finger_rdepxq1ms${side}_seed${f}/config.yaml
    python - "$C" <<'EOF'
import sys
try:
  import ruamel.yaml as y
  c = y.YAML(typ='safe').load(open(sys.argv[1]))
  a = c.get('agent') or {}
  ok = (a.get('recon_grad') is False and a.get('reward_grad') is True
        and a.get('repval_grad') is True and a.get('frozen_enc') is False
        and a.get('model_obs') == 'image'
        and (a.get('expl') or {}).get('mode') == 'task'
        and c.get('task') == 'dmc_finger_turn_hard')
except Exception as e:
  print(f'QUARANTINE {sys.argv[1]} ({type(e).__name__}: {e})')
else:
  print(('PASS ' if ok else 'QUARANTINE ') + sys.argv[1])
EOF
  done; done; } 2>&1 | tee $RUNROOT/rde_config_audit.log
grep -q QUARANTINE $RUNROOT/rde_config_audit.log \
  && { echo "CONFIG AUDIT FAILED — do not run the read"; exit 1; }
[ "$(grep -c '^PASS' $RUNROOT/rde_config_audit.log)" -eq 16 ] \
  || { echo "CONFIG AUDIT INCOMPLETE — do not run the read"; exit 1; }
# E4 pass (separate collate file — never overwrite the swamping csv).
# BEFORE proceeding, verify the job log echoes the overridden COLLATE
# path: if the env var failed to propagate, e4_measure.sbatch defaults
# to $RUNROOT/e4_fingerpx_v1.csv — the swamping csv path (recoverable
# from the committed artifact, but check the echo, not the outcome).
PROBESET=$RUNROOT/e4_probesets/fingerpx_v1 GLOB='ax1wm_finger_rdepxq1m*' \
  COLLATE=$RUNROOT/e4_fingerpx_v1_rde.csv sbatch $REPO/scripts/e4_measure.sbatch
```

## Registered read (ONE execution)

```
python -m analysis.rde_pixel_read --auc <canonical auc.csv> \
  --e4 <e4_fingerpx_v1_rde.csv> --calib <e4_fingerpx_v1cal_fpxpx.csv> \
  --output <dir>
```

Reader guards (selfcheck-verified): complete 2×8 grids on the adapt, E4,
and calibration sides; unregistered-seed, duplicate, reward_aware (=1 rde,
=0 calibration), finite-NLL, finite-nonnegative deter_std, missing-column,
qc_pass, calibration-refusal, and complete committed X2 baseline. Pinned
anchors in the reader: swamping baseline 23.34 [19.41, 27.29], bars
2.0/1.5, floor 0.6713, ALIVE_FRAC 0.10, CALIB_MIN 1e-4.

## Power and cost

n=8 per side (16 total); estimators identical to the pe wave, whose
realized precision transports: P-RD1 CI half-width ~2.8 nats vs a ~21-nat
gap to the bar; P-RD2 half-width ~18 AUC against cell means 77–83. Known
external prior, disclosed: decoder-free/recon-free Dreamer variants are
folklore-weaker at pixel control; our estimand is representation-level
legibility on a static buffer, not benchmark performance, and the
degenerate outcome has its own registered branch rather than
contaminating the null. Compute: 1 smoke (~1–2 h) + 16 fit+adapt jobs at
the 36 h pixel envelope + 16 short calibration measures + 1 E4 pass.

## Review provenance

ONE adversarial reviewer (Opus 5, user-approved), ran 2 Aug on the full
registered set: **2 BLOCKING / 6 MAJOR / 6 MINOR / 5 NIT — ALL
BLOCKINGs and MAJORs fixed and re-verified before this freeze**: B1
cleanup-glob deletion hazard (digit-anchored; narrowed globs
machine-verified); B2 trivial-floor false-fire region (floor made
decisional, NOT-BETTER-THAN-CONSTANT outcome + fixture — the
pre-review Case-1 fixture was itself an instance of the false fire); M1
COLLAPSE-QUALIFIED extended to PARTIAL-RELIEF; M2 negative-lift
disclosure branch; M3 NO-RELIEF-with-lift branch; M4 four surviving
decision-rule mutants killed by new fixtures (straddle-2.0, in-band
CI, span-0 lift, skewed calibration) — all six mutants now CAUGHT on
mirror copies; M5 config audit made fail-closed with an asserted PASS
count; M6 smoke rerouted through axis1.sbatch (exercises the ARM case +
--export propagation) with machine-checked config asserts. MINORs
applied: centered-moment Chan combiner (unit-tested in the committed
stratified_error selfcheck + the archived harness now imports the
SHIPPED function), harness rev 2 (training=True, all-8-component
comparison, value/policy outscales un-zeroed), deter-only witness scope
+ cross-arm-margin disclosures, audit gains expl.mode/model_obs.
NITs: full-precision pins, member_band ≤, calibration --horizons 1,
COLLATE-echo check. Registered reads run without `python -O` (reader
guards are asserts). Un-adopted remainder: stoch-moment witness
(deter-only scope disclosed instead — conservative direction).

## Disclosure and ordering

Known at freeze: every read through 2 Aug (incl. pe NO-RELIEF and the
committed swamping/X2 anchors pinned above). Not known: any rde fit,
adapt, calibration value, E4 row, or smoke — none exists. The f_R wave
is pending and was not consulted. Ordering: this file + code additions +
reader committed (single freeze commit) → smoke gate → calibration pass
(any time post-commit) → 16 fits+adapts → config audit → E4 pass → ONE
read. Value-blindness: post-read exploratory analyses allowed, labeled,
verdict-untouchable, per standing policy.
