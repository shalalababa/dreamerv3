# PREREG: B1 capacity-descent λ trial — 2026-08-11

User GO 2026-08-11. The first of the two registered λ-discriminating
trials of `PREREG_theory_R1_gating_20260811` **Part B** (predictions
FROZEN there before this design): if feature competition exists, it
must bind as capacity shrinks below the standard size. Mediator
(capacity) is set BY CONSTRUCTION — the f_R mediator-failure mode is
structurally excluded. Respects the standing 25m upper cap (descent
only).

## Honesty block

Known at freeze: all executed reads, incl. the size1m interaction
(+84.0 E3v2, 14/14) and the 12m/25m capacity-robustness values
(+113.8* at 25m) — quoted below only as VALUE-AWARE descriptive
anchors, never entering a rule; the frozen Part-B side-by-side
predictions (λ ⇒ interaction GROWS as capacity falls + reward-free
legibility collapses faster; R1 ⇒ proportional degradation, no
interaction-with-size). Unknown: every sub-1m quantity (no fit below
size1m exists anywhere in the portfolio).

## Design — 2 sizes × 2 arms × 2 sides × 4 seeds = 32 fit+adapt jobs

- New size presets (this freeze-commit): `size300k` (rssm deter 256,
  hidden 32, classes 4; depth 2; units 32) and `size100k` (deter 128,
  hidden 16, classes 4; units 16) in `dreamerv3/configs.yaml`;
  whitelists extended in `scripts/submit_all.sh` + `scripts/axis1.sbatch`.
  classes stays 4 ⇒ stored stoch latents keep shape; only deter is
  context-resized. Realized param counts recorded at smoke: [FILL
  size300k], [FILL size100k].
- **Buffer pairs ([YOU], login node, minutes)** — the _ctx convention
  (12m/25m precedent):
  `python -m probing.resize_replay_context resize --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_ctxk3 --deter 256 --stoch 32 4`
  `python -m probing.resize_replay_context resize --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_ctxk1 --deter 128 --stoch 32 4`
  (`--stoch 32 4` = the SOURCE stoch shape, unchanged. Reviewer-3 F10:
  the tool RECORDS shapes but does not assert them — the operator gate
  is: after each build, verify the output `manifest.json`'s
  `context_resize.sides.*.source_latent_shapes` shows deter [512] and
  stoch [32, 4]; a mismatch means the source assumption was wrong —
  delete the output and STOP, do not submit.)
- **Adapt-readout pinning (reviewer-3 F5 — the mediator repair)**: the
  size presets also shrink the FRESH adapt heads (policy/value/rew/con
  units are size-keyed and NOT loaded from the ckpt under
  frozen_readout), which would confound WM capacity with readout
  capacity. Registered fix: all 32 adapts run with
  `AXIS1_ADAPT_CONFIG=readout64_frozen` (new single-token config =
  frozen_readout + all four head widths pinned at the size1m value
  64; configs compose base → size → adapt, so the pin lands LAST).
  The reader gates every adapt config on `agent.policy.units == 64`.
  With the readout pinned, the size1m random floor transfers.
- **Smoke gate**: ONE fit (task, sk3, side0 seed99) — config audit
  (rssm deter 256, expl.mode task), param count recorded into the FILL
  slots above, then the smoke run dir is REMOVED BY HAND (rev-3 F13:
  the wave's KEEP globs are seed-restricted to 1–4, so seed99 stays
  DELETE-SAFE, but never rely on cleanup for it — the #21 seed99
  contamination lesson).
- **Registered submissions** (the axis1-bundles path; QUAD_SUFFIX
  keeps `_ctx` out of run naming; ID_PREFIX carries the size infix;
  the bundle export list carries AXIS1_ADAPT_CONFIG — rev-3 F16 fix
  in this freeze-commit):
  - task sk3: `AXIS1_EXPL_MODE=task AXIS1_ID_PREFIX=ax1sk3 AXIS1_QUAD_SUFFIX=_ctxk3 AXIS1_SIZE=size300k AXIS1_ADAPT_CONFIG=readout64_frozen AXIS1_DOMAINS=finger AXIS1_QUADS=q1 AXIS1_SEEDS="1 2 3 4" ./scripts/submit_all.sh axis1-bundles`
  - apt sk3: same with `AXIS1_EXPL_MODE=apt AXIS1_ID_PREFIX=ax1fsk3`
  - task sk1: same as task sk3 with `AXIS1_ID_PREFIX=ax1sk1 AXIS1_QUAD_SUFFIX=_ctxk1 AXIS1_SIZE=size100k`; apt sk1: `AXIS1_ID_PREFIX=ax1fsk1`.
  Names: WMs `ax1wm_finger_{,f}sk{3,1}q1s{0,1}_seed{1..4}`; adapts
  `adapt_ax1{,f}sk{3,1}q1s{side}_finger_seed{k}_ckpt500000`
  (UPDATES 500000, STEPS 1.25e5 — protocol identical to the standard
  cells except WM size, with the readout pinned).
- Measures per fit: `ridge_probe` measure on `finger_v1` (default
  protocol; the legibility panel) + the E4 default pass; adaptation
  AUC collate over the 32 adapts. Bundle: auc csv + 32 ridge jsons +
  e4 csv + fit-counter/ckpt-step jsons (registered #21-style
  producers) + per-run `config.yaml` copies.
- Cleanup: KEEP_PENDING entries
  `ax1wm_finger_*sk[13]q1s*_seed[1-4]` `adapt_ax1*sk[13]q1s*_seed[1-4]_*`
  (seed-restricted, rev-3 F13) added in this freeze-commit; the ctx
  buffer pairs live under `axis1_finger` = KEEP_SUBSTRATE (no new
  entry needed; verified no DELETE glob matches any wave name).

## Registered decision rules

Gates (fail-closed): 32-name dual fit witness — the LITERAL producer
(rev-3 F14; regex seed-anchored so seed99 cannot enter):

```
python - <<'PY'
import glob, json, os, re
root = os.environ['RUNROOT']
rx = re.compile(r'ax1wm_finger_f?sk[13]q1s[01]_seed[1-4]$')
wms = sorted(w for w in glob.glob(root + '/ax1wm_finger_*sk[13]q1s*')
             if rx.search(w))
assert len(wms) == 32, (len(wms), wms[:3])
counters, steps = {}, {}
for w in wms:
    kv = dict(l.split('=', 1) for l in
              open(w + '/OFFLINE_FIT_PROGRESS').read().strip().split('\n'))
    counters[os.path.basename(w)] = dict(update=int(kv['update']),
                                         total=int(kv['total_updates']))
    cks = [c for c in sorted(glob.glob(w + '/ckpt/*'))
           if '500000' in os.path.basename(c)
           and os.path.exists(os.path.join(c, 'done'))]
    assert cks, (w, 'no done-marked 500000 ckpt')
    steps[os.path.basename(w)] = 500000
json.dump(counters, open('fit_counters.json', 'w'))
json.dump(steps, open('ckpt_steps.json', 'w'))
print('32/32 witnessed')
PY
```

(the reader re-verifies update==total per name); per-fit config gate
(rssm deter 256 at sk3 / 128 at sk1, expl mode matches arm); per-adapt
config gate (run.from_checkpoint names the registered WM; frozen;
**policy units == 64** — the readout pin); adapt QC gate (rev-3 F6:
`qc_pass` required per row ∧ no `n_ep_100k` below the modal — the
realized-training standing rule); exact 32-run inventory (missing ⇒
REFUSE); duplicate rows REFUSE; ridge jsons REQUIRED (a missing
`--ridge_glob` cannot silently skip P-B1c) with `witness_match is not
False` per json.

Per size s: interaction I(s) = two_sample(task AUCs, apt AUCs)
(house BCa + permutation, n=8v8; `auc100k` from the collate).

- **FLOOR gate per size (registered constant)**: the task arm at size
  s must show mean auc100k > **113.7** (= the executed random-floor
  83.5 + 2×15.1, `randominit_read_20260810`); else that size is
  **FLOOR-CENSORED** (its cells carry no competition information —
  at capacities where even the task arm cannot learn, both models
  predict collapse). Both sizes censored ⇒ **INSTRUMENT-LIMITED**
  (the descent overshot; a shallower grid is a new registration).
  One size censored ⇒ P-B1b unavailable; the surviving size's I(s)
  reported, verdict **INDETERMINATE-PARTIAL**.
- **P-B1b (PRIMARY, the λ discriminator)**: DID = I(sk1) − I(sk3)
  computed on per-run values: [mean(task,sk1) − mean(apt,sk1)] −
  [mean(task,sk3) − mean(apt,sk3)]. Group bootstrap BCa 95% (resample
  runs within each of the 4 groups, B=10K rng 0) + permutation p
  (size labels permuted within arm, B=10K rng 0, **on
  main-effect-ALIGNED values** — rev-3 F4: the raw within-arm pool is
  a location mixture under a size main effect and the unaligned
  permutation is self-defeating, power 0.69→0.04 at δ=250; the
  aligned form is exact-calibrated, type-I 0.050 at every δ, power
  ~0.74 at DID=150; the observed statistic is invariant to the
  alignment).
  - CI > 0 ∧ p < .05 ⇒ **LAMBDA-SUPPORTED**: the interaction GROWS as
    capacity falls — competition binds at the boundary; the two-regime
    model is adopted per the frozen Part-B adjudication.
  - CI < 0 ∧ p < .05 ⇒ **INTERACTION-SHRINKS**: reported; consistent
    with R1-plus-approaching-floor; interpreted WITH the floor-gate
    margins (a shrink at healthy task levels is genuine R1-side
    evidence; near the floor it is compression).
  - else **INDETERMINATE** (MDE disclosed below).
- **P-B1a (per-size, secondary)**: I(sk3), I(sk1) each with CI/p —
  descriptive of whether the size1m interaction persists downward
  (the known +84/+113.8 anchors quoted alongside, value-aware).
- **P-B1c (legibility collapse, secondary)**: DID on the ridge AUROC
  (`probe.alpha_0.001.auroc`, finger_v1) with the SAME machinery:
  λ ⇒ the task−apt legibility gap WIDENS at smaller capacity; R1 ⇒ no
  size dependence. Computed UNCONDITIONALLY (rev-3 F7: a fit-level
  quantity, informative even when behavior floor-censors), but gated
  on apt legibility headroom: mean apt AUROC at sk3 must be ≥ 0.55,
  else **P-B1c = APT-FLOOR-CENSORED** (the executed size1m apt value
  is 0.401 — this censoring is the LIKELY outcome, disclosed; the leg
  is registered anyway because a descent-driven apt collapse from an
  already-low base is exactly what it would catch if headroom
  exists). Secondary weight only ("legibility leg agrees/disagrees").
- E4 rew-NLL per arm×size recorded as descriptive panels (no rule —
  the member band is calibrated at size1m and does not transfer; the
  apt-arm rew-NLL panel is empty by construction, reward-free fits
  record no rew head).
- MDE disclosure (rev-3 F4 corrected): DID at n=8/group on the AUC sd
  basis (~77) has se ≈ 54; with the ALIGNED permutation the
  registered conjunction has ~74% power at |DID| = 150 and ~99% at
  250, independent of the size main effect — only a LARGE competition
  effect is detectable, which matches the premise (λ must be large at
  the boundary to matter); INDETERMINATE licenses nothing.
- **λ-negative mapping**: per
  `PREREG_theory_R1_gating_amend1_20260811` — only
  `INTERACTION-SHRINKS` counts as λ-negative for the Part-B joint
  rule; `INDETERMINATE*`/`INSTRUMENT-LIMITED` count as neither.
- **Part-B adjudication linkage (frozen in the addendum)**: this
  wave's verdict feeds the registered both-trials rule together with
  B2 (`PREREG_nuisance_20260811`); neither wave alone retires λ.

Reader: `analysis/capdescent_read.py`, frozen with this file,
selfcheck before any run exists. ONE read execution. Reviewer:
2026-08-11 B1/B2 batch review (ONE, Opus 5 — standing auto-approval)
BEFORE freeze-commit.
