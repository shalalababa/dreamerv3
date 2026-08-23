# PREREG — Wave B: the adversarial amortized actor (the
transmission-rescue slate, part 3 of 3), 22 Aug 2026

**Program:** the Fable-ladder Tier-2 wave (design pinned in
`research_notes/paper5_uncertainty_field/WaveB_AdversarialActor_
Design_20260822.md`; GO condition — the A + C reads — met 22 Aug;
user GO same day, one reviewer pre-approved). TWO REGISTERED WINGS
(separate questions, both α=.05, no cross-wing correction —
disclosed; the Tier-2 gate reads ONLY the ambient cell):

- **AMBIENT (the Tier-2 carrier):** can a fresh amortized
  actor-critic, trained AGAINST the decoded distractor-variance
  field on the frozen Stage-1 WMs, realize more distractor
  attribution than the incumbent exploration policy? The attacker
  is matched by CLASS (amortized, the same family as the
  incumbent) and gated in-model (below) — the strongest attacker
  this program can field.
- **LOCALIZED:** the same adaptation on the A1 checkpoints
  (hetero/flat) — the matched-strength probe of the channel Wave
  C's weak CEM left open (C landed WEAK-OPTIMIZER: ratios
  0.74–0.89; this wave replaces the optimizer class entirely).

**VALUE-AWARE DISCLOSURE (binding):** priors are informed by the
executed ambient planner read (UNSTEERABLE), the A read
(POLICY-STATIONARY at matched strength; model-predicted attr
headroom ~2× the realized level, realized ≈ 0 — the off-support
gap this wave's secondary directly measures on an amortized
attacker), the C read (weak-optimizer null; share steering 8/8),
and the A1 region prior (in-region signal 1.54×). The DATA are
fresh: no advd adaptation or probe has ever run.

## 1. Machinery (built + smoked BEFORE compute)

- **`dreamerv3/agent.py` expl mode `advd`:** imagined reward =
  DECODED DISTRACTOR MEMBER-VARIANCE at imagined (s, a) — the
  se_probe member_vars idiom (disag.predict → postsplit → frozen
  decoder per member → var over members of the distractor recon,
  mean over dims) inside the training graph, sg'd, ×
  `expl.advd_scale = 100` (registered; smoke-measured raw ≈ 0.27,
  so pre-retnorm imagined reward ≈ 27 — the perc retnorm absorbs
  it, the same argument as p2e's disag_scale). Unnormalized:
  per-run argmax-equivalent (iid same-scale distractor dims).
  **The freeze IS the module list** (`self.modules = [con, pol,
  val]`; wm + disag excluded — nj.grad selects by module path);
  its PROOF is post-hoc and empirical: the probe's bitwise
  WM-identity gate (rev WB-n8 — a constructor assert on the
  literal list would be tautological, so none is claimed). Config
  preset `expl_advd` (mode + `from_checkpoint_regex
  '^(enc|dyn|dec|disag)/'` — the regex loads the substrate and
  ONLY the substrate; pol/val/con initialize fresh).
- **Producer `scripts/uncfield_advd.sbatch`:** source resolution
  via ckpt/latest WITH the stale-pointer guard; full BIND-CHECK
  (advd pins + the R3-D4 identity set + seed + steps +
  from_checkpoint path).
- **Probe `uncfield/se_advd_probe.py`** (one invocation, 1-Aug
  rule): arms random/policy/advd, fresh env per arm (CRN), online
  posterior states under the SOURCE WM, #32-form measurement,
  per-step npz + float64 json. **Integrity gates computed in the
  probe and asserted by the reader:** WM-IDENTITY (every param
  under the regex BITWISE equal between source and adapted
  checkpoints — the freeze proof) and ENV-IDENTITY
  (task/distractor/planted subtrees equal). **Registered
  imagination rows** (stride 5, horizon 12, chunk 100 — chunk is
  PINNED, rev WB-M4: the imagination sampler's batching + seeds
  depend on it; normalized attr units): from POLICY-arm states
  under BOTH policies (the dominance gate), and from ADVD-arm
  states under the actor (the imagined-vs-realized row). Both
  agent arms SAMPLE from their action distributions (Agent.policy
  ignores mode outside cemplan — rev WB-n4): the comparison is
  stochastic-vs-stochastic.
- **Reader `uncfield/se_advd_read.py`** (frozen, selfcheck PASS;
  ONE execution, KEYED TO THE CARRIER — rev WB-M3: the registered
  output file is claimed only when the AMBIENT wing adjudicates; a
  localized-only or no-wing result goes to a timestamped side file
  and does NOT consume the execution).
- **Zero-GPU staging gate `scripts/wb_advd_preflight.py`** (rev
  WB-M5; run on the staged sources BEFORE submission — executed
  22 Aug against the verified local bundles: 16/16 OK): the
  SOURCE_CKPT_PIN manifest (latest == pinned final == newest — a
  jointly stale source is invisible to the bitwise WM gate, rev
  WB-M2), env-identity vs the producer-emitted config through the
  defaults overlay (the rev WB-B1 class: the Stage-1 sources
  predate the distractor mod_*/gate_* schema — raw comparison
  would have aborted all 8 ambient probes; fixed + regression-
  fixtured in the probe selfcheck), and architecture parity (the
  validity condition of the probe's params-substitution
  imagination rows; also asserted at probe runtime).
- **Smokes executed pre-registration (CPU, local):** (i) a det
  p2e source (wb_src) + a REAL advd adaptation from its checkpoint
  — from_checkpoint restored exactly enc/dyn/dec/disag (log-
  verified key list), training ran (advd_rew_raw ≈ 0.266 at step
  4048), and (ii) the full probe smoke on that pair — all three
  arms rolled and the BITWISE WM-identity gate PASSED on real
  artifacts (the freeze held through genuine optimizer steps).
  One registered GPU smoke (SMOKE=1, pair 0) precedes the probe
  panel.

## 2. Wave (16 adaptations + 1 GPU probe smoke + 16 probes)

`ops/waves/wb_advd/spec.yaml`: ambient `se_advd_s10..17` (sources
`se_cheetah_seed10..17`), localized `se_advd_het_s44..47` (the A1
mod quadruple) + `se_advd_flat_s48..51` (FLAT scale). Adaptation
seed = source seed (the identity convention, reader-pinned). 1e5
steps each (the ~2–3 h/run figure is an EXTRAPOLATION, not a
measured GPU step rate — rev WB-n9 — so the FIRST adaptation is a
canary: reach RUNNING + step-rate check before feeding the rest,
the submit-ledger rule; SAVE_EVERY stays 300 s so the final
snapshot lands inside the milestone regex; the in-model gate plus
the per-run plateau descriptive are the sufficiency evidence).
Sources staged instance-side at `$RUNROOT/_wb_srcs` (set IN the
wave cmd so it rides the stale-producer gate — rev WB-M1; ckpt
dirs incl. latest pointers), gated by the zero-GPU preflight
BEFORE submission. Probe chain (rev WB-n7, results-sync v2): pull
the 16 adapted run dirs AND the 16 source dirs to RCC, verify,
then ALL 16 pairs in ONE GPU job
(`scripts/se_advd_probe.sbatch`; single-device rule; the driver
guards BOTH the adapted and the source ckpt pointers), then
bundle (probe logs ride cloud_logs).

## 3. Registered read (frozen `uncfield/se_advd_read.py`; ONE
execution; registered invocation `--runs "<runroot>/se_advd_*"
--output <artifacts dir>`)

- **Validity per run:** SOURCE-CKPT PIN (the committed
  16-entry manifest: the probe's source_ckpt AND the adaptation's
  from_checkpoint must both equal the pinned final source
  checkpoint, rev WB-M2), disag_target postfeat pin (rev WB-n3),
  imagination rows pinned to 400 entries (rev WB-n2), adapted-config pins (mode advd, advd_scale
  100, from_checkpoint + the registered regex — a drifted regex
  means disag was FRESH, not frozen, and the run is void; disag
  det/ens-8 pins; per-wing env pins incl. the A1 mod quadruple /
  FLAT scale; penalty inert), probe pins
  (episodes 4 / ep_len 500 / horizon 12 / stride 5 / probe seed 0
  / gate constants / one CUDA device / gpu backend), WM-IDENTITY
  ok, ENV-IDENTITY ok, run-dir + source-seed identity, newest-ckpt
  cross-check, float64 provenance (attr/intr/occ + the imagination
  rows), n_steps 2000, reset_hits 0, per-run degeneracy floor
  intrinsic(policy) > intrinsic(random). Defective → excluded; a
  wing needs its FULL panel (8, or 4+4). Fit counters
  (expect_steps 1e5) reported; non-OK annotates.
- **IN-MODEL DOMINANCE GATE (per wing, adjudicated first; rev
  WB-B2 — the bar is itself an α ≤ .05 test, since under an
  indistinguishable actor each run is a coin flip):** on
  POLICY-arm states, imagined attr under the adapted actor >
  under the incumbent policy — ambient ≥ 7/8 (P|H₀ = 9/256 =
  .035); localized POOLED ≥ 7/8 (same α) AND ≥ 3/4 in EACH
  sub-wing (balance). Fail → that wing
  NOT-ADJUDICABLE-INSTRUMENT-WEAK. Registered descriptives making
  gate strength legible on every path: per-run relative margin,
  per-episode block means (the honest inferential unit — the 400
  stride-5 starts are trajectory-correlated, so no anchor-level
  test is run), and the adaptation-plateau row from each run's
  own telemetry (expl/advd_rew_raw last-decile vs 5th-decile —
  the measured evidence for the 1e5-step sufficiency claim, rev
  WB-B2.4/n9).
- **AMBIENT PRIMARY:** Δ = attr_d(advd) − attr_d(policy) > 0 in
  ≥ 7/8 AND exact sign-flip p ≤ .05 → **TRANSMISSION-AMPLIFIED**
  (the containment story falls at matched strength; Act-2
  revives). Else → **CONTAINED-AT-MATCHED-STRENGTH**: with the
  gate green this licenses the Fable-ladder **Tier-2** wording —
  "an amortized attacker trained against the exposure field, at
  matched class and verified in-model competence, does not
  increase realized exposure (this env class / WM family)."
  "Cannot amplify exposure" unqualified remains NEVER licensed.
- **LOCALIZED PRIMARY (conjunctive, full 4+4):** interaction
  mean Δ(hetero) − mean Δ(flat) > 0 at exact C(8,4) p ≤ .05
  (floor 1/70) AND Δ > 0 in ≥ 3/4 hetero →
  **LOCALIZED-TRANSMISSION-MATCHED** (deployment optimization
  cashes the avoided region's misprice — the two-regime story
  lands at matched strength). Else →
  **LOCALIZED-CONTAINED-MATCHED** (the containment extends to the
  localized channel — closing the question C left open). Power
  note: the fire needs near-complete separation; no power against
  moderate effects.
- **Registered secondaries (no fire, every path):** the
  imagined-vs-realized row (imagined attr under the actor from
  its own states vs realized attr_d(advd) — Wave A measured this
  gap ≈ 10× for greedy replanning; whether AMORTIZED training
  closes it is a mechanism finding either way), share rows
  (composition steering under a trained attacker), occupancy rows
  (localized: does the actor ENTER the avoided region?), task
  return, intrinsic rows.

## 4. Fences & placement

Paper-5 security paragraph (the Tier-2 gate) + the Track-B
transmission section (the localized wing + the two-regime
paragraph) + the off-support-imagination mechanism thread (with
Wave A). The attacker is an instrument; no claims about adversarial
RL in general. Tier ladder as pinned in
PREREG_planner_localized_20260822 §2.

Freeze = the ONE pre-approved review adjudicated (done 22 Aug —
2 BLOCKING + 5 MAJOR + 9 minor, all resolved pre-freeze; every
selfcheck re-PASS; the preflight verified 16/16 against the real
source configs) + commit of this file + the agent/configs deltas +
`uncfield/se_advd_probe.py` + `uncfield/se_advd_read.py` +
`scripts/uncfield_advd.sbatch` + `scripts/se_advd_probe.sbatch` +
`scripts/wb_advd_preflight.py` + `ops/waves/wb_advd/spec.yaml` —
then staging + preflight → canary → the 16 adaptations → pull to
RCC → GPU probe smoke → the 16 probes → bundle → the ONE read.
