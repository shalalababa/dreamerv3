# PREREG — Routed-repair control wave (rrpctl): overwrite + matched
# budget (22 Aug 2026)

**Status: FROZEN at user commit. ONE read execution.** Licensed by the
routed-repair PROCEED (artifacts/routedrepair_read_20260822; amend1
M14 names exactly these registrations). User GO 22 Aug with the
recommendation adopted: register (a) the rgo-from-random-init control
and (c) the matched-budget comparison as ONE wave; (b) the short-refit
dose curve stays parked. Reviewer: ONE Opus pass, user pre-approved,
triggered at build. Self-scooping fence unchanged
(standalone-algorithm-paper material).

## The question

The repair arm (291.44 mean AUC100k) beat full fine-tune of the same
sunk trunk by +201.09* and sits ~2× above scratch — but M14: a 500k
rgo re-fit on the same buffer approaches a fresh rgo fit, so "repaired
the sunk trunk" is not yet separated from "overwrote it", and the
repair arm carried 2× the pretraining budget of its fine-tune control.
This wave supplies both missing comparisons:

- **Cell A (rca) — rgo-from-random-init, 500k**: does the apt init
  contribute ANYTHING beyond a fresh rgo fit of the same length?
- **Cell B (rcb) — matched budget, matched phase structure**:
  repair = [apt 500k → rgo-refit 500k]; rcb = [rgo 500k → rgo-refit
  500k]. Both 1M total updates, both two-phase, phase-2 IDENTICAL
  (same refit recipe, same buffer, same init regex); ONLY the phase-1
  objective differs. This is the cleanest attainable isolation of
  "what a sunk apt phase is worth at budget parity."

## Design — finger q1 side1, 16 new cells

| cell | phase-1 fit | phase-2 | adapt run_id |
|---|---|---|---|
| repair (EXISTING, pinned) | fq1s1 seeds 1–8 (apt 500k) | rrp rgo-refit 500k | (read 22 Aug; per-seed constants below) |
| rca (8) | — (random init) | rgo fit 500k = the EXISTING carrier fits `ax1wm_finger_rgoq1s1_seed{17..24}` | `adapt_ax1rcaq1s1_finger_seed{k}_ckpt500000` |
| rcb (8) | the same carrier rgo fits | NEW rgo-refit 500k `ax1wm_finger_rcbq1s1_seed{17..24}` | `adapt_ax1rcbq1s1_finger_seed{k}_ckpt500000` |

- rcb refit mirrors the rrp recipe EXACTLY apart from the init source:
  `AXIS1_INIT_WM=ax1wm_finger_rgoq1s1_seed{k}`,
  `AXIS1_INIT_REGEX='^(enc|dyn|dec)/'`, `AXIS1_INIT_FROZEN_ENC=False`,
  `AXIS1_ARM=rgo`, `REPLAY=$RUNROOT/axis1_finger/q1/side1`,
  `UPDATES=500000` (fresh counter — reads 500000, same as rrp).
- All adapts: unfrozen_readout, STEPS=1.25e5, U1 protocol — identical
  to the rrp/raft adapts. Vehicle: `ops/waves/rrpctl/bundle.sbatch`
  (mirrors the routedrepair bundle; arms rca adapt-only ~5 h /
  rcb refit+adapt ~12 h; job logs `rrpctl_bundle_*.out` feed the
  witness completion evidence).
- **Cross-cohort disclosure**: repair lives at seeds 1–8 (donor
  cohort, RCC 22 Aug), controls at seeds 17–24 (carrier cohort fits,
  fresh venue for adapts) — two_sample, not paired; venue/era
  differences disclosed, with the aptctl-vs-July-pin stability
  (90.35 vs 89.90) as the measured cross-era anchor for this AUC
  protocol.
- **Valuefree overlap disclosure**: rca duplicates the registered (not
  yet run) valuefree `q1uzs1` arm (same fits, same adapt protocol,
  different run_ids). Deliberate: no cross-registration coupling or
  ordering dependency; if valuefree reads first, the control-arm
  LEVEL becomes partially known — immaterial, every threshold here is
  frozen now and no discretionary choice remains. The duplicate pair
  is a free same-protocol stability check, descriptive only.
- Outcome-blindness: rgo-unfrozen adapts have never been run under
  ANY name; both control arms are genuinely unseen. The repair arm is
  known (its read is published) and enters only as pinned constants.

## Pinned constants (look-0, from executed reads — sources cited)

- REPAIR per-seed AUC100k (routedrepair_20260822_180310 collate, mode
  ax1rrpq1s1): seed1 360.9792, seed2 429.8125, seed3 257.6875, seed4
  266.6667, seed5 198.3542, seed6 263.6042, seed7 214.6146, seed8
  339.8229 (mean 291.4427).
- G_PIN = 201.0885625 (the D1 routed-repair effect; the yardstick for
  the bounded branches). **Band justification (review m1, stated
  precisely)**: G_PIN embeds the repair cohort's realization, and
  repair is one side of both E1 and E2 — but BOTH the band and the
  repair values are look-0 FROZEN constants, so the band cannot move
  with any quantity this wave produces; that immovability, not
  arm-exclusion, is what satisfies the house band rule here, and it
  is disclosed rather than claimed away.
- BOUND = G_PIN/2 = 100.5443. SCRATCH = 147.6823 (context only).

## Estimands and decision rules (frozen; two_sample = capdescent house
## form, label-permutation + BCa, α=.05; fixed-sequence gatekeeping)

- **E1 (PRIMARY — overwrite test)**: two_sample[repair − rca].
  - FIRES POSITIVE (perm p<.05 AND BCa CI>0) ⇒ **INIT-ADVANTAGE**:
    the apt init adds something beyond a fresh 500k rgo fit; E2 is
    then adjudicated.
  - FIRES NEGATIVE ⇒ **ANTI-SALVAGE-500K** (fresh rgo beats repair —
    registered surprise; E2 descriptive).
  - No fire, BCa CI ⊆ (−BOUND, +BOUND) ⇒ **OVERWRITE-SUFFICIENT-
    BOUNDED**: the init contributes less than half the routed effect
    in either direction; the claim demotes to "refit-don't-finetune"
    in every use. E2 descriptive.
  - No fire, CI wider ⇒ **NO-CALL-UNDERPOWERED** (realized MDE80
    reported; expected ≈1.4×sd_pooled ≈ 110 AUC at sd≈80 — this wave
    decides LARGE separations only, disclosed).
- **E2 (CO-PRIMARY, adjudicated ONLY on E1 INIT-ADVANTAGE; otherwise
  reported descriptively with no verdict weight)**: two_sample[repair
  − rcb].
  - FIRES POSITIVE ⇒ **SALVAGE-CONFIRMED**: the apt phase beats an
    equal-budget rgo phase — the sunk trunk has value beyond generic
    training; the deployable salvage claim gains its license (still
    scoped: finger, q1s1, 500k+500k).
  - No fire, CI ⊆ (−BOUND, +BOUND) ⇒ **BUDGET-EQUIVALENT**: a sunk
    apt trunk is worth its update budget under routing, no more —
    salvage = "you don't lose the compute", never "you'd buy it".
  - FIRES NEGATIVE ⇒ **ANTI-SALVAGE-AT-BUDGET**.
  - Else ⇒ NO-CALL at E2 (INIT-ADVANTAGE stands, budget question
    open; MDE80 reported).
- **E3 (descriptive, no α)**: paired within-seed [rcb − rca] — the
  value of the second 500k rgo phase (dose information for the parked
  short-refit registration).

## Gates (refusal = read not consumed; review B1/M1–M4/m2 applied
## pre-freeze)

Witness (`build_rescue_bundle --wave rrpctl --job_logs <dir>`): 16 fit
entries — the 8 EXISTING carrier rgo fits AND the 8 new rcb refits —
counters == 500000 exact; **rgo-arm identity pins on BOTH fit sets**
(M1: task + agent.reward_grad True + agent.repval_grad False — the
carrier fits ARE the rca arm and phase-1 of rcb, so they are pinned
too, closing the wrong-objective-refit blind spot); **init-sha
EQUALITY pairing** (M2): each rcb fit's saved `run.from_checkpoint`
must resolve inside `ax1wm_finger_rgoq1s1_seed{k}` (same seed) AND
its recorded sha must EQUAL the carrier fit's own resolved done-ckpt
sha (both non-None) — path containment alone is not identity;
**phase-2 recipe pins on rcb** (M3): run.from_checkpoint_regex ==
'^(enc|dyn|dec)/' + agent.frozen_enc == False, plus the REALIZED
replay path (parsed from axis1's own `offline_fit ... REPLAY=` job-log
line — --static_replay never enters the fit config) must end with
`axis1_finger/q1/side1`; **adapt completion witnesses ×16** under the
four-signal evidence (config run.steps 125000 + job-log OK/DONE +
metrics final ≥115000 + scores ≥100), counters == 125000;
adapt→source linkage (rca from the carrier rgo fit dir, rcb from the
rcb fit dir); STRICT modal n_ep across both modes **pinned == 96**
(M4: the repair cohort's modal — auc100k is a mean over the ≤100k
episode window, so a different modal is a different estimand window,
and the pinned constants get no within-read protection);
**_meta.problems must be EMPTY** (m2: an `--allow_problems` build can
never reach a consumed read); ONE-read guard (output artifact must
not pre-exist).

Disclosures (review m3/m4): crashed cells may resume into their
logdir — a from-zero restart doubles n_ep and refuses at the modal
pin; the one-venue line below is LOAD-BEARING, not advisory — init_sha
is computed from the absolute path in the rcb config, which must
resolve at bundle time.

## Ops

Existence precondition: 8/8 `ax1wm_finger_rgoq1s1_seed{17..24}` fits
hold a done-ckpt **whose step suffix is 500000** at the venue (review
B1: the check is the newest `ckpt/*/done` dir's step — NEVER the
`ckpt/latest` pointer, which exists after any save; the bundle
vehicle re-checks this per-cell with the same rule and additionally
neutralizes inherited AXIS1_*/FORCE_WM dials). They are also the
valuefree wave's arm — the standing archive-first precondition covers
both →
8 rcb refit+adapt cells + 8 rca adapt cells via
`ops/waves/rrpctl/bundle.sbatch` (~8×12h + 8×5h ≈ 136 lane-h ≈
55–70 GPU-h; UNVERIFIED, re-derive from the first measured cell) →
`build_rescue_bundle --wave rrpctl` → bundle + sha manifest → **[ME]
ONE read** via `analysis/rrpctl_read.py`. One $RUNROOT, one venue
before bundling (bundle-after-pull rule).
