# PREREG AMENDMENT 2 — SE wave: triggered arms + M4 operationalization (14 Aug 2026)

**Parent:** `PREREG_nfi_scale_exhibit_20260812.md` v2.2 (frozen
fc85fdee) + Amendment A1. **Context:** the Stage-1 registered read
(`artifacts/se_stage1_read_20260814/`) landed OUTCOME CELL 3 and met,
verbatim, the parent's two pre-authorized conditional triggers: the
§3 D-only arm ("N-share fires while all D-shares are at floor") and
the §3 Stage-2 behavioral arm ("P-SE1 fires on any non-C channel").
User GO recorded 14 Aug for both.

**Timing disclosure (registered honestly):** this amendment is written
AFTER the Stage-1 outcome (the triggers are outcomes). Every numeric
pin below is therefore either (a) forced by the frozen parent text,
(b) derived from PRE-Stage-1 data (the 12-Aug smoke bundle), or (c) an
arbitrary indexing convention — no Stage-1 attribution number is used
in any pin. The read RULES for both arms and for M4 are registered
here BEFORE their compute/reads execute.

---

## 1. D-only interference arm (parent §3, trigger met)

**Config:** Stage-1 verbatim except the Distractor wrapper is OFF
(`--distractor.dim 0`); the planted wrapper (D0/D1/D2 + C) unchanged;
same task, steps 5e5, size1m, expl_p2e. **Seeds 10–13**
(parent-pinned), RUN_IDs `se_donly_s10..13`, 4 runs. No cross-arm
seed-pairing statistics of any kind vs Stage-1 (parent §1 rule): the
comparison below is level-based, never per-seed differenced.

**Read (frozen reader `uncfield/se_donly_read.py`, built + selfchecked
BEFORE the read; per-run se_probe outputs at the FINAL checkpoint are
its only statistical inputs; same validity gates as the primary: S ≥
256, calibration < 0.5 on every real key, final-ckpt gate, provenance
p-equality, train seeds {10..13} distinct):**

- Per run × D channel: indicator = share_vs_real > per-run
  permutation-null median (identical statistic to Stage-1);
  θ₁ = vs_source.
- **Outcome map (registered; θ₁ materiality band [0.90, 1.10] — the
  raw "CI excludes 1.0" criterion is degenerate at n=4 because
  Stage-1 measured parity to ±0.1%, so a trivial 0.4% deviation with
  a tight CI would fire; ±10% is two orders looser than measurement
  precision yet 40× smaller than the observed N misprice of +426%,
  and Stage-1's D values (1.000) sit far inside it, so the band
  choice cannot re-adjudicate Stage 1):**
  (a) **NO-INTERFERENCE**: dup0 θ₁ seed-level BCa CI entirely within
      [0.90, 1.10] AND no D channel at 4/4 indicators → Stage-1's
      D-null stands UNQUALIFIED (N did not suppress D attribution).
  (b) **INTERFERENCE**: dup0 θ₁ CI entirely > 1.10 OR any D channel
      at 4/4 indicators → Stage-1's D-null is QUALIFIED as
      N-suppression; redundancy re-enters candidate mechanisms
      (flagship still NOT licensed without a full re-run).
  (c) **MIXED**: anything else → descriptive report; neither
      qualification licensed.
- **Power disclosure (and why the indicator arm is 4/4, not ≥3/4):**
  per-run indicators are ~Bernoulli(.5) under the parity null, so at
  n = 4 the ≥3/4 form fires with probability .3125 per channel
  (familywise ≈ .68 over 3 channels) — degenerate; verified on parity
  fixtures during the reader build. The 4/4 form has exact p = .0625
  per channel (familywise ≈ .18) and is DISCLOSED as the weak
  supporting arm; the sensitive instrument is θ₁, which Stage 1
  measured to ±0.1%. This arm is a registered ROBUSTNESS CHECK, not a
  confirmatory test. BCa on n = 4 is fragile; min/max reported
  alongside.

**Bundle spec:** per run `se_probe/` (json+npz), `config.yaml`,
`metrics.jsonl`, `ckpt/latest` (+ ckpt dir listing) — same shape as
Stage-1; manifest per results-sync v2.

## 2. Stage-2 behavioral arm (parent §3, trigger met)

**Gate pin (data-independent):** `gate_key = position`,
`gate_index = 0`, `gate_threshold = −0.13009691` = the MEDIAN of
position[0] over the ENTIRE pre-Stage-1 smoke replay buffer
(`uncfield_se_smoke_20260812_155552/se_smoke0/replay`, 14,928 steps,
computed 14 Aug before any Stage-2 compute; range [−0.322, +0.197]).
The parent's placeholder threshold 0.0 would have been DEGENERATE
(2.96% open at smoke) — the median gives a maximally non-degenerate
gate at the only outcome-independent behavioral reference available.

**Gating covers ALL appended channels** (parent §3: "both arms carry
identical appended channels; the gated arm emits them only when…"):
the planted wrapper's gate (already built) AND the Distractor wrapper,
which gains a strictly-additive gate for this arm
(`embodied/envs/distractor.py` + configs keys `distractor.gate_*`;
default `gate_key=''` verified BITWISE-INERT by selfcheck — all prior
consumers incl. Paper-1 dz configs and Paper-2 Gate-D0 unaffected; OU
recursion and rng consumption are gate-independent, so gated/ungated
share identical latent OU paths). Gating N is load-bearing: N is the
channel Stage-1 showed mispriced, so the diversion incentive rides on
it.

**Arms & seeds:** gated = seeds 20–23 (`se_gate_s20..23`), ungated =
seeds 24–27 (`se_ungate_s24..27`) — fresh, disjoint from 10–17 and
smoke 0; arms never seed-paired (parent §1). Ungated config = Stage-1
verbatim; gated config = Stage-1 + the gate triple on BOTH wrappers.
8 runs, 5e5 steps each.

**M3 read (frozen reader `uncfield/se_m3_read.py`, built + selfchecked
BEFORE the read):**
- **Occupancy estimand:** fraction of steps with
  position[gate_index] > gate_threshold over the ENTIRE synced replay
  buffer (every chunk file, every step — deterministic full scan, no
  sampling rng; validity floor ≥ 1e5 steps present, else the run is
  flagged invalid). Identical measurement in both arms (the gate
  coordinate is a REAL key present everywhere).
- **PRIMARY (directional, registered):** mean occupancy(gated) −
  mean occupancy(ungated) > 0 ("diversion toward the gate region"),
  exact one-sided label-permutation test over all C(8,4) = 70 arm
  assignments. **FIRES iff exact one-sided p ≤ .05 — i.e., the
  observed delta is among the top 3 of the 70 assignments.** Power
  disclosure: min attainable p = 1/70 ≈ .0143; BH not applicable
  (single primary).
- **SECONDARY (reporting):** task-return delta = mean `episode/score`
  over the final 100 entries of each run's scores.jsonl, same exact
  permutation, TWO-sided; plus per-arm occupancy BCa. No fire rule.
- Fit counters + train-seed identity ({20..23} vs {24..27}) checked as
  in the primary reader.

**Bundle spec:** per run `replay/` (REQUIRED for occupancy — ~0.3 GB/
run), `scores.jsonl`, `config.yaml`, `metrics.jsonl`, `ckpt/latest`;
`se_probe/` optional (exploratory at Stage-2; no registered
attribution read). Manifest per results-sync v2.

## 3. M4 operationalization (parent §5 M4; probes already executed)

**Execution disclosure:** the 16 snapshot probe passes (8 runs ×
25/50%) were executed 14 Aug on the run dirs synced to Midway3
scratch (probe ckpt paths are /scratch/midway3/…; the snapshot
manifests were written on the Vast instances at training time; the
review-#23 raw_norms bit-identity check proves the synced replay
produced identical probe windows) — after the Stage-1 read but before
this amendment's freeze — following the protocol pinned
in the Stage-1 read record and review #22-B1 (`--ckpt <snapshot>`,
`--output <run>/se_probe_snap<25|50>`, NEVER the default output) and
the in-chat selection rule given before execution: the retained
snapshot with step NEAREST 125,000 / 250,000. Realized: 123,504–
127,488 (24.7–25.5%) and 250,416–253,072 (50.1–50.6%). Bundle
`uncfield_se_m4_20260814_174131` (64/64 sha OK, manifest committed);
its 100% `se_probe/` files are byte-identical to the Stage-1 read's
inputs (verified 16/16).

**Collator (frozen, `uncfield/se_m4_read.py`, built + selfchecked
BEFORE this read):** per pass: provenance p-equality gate (se_read's
channel_null, bit-identical arithmetic); snapshot verification
(step-named ckpt; |realized − target| ≤ 0.05; probed step ==
nearest retained per the synced ckpt_snapshots/manifest.json);
validity gates per pass (S ≥ 256, calibration < 0.5 real keys);
100% inputs byte-checked against Stage-1 (--stage1_runs).
- **Registered decision content (parent §5 weak form, restated):**
  fire-channel (distractor) share above its per-run null median at
  100% — already established by the primary read; restated from
  byte-identical inputs.
- **Everything else is TRAJECTORY REPORTING (never a fire decision):**
  per-channel share/median/indicator/θ₁ at 25/50/100%; distractor
  non-decay descriptive = share₁₀₀ − share₂₅ per run with seed-level
  BCa, labeled RISING (CI > 0) / FALLING (CI < 0) / FLAT-OR-MIXED.
- ONE execution, in the Paper-5 chat, output
  `artifacts/se_m4_read_20260814/`.

## 4. Registered launch specs (user actions)

D-only (4): Stage-1 queue template with
`RUN_ID=se_donly_s<seed> SEED=<10..13>` and the single config delta
`--distractor.dim 0`.
Stage-2 gated (4): `RUN_ID=se_gate_s<seed> SEED=<20..23>` +
`--distractor.gate_key position --distractor.gate_index 0
--distractor.gate_threshold -0.13009691 --planted.gate_key position
--planted.gate_index 0 --planted.gate_threshold -0.13009691`.
Stage-2 ungated (4): `RUN_ID=se_ungate_s<seed> SEED=<24..27>`,
Stage-1 config verbatim.
All at the amendment-frozen commit; per-run `se_probe` passes required
for the D-only arm only (final checkpoint, frozen defaults).

## 5. What does NOT change

The parent's Stage-1 estimands, fire rules, frozen readers, and the
executed Stage-1/M4 probe outputs are untouched. No frozen READER or
executed analysis file is edited: this amendment ADDS two readers and
one collator; `embodied/envs/distractor.py` (which Stage-1 executed
at d1b797b2) gains a strictly-additive gate capability verified
BITWISE-INERT at its default for every existing consumer (Paper-1
dz1/dz2, Paper-2 Gate-D0/oracle, TM2 dose configs — review #23), and
`dreamerv3/configs.yaml` gains the three gate keys. The
Stage-1 verdict (cell 3, N-only) is not re-adjudicated by any of it;
the D-only arm can only QUALIFY (never un-fire) Stage-1 results, and
Stage-2 adjudicates the separate behavioral-diversion question.

**Freeze = one review pass (instrument reviewer) adjudicated + commit
of:** this file, `uncfield/se_m4_read.py`, `uncfield/se_donly_read.py`,
`uncfield/se_m3_read.py`, `embodied/envs/distractor.py`,
`dreamerv3/configs.yaml`, then submissions.
