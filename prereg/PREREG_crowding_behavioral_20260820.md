# PREREG — Crowding/rescue behavioral leg (σ=2 only), 20 Aug 2026

**Status: FROZEN at user commit. ONE read execution. First behavioral
look on this chain.** Motivation: `reviews/PreSubmission_Synthesis_
20260819.md` §4.3 — both `PREREG_sigma_ladder_20260814` and
`PREREG_rescue_load_20260815` registered NO behavioral primary, so the
crowding→rescue mechanism (§16 centrepiece) currently has zero
demonstrated transfer consequence. Two review lenses independently named
this the gap. It also gates three algorithm candidates (stamp-canary,
crowding governor, saliency reclamation — Algorithm_Ideation §4.2).

## Relation to the prior registrations (stated, not overridden)

`PREREG_rescue_load_20260815` declared its own adapts DESCRIPTIVE-ONLY
("must never be read as behavior under w_r=100"). That restriction bound
THAT registration's read, which had no behavioral estimand and no
behavioral controls. This registration asks the behavioral question as
its own registered estimand with its own paired control: both arms share
the identical σ=2 substrate and adaptation protocol; the only difference
is the pretraining loss weight w_r (1 vs 100). The prior prereg's frozen
text stands unedited; nothing there is reinterpreted.

## Trim disclosure

σ=8 is EXCLUDED by design: at σ=8 the ladder arms have merged (task
0.6786 vs apt 0.6751 probe-level), so a behavioral contrast there is an
expected null carrying no information about the rescue. σ=2 — where the
probe-level effects live (excess task loss +0.339*, rescue +0.2144
[+0.139, +0.264] 8/8) — carries the question alone. Decided before any
behavioral data exist.

## Design

- Cells: **16 unfrozen adapts** from the EXISTING 16 rescue-wave fits
  (task arm, q1_nzs2 σ=2 substrate, w_r ∈ {1, 100} × 8 seeds; fits are
  archived registered outputs — no new fits). Standard adaptation config
  (the adapt-time config carries NO w_r manipulation; w_r was a
  pretraining-time loss weight only).
- INVENTORY GATE (pre-submission FILL): ops enumerates the 16 fit dirs
  with witnessed counters and records run_ids + ckpt steps here by dated
  edit. Any missing fit ⇒ that seed-pair is dropped symmetrically and the
  realized n recorded before outcomes exist.

## Estimand and decision rule

- **PRIMARY**: paired [AUC100k(w100) − AUC100k(w1)] over seed-pairs
  (n=8), permutation p primary + BCa CI, **α=.05** two-sided. FIRE with
  positive sign ⇒ the β-side rescue has a behavioral consequence — the
  crowding chain gains its transfer leg. FIRE negative ⇒ registered
  surprising reversal (report as such). No fire ⇒ the mechanism claim
  stays probe-level; the paper's §16 language must continue to say
  representation-level only, and the three downstream algorithm
  candidates are gated CLOSED.
- SECONDARY (labeled): both arms vs the scratch-AUC anchor 147.6823 —
  context only, no α.
- MDE note: with n=8 the read reports realized MDE80 alongside any null;
  a null with MDE80 wider than the w1→w100 probe-level gap is reported
  NO-CALL-UNDERPOWERED, not evidence of absence.

## Gates

Adapt counters/witnesses ×16; fit↔adapt linkage (adapt config's
from_checkpoint path must resolve inside the paired fit dir; ckpt sha
recorded); STRICT modal n_ep; within-invocation pairing only; refusal on
any failure.

## Reader

`analysis/crowding_behavioral_read.py` — frozen (selfcheck PASS) **before
any job is submitted**; one_sample paired machinery (domains_read
lineage); literal pins {147.6823, n=8 pairs, w_r values {1,100}}.

## Ops

Inventory FILL → 16 adapts (~50 GPU-h) → collate + witness → bundle +
sha manifest → **[ME] ONE read**.
