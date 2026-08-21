# PREREG — MIN-DOSE reacher kill (algorithm Tier-1), 20 Aug 2026

**Status: FROZEN at user commit. ONE read execution. KILL-scale test** —
registered so a surviving candidate enters the algorithm paper with
confirmatory status and a dead one is cleanly dead. Candidate:
`reviews/Algorithm_Ideation_20260819.md` §3.1 (MIN-DOSE / minimal-dose
alignment, P≈0.35 — two ideation lenses converged on it independently).

**Self-scooping fence (binding)**: this wave belongs to the STANDALONE
algorithm paper. Paper 1 may cite the dose *saturation* as measured
science; no supervision-cost framing, no URLB comparison, no recipe
enters Paper 1.

## Licensing facts (look-0, disclosed)

Finger dose curve (`artifacts/dose_ext_read_20260813/` +
`dose_task_read_20260811/`): occupancy 0.054/0.131/0.249/0.323 → AUC100k
141.70/**285.03**/267.44/244.30; level-1 fires perm p=.0025; G1
replication +102.60 [+24.79, +200.21] p=.0293 — **that confirmatory pass
cleared its Pocock boundary by .0001 and the fresh-only estimate was
+47.6 p=.302**, which is exactly why this kill is required before any
recipe claim. All prior dose evidence is finger-only, curated (not
collected online).

## The kill question

Does level-1-dose reward-aware pretraining, transferred UNFROZEN, clear
scratch in a SECOND domain (reacher)? If not, the recipe has no legs
outside finger and dies.

## Design — 16 cells

- **Dose arm (8)**: build ONE level-1 curated buffer from the frozen
  axis1_reacher/q1 substrate (inventory verified 20 Aug) targeting
  occupancy ≈ 0.13; fit task-mode (reward-aware, house config) seeds 1–8;
  **unfrozen** adapt, house protocol.
- **Scratch arm (8)**: same adaptation protocol from random init, seeds
  1–8. Load-bearing: inventory 20 Aug confirmed NO reacher scratch anchor
  exists anywhere (148 reacher dirs, zero scratch-named, no csv row).
- **Curation feasibility gate (walker lesson — the read bar is an INPUT
  to the search)**: run `build_controlled_replay search-dose` on the
  reacher index FIRST; the buffer must land occupancy in **[0.08, 0.20]**
  (wide because the occupancy scale is domain-specific — disclosed). If
  the constrained pool cannot reach that window, the wave REFUSES
  pre-outcome (walker precedent; no re-sampling until it passes — that is
  gate-shopping). Realized occupancy recorded here by dated edit before
  any fit starts.

## Estimand and decision rule (frozen kill criterion)

- **PRIMARY**: two_sample [dose1_unfrozen − scratch] on AUC100k, n=8 vs
  8, permutation p primary + BCa CI, **α=.05** two-sided.
- **PROCEED** iff CI lower bound > 0. **DROP** otherwise — recorded as
  "the recipe has no legs outside finger"; no seed extension, no second
  look, no re-curation at a different dose (a dose sweep would be a NEW
  registration for the algorithm paper, licensed only by a PROCEED here).
- SECONDARY (labeled): dose arm vs the W0-wave reacher task-arm cells —
  context on where dose-1 sits relative to full-occupancy pretraining.
- MDE note: realized MDE80 printed with any null.

## Gates

Buffer manifest + occupancy witness; fit counters ×8; adapt witnesses
×16; config byte-identity within arms; STRICT modal n_ep;
within-invocation comparisons; refusal on any failure.

## Reader

`analysis/mindose_reacher_read.py` — frozen (selfcheck PASS) **before any
job is submitted**; two_sample machinery (capdescent_read lineage);
literal pins {occupancy window [0.08, 0.20], n=8/8, α=.05}.

## Ops

search-dose feasibility → dated occupancy FILL → build buffer → 8 fits +
16 adapts (~48 GPU-h) → collate + witness → bundle + sha manifest →
**[ME] ONE read**.

---

## FILL (ops, dated 2026-08-20) — the registered draw and its feasibility gate

Reserved by the "Curation feasibility gate" clause above ("Realized
occupancy recorded here by dated edit before any fit starts"). This block
records a measurement; it changes no decision rule.

**Command, as pinned by amendment 1** (run once, RCC caslake job `53746920`,
`ops/waves/mindose_reacher/search_dose.sbatch`; every flag verified against
the subparser before running so the registered draw was not spent on a typo —
`--levels 4 --beam 40 --n_episodes 200 --n_candidates 400 --dirichlet 0.3
--seed 0` are exactly the subparser defaults):

```
python -m probing.build_controlled_replay search-dose \
  --index   $RUNROOT/axis1_reacher/episodes.json \
  --ref_replay $RUNROOT/pilot_goal_reacher_seed1/replay \
  --levels 4 --beam 40 --n_episodes 200 --n_candidates 400 \
  --dirichlet 0.3 --seed 0 \
  --output  $RUNROOT/axis1_reacher/dose.json
```

`--ref_replay` is copied verbatim from the registered reacher q1 search's own
provenance (`axis1_reacher/pairs.json` records
`ref_replay = $RUNROOT/pilot_goal_reacher_seed1/replay`, `n_episodes 200`).

**Realized occupancies**

| level | occupancy | target |
|---|---|---|
| d0 | 0.000000 | 0.0000 |
| **d1** | **0.097837** | 0.0932 |
| d2 | 0.185035 | 0.1863 |
| d3 | 0.273711 | 0.2795 |

`decision: OK`; 378 feasible candidates; max pairwise `dcov` 0.0109 against
`cov_tol` 0.0144; overlap 0.11 against the 0.2 cap.

**Gate outcome: PASSES.** Level-1 occupancy **0.097837 ∈ [0.08, 0.20]**, so
the wave does NOT refuse pre-outcome. No re-draw was performed and none is
admissible (a second draw would be a second look at this gate).

**Holdout disposition**: `--holdout` omitted, per amendment 1 — the flag takes
E4 probeset manifest paths and no reacher probeset exists (`e4_probesets`
holds cup/finger/synth only). The obligation is inverted and registered: any
future reacher probeset measured on these fits must exclude this buffer's
episodes at probeset-build time. The buffer manifest carries
`holdout_disposition: "no-probeset-exists"`.
