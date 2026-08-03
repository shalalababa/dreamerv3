# high-f_R falling-limb wave — ONE read record (2026-08-03)

Registered ONE read of the curated high-f_R wave, executed once via the
frozen reader `analysis/highfr_read.py`. Adjudicates the frozen theory
predictions `PREREG_highfr_theory_20260730.md` under
`PREREG_highfr_wave_20260802.md` (freeze `39a15880`) as amended by
Amendment 1 (`be60afbe`, side0 re-target 0.41) and Amendment 2
(`080aa7b2`, `--require_chunks` availability filter).

## Provenance (verified before the read)

- Bundle `local_results/highfr_20260803_113751/`; committed manifest
  `manifests/highfr_20260803_113751.sha256` byte-identical to the
  bundle's `MANIFEST.sha256`; **sha256 sweep 151/151 OK, 0 FAILED**.
- Instrument shas at read time: `analysis/highfr_read.py`
  `dc474355…` ≡ the Amendment-1 registered final;
  `probing/curate_frew.py` `4b14eb47…` ≡ the Amendment-2 registered
  final. All three freeze commits present in history.
- 16/16 fits `OFFLINE_FIT_PROGRESS` = 500000/500000; 16/16 adapts
  `ADAPT_DONE`; all 16 wm configs `expl.mode: task`,
  `task: dmc_finger_turn_hard`, seeds 1–8 per side.
- AUC collation `exclusions.csv` contains ONLY non-wave run ids swept
  from the shared runroot (`adapt_ax1rdesmoke_finger_seed99_ckpt2000`,
  4 xc-arm adapts, 1 speedtest) — no wave run excluded.
- Note: the bundle carries no GIT_HEAD file; provenance rests on the
  committed manifest + sha-matched frozen instruments + the reader's
  hard gates (comparator seed sets asserted exactly).

## Pre-fit disclosables (registered rider quantities)

- Curation criteria honored: targets [0.41, 0.80], tol 0.05,
  fallback_min 0.60, deterministic window rule; availability filter
  ACTIVE: 288 eids excluded (3264 missing chunk files recorded in
  `q1f/frew_pairs.json` availability block).
- Realized instrument occupancies: side1 f_R = 0.7248 (gate
  [0.60, 0.85] PASS — primary cell valid); side0 f = 0.4088 (gate
  [0.36, 0.46] PASS, INCLUDED); separation 0.316.
- Source composition (`q1f/manifest.json`): **source_l1 = 1.0** — the
  maximal skew the registered composition rider anticipated. side0
  mixture = 15 collectors, ALL `/mid` tier; side1 = 16 collectors, ALL
  `/high` tier (incl. `pilot_goal/high` 17). Confound deltas otherwise
  tight (action stats ≤ 0.047, episode length 0, terminals 0).
- Spectral (version `spectral_v1_1_20260724`, 200 eps each):
  diversity_pr side1 **10.2609**, side0 9.6480, frozen ref (v200s1)
  9.2300; spectrum_pr hi 31.74 / lo 38.64 (opposite direction —
  registered descriptive, known behavior).

## Registered verdict (frozen reader, sole consumer of the outcomes)

**SATURATION — compatible, uninformative** (registered branch).

- **P-HF1 primary does NOT fire**: mean[hi-f] − mean[v200s1] =
  **+49.7 [−8.4, +102.8]** (n 8 vs 12, seed-cluster bootstrap B=10K,
  rng0). Not entirely negative — no falling limb.
- **Monotone-rise refutation does NOT fire**: chain
  f 0.3232→187.72, 0.4088→244.65, 0.7248→237.46 is not nondecreasing,
  and the primary CI is not entirely positive.
- **Saturation branch FIRES**: primary straddles 0 AND
  diversity_pr(side1) 10.2609 ≥ ref 9.2300.
- **P-HF2**: membership INTACT — side1 h0 in-regime rew-NLL
  **1.178 [1.119, 1.237]**, CI entirely ≤ 1.5 (member band). But
  diversity_lower = False ⇒ **dissociation NOT adjudicated**;
  inclusion-effect alternative not engaged.
- **P-HF3 weak form HOLDS**: grid {0.054→161.95, 0.3232→187.72,
  0.4088→244.65, 0.7248→237.46}; argmax at f = 0.4088 (interior), not
  at the highest-f cell.
- Registered descriptives: side0 − v200s1 = +56.9 [−19.7, +132.9];
  side0 NLL 0.809 [0.782, 0.837]; per-cell seed maps in `highfr.json`.

## Interpretation (labeled; NON-VERDICT, cannot amend the above)

1. **The predicted magnitude is excluded.** The registration's power
   section predicted a −88 to −108 falling limb; the realized CI
   [−8.4, +102.8] excludes that entire band. At f≈0.72 with this
   composition-included draw, a falling limb of the predicted size is
   ruled out, not merely unresolved.
2. **The mediating variable never moved in the predicted direction.**
   The theory's premise is high f_R ⇒ support starvation (lower
   diversity_pr). This pool's high-f tail is *broader*, not starved:
   diversity_pr 10.26 (side1) and 9.65 (side0) both EXCEED the natural
   buffer's 9.23, and within the wave higher f came with higher
   diversity. The wave therefore could not instantiate the starvation
   antecedent — a manipulation-validity fact about the pool, coherent
   with breadth-mediation (no starvation ⇒ no predicted deficit) but
   supplying it no support.
3. **Both curated cells sit numerically above the natural buffer**
   (+49.7 / +56.9, CIs straddle). The registered composition rider
   bars any f_R-coefficient reading: these are effects of the curated
   draws (tier-pure composition included, source_l1 = 1.0).
4. P-HF2's intact membership (1.18, deep in the member band alongside
   the archived v200s1 anchor 0.827) is consistent with the paper's
   legibility headline: task fits on high-f_R data keep reward
   in-band.

## Consequences (frozen map)

- Straddle + no diversity separation ⇒ saturation: the breadth-patch
  trade-off form is **neither supported nor refuted** by this wave; no
  revision required (the refutation branch did not fire).
- The flagship's support-breadth axis does NOT gain its falling side
  from this wave.
- The outcome-contingent grid-densification option (Amendment 1 scope
  note) does NOT trigger — it required the falling limb to fire. The
  donor-longevity copy decision is moot for that purpose.
- P-HF1 goes to the theory paper as a registered non-confirmation with
  the mediator-not-moved reading attached (interpretation §2).

## Read execution

Registered command (matches the parent registration's ONE-read form):

```
python -m analysis.highfr_read \
  --auc local_results/highfr_20260803_113751/analysis/auc/auc.csv \
  --auc_frozen local_results/volume_repl_20260729_204854/analysis/auc/auc.csv \
  --spectral_lo local_results/highfr_20260803_113751/spectral/spectral_side0.json \
  --spectral_hi local_results/highfr_20260803_113751/spectral/spectral_side1.json \
  --spectral_ref artifacts/volume_repl_20260729/diversity_q1v200_s1.json \
  --e4 local_results/highfr_20260803_113751/e4/e4_finger_v1_hf.csv \
  --output artifacts/highfr_read_20260803/
```

Outputs: `highfr.json` (full), `read_stdout.txt` (console). Executed
once, 2026-08-03.
