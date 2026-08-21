# PREREG AMENDMENT 4 — SE bootstrap-decomposition arm (NOBOOT), 20 Aug 2026

**Parent:** `PREREG_nfi_scale_exhibit_20260812.md` v2.2 + Amendments
A1/2/3. **Context:** review #25 addendum A2 (verified against
`explore.py:35-43` and the shipped run configs) found
`disag_bootstrap: True, prob 0.8` undisclosed and mechanistically
load-bearing: every ensemble member fits an independent random 80%
subsample, which manufactures a persistent aleatoric floor in a
variance bonus **by construction** — a live alternative account of
BOTH headline facts (the never-decaying 5.26× misprice; duplicates at
par). Plan2Explore's published ensembles differ by INITIALIZATION,
not resampled data, so the noboot configuration is arguably MORE
standard than Stage-1. User GO 20 Aug. This arm measures the
bootstrap's share of the misprice before a referee asks.

**Timing disclosure:** post-Stage-1 outcomes; motivated by a code
fact (the flag), not by any outcome number. Bands below are analyst
choices, disclosed as such, anchored only to already-registered
quantities (Stage-1 θ₁ 5.26 [5.02, 5.46]; the Amendment-2 par band
[0.90, 1.10]).

## 1. Arm

Stage-1 config VERBATIM except **`--agent.expl.disag_bootstrap
False`** (members then differ by initialization only — the published
P2E convention). Seeds **40–43**, RUN_IDs `se_noboot_s40..43`, 4 runs
× 5e5 steps, same wave/site discipline as prior arms. Producer: the
Stage-1 template + env override `DISAG_BOOTSTRAP` (default True =
Stage-1 hardcoded behavior, inert; same pattern as prior overrides).
Per-run `se_probe` at the final checkpoint (frozen defaults, default
output, one device class) — the §5-shaped bundle: replay/, scores,
config, metrics, ckpt/latest, se_probe/.

## 2. Registered read (frozen reader `uncfield/se_noboot_read.py`,
built + selfchecked BEFORE the arm's compute; ONE execution)

Per run: the frozen primary-reader gates verbatim (provenance
p-equality, S ≥ 256, calibration < 0.5, final-ckpt, train seeds
{40..43} distinct, config pins incl. `disag_bootstrap == False`,
dim 8 / scale 1.0 / basesd 1.215 / no gates / no mod, plus the
Stage-1-verbatim pins `expl.mode == p2e`, `disag_ens == 8`,
`disag_scale == 1000.0`, `planted.basesd == 0.0976`,
`planted.source_key == position` — #26 m11).

- **DEGENERACY GATE (review #26 B1 — adjudicated BEFORE the bands):**
  an initialization-only ensemble trained on identical data can
  collapse disagreement globally; scale-free ratios (share, θ₁, p)
  cannot see that, and a whole-instrument collapse must never be read
  as a distractor-specific result. Registered guard, computed from
  absolute levels already in the outputs: per noboot run, (i) the
  mean REAL-dim per-dim attribution (dims_d, data-variance-normalized
  ⇒ cross-run comparable) and (ii) `pse2.intrinsic_mean` must each be
  ≥ 1/10 of the Stage-1 per-run MINIMUM of the same quantity (loaded
  live from the Stage-1 bundle via --stage1_runs; the 10× generosity
  absorbs device-class level sensitivity, disclosed). Any run failing
  either → verdict **GLOBAL-DISAGREEMENT-COLLAPSE** for the read: a
  different physical result (init diversity alone does not sustain
  the disagreement signal), bands NOT adjudicable, all levels
  reported. Final `train/expl/disag_replay_rew` from metrics.jsonl
  reported alongside (descriptive).
- **PRIMARY (registered): distractor θ₁ seed-level BCa (n = 4),
  banded — θ₁ recomputed from the npz per-dim attribution and
  asserted equal to the stored value (provenance gate on the primary
  statistic, review #26 M4):**
  (a) **MISPRICE-SURVIVES-LARGE**: CI entirely > 2.0 AND per-run
      permutation p < .05 in 4/4 → a large misprice survives the
      init-only ensemble; the "standard Plan2Explore-style
      (initialization ensemble)" attribution is earned FOR THE
      FINAL-CHECKPOINT MISPRICE (a persistence-under-noboot claim
      would need its own M4-style snapshot read — not registered
      here). Registered caveat (#26 M7): cell (a) is COMPATIBLE with
      a substantial bootstrap share (e.g., CI [2.05, 2.35] would be
      ~73% of the excess-over-par removed); the point estimate is
      always the headline number. Per-run p caveat: dim-level p is
      the parent's registered anti-conservative per-run evidence.
  (b) **BOOTSTRAP-DRIVEN**: CI entirely ≤ 1.10 (par band's upper
      edge) OR (≤ 1 of 4 runs at p < .05 AND CI ≤ 2.0 — the count
      disjunct alone must not fire while a large misprice survives,
      #26 M3) → the misprice is substantially a bootstrap artifact ⇒
      MAJOR scoping correction (headline re-scoped to
      bootstrap-masked ensembles).
  (c) **ATTENUATED / MIXED**: anything else → the point estimate
      QUANTIFIES the bootstrap share; descriptive, both accounts
      partially right.
  Band disclosure: 2.0 is an analyst choice ("a large misprice
  survives"), NOT derived; the honest reporting is the CI itself,
  and cell (c) exists so no outcome forces a binary. "CI ≤ 1.10" is
  implemented as an upper-bound check (shares are non-negative, so
  the lower edge is guaranteed).
- **SECONDARY (registered, same read): dup0 θ₁ BCa vs the par band
  [0.90, 1.10]** — the A2 account also attributes duplicates-at-par
  to bootstrap (all subsamples agree on predictable targets); an
  initialization-only ensemble could disagree on duplicates. Par
  inside the band → the parity result is bootstrap-independent too;
  outside → parity re-scoped.
- **Reporting (unconditional — emitted on EVERY branch incl.
  NOT-ADJUDICABLE and collapse, #26 M6):** per-run shares, θ₁ for all
  channels, calibration, fit counters, per-key absolute attribution
  levels, intrinsic_mean, disag_replay_rew; n = 4 power disclosure
  (a decomposition measurement, not a fire test). The primary's
  p-count field is named `n_p_lt_05` (the house term "indicator"
  keeps its share-vs-median meaning, #26 m8). Comparison rows vs
  Stage-1 and the flat arm are assembled DESCRIPTIVELY in the read
  record post-hoc (#26 m15), except the Stage-1 level minima, which
  the reader loads itself for the degeneracy gate.

## 3. What does NOT change

No prior read, reader, or verdict is touched. If (b) fires, Stage-1's
numbers remain what they are — measured properties of the deployed
repo objective — but the "standard P2E" attribution narrows, exactly
as review #25 A2 anticipated. Freeze = review #26 adjudicated
(1B/6M/8m ALL adopted — the blocking global-collapse guard, the
conjunctive (b) disjunct, the θ₁ provenance gate, the (a) rename +
final-ckpt scoping, unconditional reporting) + commit of: this file,
`scripts/uncfield_se.sbatch` (DISAG_BOOTSTRAP override; env value
must be capitalized `True`/`False` — elements flags reject
lowercase), `uncfield/se_noboot_read.py` — then the 4 submissions.
