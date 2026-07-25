# PREREG: spectral domain-contrast measurement (frozen 2026-07-24)

Registers the cup-vs-finger spectral measurement pass
(`probing/spectral_measure.py`, selfcheck PASS) and its predictions,
BEFORE any measurement on real buffers exists. This closes the
standing reviewer hole: WHY does cup transfer reward-free (Phase 5a;
apt +213) while finger requires the reward-gradient path (three
reward-free nulls; rgo carrier)? The spectral-competition account
says it is a λ-spectrum fact; this pass makes that measurable.

## Disclosure / epistemic status

The behavioral cup/finger outcomes are fully known at freeze (public
to us since Phase 5a / W-chain). What is frozen here is the
MEASUREMENT-level prediction, before any spectrum has been computed:
the pass is a consistency check for the known domains and a genuine
out-of-sample prediction for pixel (X2 running, UNREAD at freeze).
No spectrum, rank, ratio, or diversity number for any real buffer
exists at freeze.

## Instrument (pinned)

`probing/spectral_measure.py measure` per frozen buffer: H=8 windowed
symlog'd proprio frames (the encoder's own input transform; per-dim
standardization deliberately NOT applied — variance salience is the
measured quantity), windows never crossing episode starts; Σ
eigenspectrum; reward direction θ from out-of-fold ridge (folds by
chunk, ridge 1e-3·tr/D); λ_need = θᵀΣθ; VARIANCE RANK = #eigenvalues
above λ_need; ratio profile λ_need/g_k at cuts {4,8,16,32,64};
f_rewarded; diversity index = participation ratio of rewarded-frame
covariance; max 500K steps uniform chunk subsample, rng 0. Targets:
the frozen W-chain collector buffers, cup and finger, hi and lo
sides (all retained buffers per side; median across buffers is the
cell statistic).

## P-SM1 — PRIMARY (domain contrast; `compare` implements it)

For EVERY side measured in both domains: median variance rank of the
reward direction in cup < in finger. Confirmed iff it holds on all
matched sides. This is the registered spectral form of "cup's needed
features are high-salience (included for free); finger's are low-λ
(need reward amplification)."

## Registered secondaries (descriptive)

- Ratio profiles λ_need/g_k per domain × side across all cuts
  (expected: cup ≥ finger at every cut; the >1/<1 absolute form is
  NOT registered — the RSSM's effective k is unknown).
- f_rewarded per side (finger: f ∝ occ, corr +1.0 known; cup:
  reward-rich lo side known — recorded for the a²=s²f bookkeeping).
- OOF R² per domain (frame-level reward learnability; expected low
  for sparse rewards — level is bookkeeping, not adjudication).
- Diversity index per buffer — feeds P-E4(b)
  (PREREG_compcapacity_theory_20260724.md): conditional on occ,
  diversity predicts transfer; adjudicated there, measured here.

## Out-of-sample extension (registered, adjudicated by X2)

Pixel domain prediction (measurement extension deferred; the X2
BEHAVIORAL read is the adjudicator): pixel input raises competing bid
mass (g) and buries λ_need ⇒ the occupancy × supervision interaction
is LARGER in pixel than proprio (this is P-C2/X2, already frozen —
restated here only to bind it to the spectral account; no new
statistic).

## Consequence map

- P-SM1 confirmed ⇒ the cup/finger boundary enters Paper 1 as
  predicted structure (spectral moderator λ_j/g_k), closing the
  domain-asymmetry hole; W2 cup seed-extension stays SKIPPED (the
  account explains the weak cup interaction).
- P-SM1 fails ⇒ the spectral account of the DOMAIN boundary is
  refuted as registered; Paper 1 must present cup/finger as an
  unexplained scope condition (honest fallback), and the theory's
  cup section (E5) loses its measurement support. The arm-level
  results (interaction, shuffle, sgb/rgo) are untouched — they do
  not depend on the cross-domain account.

## Execution

Login node / CPU, against the frozen buffers on cluster scratch;
one json per buffer + one `compare` execution once all cells are
measured. Freeze-commit this file + the script BEFORE the first
`measure` invocation on a real buffer.
