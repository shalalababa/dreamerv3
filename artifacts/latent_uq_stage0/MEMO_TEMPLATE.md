# Latent-UQ Stage 0 evidence memo — {YYYY-MM-DD}

<!--
Template for the Stage 0 deliverable (Latent-UQ idea brief Sec. 6; Gate D0
calibration addendum, EVPI theory note App. A.4 item 5 / A.7). Generated
automatically by `python -m probing.latent_uq_analysis` into
`artifacts/latent_uq_stage0/<tag>/memo.md`; this file documents the
required fields. Do not fill by hand except the human-signoff section.
-->

**Replication verdict (Biased Dreams, arXiv 2604.25416): {REPLICATES /
DOES NOT REPLICATE / MIXED-AMBIGUOUS + one-line statement}**

## Protocol

- Dumps: {label=dir per cell; cross-ensemble members flagged}
- Probe set(s): {probeset_id(s)} (frozen, hash-verified at dump time;
  a NOT-HELD-OUT flag appears here when any probed run's own replay fed the
  probe set — such error reads are in-distribution and must not gate)
- Horizons: {[1, 5, 15]} (open-loop, decoder target space); headline read:
  anchor one-step disagreement vs horizon {5} error; addendum read: anchor
  one-step disagreement vs one-step (h=1) error (pre-registered, App. A.4
  item 5; always computed regardless of --primary_horizon)
- Density proxy: mean kNN distance of anchor posterior means against
  training-buffer encodings (larger = sparser); bias signature = positive
  partials with disagreement
- SEs: stream-clustered bootstrap, n_boot={N}; verdict threshold {2} SE on
  Delta = pcorr(D,E|rho) − pcorr(D,rho|E)

## Cell: {mode_domain_seed | cross label}

| checkpoint | kind | h | rho(D,E) | rho(D,rho) | pcorr(D,E\|rho) | pcorr(D,rho\|E) | Delta (SE) | verdict |
|---|---|---|---|---|---|---|---|---|
| {step} | {anchor/path} | {h} | {+.3f (se)} | {+.3f (se)} | {+.3f} | {+.3f} | {+.3f (.3f)} | {TRACKS_ERROR / TRACKS_DENSITY / AMBIGUOUS} |

Final-checkpoint verdict (anchor, h={5}): **{verdict}**
Stage-0 calibration addendum (one-step read per EVPI note App. A.4 item 5
— pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the final checkpoint,
anchor h=1; the App. A.7 gate criterion additionally applies the dose
adjustment, evaluated in the D0 pipeline, not here): **{PASS/FAIL}**

<!-- one section per cell: within-checkpoint disag-head cells (p2e /
disag_task runs) and cross-seed or refit ensembles (--cross). -->

## Interpretation notes

- `TRACKS_DENSITY` at a cell = the attractor bias replicates there:
  disagreement is explained by training density beyond what error explains.
  Route per the brief: Stage 1 fixes become the priority and the D0
  ensemble read is suspect.
- Per-window reads are never reported; all claims are aggregates with
  stream-clustered uncertainty.
- kNN density in high-dim latent spaces is itself a proxy (brief, open
  questions); a TRACKS_DENSITY verdict should be sanity-checked against a
  second k before externalizing.

## Files

- analysis: `{output}/analysis.json`
- command: `{full command line}`
- git commit: `{sha}`

## Human signoff (fill by hand)

- Read by: {name, date}
- Replication verdict accepted / amended: {…}
- Consequences recorded in plan/runbook: {list of edits, or "none"}
