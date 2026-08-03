# PREREG: breadth-causal predictions — diversity manipulation at matched f_R (frozen 2026-08-03)

Freezes the spectral-competition model's predictions
(`Theory_SpectralTransfer_20260717.tex`; λ-structure input registered
after the volume read, `PREREG_volume_repl_20260726` consequence) for a
wave that manipulates SUPPORT BREADTH (diversity_pr) directly at
MATCHED rewarded-regime occupancy — the causal test the high-f_R wave
could not deliver (its read, `artifacts/highfr_read_20260803/`,
returned SATURATION with the mediator unmoved: the pool's high-f tail
came out broader, not starved, so the breadth channel went untested).

This file is committed BEFORE any feasibility scan, moments cache,
diversity-targeted search output, curated buffer, fit, or adaptation
run for this design exists anywhere. The feasibility machinery itself
(per-episode moment scan + deterministic diversity search) is being
built alongside this freeze but has NOT been executed against the real
index; no pool spectral quantity beyond those already published
(v200s1 9.2300, v400s1 10.63, q1f sides 9.6480/10.2609) is known.

## Design skeleton (constants fixed at wave registration, not here)

Two disjoint 200-episode cells curated from the frozen Axis-1 finger
episode index (`$RUNROOT/axis1_finger/episodes.json`, the same pool as
q1/q1v200/q1v400/q1f), BOTH matched to the natural comparator's
occupancy level (target f_R ≈ 0.32, the archived `ax1v2q1v200s1`
level) and to each other, differing as widely as the pool allows in
diversity_pr (participation ratio of the rewarded-frame symlog-proprio
covariance, `spectral_v1_1_20260724` — the SAME frozen instrument that
gated every prior curated wave):

- **side0 = lo-div** (starved support at natural legibility),
- **side1 = hi-div** (broad support at natural legibility),
- task arm, seeds 1–8 per side, IDENTICAL protocol to the
  v200/v400/q1f cells (default Axis-1 offline fit 500k, frozen-readout
  adapt, AUC100k), plus the frozen E4 h0 membership pass.
- Comparator anchor: the archived pooled-12 `ax1v2q1v200s1` cohort
  (AUC 187.72, diversity_pr 9.2300, E4 h0 rew-NLL mean 0.827) sits at
  the same f_R by construction — a third, natural-diversity point on
  the breadth axis at matched legibility.

## Frozen predictions

- **P-BD1 (PRIMARY, causal breadth effect):** at matched f_R and
  matched 200-episode volume, transfer is INCREASING in support
  breadth: mean AUC100k[hi-div] − mean AUC100k[lo-div] **> 0** (CI
  entirely positive under the same two-sample seed-cluster bootstrap
  machinery as every prior wave). Direction only; no magnitude is
  frozen (see disclosure).
- **P-BD2 (secondary, anomaly attribution — weak ordering form):** the
  natural cell sits BETWEEN the manipulated cells on the transfer
  axis: mean[lo-div] < mean[ax1v2q1v200s1] ≤ mean[hi-div] (point
  ordering; the wave registration will state its decisional form).
  This is what converts the volume anomaly's diversity DIRECTION
  (P-E4b, correlational) into a diversity ACCOUNT: breadth moves
  transfer at fixed occupancy and fixed episode count.
- **P-BD3 (mechanism dissociation):** reward membership stays INTACT
  in BOTH cells — E4 h0 in-regime rew-NLL CIs entirely ≤ 1.5 nats
  (the registered member band; anchor 0.827 at the same f_R). Matched
  f_R is the design's control for legibility, and P-BD3 is its
  manipulation check on the outcome side: a lo-div deficit WITH intact
  membership is a support effect, not an inclusion effect.
  **Registered alternative:** if the lo-div cell leaves the member
  band (CI entirely ≥ 2.0), the manipulation leaked into legibility —
  the deficit is then reported under the inclusion reading and the
  pure-breadth mediation claim is NOT licensed by this wave.

## Frozen failure modes (each is a registered outcome, not an anomaly)

- **Informative NULL:** P-BD1 CI straddles 0 at an instrument-verified
  diversity separation that clears the wave's registered minimum ⇒
  diversity_pr is NOT a causal carrier of transfer at matched f_R in
  this regime; the λ-input reading of the volume anomaly loses its
  principal support and the anomaly requires a non-breadth account
  (episode count per se, optimization dynamics, or composition).
  Theory consequence: revision of the λ-structure interpretation —
  the inclusion machinery (established by the gradient-arm chain) is
  untouched.
- **REVERSED:** P-BD1 CI entirely negative ⇒ the λ-monotonicity form
  of the breadth claim is REFUTED outright (stronger than the null).
- **Manipulation-validity HALT (the high-f_R lesson, now an explicit
  gate):** the wave registration MUST fix, before any fit exists,
  (a) a minimum instrument-verified diversity_pr separation between
  the built cells and (b) a maximum |Δ f_rewarded| between them (and a
  band around the 0.32 target). Realized values outside these gates ⇒
  the wave HALTS pre-fit and a dated amendment is required — a
  mediator that fails to move is a registered non-event, never an
  outcome.
- **Composition rider (inherited from the f_R wave, binding):**
  diversity-sorted selection from a pooled index will be
  collector/tier-skewed, possibly maximally (source_l1 → 1). Breadth
  and composition-as-realized are inseparable in this design by
  construction; a fired P-BD1 is reported as the effect of the curated
  breadth draw (composition included). Mechanism attribution rests on
  P-BD3 plus the matched-f_R control; any composition-matched
  follow-up is a NEW registration. Builder manifest `mixture` +
  `source_l1` are disclosed pre-fit.

## Magnitude and power (disclosure only; nothing frozen)

Known anchors, both confounded: the volume anomaly pairs Δdiversity_pr
≈ +1.4 with +331 AUC (but 2× episode volume); the f_R grid pairs
Δdiversity_pr ≈ +0.6 (9.65→10.26) with ≈0 AUC change (but Δf = 0.32
and tier-pure composition flip). Neither licenses a magnitude
prediction at matched f_R and matched volume — the wave registration
will state an MDE against the archived comparator variance (sd ≈ 77 at
n=12) and the realized separation, and the power statement will be
made against-interest as before.

## Value-blindness and ordering

- The feasibility pipeline (per-episode rewarded-frame moment scan +
  deterministic search) consults chunk files' proprio frames, rewards,
  and the index — pool-structure quantities only; no fit, no
  adaptation, no transfer outcome exists or is revealed. Its outputs
  (achievable separations, candidate memberships, realized occupancy
  means) are pre-fit disclosables in the same class as the f_R wave's
  curation quantities, and they MAY inform the wave registration's
  gate constants (exactly as the high-f_R pool structure informed
  Amendment 1's re-target). They cannot bias the predictions above,
  which are frozen here first.
- Ordering: commit THIS file → feasibility scan + search on the RCC
  login node (`--require_chunks`-style availability filtering
  mandatory — the pool has known purge gaps) → IF a qualifying pair
  exists: wave registration (gates, seeds, run names glob-checked
  against `runroot_cleanup.sh` DELETE-SAFE, commands, ONE reviewer —
  pre-approved by the user 2026-08-03) → freeze-commit → build →
  spectral instrument gates → fits/adapts → E4 → ONE read. IF no
  qualifying pair exists: the infeasibility is recorded with the pool
  diagnostics and this freeze stands as an untested prediction set (no
  outcome exists).
- Known at this freeze: everything through 2026-08-03 including the
  high-f_R saturation read, the volume replication read, the archived
  comparator statistics quoted above, and all prior published cell
  means. None of these is an outcome of any breadth-manipulated cell;
  no such cell exists.
