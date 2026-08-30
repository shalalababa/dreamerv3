# EVA-at-scale (se_eva_probe rev 3) — 29 Aug 2026: the registered
primary is CONSTRUCT-INVALID; five certificate-backed findings land
instead (engineering-mode read; full adjudication in
ADJUDICATION.md, every number re-verified against the 12 jsons)

RCC job 56095696, 12/12 cells, zero control refusals (the rev-2 job
56089362 refused 7/12 on a gate-multiplicity defect — measured 85.6%
projected false-refusal — fixed in rev 3 by holding controls to the
same pooled-e-value standard as findings; rev-2 outputs quarantined
_REV2_*; science quantities proven identical pre/post fix).

## The construct failure (the honest headline)

The primary (Dlogrho legC−legP on a justified↔unjustified bracket)
CANNOT measure the theta_1 misprice: both legs share the realized-
error numerator R by design, so the statistic is a pure S-ratio
effect; the "fully justified" endpoint is unreachable for any
vens>0, and pb≈1 is the GENERIC value (simulated: uncorrelated
0.983 / tracking 1.356 / anti-tracking 0.682). Checkable from code
alone — review B1 fixed the endpoint's magnitude, not its content.
**EVA does NOT certify the theta_1 misprice.** The surviving
exploratory residue (pb−1 as a vens-timing statistic; det
0.852–0.992 vs gauss 0.975–1.006) needs its own registration.

## What IS certified (12/12 unless noted)

1. **The distractor's h-step predictive is ~2.1× too NARROW —
   certified UNDER at pooled log10e 1.6e4–3.8e4, FLAT in h**
   (2.25/2.14/2.13/2.13; within-cell spread ≤0.12 vs the 0.33 fall
   an AR-misclaim requires): a STATIONARY-SCALE error, not a
   persistence misclaim. The WM is overconfident about the
   distractor in predictive space.
2. **Position is 14–30× over-dispersed (log10e 1.2e5–1.7e5)** with
   vens_share 0.3–1.2% — a LIKELIHOOD finding: the homoscedastic
   symlog-MSE head cannot span channels whose realized
   predictability differs ~31×. Both legs running is what separates
   this from any ensemble story.
3. **Instrument resolution validated by the bitwise dup**: dup0 ==
   position to 3–4 s.f. in 11/12 (the one inversion 45× inside its
   own floor).
4. **Imagination/replay dissociation at deployed scale**: L3's
   persistence miscalibration does NOT appear in realized
   innovation ratios (prediction (d) failed 0/12, null comparators
   move more) — reversed polarity vs the in-world protocol finding.
5. **No gauss repair signal in innovation space** (theta1 det 5.258
   vs gauss 5.230; rho_d if anything higher on gauss) — an
   independent corroboration of B4's no-repair verdict (per B10
   this does not test the gauss REWARD, which normalizes in latent
   space).

Predictions as registered: (a) FAILED 12/12 (opposite direction);
(b) FAILED 12/12 (the vel-pos null comparator exceeds its bar by
200–38,845×); (c) MET 11/12; (d) FAILED 12/12. Controls clean.

## What EVM's paper can and cannot say

CAN (certificates attached): per-channel likelihood miscalibration
at deployed scale; the distractor's certified 2.1× under-dispersion,
flat in h; the measured instrument floor; the imagination/replay
divergence; no repair signal across arms. CANNOT: any certified
statement about theta_1 itself; "the distractor claim is
over-dispersed" (it is the reverse); L3's persistence number as
replay-confirmed; anything about the gauss reward. A v2 primary
(per-member realized errors so R responds to vens) is the designed
follow-up if a certified misprice statement is still wanted —
GO-gated, not assumed.

# REV-7 FINAL READ (29 Aug, job 56197623, 12/12, zero refusals; the
program's last computation). Adjudicated via analysis/se_eva_read.py
+ per-cell verdicts (adjudication.json + rev7/*.json).

**Primary (q_hat with the gamma_hat attribution, registered bars):**
distractor verdicts = 8 MIXED, 1 SHARED-DOMINATED, 3
UNRESOLVED-IV-DISAGREEMENT (the honestly-previewed near-bar rows);
controls_pass 12/12, resolved 12/12 (lambda_hat 0.65-0.86 — the
instrument is strong everywhere). Raw q_hat on the distractor
0.95-2.39: a real V-proportional excess (the ensemble-mean error
scales with disagreement 1.3-2.4x beyond the coherent rate). After
the gamma_hat attribution (the reconstruction-borne share),
**q_hat_net = 0.53-1.63, centered ~0.6-0.9**: the deployed
ensemble's disagreement PARTIALLY tracks its error on the distractor
— a MIXED account; no cell certifies fully-earned (SPREAD-ACCOUNTS
0/12), one certifies shared-dominated. **A5 (timing): REPLICATES**
— pb median det 0.937 < gauss 0.992, exact MWU p=0.00808 <= 0.05,
with the F3-filtered pooling and the q_hat conjunction. **A6:
REPRODUCES** — 0/3456 bitwise mismatches on the rev-3 certified
quantities, zero floor-band violations; the comparator's
"instrument-alarm" label came from 3 h15 resolution-gate BOOLEANS
derived from synthetic floors that moved within the DECLARED ±0.05
band (the per-audit reseeding's known edge) — a spec inconsistency
(derived booleans can't be held exact over banded inputs), noted,
not an instrument perturbation. All rev-3 certified findings stand.

**What EVM carries from the EVA line, final:** (1) the certified
per-channel likelihood miscalibration (distractor 2.1x too narrow,
flat in h; position 14-30x too wide) — rev 3, reproduced bitwise;
(2) the measured q_hat/gamma_hat decomposition: the distractor's
disagreement-scaling excess is real but 30-60% reconstruction-borne,
verdict MIXED (the ensemble partially earns its disagreement);
(3) the replicated det-vs-gauss timing split (p=0.008); (4) the
imagination/replay dissociation; (5) theta_1 itself remains
DESCRIPTIVE/uncertified — stated plainly. Instrument lineage: 7
revisions, 3 adversarial reviews, 2 construct rebuilds, all
documented in eva_scale_rev6/rev7 archives + PREDICTIONS.md.
