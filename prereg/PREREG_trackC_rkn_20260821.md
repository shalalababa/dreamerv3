# PREREG — Track C: RKN-class coherent belief operator + CEI audit
pricing, 21 Aug 2026
(review #33 adjudicated same day: 6B/10M/9m ALL applied, including a
THESIS CORRECTION and a re-registered primary)

**Program:** Track C of `Plan_FollowupPrograms_20260821.md`;
standalone-paper track. Ownership: RKN architecture = Becker et al.
ICML 2019 — claimed as a HARNESS CELL + audit-pricing methodology.

**PRE-COMPUTATION DISCLOSURE (binding, #33):** review #33 executed
the registered recipe on the SEED-0 member (training on the real
pilot2 episodes, the full search, an audit pass) as part of
adversarial verification. The seed-0 outcomes are therefore KNOWN
before this freeze and are labeled PRE-COMPUTED below (the house
P-MC2 convention); members 1–3 have never been trained and remain
predictions. Known seed-0 facts: verdict EXPLOIT-SURVIVES-CIG-ONLY,
n_exploit_cig = 332, **n_exploit_pbim = 0, n_exploit_both = 0**;
q_hat ≈ [.0030, .0033, .0034, .0588, …] (logq moved exactly the adam
budget lr×steps = 3 log-units from init); q_hat[3] never descended —
**z3 is read by NO sensor and is unidentifiable** (disclosed);
heldout obs-NLL 0.43 (as-optimized); trained-member audit gammas
≈ +0.07…+0.27.

## 1. Corrected thesis (#33 F2)

Coherence-by-construction removes the promise-vs-OWN-license
inconsistency channel (the ratio/min-clip class: frac_inconsistent
≡ 0, license ≥ promise structurally) and makes the REALIZED-ΔH
farming conjunction unreachable — the covariance path converges to a
periodic orbit, so net realized entropy drop per steady loop ≈ 0,
killing the carried_dh (PBIM) tier and hence the flagship CIG+PBIM
conjunction. Coherence does NOT prevent honest-wrong carried-EIG
promising: q_hat > 0 on a truly static latent keeps the operator
promising gain on exhausted sensors forever — the COHERENT-LIAR
cell, which is exactly what CEI's interventional audits price.
Review #33 measured the fire threshold of the naive claim:
NO-EXPLOIT requires q_hat_static ≲ 1e-5 (ladder: 2.5e-3→293
exploits, 1e-4→1, 1e-5→0), ≈2.8× the registered optimizer budget —
so CIG-tier occupancy is the EXPECTED cell, not a failure, and the
old "NO member is tier" primary is arithmetically unreachable
(withdrawn).

**Structural degeneracies (disclosed + asserted in code, #33 F12):**
observation-free covariance ⇒ naive_eig ≡ carried_eig (the
EXPLOIT-NAIVE-ONLY verdict is unreachable; the verdict-distribution
comparison to the GRU bank must note the reduced reachable set);
licensed_rate's pure-update Bayes gain structurally EXCEEDS the
predict-inclusive carried_eig (equality was never the right claim,
#33 F5).

## 2. Operator + certification (selfcheck PASS 21 Aug, post-#33)

`uncfield/rknwm.py` `RKNModel`; training = adam (batch **32** = the
GRU anchor's, #33 F10; lr 1e-3; 3000 steps; float32 — recorded,
#33 F14) on the pilot2 episodes with the learnedwm per-step loss
through exact Kalman recursions. **Train/deploy identity (#33 F1):
the final A is capped ONCE by the training-time power-iteration
estimate and stored as A_eff; RKNModel consumes it verbatim** (the
old exact-SVD recap at load silently shrank A by 1.24% on the seed-0
member and cost 35%/63% heldout NLL — review-measured, fixed).
Certified: oracle equivalence (200 real steps); structural coherence
(planner _eig == realized ΔH at RANDOM parameter points, any y);
pure-update commutativity; diagonal-lens consistency exact on
one-hot sensors + the s4 overlap infinite-license artifact pinned;
**action-based overlap detection** (#33 F4: the name-substring check
missed every `_all` cycle through node 0 — 20/20 of the seed-0
top-20 were mislabeled); A_eff cap identity; power-iteration mirror
vs SVD on separated spectra; audit battery (below); an END-TO-END
tiny driver pass writing all four jsons (#33 F13).

## 3. Registered claims (members 1–3 = predictions; seed-0 rows
PRE-COMPUTED per the disclosure)

- **P-RKN1 (consistency, corrected form #33 F5/F6):** on every
  consistency row (= ALL exploit cycles ∪ top-20 by carried, capped
  400 with a truncation flag) that contains NO overlap-sensor
  action: `frac_inconsistent == 0` AND `licensed ≥ carried − 1e-9`
  ⇒ min-clip is a NO-OP on this operator (clipped == carried), as a
  CONSEQUENCE of license ≥ promise. The licensed−carried gap is
  reported per row (descriptive). Overlap rows are reported as the
  diagonal-lens artifact exhibit, never adjudicated.
- **P-RKN2′ (primary, re-registered #33 F3):** **4/4 members:
  n_exploit_pbim == 0 AND n_exploit_both == 0** — the coherent
  operator never reaches the realized-ΔH tier or the flagship
  conjunction (seed-0: PRE-COMPUTED TRUE). Registered descriptive,
  no fire: n_exploit_cig per member (expected large — the
  optimizer-budget bound above; seed-0 = 332) and the per-member
  q_hat vector (z3 row = the unidentifiability disclosure).
- **P-RKN3 (comparability, descriptive):** heldout row vs the
  **DIAGONAL exact floor** (#33 F11 — eval_exact's full-covariance
  z-NLL differs from the diagonal surrogate by a fixed −0.072-nat
  offset at the oracle point; the diag floor is emitted alongside
  and is the comparison target), and vs the published GRU rows. The
  12-Aug published floor (obs 0.30059 / z −1.82030) is asserted to
  1e-3 as a drift gate (#33 F24).
- **P-RKN4′ (CEI bridge, re-built #33 F7/F8/F9):** the audit prices
  the operator AGAINST ITS OWN DECLARED FILTER — the trained
  RKNModel stepped in the loop (its Â, Q̂, ĉ, r̂), not a
  fictitious static-dynamics filter with r̂ alone (the marginal-KL
  pricing understated κ by ≈3.5× on the seed-0 member). Per audited
  sensor: empirical κ (tail LLR drift under the TRUE generator),
  bound = d(.05,.95)/κ_emp; **PRICED requires crossed_frac ≥ 0.98**
  (else the row cannot be priced at nmax = 50 000 and reports
  BELOW-AUDIT-RESOLUTION — #33 F8); registered criteria per PRICED
  row: power ≥ 0.9 under the true generator AND
  **false-conviction ≤ 0.1 under the DECLARED generator (#33 F9 —
  both error sides simulated)** AND E[N] within [0.5, 4]× the bound
  AND Wald ratio in [0.5, 3]. Fire = all PRICED rows meet all four.
  **Scope (#33 F16, registered):** r-channel audit of the static
  one-hot readouts {s0,s1,s2,s3,s7} from the FRESH prior
  v0 = c′P0c (in-situ audits arrive at converged posteriors —
  disclosed scoping); the C-channel and the TV sensor s6 are OUT OF
  SCOPE (an r-only audit cannot see a C-liar; the seed-0 member
  learned ‖ĉ_s6‖ = 0.193 on the zero-C sensor — reported as the
  c_hat_norm disclosure rows).
- Descriptive: verdict distribution (with the reduced-reachable-set
  note), train losses, σ_pi/σ_exact per member.

## 4. Execution & read

`scripts/uncfield_rkn.sbatch` — ONE sequential CPU job (preflight
asserts the gitignored `pilot2/episodes.pkl` is staged — #33 F15;
selfcheck → train → search → heldout → audit). Timing corrected
(#33 F21): train ≈ 8.5 min/member measured locally (not 5); search
≈ seconds/member on this operator (the 4–20 min figure was the
GRU's); 4 h remains a wide margin. Adjudication = ONE pass over the
emitted jsons against §3 after results-sync collection; no re-runs
on an unwanted outcome. Seed-0 rows are reported with their
PRE-COMPUTED label.

## 5. Fences & placement

Standalone Track-C paper (coherent-operator counterfactual + audit
pricing); cites Paper 5 and CEI. Does NOT enter Paper 5. Min-clip
claims stay within the parent fences. The diagonal-lens artifact
and the naive≡carried degeneracy feed the ratio-instrument
limitations disclosure of whichever paper cites it.

Freeze = review #33 adjudicated (this file) + commit of
`uncfield/rknwm.py` + `scripts/uncfield_rkn.sbatch` — then the ONE
job.
