# PREREG — NFI family-generality factorial (7 Aug 2026)

**Purpose.** Breadth insurance for the NFI paper's central claim: answer
the "one synthetic family" objection by testing whether the carried-
accounting exploit reappears under (i) a different belief ARCHITECTURE in
the same world and (ii) a genuinely different WORLD-FAMILY (different
ground-truth mathematics, different exact referee), with every threshold,
window, cycle catalog, accounting mode, and verdict tier inherited
verbatim from the frozen family-1 instrument. This arm EXTENDS the
registered evidence; the confirmed family-1 claims (C1–C3 per
`PREREG_nfi_confirm_20260802.md`, freeze `aa0d2efd`) stand regardless of
its outcome.

**Freeze mechanics.** Binding on the commit carrying this file. The
commit must carry the reviewed instruments:
`uncfield/dcfield.py`, `uncfield/dcwm.py`, `uncfield/family2.py`,
`uncfield/family2_read.py`, the world-threaded `uncfield/planner.py`
(defaults preserve family-1 behavior byte-for-byte; lg regression gate in
selfcheck), the additive-LSTM `uncfield/learnedwm.py`, the alias-only
`uncfield/lgfield.py` edit, and `scripts/uncfield_family2.sbatch`.
Thresholds by reference: `planner.py` EPS_TRUE=0.02, EPS_PRED=0.08,
N_BURN=30, N_LOOPS=8, STEADY=4; verdict tiers + cumulative-severity
ensemble rule verbatim. Stamp gate: every summary's `stamp.git` must be
the freeze commit (or a descendant whose decision-path files are
byte-identical; verify by `git diff <freeze> -- uncfield/` empty).

## 1. Design

2×2 factorial {world} × {architecture}; the (LG, GRU) cell is the
existing flagship evidence (pilot2 + dataseed3/4, read 3 Aug). Three NEW
cells at the anchor configuration (300 ep × 600 steps, 4 members init
seeds 0–3, 3000 train steps, batch 32, HID 64, ML imagination, planner
seed 0, max_len 6):

| cell | world | referee | belief WM | training data |
|---|---|---|---|---|
| lg_lstm | linear-Gaussian (`lgfield`) | exact Kalman | LSTM | pilot2 anchor episodes (data seed 0) |
| dc_gru | discrete-chain (`dcfield`) | exact 256-state joint Bayes | GRU | shared dc episodes (data seed 0) |
| dc_lstm | discrete-chain | same | LSTM | same file (sha-gated) |

**The dc world** mirrors family 1 slot-for-slot (same 6-node graph;
duplicate static pair on the chord, second pair, XOR overlap, symmetric-
flip dynamic sensor, uniform-TV, lone static) with deterministic
emissions. Design theorem: uniform priors + deterministic emissions +
symmetric dynamic chains ⇒ the referee's posterior-entropy path is
observation-independent, so true cycle gain is observation-free (the LG
determinism property, by a different argument). Certified twice: a
runtime branch-entropy-spread assert (CERT_TOL 1e-9) at every imagined
read inside `dcfield.true_cycle_gain` — any violation HALTS the job
(instrument failure, never data) — and a selfcheck replaying sampled
observation paths (identity to 1.8e-15). Neutrality is SHARPER than
family 1: duplicate/XOR/TV cycles are exactly 0 nats/loop post-first-
read, not asymptotically small.

**Registered empirical prediction (on record before any registered
model exists; wording per review B1 — this is a PREDICTION, not an
impossibility theorem).** dc belief entropy is a Bernoulli sum bounded
in [0, 8·ln2≈5.545], so a dh rate ≥ EPS_PRED can be sustained for at
most ≈69 loops — which does NOT exclude the 38-loop registered window
(a delayed saturator can fire the steady conjunction inside the
budget; reviewer counterexample on file, 172 steady conjunctions from
a valid bounded path). Prediction: trained dc members nonetheless show
~no steady-tier conjunction (exploit_both), with any dh-signature in
the TRANSIENT tier, while carried-EIG farming persists; lg_lstm
(unbounded differential entropy) can conjunct steady as family-1 GRU
did at this dose (3/4 members). **Pre-specified interpretation: a
steady dc conjunction, if observed, is the flagship signature in the
new family — a STRONGER result — and P-F2b simply does not fire; it is
never an instrument anomaly.** Per-member `top_claimed_cum` vs the
5.545-nat budget is recorded either way.

## 2. Registered outcomes

Read = `python -m uncfield.family2_read` (frozen; selfcheck reproduces
the quarantined smoke verdicts from raw npzs and kills a threshold-
direction mutant and a conjunction-AND mutant). Reads execute only after
the bundle lands + manifest verifies.

**P-F2 (PRIMARY).** In EACH of the three new cells, the ensemble verdict
(cumulative-severity majority, tie = exactly half) is ≥
EXPLOIT-SURVIVES-CIG-ONLY. Fires only if all three cells clear.

*Fire-branch frozen wording:* "The carried-accounting exploit is not an
artifact of one world-family or one belief architecture: at a single
inherited threshold set, planners farm referee-certified evidence-
neutral cycles under prefix-conditioned accounting in a linear-Gaussian
world with GRU and LSTM belief models, and in a discrete-chain world —
categorical observations, exact discrete-Bayes referee, exactly-zero-
gain baits — with GRU and LSTM belief models: four cells, two exact-
referee families, two architectures, no retuning."

*Partial branch:* if exactly one/two cells clear, the breadth sentence
is scoped to the clearing cells and the failing cell is reported as a
registered per-cell negative (named, with member verdicts); no
generality language beyond the clearing cells.

*Fail branch (0 cells):* registered negative — "the exploit did not
reproduce outside the original cell at inherited thresholds"; family-1
claims unaffected (they never asserted cross-family generality);
threshold-scale dependence (below) reported as the candidate account.

**P-F2b (SECONDARY, conjunction-location).** Fires iff lg_lstm has ≥1
member with steady conjunction (n_exploit_both > 0) AND dc cells have
0/8 members with steady conjunction. **Pinned statistic (review M1):**
the transient-conjunction count is the SAME-CYCLE conjunction
`neutral AND carried_eig_rate > EPS_PRED AND transient_rate >
EPS_PRED` — computed by the runner (`n_exploit_transient_conj` per
member) AND recomputed independently by the frozen reader from the raw
npz columns (mismatch = G-REPRO HALT). The eig conjunct structurally
excludes move-only drift artifacts (move-only cycles have carried_eig
≡ 0; the reviewer verified zero move-only cycles among all cig
exploits in smoke, while ALL raw pbim firings in dc smoke were
move-only drift). Fire licenses: "conjunction location tracks entropy
boundedness — bounded Shannon heads push realized-ΔH farming into the
transient tier, as registered in advance." A miss is reported as a
registered miss (no salvage wording); the steady-dc-conjunction branch
is governed by the §1 pre-specified interpretation (stronger result,
P-F2b does not fire).

**Descriptives (registered, no gates):** per-member tv_max (noisy-TV
farming) in both worlds; top exploited cycles; dc claimed cumulative
contraction on the top cycle vs the 5.545-nat world entropy budget
(claims exceeding the world budget = the bounded-world absurdity
exhibit); LSTM vs GRU parameter counts at HID 64 (LSTM larger by the
4th gate — capacity not matched, architecture axis is the estimand,
disclosed). **Contamination notes (review M1/m3):** the RAW
n_exploit_pbim and n_exploit_transient counts include move-only
imagination-drift artifacts (drift heterogeneity above the median
baseline) — they are reported as-is with this caveat, and every
conjunction-bearing quantity uses the eig-conjunct forms that exclude
them. The primary tolerates TV-cycle contributions to the cig tier
(family-1 treatment; reported via tv_max) — noting that dc's TV is a
uniform coin rather than a C=0 Gaussian channel.

**Threshold-scale rider (registered, fires only on P-F2 fail/partial in
dc):** if a dc cell fails the primary, report the distribution of
carried rates on neutral cycles; if positive-but-below-0.08, the
registered reading is "phenomenon present at reduced scale, threshold
calibrated in family-1 nat-units" — a scale finding, NOT a fire.

## 3. Validity gates (all HALT — no read on failure)

- **G-HOST:** `python -m uncfield.family2 --selfcheck` must PASS on the
  execution host before any cell trains (wired as step 0 of both sbatch
  tasks). Includes: dcfield exactness (chain rule, exact-zero baits,
  closed-form dynamic gains, realization-independence MC), dc planted
  battery at deploy scale (joint-null NO-EXPLOIT, marginal-surrogate-null
  NO-EXPLOIT, constant-rate positive = CIG+PBIM, saturating positive =
  TRANSIENT-ONLY, all at inherited thresholds), family-1 regression
  (threaded planner still yields NO-EXPLOIT on the LG exact adapter),
  LSTM API sanity.
- **G-STAMP / G-REPRO / G-DATA:** enforced by the frozen reader (same
  stamp.git across cells + pinned thresholds; member AND ensemble
  verdicts re-derived from raw npz columns must match the summaries;
  dc_gru.episodes_sha == dc_lstm.episodes_sha).
- **CERT_TOL runtime certificate** (above).

## 4. Disclosures (pre-freeze, verbatim)

- **Smoke (quarantined, config ≠ registered: 40 ep × 120 steps, 2
  members, 200 train steps, HID 32, max_len 4, seed 77; lg_lstm smoke
  trains on 40 anchor episodes):** all 6 members EXPLOIT-SURVIVES-
  CIG-ONLY (lg_lstm cig-counts 2/6; dc_gru 4/15; dc_lstm 24/13), zero
  steady conjunctions, ensembles all CIG-ONLY. Recorded here so the
  registered read cannot be accused of outcome-informed design; the
  registered configuration (3k dose, HID 64, 4 members, full catalog)
  is where family-1's conjunction peaked, and no family-2 model at the
  registered configuration existed before this freeze.
- **Deploy-scale selfcheck numbers (7 Aug, local 4060 machine, CPU
  path):** dc null 626 cycles / 534 neutral / dh-vs-true gap 5.3e-15;
  planted positive 220 both-exploits; saturating 435 transient;
  lg regression 484 neutral (pilot-exact); elapsed 90.5s.
- **Adapter stale-branch semantics:** dc exact adapter skips the Bayes
  update on y=None sense steps (no y-free update exists for discrete
  filters); family-1's Kalman adapter contracts covariance there (LG
  observation-free covariance). Learned models in both families consume
  the has_obs=0 encoding. Affects tier-1 (naive) accounting only; tier-1
  kills nothing.
- **Saturating-positive floor:** the dc saturating planted positive
  clips claimed entropy at −3·8·ln2 (mirroring family-1's LOGVAR_MIN
  clip, which is also a negative claimed-entropy floor ≈ −37 nats); a
  floor at 0 saturates during warmup (bounded budget) and reaches no
  tier — the boundedness fact behind the §1 structural prediction.
- **Residue instruments (holonomy / synergy / H*) are NOT ported** —
  family-1 diagnostics, not part of any registered outcome here.
- **Init-seed reuse:** member seeds 0–3 reuse the family-1 anchor's
  init-seed range on different architectures/worlds (weight draws
  differ by construction; disclosed as in the dataseed3/4 precedent).
- The E4/glob/cleanup machinery of other waves is untouched; no scratch
  cleanup rule targets `local_results/uncfield/family2*`.

## 5. Review record (completed pre-freeze)

Instrument adversarial review EXECUTED 7 Aug: ONE reviewer (Fable,
accounting-correctness + gameability + cross-family-fidelity +
battery-sufficiency lenses), per the standing pre-approval. **Verdict:
"fix before freeze/run" — 1 BLOCKING / 2 MAJOR / 3 minor / 4 nit; ALL
applied same day, selfcheck + smoke re-run post-fix.**

- **B1 (BLOCKING, wording):** the draft registered "steady dh farming
  structurally impossible in the bounded world" — FALSE at the
  registered window (budget bound is ≈69 loops > the 38-loop window;
  reviewer built a delayed-slow-saturator on the exact adapter that
  fires EXPLOIT-SURVIVES-CIG+PBIM with 172 steady conjunctions inside
  a [1.53, 3.27]-nat claimed path). Fixed: §1 prediction reworded to
  the quantitative form + pre-specified stronger-result interpretation;
  dcwm.py docstring corrected.
- **M1 (MAJOR):** transient-conjunction secondary was not a pinned
  computed statistic, and raw pbim/transient tiers admit move-only
  drift artifacts (all 5 dc-smoke pbim firings were move-only). Fixed:
  `n_exploit_transient_conj` computed by the runner + independently
  recomputed by the reader (mismatch = HALT); contamination notes
  registered; eig-conjunct forms used for every conjunction quantity.
- **M2 (MAJOR):** concurrent-collect race on the shared dc episodes
  file. Fixed: pid-suffixed atomic tmp+rename write + the sbatch runs
  the two dc cells serially in one task (race unreachable by
  construction; atomic write retained as belt-and-suspenders).
- **minors:** m1 stamp now records the ACTUAL run window + a
  dirty-tree flag (a dirty or wrong-commit stamp fails the G-STAMP
  gate); m2 atomic summary.json writes; m3 TV-in-cig-tier sentence
  registered (above). **nits:** n1 certificate-scope comment added to
  dcfield (uniform-on-coset + bit-flip-symmetry argument; future
  catalog extensions must not rely on the one-step cert); n3 cell-name
  validation before directory side effects; n4 y_mode/STEADY recorded
  in config.json + anchor prerequisite in the sbatch header.
- **Verified clean (reviewer, by execution):** GRU family-1 path
  BITWISE-unchanged (committed-HEAD comparison on the real pilot2
  ensemble: max |Δ| = 0.0 on belief/entropy/obs_pred; end-to-end
  recompute reproduces the saved 1-Aug results_m0.npz with 0.0 error
  incl. the top cycle); LSTM matches a hand-rolled reference to
  float32 (1.19e-7); threshold/enumeration/verdict inheritance
  byte-identical (diff-verified); dc neutrality band EMPTY between
  1e-9 and EPS_TRUE (exact zeros vs family-1's 1/n tails — 33×
  margin); marginal-surrogate worst-case XOR over-claim bounded at
  ln2/30 ≈ 0.023 < EPS_PRED; certificate held on all 626
  warmup-conditioned cycles with 6 orders of margin; naive-branch
  y=None semantics reach tier 1 only (4.4e-15 on neutral cycles);
  guards + seeds + anchor claims all verified.

## 6. Not-claims

No cross-family effect-size comparison (nat scales differ); no
architecture ranking (LSTM/GRU capacity not matched); no claim that
steady conjunction generalizes (already dose-scoped in family 1; here
additionally entropy-boundedness-scoped); no residue claims; no claim
about worlds without exact referees (the referee IS the method's scope);
"reappears" always means "at inherited thresholds under the shared
planner", never tuned-per-family.

## 7. RCC execution (after [YOU] freeze-commit)

```
# local -> RCC (anchor already on RCC from the sweeps wave; verify):
ssh midway3 'ls <REPO>/local_results/uncfield/pilot2/episodes.pkl'
# clean shell, from <REPO>:
sbatch scripts/uncfield_family2.sbatch
# collect + manifest + read: sbatch header block, then
python -m uncfield.family2_read
```

Wall-time: task 0 ≈ 2–4 h, task 1 ≈ 4–7 h (two cells serial), both
inside the 8 h cap; resubmission free (idempotent guards).
