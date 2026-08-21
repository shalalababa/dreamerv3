# PREREG — APT taxonomy column (Track B2), 21 Aug 2026

**Program:** Track B of `Plan_FollowupPrograms_20260821.md` (user GO
21 Aug). The first NON-disagreement proxy cell of the channels ×
proxies matrix: APT (k-NN particle entropy in latent space, in-repo
`expl.mode: apt`) on the Stage-1 planted-channel environment. The
matrix question: does a nonparametric-entropy objective price
irreducible stochasticity the way ensemble disagreement does, and do
duplicates stay at par?

## 1. Wave

Stage-1 config with **`EXPL_CONFIG=expl_apt`** (producer override
added, default expl_p2e = Stage-1-inert), planted channels + doses
unchanged (dim 8, basesd 1.215, scale 1.0, no gates/mod). **Seeds
68–75**, RUN_IDs `se_apt_s68..75`, 8 runs × 5e5 steps, one wave/site.
**SMOKE GATE first (required before the 8 submissions):** one 2e4-step
run (`se_apt_smoke`, seed 9 — outside every registered family) to
verify the apt objective trains against the planted obs space and the
mask instrument behaves on an APT checkpoint; smoke FILLs (if any)
pinned by amendment before the wave freezes, per the parent's
convention.

## 2. Attribution (instrument built + selfchecked; post-#29 form)

`uncfield/se_apt_mask.py` — **the estimand is a POPULATION-
INTERVENTION delta** (#29 C-B1: all windows masked simultaneously;
one shared k-NN graph). Mask suite, redesigned after review #29
demonstrated that source-substitution DELETES VARIANCE into an
entropy statistic (mechanically predetermining D-family deltas):
- **distractor → batch-permutation** (marginal-preserving — the ONLY
  fire channel);
- **dup1/dup2 → TWO masks each**: source-substitution
  (spread-inclusive, DESCRIPTIVE) and FRESH-RESAMPLE
  (variance-matched content-only; null prediction ≈ 0 for i.i.d.
  noise; substitution-minus-resample = the mechanical
  cloud-contraction component);
- **dup0** → bitwise no-op anchor (hard-asserted);
- **velocity → batch-permutation CALIBRATION CONTROL** (a real
  coupled key; expected negative if APT prices real cross-window
  coupling; the distractor-vs-velocity asymmetry is the mechanism
  row).
Pins (#29 C-M5/m8/m9): realized S == 512 hard-asserted (APT
magnitudes are population-size dependent); apt_knn/apt_logc read
from the RUN config and asserted; `expl.mode == apt` gated
(--allow_nonapt exists for mechanics selfchecks only); train_seed +
task recorded. Anchor-level sign-flip p and BCa are DIAGNOSTIC, NOT
CALIBRATED (#29 C-M4 — anchors share one k-NN graph); they are never
reported beside the fire rule. Disclosure (#29 C-m11): training-time
APT scores imagined rollouts (B×K = 1024, T = 16); this instrument
is a replay-anchor PROXY for the trained functional. The saved run
config carries inert disag_* keys with no ensemble built — the
reader must not gate on them (#29 C-m12).

## 3. Registered read (frozen reader `uncfield/se_apt_read.py`,
built + selfchecked BEFORE THE READ)

- **PER-RUN STATISTIC, pinned verbatim (#29 C-M2):**
  `channels["distractor"]["delta_apt_mean"] < 0` read from
  `se_apt_mask.json`.
- **CROSS-RUN PRIMARY — SINGLE fire channel (distractor), no BH:**
  fires iff the per-run statistic is negative in **≥ 7 of 8 runs**
  (exact binomial p = 9/256 = .0352). REGISTERED CAVEAT (#29 C-M3):
  the sign-at-zero coin null is ASSERTED, not constructed — the
  mitigations are (i) the dup0 exact-zero anchor, (ii) the velocity
  calibration control, (iii) the fresh-resample nulls; **if the
  registered SMOKE shows a systematic sign on the resample controls
  (|mean| > half the distractor delta), the fire rule is amended to
  a control-relative comparator BEFORE the wave submits** — the
  amendment path is registered here, pre-outcome. **Comparator form,
  pinned 21 Aug pre-smoke (#30 M8, implemented as
  `--fire_rule control_relative` in the frozen reader): the per-run
  statistic becomes [distractor delta MINUS the mean of the two
  fresh-resample deltas] < 0, same ≥7/8 rule; the flag may be used
  ONLY after a triggering smoke report, and the reader records which
  rule ran.**
- **D-family: DESCRIPTIVE ONLY** (no fire; #29 C-B1): the dual-mask
  decomposition per channel (substitution vs resample), the dup0
  anchor, and the velocity control; the taxonomy-cell narrative rests
  on the PATTERN, with the mechanical component explicitly separated.
- **Taxonomy-cell outcome map:** (i) distractor fires with resample
  controls ≈ 0 → APT prices irreducible stochasticity via coupling,
  duplicates at par — the disagreement pattern replicates across
  proxy class; (ii) distractor does not fire → APT is immune to
  these channels (a genuinely different column); (iii) resample
  controls systematically negative → the spread mechanism dominates
  and the column is re-scoped to spread-pricing (no content claim);
  (iv) validity cells (S == 512 pin; dup0 no-op assert; fit
  counters; seeds 68–75 distinct; expl.mode == apt pin).
- **SMOKE GATE acceptance criteria, pinned (#29 C-M7):** rc = 0;
  run config `expl.mode == apt`; all 4 planted keys live with
  per-key decoder losses in metrics; instrument runs end-to-end on
  the smoke ckpt with dup0 delta == 0.0 exactly and S == the pinned
  smoke population; the resample-control sign report (feeds the
  registered amendment path above). The gate is MECHANICAL, not
  powered — it cannot establish delta resolvability at 2e4 steps.

## 4. Fences & placement

Track-B paper cell. MI/JSD-disagreement and learning-progress bonuses
are OWNED elsewhere (ideation §4) — APT is claimed as a MATRIX COLUMN
measurement, never as a novel algorithm. No Paper-5 content depends
on this wave.

Freeze = review #29 (APT half) ADJUDICATED (1B/6M/5m ALL adopted
21 Aug — the variance-confound blocker resolved by the dual-mask
redesign + velocity control; statistic pinned verbatim; S pin;
smoke criteria; single-channel fire rule) + commit of this file +
`uncfield/se_apt_mask.py` + the sbatch EXPL_CONFIG override — then
the smoke (seed 9), the resample-sign report, then the 8
submissions.
