# PREREG — Structured-noise AVOIDANCE confirmatory wave (A1), 21 Aug 2026

**Program:** Track A of `Plan_FollowupPrograms_20260821.md` (the
Paper-5 follow-up slate; user GO 21 Aug "build everything"). **This
prereg REGISTERS THE DIRECTION** that three prior observations carried
only descriptively: M3's complete-separation reversal (gated 0.481 <
ungated 0.560), the gradient arm's replication against the
amplitude-matched comparator (hetero 0.472 < flat 0.524, unique
minimum of 70), and the rankx anti-transmission disclosure (hetero
ranking −0.0822 vs flat −0.0088). All three were unregistered
directions and were reported as such; none may be cited as
confirmation of this wave's hypothesis — this wave exists because
they cannot be.

**Registered hypothesis:** agents whose epistemic objective overprices
a spatially-structured zero-information channel AVOID the region where
that channel's amplitude (and hence its fictitious epistemic value) is
highest.

## 1. Arms (fresh seeds, same wave, same site)

- **hetero2**: the Amendment-3 hetero config VERBATIM (distractor
  amplitude ramp m over the smoke range; every pin unchanged), seeds
  **44–47**, RUN_IDs `se_avoid_het_s44..47`.
- **flat2**: the Amendment-3 flat config VERBATIM
  (`DISTRACTOR_SCALE=0.35891393`, amplitude-matched, no gradient),
  seeds **48–51**, RUN_IDs `se_avoid_flat_s48..51`.
- **Registered Stage-1-verbatim pins on BOTH arms (review #27 B3 —
  the producer now carries overrides that could silently flip them):**
  `task == dmc_cheetah_run`, `expl.mode == p2e`,
  `disag_bootstrap == True`, `disag_ens == 8`,
  `disag_scale == 1000.0`, `planted.source_key == position`,
  `planted.basesd == 0.0976`; the frozen reader hard-gates every one.
- 8 runs × 5e5 steps; seeds disjoint from all prior ({0, 10–17,
  20–43}) and from the concurrent dose wave (52–67); arms never
  seed-paired. Launch via the Stage-1 producer env overrides (all
  existing; no code changes).

## 2. Registered measurements (frozen reader
`uncfield/se_avoid_read.py`, built + selfchecked BEFORE this compute)

- **PRIMARY (confirmatory):** occupancy(hetero2) − occupancy(flat2)
  in the high-amplitude region (position[0] > −0.13009691; the
  Amendment-2 full-replay estimand verbatim, ≥1e5-step floor),
  **one-sided NEGATIVE** exact label permutation over C(8,4)=70;
  **FIRES iff p ≤ .05 AND the delta is negative** (min attainable
  p = 1/70 ≈ .0143).
- **SECONDARY-1 (registered, direction pre-specified):** ranking
  contrast — mean per-run P-SE2 delta (pse2 top−base, the frozen
  probe's stored field) hetero2 − flat2 < 0, one-sided exact
  permutation. This registers, on fresh data, the direction the rankx
  DISCLOSURE could not claim.
- **REGISTERED ADJUDICATION ORDER (#27):** (0) DEGENERACY GATE —
  per-run real-dim level + intrinsic_mean ≥ 1/10 of the live Stage-1
  minima (--stage1_runs; the NOBOOT/dose pattern) else
  GLOBAL-COLLAPSE; (1) **MISPRICE GATE** (the Amendment-3 form,
  re-registered — #27 M6): distractor p_perm < .05 in ≥ 3/4 VALID
  hetero2 probes under the full frozen probe gates, else
  MISPRICE-ABSENT (nothing to avoid); (2) **GRADIENT-DELIVERY
  GATE**: among REGION-ADJUDICABLE hetero2 runs (≥ 32 anchors per
  region AND finite means; ≥ 3 of 4 required adjudicable else
  REGION-UNDERPOWERED — the floor is a per-run mechanism/delivery
  adjudicability flag and NEVER aborts or touches the primary, #27
  B1), distractor disagreement higher in the high region in ALL of
  them. DISCLOSED: this gate is near-degenerate on the pass side by
  construction (the E[m²] contrast) — its role is catching
  collapse/instrument faults, not discrimination; (3) occupancy
  validity (4+4); (4) the PRIMARY; (5) the SECONDARY.
- **MECHANISM (descriptive, never a fire, #27 M9):** the registered
  decomposition statistics are the WITHIN-RUN normalizer-invariant
  ratios dis_high/dis_low and real_high/real_low per arm (cross-arm
  raw levels are normalizer-confounded and are NOT compared); BCa
  reported. **No outcome of the mechanism leg licenses any claim or
  venue change** — the "ICML if the mechanism lands" phrasing in the
  slate plan is hereby voided; venue is decided on the PRIMARY.
- Score two-sided reporting; per-arm occupancy BCa; fit counters;
  identity pins verbatim from the reviewed Amendment-3 readers PLUS
  the §1 pin block. **Multiplicity (#27 M10): SECONDARY-1 is GATED on
  the PRIMARY — it fires only if the primary fires, and firing alone
  licenses nothing; it is reported unconditionally.** **Power
  disclosure (#27 m17): p ≤ .05 over C(8,4)=70 requires the observed
  split among the 3 most extreme assignments — near-complete
  separation; this wave has essentially no power against moderate
  effects, and a non-fire does not distinguish "no effect" from
  "moderate effect" (registered).

## 3. Outcome map

- **AVOIDANCE-CONFIRMED** (primary fires): the registered claim is
  licensed: "spatially structured fictitious information repels the
  agents that overprice it" — scoped to this substrate, objective,
  and delivered contrast; SCARECROW (A2, the algorithm leg) unlocks.
- **AVOIDANCE-NOT-CONFIRMED** (primary does not fire): the two prior
  reversals are demoted to wave-specific observations; no repulsion
  claim survives anywhere in the program; A2 does not proceed.
  This is a REAL risk — the prior observations were post-hoc and this
  is exactly the winner's-curse test of them.
- **DELIVERY-FAILED / NOT-ADJUDICABLE**: validity cells; no wording
  either way.

## 4. Instruments

`uncfield/se_region_probe.py` (NEW, built + selfchecked: per-state
per-key decoder-projected disagreement + anchor region coordinate;
se_probe mechanics verbatim where shared; npz carries its own
provenance — ckpt basename, seed, n_eval — and the reader binds it to
the run, recomputes the region labels from the stored coordinate, and
ties it to se_probe via per-state-mean ≈ dims_d, #27 M4). **Pinned
execution (#27 M7): `--n_eval 512 --burn_in 16 --seed 0` on the SAME
device class and in the SAME probe job as the `se_probe` pass, both
run post-training on all 8 runs** (reader floors: region-probe
S ≥ 256). Frozen reader as above. Bundle: replay/, scores, config,
metrics, **ckpt/latest + the ckpt dir listing** (#27 m18 — the
final-ckpt gates need it or the manual-discharge ritual repeats),
se_probe/, se_region_probe/; manifest per results-sync v2.

## 5. Paper placement

A fired primary becomes the spine of the Track-A standalone paper
(RLC/TMLR floor), citing Paper 5's arXiv for the pricing results. It
does NOT retroactively enter Paper 5 (whose avoidance treatment stays
descriptive with the unregistered-direction clause). **The fence is
asymmetric-with-duty (#27 M8): A1 may never SUPPORT a Paper-5 claim
in either outcome — but if A1 does NOT confirm, Paper 5's descriptive
avoidance paragraph MUST cite the non-confirmation as a caveat before
any subsequent submission.** One-directional insulation from bad news
is exactly what this clause forbids.

Freeze = review #27 ADJUDICATED (3B/7M/11m ALL adopted 21 Aug: the
endogenous-floor abort removed, SECONDARY-1 provenance-gated through
the frozen probe reader, the full pin block, region-npz provenance
binding + label recompute + cross-instrument tie, nan guards, the
misprice + degeneracy gates restored, probe flags pinned, the fence
made asymmetric-with-duty, mechanism ratios registered, secondary
gated; selfchecks re-PASS incl. the fabricated-probe and
region-underpowered fixtures) + commit of this file + both
instruments — then the 8 submissions.
