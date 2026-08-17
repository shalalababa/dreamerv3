# FB / successor-features read — BEHAVIORAL-ESCAPE (qualified)

**2026-08-17.** The registered ONE execution of `analysis/fb_read.py` against
bundle `local_results/fb_fits_20260817_141845` (sha-verified 101/101 against
`manifests/fb_fits_20260817_141845.sha256`). Registration:
`prereg/PREREG_fb_20260817.md` (freeze f9cef148; Amendment 1 = no_grad wrapper
59b97eb9; FILLs pinned by dated pre-fit edit: updates=500000, health band
[-455.1604080200195, -4.551604080200195]). Selfcheck PASS immediately before
execution (8 fixtures + 14 refusals). Reader invocation exactly as registered,
`--health_band -455.1604080200195 -4.551604080200195`.

## Verdict

**BEHAVIORAL-ESCAPE (qualified)** — the reader's registered branch, verbatim:
the flagship consequential-level escape LANDS; the representation leg shows no
separation from a near-ceiling raw comparator (registered weakly informative);
escape wording licensed at the behavioral level; the strongest bidirectional
form reserved for ESCAPE-CONFIRMED.

**This is the first tested objective family to escape the reward-free null.**
The theory's inclusion criterion predicted exactly this: FB's objective
factorizes occupancy (M ~ F(s,a,z)·B(s')), making any reward's value readable
by construction — and it clears the house behavior-alive floor that every
reconstruction-based reward-free arm (rif 70.7, apt 81.3) sat far below.

## P-FB1 (flagship, zero-shot behavior): ZERO-SHOT-ESCAPE

| quantity | value |
|---|---|
| mean return (n=16) | **179.79** vs floor **113.682** |
| excess | **+66.11 [BCa +21.04, +125.80], perm p = .0267, d_z 0.597** |
| random-floor control | 79.4 ∈ [35, 110] — CONTROL PASSES (fail-closed gate) |
| context pins | rif 70.665 / apt 81.301 — FB sits ~2.2–2.5× above both |

Side-stratified secondaries (n=8, registered NON-DECISIONAL): **the escape is
carried by side1** (high-occupancy): side1 mean return 242.3, excess +128.6
[+31.0, +199.2] perm p = .039, d_z 1.02; side0 mean return 117.3, excess +3.6
[−11.8, +44.7] ns — side0 sits AT the floor. Registered side control
(side1 − side0 = +125.0 [+26.1, +199.2], perm p = .023): **CONTROL-OK**,
direction as predicted by the occupancy dose.

Read-level note (post-read analysis, labeled): the side pattern is
mechanism-coherent with the paper's whole arc — even the escaping objective
escapes only where the buffer's task-relevant support is dense. The z-inference
side premise is auditable in the same table: reward-positive fraction
0.052–0.058 (side0) vs 0.311–0.330 (side1), the registered premise gate
(frac_pos1 > frac_pos0) passing 8/8 pairs.

## P-FB2 (representation): B-LESS-LEGIBLE

fb mean AUROC **0.7951** vs raw-identity comparator **0.9028**; paired diff
**−0.1077 [−0.1454, −0.0854], perm p = 3.05e-05, d_z −1.78** — B embeddings
are *less* linearly reward-legible than the raw 12-dim proprio they are built
from. Registered handling: raw is near-ceiling, so this leg was disclosed
pre-outcome as weakly informative; the verdict term is B-LESS-LEGIBLE, not a
damage finding. It blocks only the bidirectional ESCAPE-CONFIRMED wording.

Read-level note (post-read, labeled): behavioral escape without
linear-probe superiority of B is not a contradiction — the zero-shot policy
reads reward through F·z value structure, not through linear decodability of
B; the wave's claim ("readable BY the objective") is about the objective's
functional form, which the behavioral leg tests directly.

## Gates and witnesses

- updates 500000 across 16/16 (counters == config == allowed set); kwargs
  byte-identical ×16; checkout pin, action_convention prev_shifted_v1, data
  shas side-matched, fit-seed linkage, zeroshot↔emb ckpt sha byte-identity —
  ALL PASS (any failure is a refusal; the read executed).
- emb panel single-device: NVIDIA GeForce RTX 5060 Ti ×16, raw 'cpu'; bundle
  note discloses serial CUDA_VISIBLE_DEVICES=0 production.
- z-health: z_norm ≡ 7.0711 (= √50, the checkout's z-scale projection) 16/16.

## Training-health witness: recorded out-of-band 16/16 — band uninformative as pinned (disclosed)

All 16 tail-median fb_loss values fall OUTSIDE the pinned band
[−455.16, −4.55]: side0 −625.9 … −1643.6 (and **seed8 +1001.7, sign-flipped**),
side1 −2826.2 … −3969.6. Per registration this is recorded, never a refusal.

Honest accounting (post-read, labeled):

1. **The band width was my misjudgment at the FILL edit.** I pinned ×10
   log-magnitude around the 20000-update smoke anchor (−45.5) to "tolerate the
   20000→500000 extrapolation"; realized growth was ~30–90×. FB loss magnitude
   evidently scales with training (M/Q value scale growth), which the smoke
   anchor could not see. The witness therefore cannot discriminate healthy
   from unhealthy fits at 500000 updates — it is uninformative, not alarming.
2. **The verdict does not lean on it.** The witness's sole registered role is
   gating the undertraining caveat's admissibility *against a NO-ESCAPE*; the
   outcome is a fire, so the caveat mechanism is moot. Had the flagship gone
   negative, the all-16 out-of-band status would have made the undertraining
   caveat admissible — worth stating: the fire direction is the one the
   witness cannot weaken.
3. **Within the panel the scale is regular, one cell excepted**: side1
   uniformly ~2–3× more negative than side0 (tracks the frac_pos/occupancy
   gap); the lone qualitative anomaly is s0_seed8's POSITIVE +1001.7. Its
   outcomes are unremarkable (zeroshot 134.8, emb AUROC 0.815; excluding it
   would only strengthen side0's null and not move the pooled fire, which
   side1 carries). Flagged as a descriptive anomaly only.

## Disposition

- P-FB1 ZERO-SHOT-ESCAPE + P-FB2 B-LESS-LEGIBLE ⇒ **BEHAVIORAL-ESCAPE
  (qualified)** — registered wording licensed: *the FB/successor-feature
  family escapes the reward-free null at the behavioral level, as predicted
  by the inclusion criterion; no reconstruction-based family tested does.*
- The family-scope limitation registered on 17 Jul (headline restricted to
  reconstruction-based WMs after TD-MPC2 ns) now gets its positive
  complement: the boundary is not "everything non-reconstruction fails to
  inform" — it is exactly where the objective makes reward legible.
- Side-carried escape = support-dependence extends to the escaping family
  (non-decisional here; candidate follow-up estimand, NOT claimed).
