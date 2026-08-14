# B2 nuisance-injection λ trial (crowding test) — ONE registered read — 2026-08-13

- Registration: `prereg/PREREG_nuisance_20260811.md` (post-review
  REDESIGNED form; sha256
  151b00a76c5a5858409434b90eaa5175af0d7d2edab8a54e894db36bad5a5952,
  freeze-committed 8cd62845 on 2026-08-12, before any nz fit
  existed). Part-B predictions + verdict mapping frozen in
  `PREREG_theory_R1_gating_20260811` + Amendment 1.
- Bundle: `local_results/b2_nuisance_20260813_213844` —
  MANIFEST.sha256 verified (107 entries, all OK; identical to
  committed manifest) before any file was opened. 32 renamed ridge
  jsons + witness jsons + both nz buffer manifests + runroot_light.
- Reader: `analysis/nuisance_read.py` (sha256 af4867c3…86220),
  selfcheck PASS re-run immediately before execution. THIS is the
  wave's single read. All gates passed: 32-name dual witness, per-fit
  distractor config gate (dim 16, basesd 1.0/4.0 by load; expl mode
  by arm), replay-manifest σ/dims gates on BOTH nz buffers,
  ridge-json gates (run_id↔filename, load-matched probeset_id, no
  reward_override, witness_match), exact-32 inventory.

## Verdict (registered rules)

**NEGATIVE-DID: the task arm lost MORE — outside both models'
registered predictions; fact reported, no further wording.**

- **Informativeness gate PASSED**: mean apt AUROC at nz1 = **0.718**
  ≥ 0.55 — the reward-free arm had real legibility headroom on this
  wave's distractor probesets (the disclosed censoring risk from the
  0.401 anchor did not materialize; different probeset distribution).
- **P-N2a (primary)**: DID = [task−apt](nz2) − [task−apt](nz1) =
  **−0.196** [−0.258, −0.096], aligned perm p = .0001 (n=8/group).
- Panels: task 0.913 (nz1) → 0.679 (nz2), load slope **−0.234**;
  apt 0.718 → 0.681 (−0.037). Crowding barely touches the
  reward-free arm's legibility and hits the TASK arm hard.
- MDE note (in-read): per-group sd 0.051, DID se ≈ 0.036, 80%-power
  MDE ≈ 0.101 — the observed |DID| = 0.196 is ~2× the MDE; this is a
  decisively powered negative, not a fragile one.

## Part-B adjudication (frozen mapping applied)

- B2 = `NEGATIVE-DID` ⇒ **λ-negative** per Amendment-1 mapping.
- B1 (read earlier today) = `INSTRUMENT-LIMITED` ⇒ **neither**.
- Realized pair = (neither, λ-negative) = a **split/partial outcome:
  reported as-is; no wording beyond the branch definitions is
  licensed** (base adjudication text). λ-competition is NOT
  "decisively dead" (that required both λ-negative); the two-regime
  model is NOT adopted (no λ-positive); **λ remains
  UNRESOLVED-IN-SCOPE with one λ-negative leg on the scoreboard.**
- Note the registered branch definition itself: NEGATIVE-DID is
  outside BOTH models' predictions — λ predicted the reward-free arm
  loses more (a² rescue of the task arm); R1 predicted NO load×arm
  interaction. The realized fact (a large, well-powered interaction
  in the direction neither predicted) is adverse to each as written.

## Post-read notes (labeled, non-registered)

- Mechanism-coherent reading of the surprise: the task arm's
  legibility at nz1 (0.913) had the most room to fall, and its
  reward-relevant features are what the added channels compete with
  under a shared reconstruction budget; the apt arm's legibility was
  already mid-range and its support is reward-agnostic, so crowding
  redistributes it less. That story is post-hoc — it resembles a
  regression-to-ceiling account as much as a competition account, and
  ONLY a design with matched baseline legibility could separate
  them. Any such wave is a new registration (and note B2's realized
  form ALSO contradicts R1's "no interaction" as written — the theory
  ledger entry must carry this both-sided miss honestly).
- The apt-arm 0.718 here vs the 0.401 pooled anchor is a probeset
  effect (distractor-context heldout episodes, level-invariant
  filenames), not a contradiction — different probe distribution,
  value-aware note only.
