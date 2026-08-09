# PREREG: rl/sh own-label re-fit wave — 2026-08-08

Registered after the confirmed loss of the original P3 `rl`/`sh` world
models (delete-only cleanup tier, 1–2 Aug incident; user-confirmed
2026-08-08). Purpose: close review defect **D6** ("rl was never measured
against its own labels") at the population level. **This is a
REPLICATION, not a recovery**: Midway3 GPU jobs are not job-to-job
deterministic (1 Aug detprobe, g_all rank ρ≈0), so the re-fits are fresh
draws from the same training distribution — the record's disclosure that
the ORIGINAL fits were never own-label-measured stands verbatim; this
wave answers the population question those fits instantiated. No
archived verdict is re-adjudicated; the executed P3 read is untouched.

## Design (fits + measure passes only — NO adapts)

- **32 fits**: codes {sh, rl} × sides {s0, s1} × seeds 1–8, exactly the
  P3 grid (`PREREG_p3_gradient_path_20260714.md`): full task objective on
  transformed copies of the frozen `axis1_finger/q1` pair, transforms via
  `probing/relabel_replay.py` seed 0 (deterministic given the q1 buffer +
  kind + seed ⇒ label processes identical to the originals by
  construction; per-side manifest stats recorded).
- **Naming**: the reserved P3 names are reused (`ax1wm_finger_shq1s<side>_seed<k>`,
  `ax1wm_finger_rlq1s<side>_seed<k>`) — the cluster copies are deleted so
  there is no on-cluster collision. Each re-fit run dir gets a marker
  file `REFIT_20260808` at creation. These globs sit in the cleanup
  DELETE tier ("P3 arms closed"); they are added to KEEP_PENDING in the
  same freeze-commit (KEEP-PENDING wins — batch-review B4 precedent).
- **Submission**: existing plumbing, unchanged:
  `AXIS1_TRANSFORM=<sh|rl> ./scripts/submit_all.sh axis1-factorial-bundles`.
- **E4 measure passes per fit, to a NEW output dir** (`e4_finger_v1refit`,
  archived summaries never touched; any collate goes to a NEW csv, never
  merged with archived rows):
  1. true-label pass: standard `finger_v1` probeset;
  2. own-label pass: `relabel_replay transform-probeset --kind
     <shuffle|relocate> --seed 0` npz via `stratified_error measure
     --reward_override` (byte-sha manifest verify per batch-review m17).
- **Preflights (before build; halt + dated amendment on failure)**:
  (a) q1 pair chunk availability (chunkgap lesson — every episode's
  covering chunks present); (b) glob check of run names vs
  `runroot_cleanup.sh` with the KEEP_PENDING addition in place;
  (c) fit-counter verification at read time (realized-training standing
  rule: OFFLINE_FIT_PROGRESS update == total for every consumed fit).

## Registered decision rules (permutation-primary + BCa, standing rule)

Let `own` and `true` be a fit's total rew-NLL (all probeset frames, h=0)
under the own-label and true-label passes.

- **P-RS1 (rl, primary)**: paired within-fit contrast own − true over the
  16 rl fits, sides pooled (per-side n=8 descriptive). Permutation
  (sign-flip, B=10K, rng 0) + BCa 95%.
  - CI entirely < 0 and perm p < .05 ⇒ **LEARNED-BUT-UNALIGNED
    established** (D6 closed at population level; the review's
    "one-directional" caveat on the rl counterexample is discharged).
  - CI straddles 0 with own at the true-label level of the archived rl
    values (~0.9–1.2 in / ~0.38 out ordering) ⇒ **FAILURE-TO-LEARN**:
    rl does not learn its own labels either — the regime-indicator
    account is then the wrong story and §4.10 is re-corrected (dated).
  - Own significantly ABOVE true ⇒ registered anomaly, no wording
    licensed, disclose.
- **P-RS2 (sh)**: same rule on the 16 sh fits with shuffle own-labels.
- **S1 (graded secondary)**: own-label in-regime NLL level vs the
  archived task-arm member band (anchor 0.827, bar ≤ 1.5) — "learned its
  own labels as well as task learned true labels" is claimable only if
  the 16-fit CI is ≤ 1.5.
- **S2 (registered exploratory, no verdict)**: d_errin by-key fingerprint
  out-of-sample replication — `dist_to_target` out−in at h0, s0 side, for
  the fresh rl/sh fits; archived prediction ≈ −0.018 (vs task/rgo
  ≈ 0.000, `artifacts/review_response_20260808/d_errin_by_key.json`).
  Sign-negative on fresh fits = replication; reported descriptively.

Known at freeze: all archived P3/e4 rl/sh summary values (the fits that
produced them are destroyed); the d_errin by-key table; the rew_nll_out
ordering. Unknown: every own-label number (none was ever measured — that
is the defect) and all fresh-fit values.

Reader: `analysis/rlsh_ownlabel_read.py`, frozen + selfchecked BEFORE the
read (two-stage pattern, as `PREREG_swave_theory_correction_20260808`);
it implements exactly the rules above and may not weaken them. ONE read
execution. No reviewer for this registration (rides the just-reviewed
transform-probeset / --reward_override / cleanup-precedence instruments;
simple paired contrasts) — reviewer skip disclosed per standing policy.
