# rl/sh own-label re-fit read — 2026-08-10

Registration: `prereg/PREREG_rlsh_ownlabel_refit_20260808.md` (frozen
18a774fb, 08-08) + `PREREG_refit_reads_amend1_20260810.md`. Reader:
`analysis/rlsh_ownlabel_read.py` (selfcheck PASS before execution; one
schema fix — `per_key` strata are {mean,n} dicts — made before the
execution, no outcome values seen). Bundle:
`rlsh_ownlabel_ridge_e4_20260810_220721` (sha-verified, 259 files). ONE
execution; output `read.json`.

## Gates

32/32 fits (16 rl relocate, 16 sh shuffle); all measured checkpoints
fresh Aug-9/10 at step 500000 (genuinely fresh draws — no rl/sh originals
exist anywhere; Amendment 2 census); true/own passes share the same
checkpoint per run; override sidecars exact (kind per code, seed 0,
probeset sha = finger_v1 a78fe58ca526…); counters clean (32/32
update==total, the only wave with intact counters). Per-run ridge JSONs in
the bundle NOT consumed (unregistered; Amendment 1 §4).

## P-RS1 (rl, primary): **LEARNED-BUT-UNALIGNED established**

own − true (total rew-NLL, all frames, h0, n=16) =
**−0.0408 [−0.0502, −0.0335]**, sign-flip perm p < 1e-4. Per-side −0.032 /
−0.050. D6 closed at the population level; the review's "one-directional"
caveat on the rl counterexample is discharged: rl fits demonstrably learn
their own (relocated) labels better than the true ones — support that is
legible to the objective gets learned even when it is wrong.

## P-RS2 (sh): **REGISTERED-ANOMALY (own significantly ABOVE true)**

own − true = **+0.0236 [+0.0198, +0.0265]**, perm p = .025 — the
registered anomaly branch: no wording licensed, disclosed.
**[POST-READ structure, descriptive]**: the 'all' contrast hides a
regime split — in-regime own is far BELOW true for BOTH codes (rl s0
2.24 vs 5.27; sh s0 2.46 vs 4.65), while out-regime own is above true
(e.g. rl s0 0.53 vs 0.26) and out-regime frames dominate the pooled mean
(~54.6k of 60k). rl's relocation keeps enough state-dependence for the
net effect to stay negative; sh's shuffle does not. An in-regime-scoped
own-label question would need its own registration.

## S1 (level bar): **not claimable, either code**

own in-regime NLL 16-fit CI: rl 1.690 [1.586, 1.774]; sh 1.749 [1.677,
1.909] — both above the registered 1.5 bar (task anchor 0.827). "Learned
its own labels as well as task learned true labels" is NOT licensed.

## S2 (d_errin fingerprint rider, descriptive): **sign replicates, both codes**

dist_to_target out−in (h0, s0, true pass): rl **−0.0165** [−0.0170,
−0.0161]; sh **−0.0166** [−0.0170, −0.0160]; archived prediction ≈ −0.018.
Out-of-sample replication on fresh fits of the non-gradient-arm side of
the by-key fingerprint.

## Rider levels vs archived panel (descriptive)

Fresh true-label levels: rl s1 3.44 vs archived 3.43 (tight); rl s0 5.27
vs 4.41, sh s0 4.65 vs 4.62, sh s1 1.99 vs 2.03 — population-consistent
with seed-level spread; no divergence rising to disclosure-as-anomaly.
