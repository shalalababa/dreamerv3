# Amendment 1 to PREREG_p3_gradient_path_20260714.md — frozen 2026-07-17

New file per the amendment rule. Registered after the P3 seeds-1–8 read
(`artifacts/p3_wave_20260717/RESULTS.md`) and **before any seed-9–16
factorial outcome exists** (no such WM fit or adapt logdir exists on any
runroot or snapshot; verified by run-id grep at freeze).

## Disclosure of known outcomes at freeze

Known: all seeds-1–8 outcomes of every factorial arm — PRIMARY
B_rgo − B_sgb = +50.2 [−12.8, +111.2] (not confirmed); rgo simple
effect +65.0 [+4.2, +120.9] (descriptive CI excludes 0); sgb +14.8 ns;
the fired S2/S3 transform contrasts; the E4 factorial reward-NLL panel
(rgo 1.21 vs sgb/vgo ≈2.3). The direction of the re-test is therefore an
informed prediction; the test is prospective only through the 16 unknown
seed-9–16 outcomes per arm.

## Registered extension and decision rule

- **Runs:** `adapt_ax1{rgo,sgb}q1s{0,1}_finger_seed{9..16}_ckpt500000`
  (2 arms × 2 sides × 8 seeds = 32 jobs), identical protocol, buffers,
  and flag derivation as the frozen P3 design
  (`AXIS1_ARM=rgo|sgb AXIS1_SEEDS="9 10 11 12 13 14 15 16"
  ./scripts/submit_all.sh axis1-factorial-bundles`). Per-run audit rule
  unchanged (saved config flags per the arm table).
- **Primary confirmatory (re-test):** mean over seeds 1–16, fully
  paired, of [B_rgo − B_sgb] on AUC100k; cluster-bootstrap percentile
  95% CI (B=10,000, seed 0). **Decision on this CI alone.** CI > 0 ⇒
  the reward-head representation gradient alone carries part of the
  interaction (upgrades the 17-Jul descriptive claim to confirmatory);
  otherwise the both-paths/interplay branch of the original
  interpretation map stands as final.
- **Labeled subsidiary:** fresh-batch (seeds 9–16 only) contrast, same
  machinery — the winner's-curse check, exactly as in the W0 read.
- **Secondary (precision update):** rgo simple effect pooled 1–16 with
  CI; sgb simple effect pooled 1–16 (placebo tightness).
- Sensitivity suite (paired t, exact sign-flip permutation, Wilcoxon,
  d_z, LOO) robustness-only, never decisions.
- No other arm is extended: S2 (binding necessity) already fired at
  seeds 1–8 and vgo's null is decision-complete under the original map.
- E4 measure pass over the 32 new fits joins the existing factorial
  panel (descriptive; same probe set version).

## Ordering statement

At freeze: extension not submitted; no seed-9–16 factorial logdir
exists; `analysis/p3_factorial_read.py` (frozen 2026-07-16, committed)
is reused with `SEEDS = 1..16` for the pooled read — the only change is
the seed list, declared here in advance.
