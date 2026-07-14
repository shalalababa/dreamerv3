# P3 registration — gradient-path factorial + reward-label controls (frozen 2026-07-14)

Registers the follow-up to the confirmed occupancy × reward-supervision
interaction (W0 read 2026-07-13, `artifacts/w0_n16_interaction_20260713/`)
**before any factorial outcome exists**. Question: *which gradient path
into the representation carries the interaction, and does it require the
reward labels to be bound to the true reward states?*

## Design

All fits use the frozen Axis-1 protocol (offline fit 500K updates on a
built buffer side, frozen-readout adapt 125K steps, AUC₁₀₀ₖ from the
frozen pipeline), finger Q1 pair (`$RUNROOT/axis1_finger/q1`), paired
seeds 1–8, task-mode objective with the two representation-gradient
switches toggled. Heads are constructed and trained in every arm (only
gradient FLOW into enc/dyn differs), so parameter counts and loss sets
are identical across arms; `ac_grads` stays at its default False
everywhere (actor-critic never touches the representation).

| arm code | expl.mode | reward_grad | repval_grad | rep is shaped by | status |
|---|---|---|---|---|---|
| (apt) | apt | — | — | dynamics+decoder only | EXISTS (seeds 1–16) |
| `sgb` | task | **False** | **False** | nothing reward-linked (placebo) | new |
| `rgo` | task | True | **False** | reward head only | new |
| `vgo` | task | **False** | True | replay-value head only | new |
| (full) | task | True | True | both | EXISTS (seeds 1–16) |

`repval_loss` stays True in all task arms (the switch registered here is
`repval_grad`). Arm flags are derived inside `scripts/axis1.sbatch` from
the single registered code (`AXIS1_ARM`); the saved `config.yaml` records
`expl.mode` / `reward_grad` / `repval_loss` / `repval_grad` per run — that
is the per-run audit rule (P0-style), plus loss lines containing both
`rew` and `repval` in every task arm.

**Reward-label transforms** (controls; `probing/relabel_replay.py`, seed
0, built once per transform, manifest records per-side stats):

- `sh` — within-episode shuffle: preserves each episode's reward density
  (and the episode-level occ↔density correlation the battery measured at
  +1.0 in finger); destroys the frame-level reward↔state binding.
- `rl` — relocate: all positive reward mass moved to out-of-regime frames
  of the same episode (values preserved; destinations uniform without
  replacement).

Both are fitted under the FULL task objective (arm code = buffer suffix;
`AXIS1_ARM=full` internally), both sides, seeds 1–8. The apt arm is
provably invariant to both transforms (reward enters no apt loss and is
not a decoder key), so no apt-on-transformed fits are run; this is the
transform-artifact control by construction.

Run naming (reserved): adapt `adapt_ax1<code>q1s<side>_finger_seed<k>_
ckpt500000`, WM `ax1wm_finger_<code>q1s<side>_seed<k>`, code ∈
{sgb, rgo, vgo, sh, rl}; `PAIRED_SEED_SET=ax1<code>_finger_q1`. All match
the frozen RUN_RE and are excluded from dose-response population models by
the frozen `--modes` filter (ax1 prefix). Submission:
`AXIS1_ARM=<code> ./scripts/submit_all.sh axis1-factorial-bundles` /
`AXIS1_TRANSFORM=<sh|rl> ...`. 48 + 32 = 80 jobs.

## Outcomes and decision rules

Let B_arm = within-seed occupancy benefit AUC₁₀₀ₖ(s1) − AUC₁₀₀ₖ(s0).
All CIs: cluster-bootstrap percentile 95% (B=10,000, seed 0) over seeds;
decisions on the CI alone; sensitivity suite (paired t, exact sign-flip
permutation, Wilcoxon, d_z, LOO) robustness-only.

- **PRIMARY (sole confirmatory, fully prospective — both arms new):**
  mean over seeds 1–8 of [B_rgo − B_sgb]. CI > 0 ⇒ the reward-head
  representation gradient alone recreates (part of) the interaction on
  identical data; CI < 0 or spanning 0 ⇒ reward-head shaping alone is not
  demonstrated as the carrier.
- **Named secondaries** (each partially informed where an existing arm
  enters; disclosed):
  1. [B_vgo − B_sgb] — value-path counterpart (both arms new, fully
     prospective).
  2. [B_full(sh) − B_full(true)] — binding necessity. Prediction:
     negative (benefit collapses when labels are unbound). B_full(true)
     seeds 1–8 is known (+157.28 raw benefit deltas from the 10-Jul arm).
  3. [B_full(rl) − B_full(true)] — same machinery; relocation
     additionally redirects the labels (E4 checks where capacity went).
  4. Placebo check [B_sgb − B_apt(1–8)] — expected ≈ 0 (no reward-linked
     rep gradient in either); reported as estimation with CI, no
     decision. B_apt seeds 1–8 known (P0 read).
- Descriptive: additivity panel B_full vs B_rgo + B_vgo − B_sgb; E4
  stratified-error pass over all new WM fits
  (PREREG_e4_stratified_error_20260714.md signatures 3–4, incl. the
  relocation capacity-shift signature).

## Interpretation map (registered in advance)

- Primary fires + shuffle collapse (S2 < 0): headline mechanism =
  "reward-gradient hidden curriculum on reward-dense data" — P3 core
  claim, and Paper-1 mechanism section either way.
- Primary null but S1 fires: the carrier is the value path — same paper,
  different mechanism claim; no re-run needed.
- Primary and S1 both null with full arm reproducing: the interaction
  needs BOTH paths (or their interplay) — reported as-is.
- Shuffle does NOT collapse the benefit: reward-density-not-binding is
  the curriculum (label content irrelevant) — a stronger, stranger claim;
  relocation + E4 adjudicate where capacity moved.
- This factorial informs Paper-1's mechanism section regardless of the
  E3v2 (READ 2) outcome; P3's *flagship* status still follows the
  Amendment-2 §B gate.

## Disclosure and ordering

Known at freeze: all seeds-1–8/9–16 outcomes of the existing apt and full
arms (P0 + W0 reads), including B_full and B_apt per-seed deltas; the
13-Jul battery. Unknown: every `sgb`/`rgo`/`vgo`/`sh`/`rl` outcome (no
such logdir or WM fit exists anywhere; transforms not yet built on the
cluster). The primary and secondary S1 rest entirely on unknown data.
Protocol validation only: three 50-update CPU smoke fits of the arm flag
combos were run locally on a scratch replay on 2026-07-14 to verify flag
plumbing and saved-config audit fields (no adapt, no AUC, discarded);
`relabel_replay` and submit-section logic validated by selfcheck +
DRYRUN on synthetic data.
