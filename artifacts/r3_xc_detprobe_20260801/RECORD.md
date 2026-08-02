# XC determinism probe — FAIL (run 2026-08-01, recorded 2026-08-02)

Pre-wave diagnostic probe for the cross-checkpoint consumer wave
(`PREREG_r3_amend2_20260730`), recommended by the 1-Aug
repair-quarantine audit (`artifacts/r3_repair_quarantine_20260801/AUDIT.md`:
probe cross-invocation reproducibility with a one-cell twice-run BEFORE
committing a wave whose pairing assumes it). This probe is DIAGNOSTIC,
not part of the registration: no registered gate ran, no registered
outcome fires, and NO wave label exists.

## What ran (from the npz meta; integrity fields only)

The registered CONTROL configuration through the overlay code path —
`labeler_version d1fix_20260724_xc1`, `consumer_checkpoint` = the eval
checkpoint itself (`r3_cup_e1_seed31/ckpt`, late), registered regex
`^(rew|con|valens\d+)/` — executed TWICE with identical dials
(horizon 100, label_every 25, ref_stride 5, max_steps 1000, labeler
seed 0) as separate invocations: `..._late_a.npz` / `..._late_b.npz`.
Labeler seed 0 ≠ the wave's registered seed 1 and the filenames carry
no `_h{headsmat}` cell suffix, so these files can never be mistaken
for wave-grid files.

## Result (`xc_detprobe_result.json`): **FAIL**

- `episode_exact: true`, `step_exact: true` — state selection is
  perfectly reproducible across invocations (env + policy + counter
  path is deterministic, as the overlay design verified).
- `g_all_close: false`, **max |Δg_all| = 95.0** — the oracle rollout
  returns are NOT reproducible across invocations.
- `m_now_exact: false`, **64/200 states flip the plugin chooser**
  (32% — same order as the repair audit's ~25% m_now / ~50% m_real
  cross-job flip rates).

## Consequence

**The registered two-pass pairing is unexecutable on this substrate.**
The wave's pairgate requires control and swapped passes to agree on
`g_all` to atol 1e-5; the probe shows drift 7 orders of magnitude
above that on identical states, in the most favorable setting (same
machine, same checkpoint, back-to-back). Had the wave been submitted,
the registered pairgate would have tripped after 2 of 128 passes ⇒
QUARANTINE. The probe catches this pre-wave at ~2 passes' cost; the
126 remaining passes are NOT submitted. This is the same substrate
fact that invalidated the repair parent (job-to-job GPU
nondeterminism; standing rule: never design cross-invocation bitwise
pairing) — trajectory determinism holds, float determinism does not.

Path forward (queued for GO, not yet built): **xc Amendment 1 —
within-pass dual-chooser pairing**, mirroring repair Amendment 1: one
pass per (run, eval side) computes states + `g_all` ONCE and evaluates
BOTH choosers (control heads and overlaid consumer heads) on the same
stored features; per-state paired contrast
`[g_all[m_real_sw] − g_all[m_now_sw]] − [g_all[m_real_ctl] − g_all[m_now_ctl]]`
never compares floats across invocations. 64 passes instead of 128
(halves the wave cost). Requires a labeler extension + amended reader
+ dated amendment before any wave label.

## Disclosure impact (for the future amendment)

- `--consumer_checkpoint` HAS now executed — control configuration
  only (consumer ≡ eval checkpoint), one cell, twice, labeler seed 0.
  No swapped-heads pass has ever run; no cross-checkpoint statistic
  exists anywhere.
- Value-blindness: the probe compared integrity quantities only
  (episode/step identity, `g_all` closeness, `m_now` agreement); no
  estimand (per-state achieved / delta_real means) was computed or
  printed (`_xc1` stdout redaction active in both invocations). The
  npz files DO contain estimand-bearing arrays; they are kept as probe
  evidence and MUST NOT be read outside a registered read.

## Provenance

- Git head at probe time `a0b3d0c8` — verified committed ancestor on
  `causal-wm-transfer`.
- `instance_provenance.txt` synced EMPTY; the meta's `/workspace/...`
  paths show the Vast R3 substrate instance. **USER-CONFIRMED
  (2 Aug): the a/b invocations shared ONE instance session** — the
  strongest form of the failure: same machine, same session,
  back-to-back invocations. The drift is invocation-level; no
  execution-detail pinning (same job, same node, same session) can
  rescue cross-invocation float pairing.
