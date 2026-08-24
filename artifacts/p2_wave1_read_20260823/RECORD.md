# P2 Wave 1 — first-update cell, ONE read (23 Aug 2026)

Registration: PREREG_p2_wave1_firstupdate_20260821.md (frozen
8828ab6f consolidated). Bundle: p2_wave1_20260823_004802 (manifest
35/35 OK; 32 registered cells + 2 CRN dup gates, exactly 34 npz, no
unregistered wv1* files; reader gates all passed — env_seed 20260821,
repeats 4, states 400, dup gates exactly zero; 12 refusal legs armed,
none tripped).

**Procedural deviation, disclosed**: the registered ordering called
for the 4 pilot cells first + a dispersion-only `--pilot` look before
the remaining 28; ops ran all 32+2 in one job (10h45m, instance 12)
and no pilot look was ever executed. The registration says the wave
proceeds regardless of the pilot — its only power was to ADD cells
via the re-power trigger — so the read on the registered 32 is
unaffected; what was lost is the chance to have enlarged the wave
pre-spend. No reader gate involves the pilot. Recorded, not cured.

## Verdict: **MECHANISM-CLOSED-NEGATIVE**

| estimand | value | rule |
|---|---|---|
| PRIMARY d_raw (first-update damage, damage-positive), n=32 | **+0.0556 [−0.0279, +0.1479], perm p = .229** — no fire | α=.05, MC sign-flip 200000 |
| realized MDE80 | **0.1269 < the +0.1449 registered target** | adequately powered for the registered effect |
| C5 own-objective delta (rider) | −0.3054 [−0.959, −0.028], perm p = .157 — no fire, not significantly negative | descriptive/rider |

## What this settles

- **No first-update damage**: the last live mechanism candidate for
  the anti-harvest/replication-failure complex closes NEGATIVELY at a
  realized MDE80 tighter than the registered target — the registered
  branch text: "strictly more informative than CLOSED-UNATTRIBUTED."
  The 14-Aug W2 verdict (CLOSED-UNATTRIBUTED) upgrades to a closed
  negative on this channel.
- C5 (the dual-objective oracle riding the same passes) does not
  fire: the one-step-real-env own-objective contrast is a loose
  negative point — the channel-D discounting/horizon half shows no
  significant own-objective advantage either.
- **Registered consequence: the WCEM wave is now UNLOCKED**
  (PREREG_p2_cem_consumer execution was sequenced behind this read;
  the `--after_wave1` code gate may now be satisfied per its own
  procedure).
