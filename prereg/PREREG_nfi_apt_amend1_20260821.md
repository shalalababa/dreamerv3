# AMENDMENT 1 — B2 APT smoke-gate adjudication, 21 Aug 2026
(pre-outcome; before any of the 8 registered submissions)

**Parent:** `PREREG_nfi_apt_20260821.md` (frozen 771445ab), §1
("smoke FILLs (if any) pinned by amendment before the wave freezes")
and §3 (smoke-gate criteria, #29 C-M7 / #30 B2 form).

## Smoke result (`se_apt_smoke`, seed 9, 2e4 steps; gate executed by
ops via `se_apt_read --smoke_gate` — NOT the registered read)

- `instrument_gates` PASS (S == 512 pin; dup0 delta 0.0 exactly,
  vector-level; mask forms; knob pins; json↔npz provenance).
- `planted_keys_live` PASS (all 5 per-key decoder losses live+finite).
- **`amendment_trigger` = false**: resample-control means (4.7e-4,
  6.8e-4) sit under half the distractor delta (8.1e-4) ⇒ **the
  registered SIGN fire rule stands**; the control-relative comparator
  (`--fire_rule control_relative`) is NOT activated. This was the
  gate's registered decision input, and it is clean.
- `run_completed` FAIL: last step 18,976 / 20,000 = 0.9488 < 0.95.

## Adjudication (registered FILL): the gate is PASS with a disclosed
scale artifact; the wave is GO

The `run_completed` check (#30 B2) exists to catch a CRASHED run
whose replay still holds ≥512 windows. This run is directly shown
not to be that: the final checkpoint `20260821T181049F010178` is
recorded at step 18,960 in the snapshot manifest — the run stopped
at its boundary, ~1,000 steps short of the nominal target, with the
instrument passing end-to-end on it. The shortfall is the ABSOLUTE
producer stop-boundary tail, measured by ops on four completed
5e5-step SE runs (se_noboot_s40 497,536 = .9951; se_noboot_s41
499,024 = .9980; se_dose10_s53 497,472 = .9949; se_avoid_het_s44
499,264 = .9985): a ~1–2.5k-step tail is 0.2–0.5% at 5e5 and 5% at
2e4 — the 0.95 threshold is one a 2e4 smoke structurally cannot
clear. The gate's purpose (completion evidence) is discharged by the
checkpoint-step witness; the threshold, calibrated for wave-scale
runs, is simply mis-scaled for the smoke.

**Pinned consequences:**
1. The frozen reader `se_apt_read.py` is UNTOUCHED (no threshold
   edit, no `--smoke_steps` gaming). Its wave-read fit counters run
   at expect_steps = 5e5, where the measured boundary tail sits at
   0.995+ — the 8 registered runs are not at risk from this check,
   and any wave run that DOES flag at 5e5 is a real truncation
   suspect, reported as registered.
2. The fire rule for the wave is the registered SIGN rule (the
   comparator amendment path is closed by the trigger's negative
   result above).
3. Expected wave completion ratios ≈ 0.995 (the four-run measurement
   here) — the reference for interpreting the read's fit flags.
4. The 8 submissions (`EXPL_CONFIG=expl_apt`, seeds 68–75) are
   authorized. NOBOOT-incident rule applies: instances must carry a
   checkout ≥ the batch-2 freeze (the producer BIND-CHECK asserts
   `expl.mode == apt` binds within minutes; the smoke itself already
   proved EXPL_CONFIG binds on its instance).

Freeze of this amendment = commit of this file before any wave
submission.
