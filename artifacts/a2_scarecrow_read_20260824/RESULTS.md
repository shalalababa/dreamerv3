# A2 (scarecrow field-shaping, Track A) — REGISTERED READ, 24 Aug
2026: **SCARECROW-FENCES — the primary FIRES at the exact floor**

**Prereg:** `PREREG_trackA_scarecrow_20260822.md` (review #34) +
Amendment 1 (`PREREG_trackA_scarecrow_amend1_20260823.md`, full
restart + cyclic 4×5 interleave after the instance-13 breach).
**Reader:** `uncfield/se_scarecrow_read.py` (frozen; RCC copy
sha-verified byte-identical to HEAD incl. se_read/se_dose_read),
ONE execution on RCC 24 Aug 19:07, on the FULL bundle
`a2_scarecrow_20260824_184000` (the reader scans replay/ for the
occupancy primary — the light bundle is not read-complete, per the
bundle NOTES). 20/20 loaded, 0 excluded, 0 collapsed, fit counters
20/20 OK.

## Verification chain (before the read)

- Committed light manifest == RCC copy byte-exact; light bundle
  verified in place 11456/11456; all 11435 light-listed files
  hashed DIRECTLY in the FULL tree — 0 mismatches (the 20
  "missing" entries are the light bundle's own REPLAY_OMITTED.txt
  markers — benign by construction).
- **Full-manifest gap found and closed by construction:** the full
  bundle's MANIFEST.sha256 does not cover the replay/ trees of the
  8 scare/scare2 runs (7 at zero entries, s84 partial) — the
  primary's own data. Because the bundle is hard-linked from the
  pull-verified runroot, content identity was verified EXACTLY: a
  full per-file inode sweep over all 8 trees (9,536 files) shows
  every bundle replay file shares its inode with the runroot
  counterpart (0 mismatches, listings equal, link count 2), and
  the runroot copies carry ops' per-run pull verification + the
  instance-23 destroy-gate archive check. The 12 covered runs'
  replay verified by manifest count + 3 witness shas each (36/36).
  **[ops] regenerate the full MANIFEST.sha256 to close the
  archival record** — the verification above is complete but the
  manifest should not stay partial.
- Stage-1 pinned reference (#34 M14): RCC bundle
  `uncfield_se_stage1_20260813_193324`, all 24 reader-touched
  files verified against the committed manifest; n = 8 asserted.

## Devices (the A1-lesson accounting)

All 20 trainings on ONE instance (vast inst 23), Amendment-1 cyclic
4×5: every round 4 distinct arms, every lane each arm exactly once —
**arm×lane balanced by construction; no device≡arm term exists in
this wave** (the defect that caveats A1 and voided the original A2
submission). Probe panel: all 20 in ONE CPU job on midway3-0058
after ops' host-split repair; the retained originals give a
MEASURED cross-node term: byte-identical 20/20 across caslake
nodes — the CPU probe is device-deterministic (contrast the GPU
ridge probe's 0.113 AUROC).

## Primary

**occupancy(sc_scare) < occupancy(sc_ctrl): Δ = −0.0632, exact
C(8,4) one-sided permutation p = .0143 (complete separation — the
floor), full 4+4.** Per-run: scare 0.4487/0.4546/0.4736/0.4805 vs
ctrl 0.4996/0.5038/0.5254/0.5813. **A planted misprice alone — no
reward term, no penalty — steers the deployed agent's visitation
away from the fenced region.** The A1 avoidance phenomenon is now
a TOOL.

## Registered secondaries

- **S1 (scale 2.0): also fires** — Δ = −0.0736, p = .0143;
  scare2 − scare = −0.0104 (mild dose deepening, descriptive).
- **S2 (explicit penalty comparators): both fire** — pen1
  Δ = −0.479, pen2 Δ = −0.495, each p = .0143; occupancy crushed
  to 0.03–0.05.
- **S3 (descriptive, as pre-declared):** scare vs pen2
  Δ = +0.432 — the scarecrow does NOT match an explicit penalty
  (pre-declared P(win) ≈ 0.10 held). The honest sizing: the
  misprice fence delivers ≈13% of the explicit penalty's occupancy
  displacement, at zero reward-channel footprint.
- **S5 (θ₁ rows): the fence costs no global misprice** — scare
  arms 6.4–7.5 ≈ ctrl 6.4–7.6 (scare2 5.5–6.4, pen2 ≈4.4);
  substrate healthy everywhere, no collapse anywhere (S2's λ = 2.0
  did not crush exploration).
- S4 coverage rows emitted per arm (off-region coverage ≈ ctrl for
  scare arms; gated-dim off-std its own row per #34 M10).

## Consequences

- **Track A is complete and internally replicated**: A1 = the
  phenomenon (a deployed system avoids what it misprices hardest,
  p = .0143), A2 = the deliberate use (plant a misprice, fence a
  region, p = .0143 with dose response), on a device-clean layout.
  **A2 substantially defuses A1's device caveat**: the same
  directional behavior fires at the floor in a wave with no
  device≡arm term by construction.
- Field-shaping framing for the papers: uncertainty-misprice is a
  CONTROL SURFACE — steerable, dose-responsive, and invisible to
  the reward channel; but a partial fence (~13% of an explicit
  penalty), which is the honest tool claim.
- [writing chat] Track-A paper: A2 tool-demo section + the
  A1-caveat-defusal sentence; Paper-5 crosslinks: the
  endogenous-starvation rider (the ideation panel's #9 empirical
  hook — "a deployed system reduces its own visitation of the
  region it misprices hardest" — now has BOTH the phenomenon and
  the constructed form); NFI/EVM cross-reference per the venue
  plan.
- No further A2 compute registered or needed.

Artifacts: `se_scarecrow_read.json` (the consumed execution).
