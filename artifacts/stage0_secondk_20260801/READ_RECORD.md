# Stage-0 second-k — Vast-matched retry + ONE read record (2026-08-01)

Registration `PREREG_stage0_secondk_20260730.md`. After the 31-Jul
cross-machine QUARANTINE (`artifacts/stage0_secondk_quarantine_20260731/`),
the registered environment-matched retry ran on a Vast instance
matching the 10-Jul dump environment (`/workspace` paths throughout the
logs). Bundle: `local_results/stage0_secondk_20260801_212204/`
(sha-verified at sync).

## Stage record (from the instance logs)

- `secondk_smoke.out`: re-dump + `--smoke_check` PASSED on the matched
  env — "guard calibration: 5 tolerance failure(s); guard_mode=fallback"
  → "SMOKE-CHECK OK: guard_mode=fallback; proceed with the 20 full
  passes" (the registered fallback branch; on RCC this same gate had
  refused with guard_mode=invalid).
- `secondk_full.out`: full passes over 10 cells (cup/finger × seeds
  1–5) × k ∈ {5, 20} × 5 checkpoints; "guard mode: fallback";
  "integrity: 0 failures / 1450 arrays checked".
- **ONE read EXECUTED ON THE INSTANCE** (frozen
  `analysis/stage0_secondk_read.py` via the staged driver); output
  written to `/workspace/dreamerv3/artifacts/stage0_secondk_20260801/stage0_secondk.json`.

## Registered outcome: K-SENSITIVE

18/20 cell×k verdicts = TRACKS_DENSITY; 2 flips, both finger @ k20:
`p2e_finger_seed2@k20 = AMBIGUOUS`, `p2e_finger_seed5@k20 = AMBIGUOUS`
(all k5 verdicts and all cup verdicts track). Frozen consequence
(verbatim from the reader): "the committed claim is k-dependent; every
external TRACKS_DENSITY statement must carry the k=10 qualifier and
enumerate the flipped cells."

Full 20-verdict enumeration (from `secondk_full.out`, lines 1315–1334):
cup seeds 1–5 @k5+@k20 all TRACKS_DENSITY; finger seeds 1,3,4 @k5+@k20
TRACKS_DENSITY; finger seed2 k5 TRACKS / k20 AMBIGUOUS; finger seed5 k5
TRACKS / k20 AMBIGUOUS.

## Provenance — synced and VERIFIED (2 Aug)

`provenance/` (pulled from the instance before teardown) closes the
chain, all four legs machine-checked locally:

- `stage0_secondk.json` bytes match the instance-recorded sha256
  `75ab1bd2…` (recomputed locally, identical).
- The json's verdict + flip set are IDENTICAL to the synced stdout
  (`secondk_full.out`): K-SENSITIVE, flips = finger seed2/seed5 @k20.
- Instance reader sha (`stage0_secondk_read.sha256`) = `3c297e18…` =
  the registered freeze sha (ledger) = the local checked-out
  `analysis/stage0_secondk_read.py`, byte-identical.
- Instance git head (`instance_git_head.txt`) `003d7b82` is an
  ancestor on the local `causal-wm-transfer` branch — the read ran on
  a clean committed checkout.

## Disclosures

- The 1-Aug repair-quarantine audit recommended a one-cell twice-run
  same-job probe before committing to the full passes; **no such probe
  appears in the logs** — the retry proceeded without it. The risk it
  guards (cross-invocation pairing) is partially covered here by the
  smoke gate itself passing on the matched env and by the read
  executing against dumps produced in the same instance session.
- The RCC-side quarantine record and its cross-machine root-cause
  analysis stand unchanged; this retry is the remedy it recommended.
