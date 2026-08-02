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

## Pending sync (URGENT before instance teardown)

`instance_read/` synced EMPTY — the read json and provenance records
are still only on the instance. Pull before teardown:

- `/workspace/dreamerv3/artifacts/stage0_secondk_20260801/stage0_secondk.json`
- `git -C /workspace/dreamerv3 rev-parse HEAD` + `git status --short`
  (checkout provenance of the executed reader)
- `sha256sum /workspace/dreamerv3/analysis/stage0_secondk_read.py`

The verdict itself is recoverable from the synced stdout (above), so
the outcome is safe; the json + code-sha records complete provenance.

## Disclosures

- The 1-Aug repair-quarantine audit recommended a one-cell twice-run
  same-job probe before committing to the full passes; **no such probe
  appears in the logs** — the retry proceeded without it. The risk it
  guards (cross-invocation pairing) is partially covered here by the
  smoke gate itself passing on the matched env and by the read
  executing against dumps produced in the same instance session.
- The RCC-side quarantine record and its cross-machine root-cause
  analysis stand unchanged; this retry is the remedy it recommended.
