# f_R wave: donor-chunk gap halts the q1f build (2026-08-02)

Registered response: `prereg/PREREG_highfr_wave_amend2_20260802.md`
(chunk-availability filter). Parent chain: `PREREG_highfr_wave_20260802`
(freeze `39a15880`) → Amendment 1 (freeze `be60afbe`).

## Facts

- Amendment-1 curation succeeded, in-gate, pre-fit: side1 0.7293
  (registered fallback, reproduces the SHORT run); side0 0.4098
  (dev 0.0012 from target 0.41; a CLOSER window than the SHORT
  diagnostic's 0.4130 max-mean window — Amendment 1's ≈0.4130
  prediction mistook the max-mean window for the distance-minimizing
  one; constants all honored, no decision consequence).
- Build failed: `FileNotFoundError` on
  `/scratch/midway3/rickybao/dreamerv3_runs/pilot_p2e_finger_seed1/replay/20260701T215001F547386-1Hb2yS6vRENeKKyMUq5r7l-43IsWx5imPbuTprM9N2yyO-251.npz`.
- User preflight over the emitted pair: needed_chunks = 2046,
  missing_chunks = 26 (list to be archived here, see below).
- Missing chunks dated 2026-07-01. Working hypothesis: scratch purge of
  files never read since collection — prior curations materialized only
  their selected episodes' chunks (refreshing atime); this wave's
  high-occupancy tail picks episodes never before built. Not previously
  observed because every earlier curation drew from the
  previously-touched part of the pool.
- No frames fit, no adaptation, no transfer outcome exists. The failed
  build's partial output (side0 written before the crash) must be
  deleted before rebuild.

## Consequence

- Curation re-registered with `--require_chunks`
  (pool = buildable episodes only; eid numbering preserved; excluded
  eids + missing files recorded in the pairs json). All targets, tol,
  fallback, SHORT/halt semantics, gates, and downstream protocol
  unchanged; realized means may shift and the registered gates bind.
- Selfcheck reproduces the exact field failure end-to-end (PASS).

## To archive here (user, from RCC)

- [ ] `frew_pairs_unbuildable_20260802.json` — the Amendment-1 pairs
      json that referenced missing chunks (overwritten by the re-run;
      copy before re-running).
- [ ] `missing_chunks_20260802.txt` — the preflight's full 26-file list
      with per-side attribution.
- [ ] Post-re-run: the new `frew_pairs.json` (with its `availability`
      block) alongside, as usual for this wave's bundle.
