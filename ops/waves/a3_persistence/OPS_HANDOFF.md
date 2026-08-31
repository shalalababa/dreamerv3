# A3 PERSISTENCE — ops handoff (31 Aug 2026)

## What the wave is

The **eight A2 scarecrow primary-pair runs**, resumed IN PLACE from
staged copies and trained **5e5 → 1e6** (~7 h of new training each,
~56 GPU-h): 4 scare (seeds 80–83) vs 4 ctrl (seeds 76–79). It asks
whether A2's occupancy displacement (Δ = −0.0632, p = .0143 at 5e5)
survives continued training. It adjudicates a paper claim, so it is
fully registered: predictions, decision rule and reader frozen before
any compute.

**This wave is unusual in one way: it CONSUMES STAGED INPUTS.** It is
the first SE wave that resumes rather than starts. Everything
operational — the staging spec, the capacity arithmetic, the
materialized-bytes rule, the canary, the lane layout, why the
declaration order is double-balanced, why `nearest.json` is not the
completion witness — is in the **spec header**,
`ops/waves/a3_persistence/spec.yaml`. Read it; it is the authority and
nothing here duplicates it.

## Files

| file | role |
|---|---|
| `prereg/PREREG_a3_persistence_20260831.md` | the registration (design, resume protocol, windowed estimand, gates, outcome map, deviations from A2) |
| `uncfield/se_a3per_read.py` | the frozen reader — ONE execution, explicit `--output`; `--selfcheck` PASSes 12 scenario groups |
| `ops/waves/a3_persistence/spec.yaml` | the wave (runs, staging preflight + cmd, `done_when`, bundle) |
| `prereg/PREREG_trackA_scarecrow_20260822.md` + `amend1` | A2, the parent registration |
| `artifacts/a2_scarecrow_read_20260824/` | A2's consumed read — the reader re-verifies it at import |
| `scripts/uncfield_se.sbatch` | the producer, **UNCHANGED** — no delta is registered or required |

## GO / NO-GO gate

**NO-GO until the prereg, the reader and the spec are committed and
pushed.** Per the wave-submission chain: `commit → push-repo → gen`. A
`gen` from an unpushed tree is the stale-producer class that cost the
NOBOOT wave 4 runs, and this wave's arm identity rides entirely on
producer overrides that only a current checkout binds.

After `gen`, before submit: read the lane files back and **verify
arm × lane** against the layout in the spec header. The double-balanced
declaration order IS the registered layout; reordering the runs list
silently reassigns lanes and rounds. Run at **4, 2 or 8 lanes, never
3** (3 is arm-imbalanced).

## Staging (must happen before submit)

Per run, into `$RUNROOT/<new run_id>` on RCC, then out to the instance
(`push_donors.sh` — the eight run dirs are declared `inputs:`, so
`preflight_remote.sh` checks each one's presence too):

| new run_id | staged FROM (full A2 bundle on RCC) |
|---|---|
| `a3per_scare_s80..83` | `bundles/a2_scarecrow_20260824_184000/runroot/sc_scare_s80..83` |
| `a3per_ctrl_s76..79` | `bundles/a2_scarecrow_20260824_184000/runroot/sc_ctrl_s76..79` |

**Contents — exactly two directories:**

- `ckpt/` — the complete latest save (`agent.pkl`, `step.pkl`,
  `replay.pkl`, `done`) **and** the `latest` pointer file;
- `replay/` — **every** `*.npz` chunk (~1 400 per run, ~568 MB).
  Not a newest-first subset: the buffer capacity is 5,000,000 items
  against A2's ~4.95e5, so `Replay.load`'s cutoff never fires and a
  partial stage is silently a smaller buffer. The reader FATALLY
  refuses any run whose staged replay does not reproduce A2's recorded
  step count and occupancy for that seed.

**Nothing else.** `metrics.jsonl`, `scores.jsonl` and
`ckpt_snapshots/` must NOT be staged — `metrics.jsonl`'s first rows
are the registered resume witnesses, and the wave cmd **refuses to
start (exit 91)** if it is present before the manifest is taken.

**MATERIALIZED BYTES, never `cp -al`.** The A2 bundle is hard-linked
to the A2 runroot. An in-place resume mutates its inputs — it appends
replay chunks and `Checkpoint`'s `keep=1` cleanup DELETES the staged
save folder at the first post-resume save — so a hard-linked stage
would rewrite the A2 archive. Copy content, verify, then push.

~5 G total for the eight.

**`STAGED_MANIFEST.txt` is produced by the wave itself**, not by you:
the first thing each lane task does is list `replay/*.npz` into it,
once (guarded by `[ ! -f ]`, so a re-queued run cannot recapture its
own extension chunks). It defines the read's estimand, so it must
survive into the pull and the bundle — `done_when` asserts it.

## Canary first — registered, not advisory

Let the **first** queued run reach its first `metrics.jsonl` report
(~2 min of training after the buffer load) and check its first line
before feeding the rest of the wave:

```
head -1 $RUNROOT/a3per_scare_s80/metrics.jsonl
  "step"         must be >= 490000    (resumed, not restarted)
  "replay/items" must be >= 400000    (buffer repopulated)
```

A from-scratch run shows `step` ~4160 and `replay/items` ~3136 —
~100× away, so this is unambiguous. If either is wrong, **STOP**: the
staging is broken and every further lane burns 7 h producing a
fresh-start run wearing an extension's name. These are the same two
quantities the frozen reader gates FATALLY, so a wave that fails them
cannot be read anyway.

A staging refusal exits **91** from inside the logged subshell, so the
run's cloud log ends at an `A3PER-STAGING FATAL:` line with no
`DONE rc=` banner. That signature means staging, not training.

## Downstream

The read is run by the paper chat, once, on the pulled+verified
bundle:

```
python -m uncfield.se_a3per_read \
    --scare "<runroot>/a3per_scare_s8*" \
    --ctrl  "<runroot>/a3per_ctrl_s7*" \
    --output artifacts/a3_persistence_read_<date>
```

The reader consumes `replay/`, `STAGED_MANIFEST.txt`, `config.yaml`,
`metrics.jsonl` and `scores.jsonl` from each run dir — **the bundle
must carry `replay/` and `STAGED_MANIFEST.txt`** (a light bundle is
not read-complete, exactly as for A2). It needs no probe outputs and
no snapshot payload. A second invocation into the same output dir is
refused by design.

The **generality leg** (`ops/waves/a3_generality/`) is a separate
registration. Neither leg gates the other and they share no artefact.
