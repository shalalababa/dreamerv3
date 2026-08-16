# Salvage from `tm2_bridge_20260815_200027` before it was deleted

**2026-08-16, ops.** (Tracked copy: `artifacts/tm2_shardlog_salvage_20260816/`; `local_results/` is gitignored under results-sync policy v2, so the git copy is the durable one.)

The 15-Aug TM2-bridge bundle was superseded by
`tm2_bridge_20260816_173625` (485 files vs 197) once the adapt stage was
transcribed and the wave went 48/48 → 96/96. The old bundle was deleted
locally and on RCC. This directory is what did **not** carry over.

## Why the old bundle was not simply redundant

A file-by-file comparison put every one of the old bundle's 197 files in the
new one, with **four exceptions** — `_cloud_logs/worker_shard{0,1,2,3}.out`.
Those are not stale copies of the same thing:

| | |
|---|---|
| lines present only in old `worker_shard1.out` | **80** |
| lines present only in old `worker_shard2.out` | **42** |
| old is a byte-prefix of new | shard0 only |

The cause is a **filename collision across instances**. Every rented instance
writes its own `_cloud_logs/worker_shard0..3.out`, and a pull flattens them
into one runroot, so whichever instance was pulled last wins. The 15-Aug
bundle captured one instance's queue-event history; the 16-Aug bundle
captured a different (still-live) instance's. Neither is a superset of the
other, and the instance behind the older logs has since been destroyed.

Same shape as the `OFFLINE_FIT_PROGRESS` clobber recorded in
`artifacts/stale_fit_witness_20260816/` — a per-instance file whose name does
not carry the instance.

## What is here

* `worker_shard{0,1,2,3}.out` — the 15-Aug bundle's copies, verified
  byte-identical to the originals (sha256) before the bundle was removed.
* `tm2_bridge_20260815_200027.sha256` — the bundle's committed manifest, so
  the deleted contents remain fully attested.

These are ops provenance: queue START/DONE/QUEUE_ADD events. Nothing here is
science payload, and no reader consumes them. They are kept because the
standing rule is that deletion requires a verified archive of anything that
exists in only one place, and these lines did.

## What was deleted

* `local_results/tm2_bridge_20260815_200027/` (3.4 GB)
* `/scratch/midway3/rickybao/dreamerv3_runs/bundles/tm2_bridge_20260815_200027/`

The 480 science files they held are byte-identical inside
`tm2_bridge_20260816_173625`, which is manifest-verified locally.
