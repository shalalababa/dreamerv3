# The fit witness on RCC is not a completion counter — it is whichever copy was pulled last

**2026-08-16, ops session.** Found while checking wave completion after pulling
every non-SE wave from instances 1/5/7. Not a read; no verdict changes here.
The consequence is for whoever applies the 2026-08-07 standing rule
("verify fit counters in every future read"), because this is a defect in the
instrument that rule names.

## What was observed

Six adapt cells failed the wave checker on the fit-side dual witness
(`../<fit>/OFFLINE_FIT_PROGRESS~/update=500000`) while their own adapt half was
complete:

| adapt cell | RCC witness | RCC done-ckpt | adapt |
|---|---|---|---|
| `adapt_ax1fsk6q1s0_finger_seed1_ckpt500000` | 114150/500000 | **500000 (`done`)** | 112 scores |
| `adapt_ax1fsk6q1s0_finger_seed2_ckpt500000` | 115000/500000 | **500000 (`done`)** | 112 scores |
| `adapt_ax1fsk6q1s1_finger_seed1_ckpt500000` | 114250/500000 | **500000 (`done`)** | 112 scores |
| `adapt_ax1fsk6q1s1_finger_seed2_ckpt500000` | 117075/500000 | **500000 (`done`)** | 112 scores |
| `adapt_ax1fq1s0_walker_seed6_ckpt500000` | 176000/500000 | **500000 (`done`)** | 112 scores |
| `adapt_ax1fq1s0_walker_seed8_ckpt500000` | 171875/500000 | **500000 (`done`)** | 112 scores |

A fit that stopped at 114150 cannot have written a *completed* checkpoint at
500000. The two counters cannot both be describing the same process.

## Mechanism

The same run exists on more than one instance, and the copies are not equal.

`ax1wm_finger_fsk6q1s0_seed1` lives on **instance 1**, which produced it
(`update=500000`, `updated_at=2026-08-16T07:39:13`, matching its own checkpoint
directory name `20260816T073913F979417-000000500000`), and *also* on
**instance 5**, as an abandoned earlier attempt that stopped at
`update=114150` with its last checkpoint at 100000.

Pass 2 of the generated pull is `rsync -az -c --ignore-times` over a fixed list
of witness basenames. It is checksum-forced and unconditional by design — it
exists to overwrite a stale witness that an earlier mid-run pull left on RCC.
It has no notion of *which copy is newer*. So when a wave is pulled from
several instances in one sweep, the witness that survives on RCC is simply the
one from the instance pulled **last**, and an abandoned partial silently
overwrites the real one.

Instance 1 was pulled before instance 5 today. Instance 5 holds abandoned
partials of exactly those four fsk6 fits. Four stale witnesses resulted.

Two further notes on the same rsync rules:

* Pass 2's `--exclude='*'` is emitted **before** the wave-scoped
  `--include=/<run_id>` rules, so those trailing rules are dead (rsync applies
  filters in order, first match wins). Pass 2 therefore syncs witnesses for
  *every* directory on the instance, not just the wave's. That is what lets one
  wave's pull clobber another wave's witnesses.
* `-a` preserves the source mtime, so a clobbered witness carries the *older*
  file's timestamp. mtime on RCC cannot be used to tell when it was replaced.

## Scope

A scan of every `OFFLINE_FIT_PROGRESS` under `$RUNROOT` (`rcc_scan.json`):

* **63 fit directories** have `update < total_updates`.
* **63 of 63** carry a completed checkpoint at or above `total_updates`.
* **0 of 63** are genuinely short.

Affected families span weeks and many waves — `cup_*`, `finger_s12*`,
`finger_s25*`, `finger_fs12*`, `finger_fs25*`, `finger_pxpx*`, `finger_fpxpx*`,
`finger_swv*`, `finger_swr*`, `finger_bdq1d*`, `walker_fq1s0`. This is not
something today's sweep introduced; today's sweep made it visible by pulling
every wave from every instance at once.

## What this means for reads

The 2026-08-07 realized-training discovery ran the other way: runs were
walltime-truncated and the fit counters were **never recorded**, so completion
was overstated. The standing rule that followed — verify fit counters in every
read — is right in spirit, but as applied to `OFFLINE_FIT_PROGRESS` **on RCC**
it now errs in the opposite direction: it understates completion, because the
file may be a different process's leftover.

Neither counter is self-sufficient:

* `OFFLINE_FIT_PROGRESS` is authoritative **on the producing instance** and
  unreliable on RCC once duplicates exist.
* `ckpt/<ts>-<updates>/done` is a positive, forgery-resistant witness — the
  directory cannot exist without the updates having run — but it is silent
  about a fit that stopped *between* save points, which is exactly the 8-Aug
  hole.

**Suggested reading (a read's call, not ops'):** treat a completed checkpoint at
`total_updates` as sufficient evidence of a full fit, and treat
`OFFLINE_FIT_PROGRESS` as corroborating only when it is *not* below a
`done` checkpoint it should dominate. A witness below such a checkpoint is
evidence about the sync, not about the training.

## The 2026-08-07 truncation disclosure sits on this same witness

**Flagged for the owning chat (Papers 1–3). Ops does not adjudicate it.**

The 08-07 realized-training addendum reports "12m: 15/32 full, 17 truncated
(104,550–431,425)" and "25m: 22/32 full, 10 truncated (111,900–347,250)".
Those endpoints are *exactly* the endpoints of today's stale-witness scan for
the `s12`/`s25`/`fs12`/`fs25` families. Cross-checking
`realized_training_sensitivity_20260807.json` against the scan, run by run:

* **27 of 27** cells the addendum labels truncated still read below
  `total_updates` today, at the identical value — the addendum and today's
  scan are reading the same bytes.
* **27 of 27** carry a completed `ckpt/<ts>-000000500000/done`.

There is one alternative that would have preserved the addendum's reading:
the 500000 checkpoint could be a leftover from an *earlier* run of the same
`run_id`, admitted by the very `latest_ckpt()` + idempotent-fit-skip hole the
addendum identifies. Under that story the capacity-wave fit really was killed
at 104,550 and the checkpoint belongs to a different configuration.

**That story predicts the checkpoint PREDATES the frozen witness.** It does
not. Reading checkpoint directory names as UTC (the conservative direction —
if they are CDT the gaps only widen):

| done@500000 written | count |
|---|---|
| **after** the witness froze | **27** |
| before the witness froze | **0** |

Smallest gap 0.99 h (`s12q1s1_seed1`), median 4.97 h, largest 59.05 h
(`fs12q1s0_seed8`). The fits kept running and kept checkpointing for hours
after their progress files stopped advancing.

So the more likely reading is that those 27 fits **completed**, and the
addendum measured a sync artifact rather than walltime truncation. If that
holds, the consequences run the other way from the disclosure as written: the
"full-only" sensitivity analyses were re-running the primary on an arbitrary
subset, and the disclosure carried in C1/C3/C4 and the figure captions
describes something that did not happen.

What I have not done, and cannot do from ops: check whether each of those
500000 checkpoints carries the *capacity config the cell is supposed to have*.
That is the one remaining way the addendum could still be right, and it is a
read's job. Evidence for the check is in `capacity_ckpt_vs_witness.json`
(all 64 capacity fit dirs, witness + every done checkpoint with timestamps)
and `ckpt_vs_witness.py`.

## What was repaired, and what was not

**Repaired (4).** The four `fsk6` witnesses were re-pulled from instance 1, the
producing machine, and now read `update=500000`. These are the authentic bytes
recovered from the producer, not a hand-edit. `b1prime`'s fit stage went
12/16 → 16/16; its remaining 16 pending cells are the ridge panel, blocked on
the unrelated V100 backend failure.

**Not repaired (2).** `ax1wm_walker_fq1s0_seed6` and `_seed8` were produced by
an instance that no longer exists; instance 5 holds only the abandoned partial.
The fresh witness bytes are unrecoverable. Their fits are attested by a
completed 500000 checkpoint and by the adapt that consumed it
(`_ckpt500000`, 112 scores). `walker_domain` therefore reports **62/64** and
was deliberately **left unbundled** — whether those two cells count is a
science call for the owning chat, and bundling would have implied an answer.

**Not touched (57).** The remaining historical cases were left exactly as they
are. Their producing instances are gone, so no authentic witness can be
recovered, and writing a plausible one would be fabrication.

## Recurrence

Instance 5 still holds the abandoned fsk6 partials, so **any future pull of
`b1prime` from instance 5 will re-plant the four stale witnesses.** They were
not deleted (standing rule: no deletion without a verified archive, and these
partials are not archived as such). Until pass 2 learns to prefer the newer
witness, pull `b1prime` from instance 1 only.
