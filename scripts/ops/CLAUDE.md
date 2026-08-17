# Ops session scope

You are the **ops session**: you submit, monitor, sync, and bundle experiment
waves. You are not the analysis session. Read
`temp_files/ops/dv3ops_design_20260814.md` before non-trivial work.

## Hard rules

1. **Never invent a run_id.** run_ids are consumed by frozen readers, prereg
   documents, and `STUDY_LEDGER.md`. They come from a wave spec, verbatim or
   template-expanded. Print the expanded set and its count, and get approval,
   before anything is submitted.
2. **Never edit an executed frozen reader.** If a read has run, its script is
   evidence. Write a new one.
3. **Never delete anything unarchived.** Standing rule until publication.
   Cluster deletion requires a verified local archive (sha + srcstat); there is
   no delete-only tier.
4. **No AI co-author trailers in commits.**
5. **Markers are not evidence.** Completion means realized counters — ckpt
   step, score-line counts, `n_ep` vs modal — not `ADAPT_DONE` existing. A run
   that exits 0 and writes its marker can still have trained nothing.
6. **Unrecognized failure → escalate.** Do not invent a fix for a failure mode
   you do not recognize. Report what you see and say you do not know.
7. **Never change a config knob to make a retry pass.** Batch size, memory
   fraction, and steps are science, not ops: changing them on 5 of 16 cells
   silently breaks comparability across arms. Same config on different hardware
   is fine; a different config is the user's call.

## Command discipline

Use `dv3ops` verbs. When no verb exists, copy the nearest canonical pattern
from `temp_files/ops/instance_sample_commands.md` and substitute only the
intended variables. Do not derive a new command shape from first principles —
past deviations caused wrong sync roots, env shadowing, duplicate queues, and
lost rented-instance time. Call out any deviation as DEVIATION, with why,
before the command.

For paid instances, prefer boring and slightly heavier over clever and
fragile: saving a few MB is not worth risking hours.

## Gotchas that have actually cost work

**ssh flattens arguments.** `ssh host 'bash -s' -- "$a" "$b"` joins everything
into one string that the remote shell re-parses, so any argument with a space
splits and shifts every later positional. On 2026-07-24 this silently queued 2
commands instead of 8 and logged to the wrong shard. Use `dv3_ssh_script` /
`dv3_ssh_dv3` from `lib/common.sh`, which pre-quote with `printf %q`. Always
check the echoed counts after submitting.

**The fork is not on a fresh instance.** The onstart template creates
`/workspace/dreamerv3` empty, so `cd "$REPO"` always succeeds even when nothing
was synced. An unsynced fork appears as scattered "No such file or directory"
on whatever relative path is touched first, never as one clear message. Run
`dv3ops push-repo N` first, or check `dv3ops versions`.

**TD-MPC2 lives in a separate env.** Every queued task re-sources
`/root/.dreamer_vast_env`, which puts `/venv/main` first on PATH — so a bare
`python` in a task line always means the Dreamer env. Use
`"$TM2_CONDA_ENV/bin/python"`. Also: create the env, seed every dep the smoke
check needs, and only then run `tdmpc2_env_setup.sh`; never put a repair step
after the setup script inside one `set -e` block, or the repair is silently
skipped and the failure reports the dep you just installed.

**Replay buffers are parent-directory units.** Sync
`$RUNROOT/axis1_finger/<quad>/` to the matching remote parent, including
`manifest.json` — `axis1.sbatch` requires `$(dirname "$REPLAY")/manifest.json`.

**RCC caps submitted jobs at 12.** `submit_all.sh` respects `MAX_JOBS=12`;
direct `sbatch` loops must too. Slurm `.out` logs stay in `$REPO`.

**GPU jobs are not job-to-job deterministic** on Midway3. Never design a
comparison that depends on bitwise pairing across invocations.

## Division of labour

- **This session (RCC login node):** gen, preflight, submit, monitor, check,
  requeue, pull, bundle, digest. Writes `ops/waves/**` and `ops/state/**`.
- **Local WSL session:** analysis, `pull-local`, manifest verification, git
  commits, and all code changes outside `scripts/ops/` and `ops/`.

Keep code edits out of this session unless they are in `scripts/ops/`; two
writers on the same repo produce conflicts, and the analysis session owns the
scientific code.

## Bundle AFTER the pull, never before (17 Aug 2026)

**Full pull instance → RCC first, then build the bundle on RCC.** Never build
a bundle on the instance and sync only that bundle.

Read bundles are deliberately a SUBSET of the runroot. `runroot_light` specs
ship no checkpoint bytes at all — FB carries checkpoint identity through
`ckpt_sha256` linkage gates instead. So if the bundle is the only thing that
ever leaves the instance, everything outside the bundle spec dies with the
instance, and every check still reads green: the bundle verifies perfectly
against its own manifest.

This happened. The FB wave was bundled instance-side, so RCC held a 1 MB
bundle while 16 trained `fb_ckpt.pt`, 17 embedding dirs and 2 exports (584 MB)
existed nowhere else. It surfaced only because the user asked whether
everything was synced before destroying the instance. The same prereg holds a
graft/distill leg open as a future registration, and that leg needs exactly
those checkpoints.

`ops/waves/se_arms/build_read_bundle.sh` is the correct shape: it reads
`$RUNROOT` on RCC. `ops/waves/fb_fits/run_emb_panel.sh` and its bundle builder
ran instance-side and are the exception to fix, not the pattern to copy.

**Verify transfers on FILE bytes, not `du -sb`.** Directory apparent size is
filesystem-dependent (instance overlayfs vs GPFS), so a whole-tree `du -sb`
reports a mismatch on every run even when the data is identical — it compares
filesystems, not contents. Use `find . -type f | wc -l` plus
`find . -type f -printf '%s\n' | awk '{s+=$1} END{print s}'` on both sides,
with shas on the small witnesses. Reserve full-tree sha for small archives.
