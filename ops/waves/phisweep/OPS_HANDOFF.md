# OPS HANDOFF — PHI-SWEEP

Registered 30 Aug 2026. Prereg `prereg/PREREG_phisweep_20260830.md`;
spec `ops/waves/phisweep/spec.yaml`; reader
`uncfield/se_phisweep_read.py` (selfchecked, frozen, ONE execution).
**Nothing has been submitted.** This is the chain to run when the wave
is authorized.

## What it costs

| | |
|---|---|
| runs | **8** (2 phi arms x 4 seeds) |
| per run | dmc_cheetah_run, 5e5 steps, ~7 h, ~4.6 G |
| total | **~56 GPU-h**, ~36.8 G on disk (~18.4 G in flight at 4 lanes) |
| instance | **ONE box**, `per_gpu: 1` (SE = 15243 MiB of a 16311 MiB card) |
| co-tenancy | may share the **box** with the A3 wave — different GPUs, never the same GPU. Both phi arms stay on ONE box (22-Aug memory: rented boxes run different drivers on the same card model). |
| probes | NOT in this wave. The 8 `se_probe` passes run afterwards in ONE CPU job, arm block == seed block (B1 rule). |

## The arm pins (what must be true on the metal)

| arm | seeds | `DISTRACTOR_THETA` | `DISTRACTOR_SCALE` |
|---|---|---|---|
| phi = 0.80 | 140–143 | `0.20` | `1.3764944032233706` |
| phi = 0.95 | 144–147 | `0.05` | `0.7163503994113789` |

`phi = 1 - theta`. The scales are the s/s_0 control
(`s = sqrt((1-phi^2)/0.19)`), passed at FULL precision — the reader
pins them to 1e-6 and refuses the read otherwise. Seeds 140–147 are
fresh (nothing above 139 is registered anywhere).

## Submission chain (house order; every step gates the next)

1. **commit** the four registered files: the prereg, the reader, the
   spec, and the producer delta in `scripts/uncfield_se.sbatch`
   (`--distractor.theta "${DISTRACTOR_THETA:-0.1}"` + its BIND-CHECK
   `want()` line, both default-inert). *No AI co-author trailer.*
2. **`dv3ops push-repo <instance>`** — load-bearing here, not routine.
   The lane command sets `DISTRACTOR_THETA`, and a checkout that
   predates the producer delta would simply not read it: the run would
   train at phi=0.9 under a phi=0.80 run_id. That is the 21-Aug NOBOOT
   incident exactly (4 runs / ~25 GPU-h as the wrong arm).
3. **`dv3ops gen ops/waves/phisweep`** — then **read the lane files
   back** and assert each command body is non-empty and carries
   `DISTRACTOR_THETA=` and `DISTRACTOR_SCALE=`. Prune any stale
   `generated/` first.
4. **`dv3ops preflight <instance>`** — must print `PREFLIGHT PASS`.
   Its stale-producer gate greps the instance's own
   `scripts/uncfield_se.sbatch` for `${DISTRACTOR_THETA`; a `bad
   STALE PRODUCER` line means go back to step 2. **Never hand-queue a
   raw `.cmds` file** — that bypasses this gate, which is the only
   check that can see the failure.
5. **Canary.** Submit **one** run (`se_phi095_s144` — the arm whose
   prediction is the extreme one) and let it actually REACH RUNNING and
   clear BIND-CHECK before feeding the rest. A submit ledger proves
   acceptance, never execution. Confirm in its log:
   `BIND-CHECK PASS: all wave-critical overrides bound`, and that
   `config.yaml` carries `theta: 0.05` and `scale: 0.716...`.
6. **Feed the remaining 7**, preserving the declaration order — the
   BLOCKED order IS the registered lane layout (see the spec's LANE
   LAYOUT block; reordering silently makes lanes arm-pure).
7. **Watchdog** keyed to realized counters (8 `TRAINING_DONE`), not to
   idle time. Hourly `dv3ops board` heartbeat overnight.
8. **Pull continuously**, per run as it finishes — `dv3ops pull`, then
   verify (file count + file bytes + witness shas, as a SUPERSET not
   an equality), then delete instance-side before that lane starts its
   next run. `replay/` is REQUIRED in this bundle (the reader's
   coverage proxy) — never trim it instance-side.
9. **Release the box** only after: 8/8 realized, verified pull, the
   downstream consumer's gates run on the RCC copy, and the box is not
   apparatus for A3. Then `dv3ops destroy --yes`.
10. **Bundle + manifest + NOTES**, `dv3ops bundle` on the RCC run root
    (bundle AFTER the pull, never before), commit the sha256 manifest,
    sync the local archive copy to `/mnt/c/dv3_archive`.
11. **Probe pass** — the 8 `se_probe` runs in ONE CPU job.
12. **Run the reader's gates on the RCC copy before handing over**: a
    dry `read_run` over the 8 dirs must not raise. The read itself is
    ONE execution with an explicit `--output`; a gate failure refuses
    the whole read *before* the output json is written, so a
    repair-and-rerun is safe.

## What comes back to this chat

The read, and only the read:

```
python -m uncfield.se_phisweep_read \
    --runs "<runroot>/se_phi*_s14*" \
    --stage1_runs "<stage1 bundle>/se_cheetah_seed*" \
    --output artifacts/phisweep_read_<date>
```

Hand back `artifacts/phisweep_read_<date>/se_phisweep_read.json` plus
the one-line console verdict. The interpretation stays in the Paper-5
chat (reads stay with the paper chats). The registered outcome cells
are **LAW-CONFIRMED / LAW-REFUTED / LAW-SPLIT / NOT-ADJUDICABLE /
GLOBAL-COLLAPSE**; PRIMARY-2 (the raw-vs-symlog convention) is
pre-declared under-powered and its expected cell is
`CONVENTION-NOT-ADJUDICATED`.

Ledger: add the artifact record to `STUDY_LEDGER.md` in the **same
session** that writes it.
