# A3 GENERALITY — ops handoff (30 Aug 2026)

## What the wave is

8 SE training runs on `dmc_finger_spin` (5e5 steps each) that transplant
the A1 avoidance design to a second environment at a dose- and
contrast-matched manipulation: 4 **hetero** (seeds 210–213, the solved
mod quadruple at `scale` 1.0) vs 4 **flat** (seeds 214–217,
amplitude-matched comparator, no gradient). It adjudicates a paper
claim, so it is fully registered: predictions, decision rule and reader
are frozen before any compute.

Everything operational — lane layout, instance policy, disk, the
`GATE_THRESHOLD` trap, why the declaration order is blocked — is in the
**spec header**, `ops/waves/a3_generality/spec.yaml`. Read it; it is the
authority and nothing here duplicates it.

## Files

| file | role |
|---|---|
| `prereg/PREREG_a3_generality_20260830.md` | the registration (design, primary, gates, outcome map, deviations from A1) |
| `uncfield/se_a3gen_read.py` | the frozen reader — ONE execution, explicit `--output`; `--selfcheck` PASSes 11 scenario groups |
| `ops/waves/a3_generality/spec.yaml` | the wave (runs, cmd, done_when, bundle) |
| `prereg/PREREG_a3_calib_amend1_20260830.md` §6 | the pin authority for every constant |
| `artifacts/a3_calib_amend1_20260830/` | the calibration read the pins come from |

## Changes made to the drafted spec at registration (30 Aug)

Applied in place, before freeze — see the report accompanying this
handoff for rationale:

1. added the `ckpt_snapshots/nearest.json` milestone predicate to
   `done_when` (TRAINING_DONE is a lying marker for this producer);
2. per-run arm-identity `done_when` pattern (`${armpat}`) replacing an
   alternation that matched either arm;
3. explicit inert overrides `DISTRACTOR_THETA=0.1`, `DISAG_HEAD=det`,
   `DISAG_BOOTSTRAP=True`, `EXPL_CONFIG=expl_p2e` (all producer
   defaults, all BIND-CHECKed, behaviourally identical);
4. header notes: `GATE_THRESHOLD` is a reader constant and must stay
   `0.0` in the cmd; the replayed lane layout at 2/3/4/8 lanes;
   the seed-disjointness verification; the producer BIND-CHECK audit.

`spec.yaml` loads, expands and substitutes cleanly after the edits
(`wavespec.load` + `balanced_order → pair_runs → assign_lanes` replayed).

## GO / NO-GO gate

**NO-GO until all three files above are committed and pushed.** Per the
wave-submission chain: `commit → push-repo → gen`. A `gen` from an
unpushed tree is exactly the stale-producer/stale-checkout class that
cost the NOBOOT wave 4 runs, and this wave's arm identity rides on
producer overrides that only a current checkout binds.

After `gen`, before submit: read the lane files back and **verify
arm × lane** against the layout in the spec header (blocked declaration
order is the registered layout, and reordering the runs list silently
reassigns lanes).

Nothing else in this wave is the analysis session's to schedule.

## Downstream

The read is run by the paper chat, once, on the pulled+verified bundle:

```
python -m uncfield.se_a3gen_read \
    --hetero "<runroot>/se_a3gen_het_s21*" \
    --flat   "<runroot>/se_a3gen_flat_s21*" \
    --output artifacts/a3_gen_read_<date>
```

The reader consumes `replay/`, `config.yaml`, `metrics.jsonl` and
`scores.jsonl` from each run dir. It needs **no probe outputs** — the
probe-dependent legs of A1 are out of scope for this leg and the
reasons are registered in the prereg §7. A second invocation into the
same output dir is refused by design.

The **persistence leg** (extending the A2 checkpoints) is a separate
future registration and is not part of this wave.
