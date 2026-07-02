# Cluster launch scripts

Machine-specific Slurm launchers for the world-model transfer study
(`research_notes/Research_Execution_Runbook_v2_20260701.tex`). Not committed.

## First: confirm the environment (project memory is ~6 weeks old)

Edit `env.sh` (or override from the shell) and re-check on first login:

- `SLURM_ACCOUNT` (`pi-chenyuxin`), `SLURM_PARTITION` (`gpu`), `SLURM_GRES`
- `CONDA_ENV` (`/scratch/midway3/$USER/conda_envs/dreamerv3`)
- `RUNROOT` (`/scratch/midway3/$USER/dreamerv3_runs`)
- `MUJOCO_GL=egl`

Sanity check locally/first (Phase 0): `python -c "import jax, portal, dm_control,
mujoco, elements, embodied; import dreamerv3.main; print('ok')"`.

## Workflow (phase -> command)

All commands run from the repo root. Add `DRYRUN=1` to print without submitting.
`submit_all.sh` is idempotent by default: it skips any `RUN_ID` that is already
queued/running in Slurm or has a non-empty `$RUNROOT/<RUN_ID>` directory. Use
`FORCE=1` only when you intentionally want to resubmit into an existing run ID.
It also respects RCC's submitted-job cap: by default `MAX_JOBS=12`, so it fills
available slots and stops. Re-run the same command after jobs finish; already
submitted/completed run IDs are skipped.

```bash
# Phase 3 -- Gate 0 pilots (p2e/apt/random/goal x cup/finger), ~1e5 steps each
./scripts/submit_all.sh pilots
# then, when they finish, evaluate feasibility (per domain):
python -m probing.gate0 --task dmc_cup_catch --window 50 \
  --replay p2e=$RUNROOT/pilot_p2e_cup_seed1/replay \
           apt=$RUNROOT/pilot_apt_cup_seed1/replay \
           random=$RUNROOT/pilot_random_cup_seed1/replay \
           goal=$RUNROOT/pilot_goal_cup_seed1/replay \
  --output $RUNROOT/gate0_cup
#   -> gate0.json / gate0.png with a GO / NO-GO verdict (plan Sec. 3.3).

# Phase 4 -- reward-free pretraining WITH checkpoint retention (walker+cup+finger)
./scripts/submit_all.sh pretrain           # defaults to SEEDS="1 2 3 4 5"
# If 12 jobs are already active/submitted, it stops without submitting more.

# Phase 5 -- frozen-readout dose-response from the retained snapshots
python -m probing.checkpoint_watcher --select \
  --run_logdir $RUNROOT/pretrain_p2e_cup_seed1     # writes nearest.json
./scripts/submit_all.sh adapt pretrain_p2e_cup_seed1 dmc_cup_catch 1.25e5
```

## Pieces

| file | role |
|---|---|
| `env.sh` | central config (Slurm, paths, conda) + `manifest_append` helper |
| `pilot.sbatch` | Phase 3 short collection; `MODE=goal` = reward-on goal-reacher |
| `pretrain.sbatch` | Phase 4 pretraining + `probing.checkpoint_watcher` sidecar |
| `adapt.sbatch` | Phase 5 frozen-readout adapt from a snapshot dir |
| `submit_all.sh` | expands the sweep grids; `pilots` / `pretrain` / `adapt` |
| `runs.csv` | run manifest (auto-appended by the sbatch scripts) |

## Notes

- **Retention.** Saves are clock-based (`run.save_every`, seconds), so
  `pretrain.sbatch` sets it small (300 s) and runs the watcher, which copies
  every completed save keyed by exact env step. Milestones map to the nearest
  snapshot via `--select` (`nearest.json`); the dose-response x-axis uses the
  *actual* step, not the nominal milestone.
- **Goal-reacher (Gate 0).** Without the reward-on `goal` pilot the
  high-occupancy corner is empty (exploration alone rarely enters the regime --
  reward is sparse on cup/finger/reacher). For a faster substitute on
  reacher/cup a scripted controller is acceptable.
- **Gate 0 fallback ladder** (if NO-GO): (a) add targeted auxiliary collection
  to populate the empty corner; (b) switch/add domain
  (`dmc_manipulator_bring_ball`, `dmc_reacher_hard`); (c) rescope the causal
  question to forward- vs value-sensitive accuracy (separable on Walker, no
  occupancy dissociation needed).
- **Rendering off for proprio.** Data-collection jobs pass
  `--env.dmc.render False` (new `render` toggle in `embodied/envs/dmc.py`): the
  agent never consumes the image in proprio mode, so rendering is pure per-step
  overhead. Set `RENDER=True` to restore eval videos. This also sidesteps a
  local-only WSL2 crash (MuJoCo EGL context vs CUDA); the cluster is unaffected
  either way.
- **Compute report.** Capture SU / wall-clock per job (plan Sec. 5.1); the
  `.out` files record wall-clock, and the manifest has an `su` column to fill in.
