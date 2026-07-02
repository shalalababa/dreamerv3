#!/bin/bash
# ---------------------------------------------------------------------------
# Central cluster configuration -- source this from every sbatch script.
# CONFIRM every value on first login (project memory is ~6 weeks old; the
# runbook Preflight says re-check account/partition/paths and the conda env).
# Override any value from the environment: e.g. `RUNROOT=/foo sbatch pretrain.sbatch`.
# ---------------------------------------------------------------------------

# --- Slurm ---
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-pi-chenyuxin}"
export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
export SLURM_GRES="${SLURM_GRES:-gpu:1}"
export SLURM_TIME="${SLURM_TIME:-08:00:00}"
export SLURM_CPUS="${SLURM_CPUS:-8}"
export SLURM_MEM="${SLURM_MEM:-32G}"

# --- Paths ---
# Repo root (this file lives in <repo>/scripts). Resolve to an absolute path.
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RUNROOT="${RUNROOT:-/scratch/midway3/$USER/dreamerv3_runs}"
export CONDA_ENV="${CONDA_ENV:-/scratch/midway3/$USER/conda_envs/dreamerv3}"
export MANIFEST="${MANIFEST:-$RUNROOT/runs.csv}"

# --- Runtime ---
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Activate the conda env (works whether or not `conda` is a shell function yet).
activate_env() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    # shellcheck disable=SC1091
    source "$CONDA_ENV/bin/activate" 2>/dev/null || \
      export PATH="$CONDA_ENV/bin:$PATH"
  fi
}

# Append one row to the run manifest (creates the header once).
# Args: run_id phase task expl_mode seed steps ckpt_policy logdir status
# The v3 analysis columns (axis in {dose,axis1,axis2,calib,tdmpc2,d0}; a
# paired_seed_set id for within-seed contrasts) come from the env so callers
# don't need extra positional args: export AXIS/PAIRED_SEED_SET (via --export).
manifest_append() {
  mkdir -p "$(dirname "$MANIFEST")"
  if [ ! -f "$MANIFEST" ]; then
    echo "run_id,phase,task,expl_mode,seed,steps,ckpt_policy,logdir,status,su,notes,axis,paired_seed_set" > "$MANIFEST"
  fi
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,,,${AXIS:-},${PAIRED_SEED_SET:-}" >> "$MANIFEST"
}
