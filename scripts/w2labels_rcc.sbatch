#!/bin/bash
# Wave-2 labeling on RCC — 8 array tasks, ONE SEED EACH, 4 passes per task.
#
# Shape (user decision 25 Aug): 8 jobs x 1 GPU x 4 passes. Single-GPU jobs
# backfill across the five v100 nodes; a 2-GPU job would need both GPUs free on
# one node at once, which is the hard constraint on this cluster.
#
# One job per SEED means every task carries a complete cup/finger x e1/e4 block:
# no task can be arm-pure, device cannot correlate with arm, and losing a task
# costs one whole seed with all four arms intact (n=8 -> n=7 cleanly) rather
# than a partial arm.
#
# --time: the producer's own default is 12h, which at 4 passes/GPU would be
# killed mid-run. QOS MaxWall is 1-12:00:00 = 36h; we request the full 36h.
# --cpus-per-task=12: nodes are 48 CPU / 4 GPU, so 12 packs 4 jobs per node
# against its 4 GPUs. 3 cores per pass, and the pools are capped below.
#SBATCH --job-name=w2lab_rcc
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --array=59-66
#SBATCH --output=%x_%A_%a.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"
R=/scratch/midway3/rickybao/dreamerv3_runs
SEED="${SLURM_ARRAY_TASK_ID:?array task id = the seed}"
export W1_LABELS_DIR="${W1_LABELS_DIR:-$R/w1_labels_rcc}"
LOGD="$R/_cloud_logs_rcc"; mkdir -p "$W1_LABELS_DIR" "$LOGD"

# THE FIX for the inst30 pathology: unbounded BLAS/torch pools sized themselves
# to the whole box (162 threads/proc, 5184 on 64 threads, 300:1 involuntary
# context switches). Cap them so 4 passes share 12 cores cleanly.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
       NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3

echo "[$(date)] w2lab_rcc seed=$SEED node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

pids=(); names=()
for spec in cup:e1 finger:e4 cup:e4 finger:e1; do
  dom="${spec%%:*}"; dose="${spec##*:}"
  nm="w1tm2_${dom}_${dose}_seed${SEED}_late"
  TM2R3_STAGE=w2labels TM2R3_DOM="$dom" TM2R3_DOSE="$dose" TM2R3_SEED="$SEED" \
    bash scripts/tm2r3.sbatch > "$LOGD/${nm}.out" 2>&1 &
  pids+=($!); names+=("$nm")
  sleep 5   # stagger CUDA init so 4 procs do not collide on context creation
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date)] OK   ${names[$i]}"
  else echo "[$(date)] FAIL ${names[$i]} -- see $LOGD/${names[$i]}.out"; rc=1; fi
done

# realized-counter gate: never claim success from the loop alone
n=0
for spec in cup:e1 finger:e4 cup:e4 finger:e1; do
  dom="${spec%%:*}"; dose="${spec##*:}"
  [ -s "$W1_LABELS_DIR/w1tm2_${dom}_${dose}_seed${SEED}_late.npz" ] && n=$((n+1))
done
echo "[$(date)] seed $SEED wrote $n/4 npz (rc=$rc)"
[ "$n" -eq 4 ] || exit 1
exit $rc
