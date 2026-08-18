#!/usr/bin/env bash

#SBATCH --job-name=htsp_eval
#SBATCH --account=etechnik_gpu
#SBATCH --partition=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CODE_DIR="${PROJECT_DIR}/Code"
DATASET_DIR="${PROJECT_DIR}/datasets/HTSP/htsp_datasets"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/HTSP/h-tsp-ckpt"
SLURM_LOG_DIR="${PROJECT_DIR}/slurm_logs"

IMAGE="/beegfs/zoghian/containers/htsp_apptainer.sif"

GRAPH_SIZE="${GRAPH_SIZE:-1000}"
DATASET_NAME="${DATASET_NAME:-tsp1000_test_concorde.txt}"
UPPER_CHECKPOINT_RELATIVE="${UPPER_CHECKPOINT_RELATIVE:-upper-level/tsp1000/best.ckpt}"
LOWER_CHECKPOINT_RELATIVE="${LOWER_CHECKPOINT_RELATIVE:-lower-level/lower200/best.ckpt}"

mkdir -p "${SLURM_LOG_DIR}"

srun apptainer exec \
    --nv \
    --bind "${CODE_DIR}:/workspace/Code" \
    --bind "${DATASET_DIR}:/workspace/datasets" \
    --bind "${CHECKPOINT_DIR}:/workspace/checkpoints" \
    --bind "${SLURM_LOG_DIR}:/workspace/slurm_logs" \
    --pwd /workspace/slurm_logs \
    "${IMAGE}" \
    python -u /workspace/Code/evaluate.py \
        --data_path "/workspace/datasets/${DATASET_NAME}" \
        --graph_size "${GRAPH_SIZE}" \
        --lower_model "/workspace/checkpoints/${LOWER_CHECKPOINT_RELATIVE}" \
        --upper_model "/workspace/checkpoints/${UPPER_CHECKPOINT_RELATIVE}"
