#!/usr/bin/env bash

#SBATCH --job-name=htsp_eval
#SBATCH --account=etechnik_gpu
#SBATCH --partition=gpushort

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16

#SBATCH --time=02:00:00

#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail


# ============================================================
# 1. Path
# ============================================================

# Path of slurm script
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

CODE_DIR="${PROJECT_DIR}/Code"
DATASET_DIR="${PROJECT_DIR}/datasets/HTSP/htsp_datasets"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/HTSP/h-tsp-ckpt"
SLURM_LOG_DIR="${PROJECT_DIR}/slurm_logs"

IMAGE="/beegfs/zoghian/containers/htsp_apptainer.sif"
IMAGE="$(realpath "${IMAGE}")"


# ============================================================
# 2. evaluation config
# ============================================================

GRAPH_SIZE="${GRAPH_SIZE:-1000}"

DATASET_NAME="${DATASET_NAME:-tsp1000_test_concorde.txt}"

UPPER_CHECKPOINT_RELATIVE="${UPPER_CHECKPOINT_RELATIVE:-upper-level/tsp1000/best.ckpt}"

LOWER_CHECKPOINT_RELATIVE="${LOWER_CHECKPOINT_RELATIVE:-lower-level/lower200/best.ckpt}"

DATASET_HOST="${DATASET_DIR}/${DATASET_NAME}"
UPPER_CHECKPOINT_HOST="${CHECKPOINT_DIR}/${UPPER_CHECKPOINT_RELATIVE}"
LOWER_CHECKPOINT_HOST="${CHECKPOINT_DIR}/${LOWER_CHECKPOINT_RELATIVE}"


# ============================================================
# 3. Own path
# ============================================================

RUN_NAME="htsp_g${GRAPH_SIZE}_${SLURM_JOB_ID}"


mkdir -p "${SLURM_LOG_DIR}"


# ============================================================
# 4. check inputs
# ============================================================

check_file() {
    local file="$1"
    local description="$2"

    if [[ ! -f "${file}" ]]; then
        echo "FEHLER: ${description} nicht gefunden:" >&2
        echo "  ${file}" >&2
        exit 1
    fi
}

check_directory() {
    local directory="$1"
    local description="$2"

    if [[ ! -d "${directory}" ]]; then
        echo "FEHLER: ${description} nicht gefunden:" >&2
        echo "  ${directory}" >&2
        exit 1
    fi
}

check_directory "${CODE_DIR}" "Code path"
check_directory "${DATASET_DIR}" "dataset path"
check_directory "${CHECKPOINT_DIR}" "Checkpoint path"

check_file "${CODE_DIR}/evaluate.py" "evaluate.py"
check_file "${IMAGE}" "Apptainer image"
check_file "${DATASET_HOST}" "Dataset"
check_file "${LOWER_CHECKPOINT_HOST}" "Lower-Level-Checkpoint"
check_file "${UPPER_CHECKPOINT_HOST}" "Upper-Level-Checkpoint"


# ============================================================
# 5. Output job information
# ============================================================

echo "============================================================"
echo "H-TSP-Evaluation"
echo "============================================================"
echo "Job-ID:                  ${SLURM_JOB_ID}"
echo "Jobname:                 ${SLURM_JOB_NAME}"
echo "Compute-Node:            $(hostname)"
echo "Start:               $(date --iso-8601=seconds)"
echo
echo "Project directory:      ${PROJECT_DIR}"
echo "Code directory:         ${CODE_DIR}"
echo "Container:               ${IMAGE}"
echo
echo "Graph size:              ${GRAPH_SIZE}"
echo "Dataset:               ${DATASET_HOST}"
echo "Lower-Level-Checkpoint:  ${LOWER_CHECKPOINT_HOST}"
echo "Upper-Level-Checkpoint:  ${UPPER_CHECKPOINT_HOST}"
echo
echo "Ergebnisverzeichnis:     ${RUN_OUTPUT_DIR}"
echo "Programmlog directory: ${RUN_LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES:-not available}"
echo "============================================================"

nvidia-smi


# ============================================================
# 6. GPU access test
# ============================================================

srun apptainer exec \
    --nv \
    "${IMAGE}" \
    python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch can not see any GPU."
    )

print("Used GPU:", torch.cuda.get_device_name(0))
PY


# ============================================================
# 7. H-TSP start
# ============================================================

srun apptainer exec \
    --nv \
    --bind "${CODE_DIR}:/workspace/Code" \
    --bind "${DATASET_DIR}:/workspace/datasets" \
    --bind "${CHECKPOINT_DIR}:/workspace/checkpoints" \
    --pwd /workspace/Code \
    "${IMAGE}" \
    python -u evaluate.py \
        --data_path "/workspace/datasets/${DATASET_NAME}" \
        --graph_size "${GRAPH_SIZE}" \
        --lower_model "/workspace/checkpoints/${LOWER_CHECKPOINT_RELATIVE}" \
        --upper_model "/workspace/checkpoints/${UPPER_CHECKPOINT_RELATIVE}" \


# ============================================================
# 8. Eval information
# ============================================================

echo "============================================================"
echo "Evaluation successfull."
echo "End time:        $(date --iso-8601=seconds)"
echo "Result:     ${RUN_OUTPUT_DIR}"
echo "Programmlogs:   ${RUN_LOG_DIR}"
echo "============================================================"