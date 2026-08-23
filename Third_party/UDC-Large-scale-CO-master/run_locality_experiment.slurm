#!/usr/bin/env bash

#SBATCH --job-name=lkh3_locality
#SBATCH --account=etechnik
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
REPO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
SLURM_LOG_DIR="${PROJECT_DIR}/slurm_logs"

if [[ ! -d "${SLURM_LOG_DIR}" ]]; then
    echo "ERROR: Existing SLURM log directory not found: ${SLURM_LOG_DIR}" >&2
    echo "Submit this script from Third_party/UDC-Large-scale-CO-master." >&2
    exit 1
fi

IMAGE="${UDC_IMAGE:-/beegfs/${USER}/containers/udc-atsp_1.0.sif}"
LKH_BINARY="${LKH_BINARY:-/usr/local/bin/LKH}"

INSTANCES="${INSTANCES:-64}"
SEGMENT_SIZE="${SEGMENT_SIZE:-}"
K_VALUES="${K_VALUES:-1 2 4 8 16}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-1234}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
LKH_RUNS="${LKH_RUNS:-1}"
LKH_MAX_TRIALS="${LKH_MAX_TRIALS:-}"
LKH_SCALE="${LKH_SCALE:-1000000}"
LKH_PRECISION="${LKH_PRECISION:-1}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK}}"

CTRL_DATA_PATH="${CTRL_DATA_PATH:-/workspace/project/data/ATSP_data/Ctrl_ATSP_data}"
MATNET_DATA_PATH="${MATNET_DATA_PATH:-/workspace/project/data/ATSP_data/Matnet_atsp250_128instances}"
RRNCO_DATA_PATH="${RRNCO_DATA_PATH:-/workspace/project/data/ATSP_data/RRNCO_atsp250_64instances/atsp_n250_64_data}"

RUN_ID="${LOCALITY_RUN_ID:-${SLURM_JOB_ID}}"
OUTPUT_DIR="/workspace/project/Third_party/UDC-Large-scale-CO-master/locality_results/${RUN_ID}"

export OMP_NUM_THREADS=1

read -r -a K_VALUE_ARGS <<< "${K_VALUES}"
OPTIONAL_ARGS=()
if [[ -n "${SEGMENT_SIZE}" ]]; then
    OPTIONAL_ARGS+=(--segment-size "${SEGMENT_SIZE}")
fi
if [[ -n "${LKH_MAX_TRIALS}" ]]; then
    OPTIONAL_ARGS+=(--max-trials "${LKH_MAX_TRIALS}")
fi

echo "============================================================"
echo "LOCALITY EXPERIMENT - SLURM CONFIGURATION"
echo "============================================================"
echo "job id             : ${SLURM_JOB_ID}"
echo "node                : ${SLURMD_NODENAME}"
echo "image               : ${IMAGE}"
echo "LKH binary          : ${LKH_BINARY}"
echo "LKH RUNS            : ${LKH_RUNS}"
echo "LKH MAX_TRIALS      : ${LKH_MAX_TRIALS:-2*N (resolved by Python)}"
echo "LKH PRECISION       : ${LKH_PRECISION}"
echo "LKH scale           : ${LKH_SCALE}"
echo "experiment seed     : ${EXPERIMENT_SEED}"
echo "instances/dataset   : ${INSTANCES}"
echo "segment size        : ${SEGMENT_SIZE:-repository discovery (50)}"
echo "node selection      : none; strict 250x250 matrices required"
echo "K values            : ${K_VALUES}"
echo "bootstrap samples   : ${BOOTSTRAP_SAMPLES}"
echo "parallel workers    : ${WORKERS}"
echo "ctrl path           : ${CTRL_DATA_PATH}"
echo "matnet path         : ${MATNET_DATA_PATH}"
echo "rrnco path          : ${RRNCO_DATA_PATH}"
echo "output directory    : ${OUTPUT_DIR}"
echo "============================================================"

srun --cpu-bind=cores apptainer exec \
    --bind "${REPO_DIR}:/workspace/project" \
    --pwd /workspace/project/Third_party/UDC-Large-scale-CO-master \
    "${IMAGE}" \
    python -u locality_experiment.py \
        --datasets ctrl_atsp_data matnet rrnco \
        --ctrl-path "${CTRL_DATA_PATH}" \
        --matnet-path "${MATNET_DATA_PATH}" \
        --rrnco-path "${RRNCO_DATA_PATH}" \
        --instances "${INSTANCES}" \
        --k-values "${K_VALUE_ARGS[@]}" \
        --incoming auto \
        --seed "${EXPERIMENT_SEED}" \
        --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
        --workers "${WORKERS}" \
        --lkh-binary "${LKH_BINARY}" \
        --runs "${LKH_RUNS}" \
        --lkh-precision "${LKH_PRECISION}" \
        --scale "${LKH_SCALE}" \
        --output-dir "${OUTPUT_DIR}" \
        "${OPTIONAL_ARGS[@]}"
