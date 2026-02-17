#!/bin/bash
set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

# Resolve fold
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="${SLURM_ARRAY_TASK_ID}"
fi
FOLD="${FOLD:-0}"

# Set USE_BEST=1 to load best checkpoint instead of final checkpoint
USE_BEST="${USE_BEST:-0}"

NETWORK="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerV2}"
TASK="601"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_train" ]; then
    echo "ERROR: nnUNet_train not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Validation ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        Task${TASK}_TotalSegmentatorV1"
echo "Fold:        ${FOLD}"
echo "Use best:    ${USE_BEST}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Started:     $(date)"
echo ""

VALBEST_FLAG=""
if [ "${USE_BEST}" = "1" ]; then
    VALBEST_FLAG="--valbest"
fi

uv run nnUNet_train \
    "${NETWORK}" \
    "${TRAINER}" \
    "${TASK}" \
    "${FOLD}" \
    -p "${PLANS}" \
    --validation_only \
    ${VALBEST_FLAG}

echo ""
echo "Validation fold ${FOLD} complete: $(date)"
echo "Results saved to: ${RESULTS_FOLDER}/nnUNet/${NETWORK}/Task${TASK}_TotalSegmentatorV1/${TRAINER}__${PLANS}/fold_${FOLD}/validation_raw/"
