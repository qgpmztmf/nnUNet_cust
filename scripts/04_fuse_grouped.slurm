#!/bin/bash
#SBATCH --job-name=nnunet_fuse_grp
#SBATCH --partition=batch-milan
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/fuse_grouped_%j.out
#SBATCH --error=logs/fuse_grouped_%j.err

# Fuse predictions from 4 grouped models (Task611-614) into 104-class segmentation.
# Reads hard-label .nii.gz predictions from each model's validation_raw folder.
#
# Usage:
#   sbatch --export=ALL,FOLD=0 scripts/04_fuse_grouped.slurm
#
# Optional: use softmax fusion (requires --save_softmax during inference)
#   sbatch --export=ALL,FOLD=0,USE_SOFTMAX=1 scripts/04_fuse_grouped.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

FOLD="${FOLD:-0}"
USE_SOFTMAX="${USE_SOFTMAX:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
PLANS="nnUNetPlansv2.1"

RESULTS_BASE="${DATA_BASE}/nnUNet_results/nnUNet/${NETWORK}"
RAW_DATA_BASE="${DATA_BASE}/nnUNet_raw_data"
FUSED_OUTPUT="${DATA_BASE}/nnUNet_results/fused_grouped/fold_${FOLD}"

cd "${REPO_DIR}"

echo "=== Fuse Grouped Predictions ==="
echo "Fold:        ${FOLD}"
echo "Softmax:     ${USE_SOFTMAX}"
echo "Started:     $(date)"
echo ""

# Build arguments: mapping files and prediction directories
MAPPING_ARGS=""
PRED_ARGS=""
for TASK_ID in 611 612 613 614; do
    TASK_DIR=$(ls -d "${RESULTS_BASE}/Task${TASK_ID}_"* 2>/dev/null | head -1)
    if [ -z "${TASK_DIR}" ]; then
        echo "ERROR: No results found for Task${TASK_ID}" >&2
        exit 1
    fi

    PRED_DIR="${TASK_DIR}/${TRAINER}__${PLANS}/fold_${FOLD}/validation_raw"
    MAP_FILE="${RAW_DATA_BASE}/Task${TASK_ID}_TotalSegGroup$((TASK_ID - 610))/class_mapping.json"

    if [ ! -d "${PRED_DIR}" ]; then
        echo "ERROR: Prediction dir not found: ${PRED_DIR}" >&2
        exit 1
    fi
    if [ ! -f "${MAP_FILE}" ]; then
        echo "ERROR: Mapping file not found: ${MAP_FILE}" >&2
        exit 1
    fi

    MAPPING_ARGS="${MAPPING_ARGS} ${MAP_FILE}"
    PRED_ARGS="${PRED_ARGS} ${PRED_DIR}"

    echo "  Task${TASK_ID}: ${PRED_DIR}"
done

SOFTMAX_FLAG=""
if [ "${USE_SOFTMAX}" = "1" ]; then
    SOFTMAX_FLAG="--use_softmax"
fi

echo ""
echo "Output: ${FUSED_OUTPUT}"
echo ""

uv run python fuse_group_predictions.py \
    --mapping_files ${MAPPING_ARGS} \
    --pred_dirs ${PRED_ARGS} \
    --output_dir "${FUSED_OUTPUT}" \
    ${SOFTMAX_FLAG}

echo ""
echo "Fusion complete: $(date)"
echo "Fused predictions written to: ${FUSED_OUTPUT}"
