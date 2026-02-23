#!/bin/bash
#SBATCH --job-name=pred_test_601
#SBATCH --partition=gpu-h200-35g-ia
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/pred_test_601_fold%a_%j.out
#SBATCH --error=logs/pred_test_601_fold%a_%j.err

# Run nnUNet_predict on imagesTs for Task601 (nnUNetTrainerV2_fast__nnUNetPlansv2.1).
# One SLURM array job per fold; saves both hard labels (.nii.gz) and softmax (.npz).
#
# --- Predict all 5 folds ---
#   sbatch --array=0-4 scripts/13_predict_test_601.slurm
#
# --- Predict a single fold ---
#   sbatch --array=2 scripts/13_predict_test_601.slurm
#
# After all 5 folds complete, run:
#   sbatch scripts/14_ensemble_test_601.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
PLANS="nnUNetPlansv2.1"
TASK="601"
TASK_NAME="Task601_TotalSegmentatorV1"

INPUT_DIR="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/imagesTs"
OUTPUT_DIR="${RESULTS_FOLDER}/nnUNet/${NETWORK}/${TASK_NAME}/${TRAINER}__${PLANS}/test_predictions/fold_${FOLD}"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "=== nnUNet Test Prediction (Task601) ==="
echo "Trainer:     ${TRAINER}"
echo "Plans:       ${PLANS}"
echo "Fold:        ${FOLD}"
echo "Input:       ${INPUT_DIR}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Started:     $(date)"
echo ""

uv run nnUNet_predict \
    -i  "${INPUT_DIR}" \
    -o  "${OUTPUT_DIR}" \
    -t  "${TASK}" \
    -m  "${NETWORK}" \
    -tr "${TRAINER}" \
    -p  "${PLANS}" \
    -f  "${FOLD}" \
    --save_npz \
    --disable_tta

echo ""
echo "Prediction fold ${FOLD} complete: $(date)"
echo "Output: ${OUTPUT_DIR}"
