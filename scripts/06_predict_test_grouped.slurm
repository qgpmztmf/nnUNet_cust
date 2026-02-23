#!/bin/bash
#SBATCH --job-name=nnunet_pred_test
#SBATCH --partition=gpu-h200-35g-ia
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/pred_test_grp_task%x_fold%a_%j.out
#SBATCH --error=logs/pred_test_grp_task%x_fold%a_%j.err

# Run nnUNet_predict on imagesTs for one task × fold.
# Saves both hard labels (.nii.gz) and softmax (.npz).
#
# --- Predict all 5 folds for one task ---
#   sbatch --array=0-4 --export=ALL,TASK=611 scripts/06_predict_test_grouped.slurm
#
# --- Predict ALL 4 tasks × 5 folds ---
#   for T in 611 612 613 614; do
#       sbatch --array=0-4 --export=ALL,TASK=$T scripts/06_predict_test_grouped.slurm
#   done

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
OUT_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data/nnUNet_results/test_predictions"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
TASK="${TASK:?ERROR: TASK must be set (e.g. 611, 612, 613, 614)}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
PLANS="nnUNetPlansv2.1"

# Locate imagesTs for this task
IMAGES_TS=$(ls -d "${DATA_BASE}/nnUNet_raw_data/Task${TASK}_"*/imagesTs 2>/dev/null | head -1)
if [ -z "${IMAGES_TS}" ]; then
    echo "ERROR: imagesTs not found for Task${TASK}" >&2
    exit 1
fi

OUTPUT_DIR="${OUT_BASE}/Task${TASK}/fold_${FOLD}"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "=== nnUNet Test Prediction ==="
echo "Task:        ${TASK}"
echo "Fold:        ${FOLD}"
echo "Input:       ${IMAGES_TS}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Started:     $(date)"
echo ""

uv run nnUNet_predict \
    -i  "${IMAGES_TS}" \
    -o  "${OUTPUT_DIR}" \
    -t  "${TASK}" \
    -m  "${NETWORK}" \
    -tr "${TRAINER}" \
    -p  "${PLANS}" \
    -f  "${FOLD}" \
    --save_npz \
    --disable_tta

echo ""
echo "Prediction Task${TASK} fold ${FOLD} complete: $(date)"
echo "Output: ${OUTPUT_DIR}"
