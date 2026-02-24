#!/bin/bash
#SBATCH --job-name=nnunet_pred_test_601_r1
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z/predict_test_601_round1_fold%a_%j.out
#SBATCH --error=/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z/predict_test_601_round1_fold%a_%j.err

# Auto-generated test-set prediction script — Round 1
# Task:    601
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Plans:   nnUNetPlansv2.1
#
# Submit all 5 folds:
#   sbatch --array=0-4 predict_test_601_round1.slurm
#
# After all folds complete, run ensemble + eval:
#   sbatch ensemble_eval_test_task601_round1.slurm
#
# Or chain automatically:
#   PRED_JOB=$(sbatch --array=0-4 --parsable predict_test_601_round1.slurm)
#   sbatch --dependency=afterok:${PRED_JOB} ensemble_eval_test_task601_round1.slurm

set -euo pipefail

REPO_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
TASK="601"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
PLANS="nnUNetPlansv2.1"
TASK_NAME="Task${TASK}_TotalSegmentatorV1"

INPUT_DIR="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/imagesTs"
MODEL_DIR="${RESULTS_FOLDER}/nnUNet/${NETWORK}/${TASK_NAME}/${TRAINER}__${PLANS}"
OUTPUT_DIR="${MODEL_DIR}/test_predictions/fold_${FOLD}"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "=== nnUNet Test Prediction Round 1 ==="
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
echo "Prediction fold ${FOLD} round 1 complete: $(date)"
echo "Output: ${OUTPUT_DIR}"
