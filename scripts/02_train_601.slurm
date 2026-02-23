#!/bin/bash
#SBATCH --job-name=nnunet_train_601
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/train_601_fold%a_%j.out
#SBATCH --error=logs/train_601_fold%a_%j.err

# Train nnUNet 3D full-resolution on Task601_TotalSegmentatorV1 (104 classes, unfixed)
# using the custom nnUNetTrainerV2_fast trainer.
#
# --- Submit all 5 folds in parallel ---
#   sbatch --array=0-4 scripts/02_train_601.slurm
#
# --- Submit a single fold ---
#   sbatch --array=2 scripts/02_train_601.slurm
#
# --- Continue an interrupted training ---
#   sbatch --array=0-4 --export=ALL,CONTINUE=1 scripts/02_train_601.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

# Resolve fold from array task ID
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

# Set CONTINUE=1 to resume an interrupted run
CONTINUE="${CONTINUE:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
TASK="601"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

echo "=== nnUNet Training Task601 ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        Task${TASK}_TotalSegmentatorV1"
echo "Fold:        ${FOLD}"
echo "Continue:    ${CONTINUE}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

CONTINUE_FLAG=""
if [ "${CONTINUE}" = "1" ]; then
    CONTINUE_FLAG="-c"
fi

uv run nnUNet_train \
    "${NETWORK}" \
    "${TRAINER}" \
    "${TASK}" \
    "${FOLD}" \
    -p "${PLANS}" \
    ${CONTINUE_FLAG}

echo ""
echo "Training fold ${FOLD} complete: $(date)"
