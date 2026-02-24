#!/bin/bash
#SBATCH --job-name=nnunet_train_601_r2
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_2_20260224T134301Z/train_601_round2_fold%a_%j.out
#SBATCH --error=/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_2_20260224T134301Z/train_601_round2_fold%a_%j.err

# Auto-generated SLURM training script — Round 2
# Task:    Task601_TotalSegmentatorV1
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Plans:   nnUNetPlansv2.1
#
# Hyperparameter overrides are read automatically from:
#   <REPO_DIR>/agent/doc/hyperparameter_reference.json
#
# Submit all 5 folds:
#   sbatch --array=0-4 train_601_round2.slurm
#
# Resume interrupted folds:
#   sbatch --array=0-4 --export=ALL,CONTINUE=1 train_601_round2.slurm

set -euo pipefail

REPO_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"
export NNUNET_HP_REF="${REPO_DIR}/agent/doc/hyperparameter_reference.json"

mkdir -p "${REPO_DIR}/logs"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
CONTINUE="${CONTINUE:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
TASK="601"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

echo "=== nnUNet Training Round 2 ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        Task${TASK}_TotalSegmentatorV1"
echo "Round:       2"
echo "Fold:        ${FOLD}"
echo "Continue:    ${CONTINUE}"
echo "HP_REF:      ${NNUNET_HP_REF}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:        ${SLURM_NODELIST:-local}"
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
echo "Training fold ${FOLD} round 2 complete: $(date)"
