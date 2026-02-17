#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --nodelist=gpu64
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_fold%a_%j.out
#SBATCH --error=logs/train_fold%a_%j.err

# Train nnUNet 3D full-resolution model on Task601_TotalSegmentatorV1 (104 classes).
# Runs one fold per job. Submit all 5 folds in parallel using a job array, or
# pass FOLD directly.
#
# --- Submit a single fold ---
#   sbatch --export=ALL,FOLD=0 scripts/02_train.slurm
#
# --- Submit all 5 folds as an array ---
#   sbatch --array=0-4 scripts/02_train.slurm
#   (FOLD is set automatically from SLURM_ARRAY_TASK_ID when using --array)
#
# --- Continue an interrupted training ---
#   sbatch --export=ALL,FOLD=0,CONTINUE=1 scripts/02_train.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

# Resolve fold: prefer explicit FOLD env var, then fall back to array task ID
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="${SLURM_ARRAY_TASK_ID}"
fi
FOLD="${FOLD:-0}"

# Set to 1 to continue an interrupted training
CONTINUE="${CONTINUE:-0}"

NETWORK="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerV2}"
TASK="603"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_train" ]; then
    echo "ERROR: nnUNet_train not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Training ==="
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
