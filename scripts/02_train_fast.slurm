#!/bin/bash
#SBATCH --job-name=nnunet_train_fast
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/train_fast_fold%a_%j.out
#SBATCH --error=logs/train_fast_fold%a_%j.err

# Train nnUNet with nnUNetTrainerV2_fast (batch_size=32, 16 iters/epoch).
#
# --- Submit a single fold ---
#   sbatch --export=ALL,FOLD=0 scripts/02_train_fast.slurm
#
# --- Submit all 5 folds as an array ---
#   sbatch --array=0-4 scripts/02_train_fast.slurm
#
# --- Continue an interrupted training ---
#   sbatch --export=ALL,FOLD=0,CONTINUE=1 scripts/02_train_fast.slurm

# --nodelist=gpu64
# --partition=gpu-b300-288g-ellis
set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

# Augmentation workers (keep moderate to avoid OOM)
export nnUNet_n_proc_DA=16

mkdir -p "${REPO_DIR}/logs"

# Resolve fold: prefer explicit FOLD env var, then fall back to array task ID
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="${SLURM_ARRAY_TASK_ID}"
fi
FOLD="${FOLD:-1}"

# Set to 1 to continue an interrupted training
CONTINUE="${CONTINUE:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
TASK="${TASK:-601}"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_train" ]; then
    echo "ERROR: nnUNet_train not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Training (fast) ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        Task${TASK}_TotalSegmentatorV1"
echo "Fold:        ${FOLD}"
echo "Continue:    ${CONTINUE}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
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
