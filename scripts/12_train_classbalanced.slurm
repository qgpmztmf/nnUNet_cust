#!/bin/bash
#SBATCH --job-name=cb_train
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/train_cb_task%x_fold%a_%j.out
#SBATCH --error=logs/train_cb_task%x_fold%a_%j.err

# Train nnUNet with class-balanced patch sampling (nnUNetTrainerV2_classBalanced).
# Sampling split per batch item: 10% class-balanced / 23% fg-oversample / 67% random.
#
# --- Train all 5 folds of Task612 ---
#   sbatch --array=0-4 --export=ALL,TASK=612 scripts/12_train_classbalanced.slurm
#
# --- Train all 5 folds of Task614 ---
#   sbatch --array=0-4 --export=ALL,TASK=614 scripts/12_train_classbalanced.slurm
#
# --- Train both tasks (10 jobs total) ---
#   for T in 612 614; do
#       sbatch --array=0-4 --export=ALL,TASK=$T scripts/12_train_classbalanced.slurm
#   done
#
# --- Continue an interrupted fold ---
#   sbatch --array=2 --export=ALL,TASK=612,CONTINUE=1 scripts/12_train_classbalanced.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

export nnUNet_n_proc_DA=16

mkdir -p "${REPO_DIR}/logs"

# Fold from SLURM array index
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="${SLURM_ARRAY_TASK_ID}"
fi
FOLD="${FOLD:-0}"

CONTINUE="${CONTINUE:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_classBalanced"
TASK="${TASK:?ERROR: TASK must be set via --export (e.g. 612 or 614)}"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_train" ]; then
    echo "ERROR: nnUNet_train not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Class-Balanced Training ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        ${TASK}"
echo "Fold:        ${FOLD}"
echo "Continue:    ${CONTINUE}"
echo "Sampling:    10% class-balanced / 23% fg-oversample / 67% random"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "DA workers:  ${nnUNet_n_proc_DA}"
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
echo "Training Task${TASK} fold ${FOLD} complete: $(date)"
