#!/bin/bash
#SBATCH --job-name=nnunet_grp
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/train_grp_task%x_fold%a_%j.out
#SBATCH --error=logs/train_grp_task%x_fold%a_%j.err

# Train nnUNet on one of the grouped tasks (Task611-614) using nnUNetTrainerV2_fast.
# TASK must be set via --export. Fold is taken from the array index.
#
# --- Train all 5 folds of one task ---
#   sbatch --array=0-4 --export=ALL,TASK=611 scripts/02_train_grouped.slurm
#
# --- Train ALL 4 tasks × 5 folds (20 jobs) ---
#   for T in 611 612 613 614; do
#       sbatch --array=0-4 --export=ALL,TASK=$T scripts/02_train_grouped.slurm
#   done
#
# --- Continue an interrupted training ---
#   sbatch --array=2 --export=ALL,TASK=611,CONTINUE=1 scripts/02_train_grouped.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

export nnUNet_n_proc_DA=16

mkdir -p "${REPO_DIR}/logs"

# Resolve fold from array task ID
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="${SLURM_ARRAY_TASK_ID}"
fi
FOLD="${FOLD:-0}"

CONTINUE="${CONTINUE:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
TASK="${TASK:?ERROR: TASK must be set (e.g. 611, 612, 613, 614)}"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_train" ]; then
    echo "ERROR: nnUNet_train not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Grouped Training (fast) ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        ${TASK}"
echo "Fold:        ${FOLD}"
echo "Continue:    ${CONTINUE}"
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
