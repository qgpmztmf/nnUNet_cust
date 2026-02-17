#!/bin/bash
#SBATCH --job-name=nnunet_val_grp
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/validate_grp_task%x_fold%a_%j.out
#SBATCH --error=logs/validate_grp_task%x_fold%a_%j.err

# Validate a grouped task model. Loads checkpoint and evaluates on held-out fold.
#
# --- Validate all 5 folds of one task ---
#   sbatch --array=0-4 --export=ALL,TASK=611 scripts/03_validate_grouped.slurm
#
# --- Validate ALL 4 tasks × 5 folds ---
#   for T in 611 612 613 614; do
#       sbatch --array=0-4 --export=ALL,TASK=$T scripts/03_validate_grouped.slurm
#   done
#
# --- Use best checkpoint instead of final ---
#   sbatch --array=0-4 --export=ALL,TASK=611,USE_BEST=1 scripts/03_validate_grouped.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="${SLURM_ARRAY_TASK_ID}"
fi
FOLD="${FOLD:-0}"

USE_BEST="${USE_BEST:-0}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
TASK="${TASK:?ERROR: TASK must be set (e.g. 611, 612, 613, 614)}"
PLANS="nnUNetPlansv2.1"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_train" ]; then
    echo "ERROR: nnUNet_train not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Grouped Validation ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        ${TASK}"
echo "Fold:        ${FOLD}"
echo "Use best:    ${USE_BEST}"
echo "GPU:         ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Started:     $(date)"
echo ""

VALBEST_FLAG=""
if [ "${USE_BEST}" = "1" ]; then
    VALBEST_FLAG="--valbest"
fi

uv run nnUNet_train \
    "${NETWORK}" \
    "${TRAINER}" \
    "${TASK}" \
    "${FOLD}" \
    -p "${PLANS}" \
    --validation_only \
    ${VALBEST_FLAG}

echo ""
echo "Validation Task${TASK} fold ${FOLD} complete: $(date)"
echo "Results: ${RESULTS_FOLDER}/nnUNet/${NETWORK}/Task${TASK}_*/nnUNetTrainerV2_fast__${PLANS}/fold_${FOLD}/validation_raw/"
