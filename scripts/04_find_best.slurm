#!/bin/bash
#SBATCH --job-name=nnunet_find_best
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/find_best_%j.out
#SBATCH --error=logs/find_best_%j.err

# Runs nnUNet_find_best_configuration for Task601 (nnUNetTrainerV2_fast, 3d_fullres).
# Aggregates 5-fold cross-validation results, determines the best ensemble strategy,
# and writes postprocessing.json. No GPU required.
#
# Usage:
#   sbatch scripts/04_find_best.slurm
#
# Optional: disable strict mode (allows missing folds)
#   sbatch --export=ALL,STRICT=0 scripts/04_find_best.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
TASK="${TASK:-601}"
STRICT="${STRICT:-1}"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_find_best_configuration" ]; then
    echo "ERROR: nnUNet_find_best_configuration not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet Find Best Configuration ==="
echo "Network:     ${NETWORK}"
echo "Trainer:     ${TRAINER}"
echo "Task:        Task${TASK}_TotalSegmentatorV1"
echo "Strict:      ${STRICT}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

STRICT_FLAG=""
if [ "${STRICT}" = "1" ]; then
    STRICT_FLAG="--strict"
fi

uv run nnUNet_find_best_configuration \
    -m "${NETWORK}" \
    -t "${TASK}" \
    -tr "${TRAINER}" \
    ${STRICT_FLAG}

echo ""
echo "Done: $(date)"
echo "Results: ${RESULTS_FOLDER}/nnUNet/${NETWORK}/Task${TASK}_TotalSegmentatorV1/${TRAINER}__nnUNetPlansv2.1/"
