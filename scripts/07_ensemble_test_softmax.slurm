#!/bin/bash
#SBATCH --job-name=ensemble_test_sfx
#SBATCH --partition=batch-milan
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ensemble_test_softmax_%j.out
#SBATCH --error=logs/ensemble_test_softmax_%j.err

# Average per-fold softmax .npz files across all 5 folds for Tasks 611-614.
# Output: one ensembled .npz per case per task at test_ensemble/Task{id}/.
#
# Usage:
#   sbatch scripts/07_ensemble_test_softmax.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

PRED_BASE="${DATA_BASE}/nnUNet_results/test_predictions"
OUTPUT_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_results/test_predictions/test_ensemble"

mkdir -p "${REPO_DIR}/logs"

echo "=== Ensemble Test Softmax ==="
echo "Pred base:   ${PRED_BASE}"
echo "Output base: ${OUTPUT_BASE}"
echo "Tasks:       611 612 613 614"
echo "Folds:       0 1 2 3 4"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

cd "${REPO_DIR}"

uv run python scripts/ensemble_test_softmax.py \
    --pred_base   "${PRED_BASE}" \
    --tasks       611 612 613 614 \
    --folds       0 1 2 3 4 \
    --output_base "${OUTPUT_BASE}"

echo ""
echo "Done: $(date)"
echo "Output: ${OUTPUT_BASE}"
