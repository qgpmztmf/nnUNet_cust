#!/bin/bash
#SBATCH --job-name=nnunet_ensemble_test
#SBATCH --partition=batch-milan
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ensemble_test_%j.out
#SBATCH --error=logs/ensemble_test_%j.err

# Average softmax .npz predictions across 5 folds for each task (611-614).
# Must run after 06_predict_test_grouped.slurm completes for all folds.
#
# Usage (standalone):
#   sbatch scripts/07_ensemble_test.slurm
#
# Usage (chained, via submission script):
#   sbatch --dependency=afterok:<predict_job_ids> scripts/07_ensemble_test.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
PRED_BASE="/scratch/work/tianmid1/data/test_predictions"
ENSEMBLE_BASE="/scratch/work/tianmid1/data/test_ensemble"

cd "${REPO_DIR}"

echo "=== Ensemble Test Softmax ==="
echo "Pred base:    ${PRED_BASE}"
echo "Output base:  ${ENSEMBLE_BASE}"
echo "Node:         ${SLURM_NODELIST:-local}"
echo "Started:      $(date)"
echo ""

uv run python scripts/ensemble_test_softmax.py \
    --pred_base    "${PRED_BASE}" \
    --tasks        611 612 613 614 \
    --folds        0 1 2 3 4 \
    --output_base  "${ENSEMBLE_BASE}"

echo ""
echo "Ensemble complete: $(date)"
echo "Output: ${ENSEMBLE_BASE}/"
