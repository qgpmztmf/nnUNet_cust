#!/bin/bash
#SBATCH --job-name=nnunet_fuse_test
#SBATCH --partition=batch-milan
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/fuse_test_%j.out
#SBATCH --error=logs/fuse_test_%j.err

# Fuse ensembled softmax from 4 group models into 104-class predictions.
# Must run after 07_ensemble_test.slurm completes.
#
# Usage (standalone):
#   sbatch scripts/08_fuse_test.slurm
#
# Usage (chained, via submission script):
#   sbatch --dependency=afterok:<ensemble_job_id> scripts/08_fuse_test.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
ENSEMBLE_BASE="/scratch/work/tianmid1/data/test_ensemble"
FUSED_OUTPUT="/scratch/work/tianmid1/data/test_fused"

cd "${REPO_DIR}"

echo "=== Fuse Test Predictions ==="
echo "Ensemble base: ${ENSEMBLE_BASE}"
echo "Output:        ${FUSED_OUTPUT}"
echo "Node:          ${SLURM_NODELIST:-local}"
echo "Started:       $(date)"
echo ""

uv run python fuse_group_predictions.py \
    --mapping_files \
        "${DATA_BASE}/nnUNet_raw_data/Task611_TotalSegGroup1/class_mapping.json" \
        "${DATA_BASE}/nnUNet_raw_data/Task612_TotalSegGroup2/class_mapping.json" \
        "${DATA_BASE}/nnUNet_raw_data/Task613_TotalSegGroup3/class_mapping.json" \
        "${DATA_BASE}/nnUNet_raw_data/Task614_TotalSegGroup4/class_mapping.json" \
    --pred_dirs \
        "${ENSEMBLE_BASE}/Task611" \
        "${ENSEMBLE_BASE}/Task612" \
        "${ENSEMBLE_BASE}/Task613" \
        "${ENSEMBLE_BASE}/Task614" \
    --output_dir "${FUSED_OUTPUT}" \
    --use_softmax

echo ""
echo "Fusion complete: $(date)"
echo "Output: ${FUSED_OUTPUT}/"
