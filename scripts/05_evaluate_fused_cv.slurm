#!/bin/bash
#SBATCH --job-name=eval_fused_cv
#SBATCH --partition=batch-milan
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/eval_fused_cv_%j.out
#SBATCH --error=logs/eval_fused_cv_%j.err

# Evaluates 5-fold fused group predictions (Task611-614) against Task601 GT,
# producing a single cross-validation cv_summary.json.
#
# Usage:
#   sbatch scripts/05_evaluate_fused_cv.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

mkdir -p "${REPO_DIR}/logs"

FUSED_DIR="${DATA_BASE}/nnUNet_results/fused_grouped"
GT_DIR="${DATA_BASE}/nnUNet_preprocessed/Task601_TotalSegmentatorV1/gt_segmentations"
OUTPUT="${FUSED_DIR}/cv_summary.json"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Evaluate Fused CV ==="
echo "Fused dir:   ${FUSED_DIR}"
echo "GT dir:      ${GT_DIR}"
echo "Output:      ${OUTPUT}"
echo "Threads:     ${THREADS}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

cd "${REPO_DIR}"

uv run python scripts/evaluate_fused_cv.py \
    --fused_dir "${FUSED_DIR}" \
    --gt_dir    "${GT_DIR}" \
    --output    "${OUTPUT}" \
    --num_threads "${THREADS}"

echo ""
echo "Done: $(date)"
echo "Output: ${OUTPUT}"
