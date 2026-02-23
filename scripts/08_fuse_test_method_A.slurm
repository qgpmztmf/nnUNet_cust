#!/bin/bash
#SBATCH --job-name=fuse_test_A
#SBATCH --partition=batch-milan
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/fuse_test_A_%j.out
#SBATCH --error=logs/fuse_test_A_%j.err

# Method A fusion: fuse pre-ensembled (5-fold averaged) softmax from
# Task611-614 into a single 104-class label map per test case.
#
# Fusion rule (per voxel):
#   Background : product of all 4 models' bg probs (log-domain)
#   Foreground : copy each model's local channel → global channel (1..104)
#   Normalize  : divide by sum so all 105 channels sum to 1
#
# Input:  test_ensemble/Task{611-614}/{case_id}.npz  (fold-averaged softmax)
# Output: test_fused_A/{case_id}.nii.gz              (uint16 label map, 0-104)
#
# Usage:
#   sbatch scripts/08_fuse_test_method_A.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"

ENSEMBLE_ROOT="${DATA_BASE}/nnUNet_results/test_predictions/test_ensemble"
OUTPUT_ROOT="${DATA_BASE}/nnUNet_results/test_predictions/test_fused_A"
CASES_LIST="${REPO_DIR}/test_cases.txt"
RAW_DATA="${DATA_BASE}/nnUNet_raw_data"
REF_PATTERN="${DATA_BASE}/nnUNet_raw_data/Task601_TotalSegmentatorV1/imagesTs/{case_id}_0000.nii.gz"

mkdir -p "${REPO_DIR}/logs"

echo "=== Fuse Test Predictions (Method A) ==="
echo "Ensemble root: ${ENSEMBLE_ROOT}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "Cases:         $(wc -l < "${CASES_LIST}")"
echo "Node:          ${SLURM_NODELIST}"
echo "Started:       $(date)"
echo ""

cd "${REPO_DIR}"

uv run python fuse.py \
    --cases_list    "${CASES_LIST}" \
    --input_root    "${ENSEMBLE_ROOT}" \
    --output_root   "${OUTPUT_ROOT}" \
    --nnunet_raw_data "${RAW_DATA}" \
    --method        A \
    --save_nii \
    --ref_pattern   "${REF_PATTERN}" \
    --sanity

echo ""
echo "Done: $(date)"
echo "Output: ${OUTPUT_ROOT}"
