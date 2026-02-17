#!/bin/bash
#SBATCH --job-name=nnunet_convert_mw
#SBATCH --partition=batch-milan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/convert_multiwindow_%j.out
#SBATCH --error=logs/convert_multiwindow_%j.err

# Convert TotalSegmentator v1 to nnUNet format with 4-channel HU windowing (Task604).
# Produces 4 NIfTI files per case (brain/lung/bone/soft_tissue windows).
#
# Usage: sbatch scripts/01_convert_multiwindow.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
RAW_DATASET="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"

echo "=== Convert TotalSegmentator to 4-channel HU windowing ==="
echo "Input:       ${RAW_DATASET}"
echo "Output:      ${DATA_BASE}/nnUNet_raw_data/Task604_TotalSegmentatorV1"
echo "CPU cores:   ${SLURM_CPUS_PER_TASK}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

uv run python convert_totalseg_v1_to_nnunet_multiwindow.py \
    --input_dir "${RAW_DATASET}" \
    --output_dir "${DATA_BASE}/nnUNet_raw_data/Task604_TotalSegmentatorV1" \
    --target_spacing 3.0 3.0 3.0 \
    --num_cores "${SLURM_CPUS_PER_TASK}"

echo ""
echo "Conversion complete: $(date)"
