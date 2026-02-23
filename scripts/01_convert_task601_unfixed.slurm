#!/bin/bash
#SBATCH --job-name=convert_601_unfixed
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=logs/convert_601_unfixed_%j.out
#SBATCH --error=logs/convert_601_unfixed_%j.err

# Convert Task601 using the ORIGINAL (unfixed) TotalSegmentator v1 dataset.
# Corrupted out-of-FOV voxels are still present in 76 cases.
# Usage: sbatch scripts/01_convert_task601_unfixed.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"

echo "=== Convert Task601 TotalSegmentatorV1 (UNFIXED) ==="
echo "Input:    ${DATA_BASE}/TotalSegmentator_v1/Totalsegmentator_dataset"
echo "Output:   ${DATA_BASE}/nnUNet_raw_data/Task601_TotalSegmentatorV1"
echo "Cores:    ${SLURM_CPUS_PER_TASK}"
echo "Node:     ${SLURM_NODELIST}"
echo "Started:  $(date)"
echo ""

uv run python convert_totalseg_v1_to_nnunet.py \
    --input_dir  "${DATA_BASE}/TotalSegmentator_v1/Totalsegmentator_dataset" \
    --output_dir "${DATA_BASE}/nnUNet_raw_data/Task601_TotalSegmentatorV1" \
    --target_spacing 3.0 3.0 3.0 \
    --num_cores "${SLURM_CPUS_PER_TASK}"

echo ""
echo "Conversion complete: $(date)"
