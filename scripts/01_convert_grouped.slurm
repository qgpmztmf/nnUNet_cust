#!/bin/bash
#SBATCH --job-name=nnunet_train_fast
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/convert_grouped_%j.out
#SBATCH --error=logs/convert_grouped_%j.err

# Convert TotalSegmentator v1 to 4 grouped nnUNet tasks (Task611-614).
# Each group has its own HU window and remapped labels.
#
# Usage: sbatch scripts/01_convert_grouped.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
RAW_DATASET="${DATA_BASE}/TotalSegmentator_v1/Totalsegmentator_dataset"
SPLIT_FILE="/m/triton/work/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_raw/split_files/split_group4.json"
WINDOWS_FILE="/m/triton/work/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_raw/split_files/hu_windows.json"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"

echo "=== Convert TotalSegmentator to 4 Grouped Tasks ==="
echo "Input:       ${RAW_DATASET}"
echo "Output base: ${DATA_BASE}/nnUNet_raw_data"
echo "Split file:  ${SPLIT_FILE}"
echo "Windows:     ${WINDOWS_FILE}"
echo "CPU cores:   ${SLURM_CPUS_PER_TASK}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

uv run python convert_totalseg_v1_grouped.py \
    --input_dir "${RAW_DATASET}" \
    --output_base "${DATA_BASE}/nnUNet_raw_data" \
    --split_file "${SPLIT_FILE}" \
    --windows_file "${WINDOWS_FILE}" \
    --task_base_id 611 \
    --target_spacing 3.0 3.0 3.0 \
    --num_cores "${SLURM_CPUS_PER_TASK}"

echo ""
echo "Conversion complete: $(date)"
