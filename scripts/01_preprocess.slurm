#!/bin/bash
#SBATCH --job-name=nnunet_preprocess
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Preprocessing: plan and preprocess Dataset601 (TotalSegmentatorV1, 104 classes)
# Run once before training. Creates plans file and preprocessed arrays.
# Usage: sbatch scripts/01_preprocess.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"

# Verify the package is installed before starting
if [ ! -f ".venv/bin/nnUNet_plan_and_preprocess" ]; then
    echo "ERROR: nnUNet_plan_and_preprocess not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet plan and preprocess ==="
echo "Task:        Task601_TotalSegmentatorV1"
echo "CPU cores:   ${SLURM_CPUS_PER_TASK}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

# -t 601: task ID
# -tl: number of processes for low-resolution preprocessing
# -tf: number of processes for full-resolution preprocessing
uv run nnUNet_plan_and_preprocess \
    -t 601 \
    -tl 8 \
    -tf 8

echo ""
echo "Preprocessing complete: $(date)"
echo "Plans written to: ${nnUNet_preprocessed}/Task601_TotalSegmentatorV1/"
