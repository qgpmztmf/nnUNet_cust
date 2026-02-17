#!/bin/bash
#SBATCH --job-name=nnunet_preprocess_mini
#SBATCH --partition=batch-milan
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/preprocess_mini_%j.out
#SBATCH --error=logs/preprocess_mini_%j.err

# Preprocessing for mini dataset Task602_TotalSegMini (5 train + 1 test).
# Usage: sbatch scripts/01_preprocess_mini.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"

if [ ! -f ".venv/bin/nnUNet_plan_and_preprocess" ]; then
    echo "ERROR: nnUNet_plan_and_preprocess not found. Run 'bash scripts/00_setup.sh' first." >&2
    exit 1
fi

echo "=== nnUNet plan and preprocess (mini) ==="
echo "Task:        Task602_TotalSegMini"
echo "CPU cores:   ${SLURM_CPUS_PER_TASK}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

uv run nnUNet_plan_and_preprocess \
    -t 602 \
    -tl 4 \
    -tf 4

echo ""
echo "Preprocessing complete: $(date)"
echo "Plans written to: ${nnUNet_preprocessed}/Task602_TotalSegMini/"
