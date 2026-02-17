#!/bin/bash
#SBATCH --job-name=nnunet_preproc604
#SBATCH --partition=batch-milan
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/preprocess_604_%j.out
#SBATCH --error=logs/preprocess_604_%j.err

# Preprocessing: plan and preprocess Task604 (4-channel HU windowed TotalSegmentatorV1)
# Run after 01_convert_multiwindow.slurm completes.
# Usage: sbatch scripts/01_preprocess_604.slurm

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

echo "=== nnUNet plan and preprocess (Task604 - 4ch HU windows) ==="
echo "Task:        Task604_TotalSegmentatorV1"
echo "CPU cores:   ${SLURM_CPUS_PER_TASK}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

uv run nnUNet_plan_and_preprocess \
    -t 604 \
    -tl 8 \
    -tf 8

echo ""
echo "Preprocessing complete: $(date)"
echo "Plans written to: ${nnUNet_preprocessed}/Task604_TotalSegmentatorV1/"
