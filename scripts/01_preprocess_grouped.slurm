#!/bin/bash
#SBATCH --job-name=nnunet_preproc_grp
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --output=logs/preprocess_grouped_%j.out
#SBATCH --error=logs/preprocess_grouped_%j.err

# Preprocess all 4 grouped tasks (Task611-614) sequentially.
# Run after 01_convert_grouped.slurm completes.
#
# Usage: sbatch scripts/01_preprocess_grouped.slurm

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

echo "=== nnUNet plan and preprocess (Grouped Tasks 611-614) ==="
echo "CPU cores:   ${SLURM_CPUS_PER_TASK}"
echo "Node:        ${SLURM_NODELIST}"
echo "Started:     $(date)"
echo ""

for TASK_ID in 611 612 613 614; do
    echo "--- Preprocessing Task ${TASK_ID} ---"
    echo "Started: $(date)"

    uv run nnUNet_plan_and_preprocess \
        -t "${TASK_ID}" \
        -tl 16 \
        -tf 16

    echo "Task ${TASK_ID} done: $(date)"
    echo ""
done

echo "All preprocessing complete: $(date)"
