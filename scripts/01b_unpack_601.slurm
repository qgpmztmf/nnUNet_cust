#!/bin/bash
#SBATCH --job-name=unpack_601
#SBATCH --partition=batch-milan
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/unpack_601_%j.out
#SBATCH --error=logs/unpack_601_%j.err

# Pre-unpack Task601 npz files to npy so all 5 training folds can start in parallel
# without race conditions.
# Usage: sbatch scripts/01b_unpack_601.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
PREPROC="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_preprocessed/Task601_TotalSegmentatorV1"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

echo "=== Unpacking Task601 dataset ==="
echo "Node:    ${SLURM_NODELIST}"
echo "Started: $(date)"
echo ""

uv run python3 - << 'EOF'
from nnunet.training.dataloading.dataset_loading import unpack_dataset

preproc = "/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_preprocessed/Task601_TotalSegmentatorV1"

for stage in ["nnUNetData_plans_v2.1_stage0", "nnUNetData_plans_v2.1_2D_stage0"]:
    d = f"{preproc}/{stage}"
    print(f"Unpacking {stage} ...", flush=True)
    unpack_dataset(d)
    print(f"Done: {stage}", flush=True)
EOF

echo ""
echo "Unpacking complete: $(date)"
