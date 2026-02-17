#!/bin/bash
# One-time setup script. Run this before preprocessing or training.
# Usage: bash scripts/00_setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
RAW_DATA_DIR="${DATA_BASE}/nnUNet_raw_data"
PREPROCESSED_DIR="${DATA_BASE}/nnUNet_preprocessed"
RESULTS_DIR="${DATA_BASE}/nnUNet_results"

echo "=== Step 1: Install nnUNet package ==="
cd "${REPO_DIR}"
"${REPO_DIR}/.venv/bin/pip" install -e . --quiet
echo "nnUNet installed."

echo ""
echo "=== Step 2: Create directory structure ==="
# nnUNet v1 looks for {nnUNet_raw_data_base}/nnUNet_raw_data/Task601_*/
# The data is stored as Dataset601_TotalSegmentatorV1 — create a symlink.
mkdir -p "${RAW_DATA_DIR}"
mkdir -p "${PREPROCESSED_DIR}"
mkdir -p "${RESULTS_DIR}"

TASK_LINK="${RAW_DATA_DIR}/Task601_TotalSegmentatorV1"
DATASET_DIR="${DATA_BASE}/nnUNet_raw/Dataset601_TotalSegmentatorV1"

if [ ! -e "${TASK_LINK}" ]; then
    ln -s "${DATASET_DIR}" "${TASK_LINK}"
    echo "Created symlink: ${TASK_LINK} -> ${DATASET_DIR}"
else
    echo "Symlink already exists: ${TASK_LINK}"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Add the following exports to your ~/.bashrc or pass them to Slurm jobs:"
echo ""
echo "  export nnUNet_raw_data_base=${DATA_BASE}"
echo "  export nnUNet_preprocessed=${PREPROCESSED_DIR}"
echo "  export RESULTS_FOLDER=${RESULTS_DIR}"
echo ""
echo "Next steps:"
echo "  sbatch scripts/01_preprocess.slurm"
echo "  sbatch --export=ALL,FOLD=0 scripts/02_train.slurm"
