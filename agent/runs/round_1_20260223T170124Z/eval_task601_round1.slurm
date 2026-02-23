#!/bin/bash
#SBATCH --job-name=nnunet_eval_601_r1
#SBATCH --partition=batch
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=eval_601_round1_%j.out
#SBATCH --error=eval_601_round1_%j.err

# Auto-generated evaluation script — Round 1
# Task:    Task601_TotalSegmentatorV1
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Plans:   nnUNetPlansv2.1
#
# Run AFTER ALL training folds have completed:
#   sbatch eval_601_round1.slurm
#
# Produces:
#   $RESULTS_FOLDER/.../cv_niftis_postprocessed/summary.json  → val_summary.json
# (no test-set inference; run with --use-test-data to enable)
# Both are copied into the run directory for the next agent round.

set -euo pipefail

REPO_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
RUN_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

TASK="601"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
PLANS="nnUNetPlansv2.1"

MODEL_DIR="${RESULTS_FOLDER}/nnUNet/${NETWORK}/Task${TASK}_TotalSegmentatorV1/${TRAINER}__${PLANS}"

echo "=== nnUNet Evaluation Round 1 ==="
echo "Task:    Task${TASK}_TotalSegmentatorV1"
echo "Network: ${NETWORK}"
echo "Trainer: ${TRAINER}"
echo "Model:   ${MODEL_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Started: $(date)"
echo ""

cd "${REPO_DIR}"

# ── Step 1: Consolidate folds and determine postprocessing ────────────────────
echo "Running nnUNet_determine_postprocessing..."
uv run nnUNet_determine_postprocessing \
    -m "${NETWORK}" \
    -t "${TASK}" \
    -tr "${TRAINER}" \
    -pl "${PLANS}"

echo ""
echo "nnUNet_determine_postprocessing complete: $(date)"
echo ""

# ── Step 2: Copy summaries into run_dir for the next agent round ──────────────

VAL_SUMMARY="${MODEL_DIR}/cv_niftis_postprocessed/summary.json"

if [[ -f "${VAL_SUMMARY}" ]]; then
    cp "${VAL_SUMMARY}" "${RUN_DIR}/val_summary.json"
    echo "Copied → ${RUN_DIR}/val_summary.json"
else
    echo "WARNING: val_summary not found: ${VAL_SUMMARY}"
fi

TEST_SUMMARY="${MODEL_DIR}/summary.json"

if [[ -f "${TEST_SUMMARY}" ]]; then
    cp "${TEST_SUMMARY}" "${RUN_DIR}/test_summary.json"
    echo "Copied → ${RUN_DIR}/test_summary.json"
else
    echo "INFO: test_summary not found (use --use-test-data for explicit test inference)"
fi

echo ""
echo "=== Evaluation round 1 complete: $(date) ==="
