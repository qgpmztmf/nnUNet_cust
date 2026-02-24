#!/bin/bash
#SBATCH --job-name=nnunet_eval_601_r2
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=eval_601_round2_%j.out
#SBATCH --error=eval_601_round2_%j.err

# Auto-generated evaluation script — Round 2
# Task:    Task601_TotalSegmentatorV1
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Plans:   nnUNetPlansv2.1
#
# Run AFTER ALL training folds have completed:
#   sbatch eval_601_round2.slurm
#
# Produces:
#   $RESULTS_FOLDER/.../cv_niftis_postprocessed/summary.json  → val_summary.json
#   $RESULTS_FOLDER/.../test_predictions/summary.json  → test_summary.json
# (test-set inference enabled via --use-test-data)
# Both are copied into the run directory for the next agent round.

set -euo pipefail

REPO_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
RUN_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_2_20260224T134301Z"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

TASK="601"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
PLANS="nnUNetPlansv2.1"

MODEL_DIR="${RESULTS_FOLDER}/nnUNet/${NETWORK}/Task${TASK}_TotalSegmentatorV1/${TRAINER}__${PLANS}"

echo "=== nnUNet Evaluation Round 2 ==="
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

# ── Step 2: Test-set inference + evaluation ───────────────────────────────────
IMAGES_TS="${DATA_BASE}/nnUNet_raw_data/Task${TASK}_TotalSegmentatorV1/imagesTs"
LABELS_TS="${DATA_BASE}/nnUNet_raw_data/Task${TASK}_TotalSegmentatorV1/labelsTs"
PRED_DIR="${MODEL_DIR}/test_predictions"

mkdir -p "${PRED_DIR}"

echo "Running nnUNet_predict on test set..."
uv run nnUNet_predict \
    -i "${IMAGES_TS}" \
    -o "${PRED_DIR}" \
    -t "${TASK}" \
    -m "${NETWORK}" \
    -tr "${TRAINER}" \
    -p "${PLANS}" \
    -f 0 1 2 3 4

echo ""
echo "nnUNet_predict complete: $(date)"
echo ""

echo "Running nnUNet_evaluate_folder on test predictions..."
uv run nnUNet_evaluate_folder \
    -ref "${LABELS_TS}" \
    -pred "${PRED_DIR}"

echo ""
echo "nnUNet_evaluate_folder complete: $(date)"
echo ""

# ── Step 3: Copy summaries into run_dir for the next agent round ──────────────

VAL_SUMMARY="${MODEL_DIR}/cv_niftis_postprocessed/summary.json"

if [[ -f "${VAL_SUMMARY}" ]]; then
    cp "${VAL_SUMMARY}" "${RUN_DIR}/val_summary.json"
    echo "Copied → ${RUN_DIR}/val_summary.json"
else
    echo "WARNING: val_summary not found: ${VAL_SUMMARY}"
fi

TEST_SUMMARY="${PRED_DIR}/summary.json"

if [[ -f "${TEST_SUMMARY}" ]]; then
    cp "${TEST_SUMMARY}" "${RUN_DIR}/test_summary.json"
    echo "Copied → ${RUN_DIR}/test_summary.json"
else
    echo "WARNING: test_summary not found: ${TEST_SUMMARY}"
fi

echo ""
echo "=== Evaluation round 2 complete: $(date) ==="
