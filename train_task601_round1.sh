#!/usr/bin/env bash
set -euo pipefail
set -x

# ---------------------------------------------------------------------------
# Auto-generated training script — Round 1
# Task:    Task601_TotalSegmentatorV1
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Generated: 2026-02-22
# Hyperparameter source:
#   /home/tianmid1/tianmid1/nnUNet_cust/documentation/hyperparameter_reference.json
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_REF="${REPO_DIR}/documentation/hyperparameter_reference.json"
ROUND=1
LOG_FILE="${REPO_DIR}/nnUNet_training_round${ROUND}.log"
CMD_FILE="${REPO_DIR}/nnUNet_training_round${ROUND}_cmd.txt"

# ---------------------------------------------------------------------------
# Step 1 — Validate hyperparameter_reference.json
# ---------------------------------------------------------------------------
if [[ ! -f "${HP_REF}" ]]; then
    echo "ERROR: hyperparameter_reference.json not found at: ${HP_REF}" >&2
    exit 1
fi

if ! python3 -c "import json, sys; json.load(open(sys.argv[1]))" "${HP_REF}" 2>/dev/null; then
    echo "ERROR: hyperparameter_reference.json is not valid JSON." >&2
    exit 1
fi

ACTIVE_COUNT=$(python3 - <<'PYEOF'
import json, sys
hp = json.load(open(sys.argv[1]))
count = sum(1 for v in hp.values() if isinstance(v, dict) and "active_value" in v)
print(count)
PYEOF
"${HP_REF}")

if [[ "${ACTIVE_COUNT}" -eq 0 ]]; then
    echo "WARNING: No active_value fields found in hyperparameter_reference.json." \
         "Trainer will fall back to defaults."
else
    echo "INFO: Found ${ACTIVE_COUNT} active_value override(s) in hyperparameter_reference.json:"
    python3 - <<'PYEOF'
import json, sys
hp = json.load(open(sys.argv[1]))
for k, v in hp.items():
    if isinstance(v, dict) and "active_value" in v:
        print(f"  {k}: {v['active_value']}")
PYEOF
    "${HP_REF}"
fi

# ---------------------------------------------------------------------------
# Step 2 — Environment variables
# ---------------------------------------------------------------------------
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"
export nnUNet_n_proc_DA=16

# Path to the hyperparameter config consumed by nnUNetTrainerV2_configurable
export NNUNET_HP_REF="${HP_REF}"

# ---------------------------------------------------------------------------
# Step 3 — Output directory guard
# ---------------------------------------------------------------------------
OUTPUT_DIR="${RESULTS_FOLDER}/nnUNet/3d_fullres/Task601_TotalSegmentatorV1/nnUNetTrainerV2_configurable__nnUNetPlansv2.1"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Step 4 — Record metadata
# ---------------------------------------------------------------------------
START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Training start: ${START_TS}" | tee -a "${LOG_FILE}"
echo "Round:          ${ROUND}"    | tee -a "${LOG_FILE}"
echo "HP_REF mtime:   $(stat -c '%y' "${HP_REF}")" | tee -a "${LOG_FILE}"
echo "Active values:  ${ACTIVE_COUNT}" | tee -a "${LOG_FILE}"
echo "DATA_BASE:      ${DATA_BASE}"    | tee -a "${LOG_FILE}"
echo "RESULTS_FOLDER: ${RESULTS_FOLDER}" | tee -a "${LOG_FILE}"
echo "---" | tee -a "${LOG_FILE}"

# ---------------------------------------------------------------------------
# Step 5 — Build and record the training command template
# ---------------------------------------------------------------------------
TRAIN_CMD="uv run nnUNet_train 3d_fullres nnUNetTrainerV2_configurable 601 FOLD -p nnUNetPlansv2.1"
echo "Command template (FOLD substituted per iteration):" | tee -a "${LOG_FILE}"
echo "  ${TRAIN_CMD}" | tee -a "${LOG_FILE}"
echo "${TRAIN_CMD}" > "${CMD_FILE}"
echo "Command saved to: ${CMD_FILE}"

# ---------------------------------------------------------------------------
# Step 6 — Train all folds
# ---------------------------------------------------------------------------
FOLDS=(0 1 2 3 4)

for FOLD in "${FOLDS[@]}"; do
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Starting fold ${FOLD} at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"

    FOLD_CMD="uv run nnUNet_train 3d_fullres nnUNetTrainerV2_configurable 601 ${FOLD} -p nnUNetPlansv2.1"

    # Continue from checkpoint if a previous run was interrupted (-c flag)
    CHECKPOINT="${OUTPUT_DIR}/fold_${FOLD}/model_latest.model"
    if [[ -f "${CHECKPOINT}" ]]; then
        echo "INFO: Checkpoint found for fold ${FOLD}. Resuming with -c." | tee -a "${LOG_FILE}"
        FOLD_CMD="${FOLD_CMD} -c"
    fi

    echo "Executing: ${FOLD_CMD}" | tee -a "${LOG_FILE}"

    # Run from repo root (required for uv run)
    (cd "${REPO_DIR}" && eval "${FOLD_CMD}") 2>&1 | tee -a "${LOG_FILE}"

    echo "Fold ${FOLD} finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${LOG_FILE}"
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
END_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "All folds complete." | tee -a "${LOG_FILE}"
echo "Training end: ${END_TS}" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
