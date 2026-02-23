#!/bin/bash
set -e
set -x

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
HYPERPARAMETER_REFERENCE_PATH="/home/tianmid1/tianmid1/nnUNet_cust/documentation/hyperparameter_reference.json"
TASK_NAME="Task601_TotalSegmentatorV1"
TASK_ID="601"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
ROUND_NUMBER="2"

SCRIPT_NAME="train_task601_round2.sh"
LOG_FILE="nnUNet_training_round2.log"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# ------------------------------------------------------------
# Step 1: Validate Configuration
# ------------------------------------------------------------
echo "[${TIMESTAMP}] Starting training script: ${SCRIPT_NAME}" | tee -a "${LOG_FILE}"
echo "Validating hyperparameter reference file..." | tee -a "${LOG_FILE}"

if [[ ! -f "${HYPERPARAMETER_REFERENCE_PATH}" ]]; then
    echo "ERROR: Hyperparameter reference file not found at ${HYPERPARAMETER_REFERENCE_PATH}" | tee -a "${LOG_FILE}"
    exit 1
fi

if ! python3 -c "import json; json.load(open('${HYPERPARAMETER_REFERENCE_PATH}'))" 2>/dev/null; then
    echo "ERROR: Hyperparameter reference file is not valid JSON." | tee -a "${LOG_FILE}"
    exit 1
fi

ACTIVE_COUNT=$(python3 -c "
import json
with open('${HYPERPARAMETER_REFERENCE_PATH}') as f:
    data = json.load(f)
count = sum(1 for v in data.values() if 'active_value' in v)
print(count)
")

if [[ "${ACTIVE_COUNT}" -eq 0 ]]; then
    echo "WARNING: No active_value fields found in hyperparameter reference. Training will use defaults." | tee -a "${LOG_FILE}"
else
    echo "Found ${ACTIVE_COUNT} active hyperparameter overrides." | tee -a "${LOG_FILE}"
fi

# ------------------------------------------------------------
# Step 2: Set Environment (adjust if needed)
# ------------------------------------------------------------
# If using a conda environment, activate it here:
# conda activate your_nnunet_env

# Set nnUNet environment variables if required
export nnUNet_raw="/home/tianmid1/tianmid1/nnUNet_cust/nnUNet_raw"
export nnUNet_preprocessed="/home/tianmid1/tianmid1/nnUNet_cust/nnUNet_preprocessed"
export nnUNet_results="/home/tianmid1/tianmid1/nnUNet_cust/nnUNet_results"

# ------------------------------------------------------------
# Step 3: Launch Training
# ------------------------------------------------------------
echo "Launching nnUNet training for round ${ROUND_NUMBER}..." | tee -a "${LOG_FILE}"
echo "Task: ${TASK_NAME} (ID: ${TASK_ID})" | tee -a "${LOG_FILE}"
echo "Network: ${NETWORK}" | tee -a "${LOG_FILE}"
echo "Trainer: ${TRAINER}" | tee -a "${LOG_FILE}"
echo "Hyperparameters will be read from: ${HYPERPARAMETER_REFERENCE_PATH}" | tee -a "${LOG_FILE}"

COMMAND="nnUNetv2_train ${TASK_ID} ${NETWORK} ${TRAINER} --fold all"
echo "Running command: ${COMMAND}" | tee -a "${LOG_FILE}"

# Execute training
${COMMAND} 2>&1 | tee -a "${LOG_FILE}"

# ------------------------------------------------------------
# Step 4: Finalize
# ------------------------------------------------------------
END_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
echo "[${END_TIMESTAMP}] Training script finished." | tee -a "${LOG_FILE}"