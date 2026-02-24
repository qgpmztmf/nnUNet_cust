#!/bin/bash
set -e
echo "Training script for nnUNet Auto-Tuning Pipeline"
echo "================================================"

# Configuration
HYPERPARAMETER_REFERENCE_PATH="/home/tianmid1/tianmid1/nnUNet_cust/agent/doc/hyperparameter_reference.json"
TASK_NAME="Task601_TotalSegmentatorV1"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
ROUND_NUMBER=2
SCRIPT_NAME="train_task601_round2.sh"
LOG_FILE="nnUNet_training_round2.log"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Step 1: Validate Configuration
echo "[$TIMESTAMP] Starting training round $ROUND_NUMBER for $TASK_NAME"
echo "[$TIMESTAMP] Validating hyperparameter configuration..."

if [ ! -f "$HYPERPARAMETER_REFERENCE_PATH" ]; then
    echo "ERROR: Hyperparameter reference file not found at $HYPERPARAMETER_REFERENCE_PATH"
    exit 1
fi

# Check if JSON is readable and has active values
ACTIVE_COUNT=$(python3 -c "
import json
try:
    with open('$HYPERPARAMETER_REFERENCE_PATH', 'r') as f:
        data = json.load(f)
    active_params = [k for k, v in data.items() if 'active_value' in v and v['active_value'] is not None]
    print(len(active_params))
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
")

if [ "$ACTIVE_COUNT" -eq "0" ]; then
    echo "WARNING: No active_value found in hyperparameter_reference.json"
    echo "Training will proceed with default parameters from trainer."
fi

echo "[$TIMESTAMP] Configuration validated. Found $ACTIVE_COUNT active parameters."

# Create log directory if needed
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ] && [ "$LOG_DIR" != "." ]; then
    mkdir -p "$LOG_DIR"
fi

# Save command to log
echo "[$TIMESTAMP] Command: $0 $@" >> "$LOG_FILE"
echo "[$TIMESTAMP] Hyperparameter reference: $HYPERPARAMETER_REFERENCE_PATH" >> "$LOG_FILE"

# Set environment variables (if required by your setup)
# Uncomment and modify if needed:
# export nnUNet_raw="/path/to/nnUNet_raw"
# export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
# export nnUNet_results="/path/to/nnUNet_results"

# Activate Python environment (modify as needed for your setup)
# Example for conda:
# source /home/tianmid1/miniconda3/etc/profile.d/conda.sh
# conda activate nnunet

# Launch training
echo "[$TIMESTAMP] Starting nnUNet training..."
echo "[$TIMESTAMP] Task: $TASK_NAME, Network: $NETWORK, Trainer: $TRAINER" | tee -a "$LOG_FILE"

# Using nnUNetv2_train command
nnUNetv2_train \
    $TASK_NAME \
    $NETWORK \
    $TRAINER \
    --fold all \
    2>&1 | tee -a "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] Training completed successfully." | tee -a "$LOG_FILE"
else
    echo "[$TIMESTAMP] Training failed with exit code $TRAINING_EXIT_CODE." | tee -a "$LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi

echo "[$TIMESTAMP] Training round $ROUND_NUMBER finished." | tee -a "$LOG_FILE"