#!/bin/bash
# Submit the full test inference pipeline with SLURM job dependencies:
#   Step 1: 06_predict_test_grouped.slurm  (20 GPU jobs: 4 tasks × 5 folds)
#   Step 2: 07_ensemble_test.slurm         (CPU, waits for all Step 1)
#   Step 3: 08_fuse_test.slurm             (CPU, waits for Step 2)
#
# Usage:
#   bash scripts/07_predict_ensemble_fuse_test.sh

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
cd "${REPO_DIR}"
mkdir -p logs

echo "=== Submit Test Inference Pipeline ==="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Predict — one array job per task (5 folds each)
# ---------------------------------------------------------------------------
echo "--- Step 1: Predicting (4 tasks × 5 folds) ---"
PREDICT_JOB_IDS=()

for TASK in 611 612 613 614; do
    JID=$(sbatch \
        --array=0-4 \
        --export=ALL,TASK=${TASK} \
        --parsable \
        scripts/06_predict_test_grouped.slurm)
    echo "  Task${TASK}: job array ${JID}"
    PREDICT_JOB_IDS+=("${JID}")
done

# Build dependency: afterok:JID1:JID2:JID3:JID4
PREDICT_DEP="afterok$(printf ':%s' "${PREDICT_JOB_IDS[@]}")"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Ensemble — average softmax across folds
# ---------------------------------------------------------------------------
echo "--- Step 2: Ensemble (waits for all Step 1) ---"
ENSEMBLE_JID=$(sbatch \
    --dependency="${PREDICT_DEP}" \
    --parsable \
    scripts/07_ensemble_test.slurm)
echo "  Ensemble job: ${ENSEMBLE_JID}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Fuse — combine 4 group softmax into 104-class output
# ---------------------------------------------------------------------------
echo "--- Step 3: Fuse (waits for Step 2) ---"
FUSE_JID=$(sbatch \
    --dependency="afterok:${ENSEMBLE_JID}" \
    --parsable \
    scripts/08_fuse_test.slurm)
echo "  Fuse job: ${FUSE_JID}"
echo ""

echo "=== Pipeline submitted ==="
echo "  Step 1 (predict):   ${PREDICT_JOB_IDS[*]}"
echo "  Step 2 (ensemble):  ${ENSEMBLE_JID}"
echo "  Step 3 (fuse):      ${FUSE_JID}"
echo ""
echo "  Final output: /scratch/work/tianmid1/data/test_fused/"
echo ""
echo "Monitor with:  squeue -u \$USER"
