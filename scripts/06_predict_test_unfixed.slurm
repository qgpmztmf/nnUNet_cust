#!/bin/bash
#SBATCH --job-name=predict_test_unfixed
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/predict_test_unfixed_%j.out
#SBATCH --error=logs/predict_test_unfixed_%j.err

# Three-step pipeline for Task601 unfixed model on the held-out test set:
#
#   Step 1 — Symlink the backup model to the standard nnUNet results path
#   Step 2 — nnUNet_predict on imagesTs with all 5 folds + softmax (.npz) output
#   Step 3 — Evaluate hard-label predictions against labelsTs (Dice, summary.json)
#
# Usage:
#   sbatch scripts/06_predict_test_unfixed.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
PLANS="nnUNetPlansv2.1_unfixed"
TASK="601"
TASK_NAME="Task601_TotalSegmentatorV1"

INPUT_DIR="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/imagesTs"
GT_DIR="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/labelsTs"
OUTPUT_DIR="${DATA_BASE}/nnUNet_results/test_predictions/${TASK_NAME}/${TRAINER}__${PLANS}"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

BACKUP_MODEL="${RESULTS_FOLDER}/nnUNet/${NETWORK}/${TASK_NAME}/backup/${TRAINER}__${PLANS}"
STANDARD_MODEL="${RESULTS_FOLDER}/nnUNet/${NETWORK}/${TASK_NAME}/${TRAINER}__${PLANS}"

echo "=== Predict Test Set: ${TASK_NAME} (unfixed) ==="
echo "Trainer:  ${TRAINER}"
echo "Plans:    ${PLANS}"
echo "Input:    ${INPUT_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo "GT:       ${GT_DIR}"
echo "GPU:      ${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Node:     ${SLURM_NODELIST}"
echo "Started:  $(date)"
echo ""

cd "${REPO_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Symlink backup model to standard nnUNet results path
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 1: Symlinking backup model ──"

if [ ! -e "${BACKUP_MODEL}" ]; then
    echo "ERROR: Backup model not found: ${BACKUP_MODEL}" >&2
    exit 1
fi

if [ -e "${STANDARD_MODEL}" ]; then
    echo "  Path already exists: ${STANDARD_MODEL}"
    if [ -L "${STANDARD_MODEL}" ]; then
        echo "  (existing symlink — OK)"
    else
        echo "  WARNING: not a symlink — using as-is"
    fi
else
    ln -s "${BACKUP_MODEL}" "${STANDARD_MODEL}"
    echo "  Symlink created: ${STANDARD_MODEL} -> ${BACKUP_MODEL}"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: nnUNet_predict on imagesTs (all 5 folds, save softmax npz)
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 2: Running nnUNet_predict ──"
echo "  Folds:   0 1 2 3 4 (5-fold ensemble)"
echo "  Softmax: enabled (-z)"
echo ""

mkdir -p "${OUTPUT_DIR}"

uv run nnUNet_predict \
    -i  "${INPUT_DIR}" \
    -o  "${OUTPUT_DIR}" \
    -t  "${TASK}" \
    -tr "${TRAINER}" \
    -m  "${NETWORK}" \
    -p  "${PLANS}" \
    -f  0 1 2 3 4 \
    -z \
    --num_threads_preprocessing "${SLURM_CPUS_PER_TASK}" \
    --num_threads_nifti_save    "${SLURM_CPUS_PER_TASK}" \
    --disable_tta

echo ""
echo "Prediction complete: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate hard-label predictions against GT (labelsTs)
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 3: Evaluating predictions ──"

export OUTPUT_DIR GT_DIR SUMMARY_JSON

uv run python - << PYEOF
import sys, json, os
from pathlib import Path

sys.path.insert(0, '/scratch/work/tianmid1/nnUNet_cust')
from nnunet.evaluation.evaluator import aggregate_scores

output_dir = os.environ['OUTPUT_DIR']
gt_dir     = os.environ['GT_DIR']
summary    = os.environ['SUMMARY_JSON']
n_threads  = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))

pred_files = sorted(Path(output_dir).glob('*.nii.gz'))
if not pred_files:
    print('ERROR: no .nii.gz predictions found in', output_dir, file=sys.stderr)
    sys.exit(1)

pairs, missing = [], []
for pf in pred_files:
    gt = Path(gt_dir) / pf.name
    if not gt.exists():
        missing.append(pf.name)
        continue
    pairs.append((str(pf), str(gt)))

if missing:
    print(f'[WARN] No GT for {len(missing)} files: {missing[:5]}', file=sys.stderr)

print(f'Evaluating {len(pairs)} cases vs GT ({n_threads} threads)...')
labels = list(range(105))  # 0=background, 1-104=organs

aggregate_scores(
    pairs,
    labels=labels,
    json_output_file=summary,
    json_name='Task601_unfixed_test',
    json_description='Task601 nnUNetTrainerV2_fast (unfixed) — 5-fold ensemble on imagesTs',
    json_author='auto',
    json_task='Task601_unfixed_test',
    num_threads=n_threads,
)
print('Done. Summary:', summary)

# Print mean Dice per class
with open(summary) as f:
    d = json.load(f)
mean_vals = d['results']['mean']
print(f"\n{'Class':>5}  {'Mean Dice':>9}")
print('-' * 20)
for i in range(1, 105):
    dice = mean_vals.get(str(i), {}).get('Dice', float('nan'))
    print(f"{i:>5}  {dice:>9.4f}")
overall = mean_vals.get('mean', {}).get('Dice', float('nan'))
print('-' * 20)
print(f"{'Mean':>5}  {overall:>9.4f}")
PYEOF

echo ""
echo "=== All steps complete: $(date) ==="
echo "Predictions: ${OUTPUT_DIR}"
echo "Summary:     ${SUMMARY_JSON}"
