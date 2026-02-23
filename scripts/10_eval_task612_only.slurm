#!/bin/bash
#SBATCH --job-name=eval_t612
#SBATCH --partition=batch-milan
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/eval_t612_%j.out
#SBATCH --error=logs/eval_t612_%j.err

# Evaluate Task612 (SoftTissue) model alone on the test set.
#
#   Step 1 — Convert Task612 pre-ensembled softmax → patient-space label maps
#             Uses fuse.py with --task_ids 612 (single-model, no cross-task fusion).
#             Output labels are in global space (0-104) but only Task612's 24 fg
#             classes can be predicted; all other 80 classes will be 0.
#
#   Step 2 — Evaluate against GT (labelsTs) for Task612's 24 global classes only.
#             Classes: 1-5, 8-12, 50, 53-54, 94-104
#             Output: test_task612_only/summary.json
#
# Usage:
#   sbatch scripts/10_eval_task612_only.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"
RAW_DATA="${DATA_BASE}/nnUNet_raw_data"

ENSEMBLE_ROOT="${DATA_BASE}/nnUNet_results/test_predictions/test_ensemble"
REF_NII_ROOT="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data/nnUNet_results/test_predictions"
GT_DIR="${RAW_DATA}/Task601_TotalSegmentatorV1/labelsTs"
OUTPUT_DIR="${DATA_BASE}/nnUNet_results/test_predictions/test_task612_only"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
CASES_LIST="${REPO_DIR}/test_cases.txt"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"

echo "=== Task612-Only Evaluation ==="
echo "Ensemble root: ${ENSEMBLE_ROOT}"
echo "Ref nii root:  ${REF_NII_ROOT}"
echo "GT dir:        ${GT_DIR}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Cases:         $(wc -l < "${CASES_LIST}")"
echo "Node:          ${SLURM_NODELIST}"
echo "Threads:       ${SLURM_CPUS_PER_TASK}"
echo "Started:       $(date)"
echo ""

cd "${REPO_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Task612 softmax → patient-space NIfTI label maps
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 1: Converting Task612 softmax → NIfTI ──"

uv run python fuse.py \
    --cases_list      "${CASES_LIST}" \
    --input_root      "${ENSEMBLE_ROOT}" \
    --ref_nii_root    "${REF_NII_ROOT}" \
    --output_root     "${OUTPUT_DIR}" \
    --nnunet_raw_data "${RAW_DATA}" \
    --task_ids        612 \
    --method          A \
    --save_nii \
    --sanity

echo ""
echo "Conversion complete: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate Task612's 24 global classes against GT
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 2: Evaluating Task612 classes only ──"

export OUTPUT_DIR GT_DIR SUMMARY_JSON

uv run python - << 'PYEOF'
import sys, os, json
from pathlib import Path

sys.path.insert(0, '/scratch/work/tianmid1/nnUNet_cust')
from nnunet.evaluation.evaluator import aggregate_scores
from fuse import load_class_map_from_nnunet

raw_data    = Path('/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_raw_data')
output_dir  = Path(os.environ['OUTPUT_DIR'])
gt_dir      = Path(os.environ['GT_DIR'])
summary_path = os.environ['SUMMARY_JSON']
n_threads   = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))

# Task612 global classes only
maps = load_class_map_from_nnunet(raw_data, [612])
t612_global = sorted(maps[612].values())
eval_labels = [0] + t612_global   # include bg (0) for completeness
print(f"Evaluating {len(t612_global)} foreground classes: {t612_global}")

# Build (pred, gt) pairs
pred_files = sorted(output_dir.glob('*.nii.gz'))
print(f"Found {len(pred_files)} prediction files in {output_dir}")

pairs, missing = [], []
for pf in pred_files:
    gt_name = pf.name.replace('TotalSeg_', 'TotalSegmentator_')
    gt_path = gt_dir / gt_name
    if not gt_path.exists():
        missing.append(f"{pf.name} -> {gt_name}")
        continue
    pairs.append((str(pf), str(gt_path)))

if missing:
    print(f"[WARN] No GT for {len(missing)} file(s):", file=sys.stderr)
    for m in missing[:5]:
        print(f"  {m}", file=sys.stderr)

print(f"Evaluating {len(pairs)} pairs ({n_threads} threads)...")
aggregate_scores(
    pairs,
    labels=eval_labels,
    json_output_file=summary_path,
    json_name='Task612_only_test',
    json_description='Task612 SoftTissue model alone — test set evaluation',
    json_author='auto',
    json_task='Task612_only_test',
    num_threads=n_threads,
)
print(f"Summary saved: {summary_path}")
PYEOF

echo ""
echo "Evaluation complete: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Print per-class Dice summary
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 3: Per-class Dice summary ──"
echo ""

uv run python scripts/print_dice_summary.py "${SUMMARY_JSON}"

echo ""
echo "=== All steps complete: $(date) ==="
echo "Predictions: ${OUTPUT_DIR}"
echo "Summary:     ${SUMMARY_JSON}"
