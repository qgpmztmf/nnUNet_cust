#!/bin/bash
#SBATCH --job-name=fuse_eval_A
#SBATCH --partition=batch-milan
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/fuse_eval_A_%j.out
#SBATCH --error=logs/fuse_eval_A_%j.err

# Two-step pipeline: fuse + evaluate grouped model test predictions (Method A).
#
#   Step 1 — Fuse pre-ensembled (5-fold averaged) softmax from Task611-614.
#             Pipeline per case:
#               a) Load softmax .npz from test_ensemble (nnUNet internal space)
#               b) Fuse 4 models: bg = product of bg probs; fg = copy local→global
#               c) Normalise to sum-to-1
#               d) Argmax → label map (D,H,W) in internal space
#               e) Resample via scipy.ndimage.zoom (order=0, nearest-neighbour)
#                  from internal (D,H,W) → patient (X,Y,Z) using reference .nii.gz
#             Output: test_fused_A/TotalSeg_XXXX.nii.gz  (uint16, labels 0-104,
#                     same shape and affine as GT)
#
#   Step 2 — Evaluate fused label maps against labelsTs (aggregate_scores).
#             Name mapping: pred TotalSeg_XXXX ↔ GT TotalSegmentator_XXXX
#             Output: test_fused_A/summary.json
#
# Usage:
#   sbatch scripts/09_fuse_and_eval_method_A.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"
RAW_DATA="${DATA_BASE}/nnUNet_raw_data"

ENSEMBLE_ROOT="${DATA_BASE}/nnUNet_results/test_predictions/test_ensemble"
# Per-fold hard-label .nii.gz — used only to get patient-space affine/shape
REF_NII_ROOT="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data/nnUNet_results/test_predictions"
GT_DIR="${RAW_DATA}/Task601_TotalSegmentatorV1/labelsTs"
OUTPUT_DIR="${DATA_BASE}/nnUNet_results/test_predictions/test_fused_A"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
CASES_LIST="${REPO_DIR}/test_cases.txt"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"

echo "=== Fuse + Evaluate Grouped Test Predictions (Method A) ==="
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
# Step 1: Fuse softmax → patient-space NIfTI label maps
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 1: Fusing softmax (Method A) ──"
echo "  softmax (internal space) → argmax → zoom → patient space .nii.gz"
echo ""

uv run python fuse.py \
    --cases_list      "${CASES_LIST}" \
    --input_root      "${ENSEMBLE_ROOT}" \
    --ref_nii_root    "${REF_NII_ROOT}" \
    --output_root     "${OUTPUT_DIR}" \
    --nnunet_raw_data "${RAW_DATA}" \
    --method          A \
    --save_nii \
    --sanity

echo ""
echo "Fusion complete: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate against GT (aggregate_scores → summary.json)
# ─────────────────────────────────────────────────────────────────────────────
echo "── Step 2: Evaluating predictions ──"

export OUTPUT_DIR GT_DIR SUMMARY_JSON

uv run python - << 'PYEOF'
import sys, os, json
from pathlib import Path

sys.path.insert(0, '/scratch/work/tianmid1/nnUNet_cust')
from nnunet.evaluation.evaluator import aggregate_scores

output_dir   = Path(os.environ['OUTPUT_DIR'])
gt_dir       = Path(os.environ['GT_DIR'])
summary_path = os.environ['SUMMARY_JSON']
n_threads    = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))

# Build (pred, gt) pairs
# Predictions : TotalSeg_XXXX.nii.gz
# GT files    : TotalSegmentator_XXXX.nii.gz
pred_files = sorted(output_dir.glob('*.nii.gz'))
print(f"Found {len(pred_files)} prediction files in {output_dir}")

pairs, missing = [], []
for pf in pred_files:
    gt_name = pf.name.replace('TotalSeg_', 'TotalSegmentator_')
    gt_path = gt_dir / gt_name
    if not gt_path.exists():
        missing.append(f"{pf.name} → {gt_name}")
        continue
    pairs.append((str(pf), str(gt_path)))

if missing:
    print(f"[WARN] No GT for {len(missing)} file(s):", file=sys.stderr)
    for m in missing[:5]:
        print(f"  {m}", file=sys.stderr)

print(f"Evaluating {len(pairs)} pairs ({n_threads} threads)...")
aggregate_scores(
    pairs,
    labels=list(range(105)),
    json_output_file=summary_path,
    json_name='Task611-614_fused_methodA_test',
    json_description='Task611-614 grouped models, 5-fold ensemble, Method A fusion — test set',
    json_author='auto',
    json_task='Task611-614_fused_methodA_test',
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
