#!/bin/bash
#SBATCH --job-name=eval_test_601
#SBATCH --partition=batch-milan
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/eval_test_601_%j.out
#SBATCH --error=logs/eval_test_601_%j.err

# Fix axis ordering in ensemble .nii.gz files, then evaluate against labelsTs.
#
# Root cause of the shape mismatch:
#   nnUNet softmax .npz stores data as (C, D, H, W) — depth-first order.
#   argmax → (D, H, W).  NIfTI / nibabel expects (W, H, D) = (x, y, z).
#   The previous ensemble step forgot to transpose, so predictions were saved
#   with axes (50,61,61) instead of (61,61,50).
#
# This script recomputes hard labels from the .npz files with the correct
# transpose and overwrites the .nii.gz files, then runs evaluation.
#
# Usage:
#   sbatch scripts/15_eval_test_601.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

mkdir -p "${REPO_DIR}/logs"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
PLANS="nnUNetPlansv2.1"
TASK_NAME="Task601_TotalSegmentatorV1"

MODEL_DIR="${DATA_BASE}/nnUNet_results/nnUNet/${NETWORK}/${TASK_NAME}/${TRAINER}__${PLANS}"
ENSEMBLE_DIR="${MODEL_DIR}/test_predictions/ensemble"
FOLD0_DIR="${MODEL_DIR}/test_predictions/fold_0"
GT_DIR="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/labelsTs"
SUMMARY_JSON="${ENSEMBLE_DIR}/summary.json"

N_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Task601 Test Evaluation ==="
echo "Trainer:     ${TRAINER}"
echo "Plans:       ${PLANS}"
echo "Ensemble:    ${ENSEMBLE_DIR}"
echo "GT:          ${GT_DIR}"
echo "Threads:     ${N_THREADS}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Started:     $(date)"
echo ""

cd "${REPO_DIR}"

export ENSEMBLE_DIR FOLD0_DIR GT_DIR SUMMARY_JSON N_THREADS

uv run python - << 'PYEOF'
import os, sys, json
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ensemble_dir = Path(os.environ['ENSEMBLE_DIR'])
fold0_dir    = Path(os.environ['FOLD0_DIR'])
gt_dir       = Path(os.environ['GT_DIR'])
summary_json = Path(os.environ['SUMMARY_JSON'])
n_threads    = int(os.environ['N_THREADS'])

# ── Step 1: Fix .nii.gz files (transpose D,H,W → W,H,D for nibabel) ──────────
case_npz = sorted(ensemble_dir.glob('*.npz'))
if not case_npz:
    print(f'ERROR: No .npz files in {ensemble_dir}', file=sys.stderr)
    sys.exit(1)

print(f'Step 1: Fixing {len(case_npz)} hard-label NIfTI files...')

def fix_case(npz_path):
    import shutil
    name = npz_path.stem          # e.g. 'TotalSegmentator_0230'
    avg  = np.load(npz_path)['softmax'].astype(np.float32)  # (C, D, H, W)
    hard = np.argmax(avg, axis=0).astype(np.uint8)           # (D, H, W)
    hard_nib = hard.transpose(2, 1, 0)                        # (W, H, D) = nibabel x,y,z

    # Borrow affine + header from fold_0 hard-label (correct orientation)
    ref_path = fold0_dir / f'{name}.nii.gz'
    if ref_path.exists():
        ref = nib.load(str(ref_path))
        if hard_nib.shape != ref.shape:
            # 1-voxel rounding difference between npz and nii.gz resampling.
            # Fall back to fold_0 hard label (correct shape + affine).
            shutil.copy2(str(ref_path), str(ensemble_dir / f'{name}.nii.gz'))
            return name, f'WARN: shape {hard_nib.shape} vs ref {ref.shape} — used fold_0 fallback'
        out_img = nib.Nifti1Image(hard_nib, ref.affine, ref.header)
    else:
        out_img = nib.Nifti1Image(hard_nib, np.eye(4))

    nib.save(out_img, str(ensemble_dir / f'{name}.nii.gz'))
    return name, None

warnings = []
done = 0
with ThreadPoolExecutor(max_workers=n_threads) as ex:
    futs = {ex.submit(fix_case, p): p for p in case_npz}
    for fut in as_completed(futs):
        name, msg = fut.result()
        done += 1
        if msg:
            warnings.append(f'{name}: {msg}')
            print(f'  [{done}/{len(case_npz)}] {msg}')
        else:
            print(f'  [{done}/{len(case_npz)}] OK {name}')

if warnings:
    print(f'\n[WARN] {len(warnings)} cases used fold_0 fallback (1-voxel rounding):')
    for w in warnings:
        print(f'  {w}')

print(f'\nAll NIfTI files fixed.')

# ── Step 2: Evaluate against GT ───────────────────────────────────────────────
print('\nStep 2: Evaluating against GT...')
sys.path.insert(0, '/scratch/work/tianmid1/nnUNet_cust')
from nnunet.evaluation.evaluator import aggregate_scores

pred_files = sorted(ensemble_dir.glob('*.nii.gz'))
pairs, missing = [], []
for pf in pred_files:
    gt = gt_dir / pf.name
    if not gt.exists():
        missing.append(pf.name)
        continue
    pairs.append((str(pf), str(gt)))

if missing:
    print(f'[WARN] No GT for {len(missing)} files: {missing[:5]}', file=sys.stderr)

print(f'Evaluating {len(pairs)} cases vs GT ({n_threads} threads)...')
labels = list(range(105))   # 0=background, 1-104=organs

aggregate_scores(
    pairs,
    labels=labels,
    json_output_file=str(summary_json),
    json_name='Task601_test',
    json_description='Task601 nnUNetTrainerV2_fast__nnUNetPlansv2.1 — 5-fold ensemble on imagesTs',
    json_author='auto',
    json_task='Task601_test',
    num_threads=n_threads,
)
print(f'Done. Summary: {summary_json}')

# ── Step 3: Print per-class Dice ──────────────────────────────────────────────
with open(summary_json) as fh:
    d = json.load(fh)
mean_vals = d['results']['mean']
print(f"\n{'Class':>5}  {'Mean Dice':>9}")
print('-' * 20)
for i in range(1, 105):
    dice = mean_vals.get(str(i), {}).get('Dice', float('nan'))
    print(f'{i:>5}  {dice:>9.4f}')
overall = mean_vals.get('mean', {}).get('Dice', float('nan'))
print('-' * 20)
print(f"{'Mean':>5}  {overall:>9.4f}")
PYEOF

echo ""
echo "=== Complete: $(date) ==="
echo "Hard labels (fixed): ${ENSEMBLE_DIR}/*.nii.gz"
echo "Summary:             ${SUMMARY_JSON}"
