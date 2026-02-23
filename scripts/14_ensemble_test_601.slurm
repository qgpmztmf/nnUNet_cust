#!/bin/bash
#SBATCH --job-name=ensemble_601
#SBATCH --partition=batch-milan
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=logs/ensemble_test_601_%j.out
#SBATCH --error=logs/ensemble_test_601_%j.err

# Average per-fold softmax .npz files across all 5 folds for Task601.
# Saves one ensembled softmax .npz + one hard-label .nii.gz per test case.
# Then evaluates hard labels against labelsTs and prints per-class Dice.
#
# Must run AFTER 13_predict_test_601.slurm (all 5 folds) has completed.
#
# Usage:
#   sbatch scripts/14_ensemble_test_601.slurm
#
# Chain after prediction:
#   PRED_JOB=$(sbatch --array=0-4 --parsable scripts/13_predict_test_601.slurm)
#   sbatch --dependency=afterok:${PRED_JOB} scripts/14_ensemble_test_601.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

mkdir -p "${REPO_DIR}/logs"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_fast"
PLANS="nnUNetPlansv2.1"
TASK_NAME="Task601_TotalSegmentatorV1"

MODEL_DIR="${RESULTS_FOLDER}/nnUNet/${NETWORK}/${TASK_NAME}/${TRAINER}__${PLANS}"
FOLD_PRED_BASE="${MODEL_DIR}/test_predictions"
ENSEMBLE_DIR="${MODEL_DIR}/test_predictions/ensemble"
GT_DIR="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/labelsTs"
SUMMARY_JSON="${ENSEMBLE_DIR}/summary.json"

N_FOLDS=5
N_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Task601 Test Ensemble ==="
echo "Trainer:     ${TRAINER}"
echo "Plans:       ${PLANS}"
echo "Fold dirs:   ${FOLD_PRED_BASE}/fold_0 ... fold_4"
echo "Output:      ${ENSEMBLE_DIR}"
echo "GT:          ${GT_DIR}"
echo "Threads:     ${N_THREADS}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Started:     $(date)"
echo ""

cd "${REPO_DIR}"

export MODEL_DIR FOLD_PRED_BASE ENSEMBLE_DIR GT_DIR SUMMARY_JSON N_FOLDS N_THREADS

uv run python - << 'PYEOF'
import os, sys, json
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

fold_pred_base = Path(os.environ['FOLD_PRED_BASE'])
ensemble_dir   = Path(os.environ['ENSEMBLE_DIR'])
gt_dir         = Path(os.environ['GT_DIR'])
summary_json   = Path(os.environ['SUMMARY_JSON'])
n_folds        = int(os.environ['N_FOLDS'])
n_threads      = int(os.environ['N_THREADS'])

ensemble_dir.mkdir(parents=True, exist_ok=True)

# ── Collect case names from fold_0 ────────────────────────────────────────────
fold0_dir = fold_pred_base / 'fold_0'
case_npz = sorted(fold0_dir.glob('*.npz'))
if not case_npz:
    print(f'ERROR: No .npz files found in {fold0_dir}', file=sys.stderr)
    sys.exit(1)
case_names = [f.stem for f in case_npz]          # e.g. 'TotalSeg_0001'
print(f'Found {len(case_names)} cases.')

# ── Verify all folds present ──────────────────────────────────────────────────
for f in range(n_folds):
    d = fold_pred_base / f'fold_{f}'
    if not d.exists():
        print(f'ERROR: Missing fold directory: {d}', file=sys.stderr)
        sys.exit(1)

# ── Helper: load softmax array from .npz ─────────────────────────────────────
def load_softmax(path):
    data = np.load(path)
    # nnUNet saves key 'softmax' with shape (C, D, H, W), float16 or float32
    return data['softmax'].astype(np.float32)

# ── Per-case: average softmax across folds, write .npz + .nii.gz ─────────────
def process_case(case_name):
    arrays = []
    for f in range(n_folds):
        npz_path = fold_pred_base / f'fold_{f}' / f'{case_name}.npz'
        if not npz_path.exists():
            return case_name, f'MISSING fold_{f}'
        arrays.append(load_softmax(npz_path))

    avg = np.mean(arrays, axis=0)          # (C, D, H, W)
    hard = np.argmax(avg, axis=0).astype(np.uint8)  # (D, H, W)

    # Save averaged softmax .npz
    out_npz = ensemble_dir / f'{case_name}.npz'
    np.savez_compressed(str(out_npz), softmax=avg.astype(np.float16))

    # Save hard-label NIfTI — borrow affine/header from fold_0 prediction
    ref_nii_path = fold_pred_base / 'fold_0' / f'{case_name}.nii.gz'
    if ref_nii_path.exists():
        ref = nib.load(str(ref_nii_path))
        out_nii = nib.Nifti1Image(hard, ref.affine, ref.header)
    else:
        out_nii = nib.Nifti1Image(hard, np.eye(4))
    nib.save(out_nii, str(ensemble_dir / f'{case_name}.nii.gz'))

    return case_name, None

print(f'Averaging softmax across {n_folds} folds ({n_threads} threads)...')
errors = []
done = 0
with ThreadPoolExecutor(max_workers=n_threads) as ex:
    futs = {ex.submit(process_case, c): c for c in case_names}
    for fut in as_completed(futs):
        name, err = fut.result()
        done += 1
        if err:
            errors.append(f'{name}: {err}')
            print(f'  [{done}/{len(case_names)}] ERROR {name}: {err}')
        else:
            print(f'  [{done}/{len(case_names)}] OK {name}')

if errors:
    print(f'\nWARNING: {len(errors)} errors occurred.', file=sys.stderr)

print(f'\nEnsemble complete. Output: {ensemble_dir}')

# ── Evaluate hard labels against GT ──────────────────────────────────────────
print('\nEvaluating against GT...')
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

# Print per-class mean Dice
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
echo "=== All steps complete: $(date) ==="
echo "Softmax + hard labels: ${ENSEMBLE_DIR}"
echo "Summary:               ${SUMMARY_JSON}"
