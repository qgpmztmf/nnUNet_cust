#!/bin/bash
#SBATCH --job-name=nnunet_ens_eval_601_r1
#SBATCH --partition=batch
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z/ensemble_eval_test_601_round1_%j.out
#SBATCH --error=/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z/ensemble_eval_test_601_round1_%j.err

# Auto-generated test ensemble + evaluation script — Round 1
# Task:    601
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Plans:   nnUNetPlansv2.1
#
# Submit AFTER ALL 5 prediction folds have completed:
#   sbatch ensemble_eval_test_601_round1.slurm
#
# Produces:
#   test_predictions/ensemble/summary.json  → /scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z/test_summary.json

set -euo pipefail

REPO_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"
RUN_DIR="/scratch/elec/t41026-hintlab/tianmid1/nnUNet_cust/agent/runs/round_1_20260223T170124Z"

export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"

TASK="601"
ROUND_N="1"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
PLANS="nnUNetPlansv2.1"
TASK_NAME="Task${TASK}_TotalSegmentatorV1"

MODEL_DIR="${RESULTS_FOLDER}/nnUNet/${NETWORK}/${TASK_NAME}/${TRAINER}__${PLANS}"
FOLD_PRED_BASE="${MODEL_DIR}/test_predictions"
ENSEMBLE_DIR="${MODEL_DIR}/test_predictions/ensemble"
LABELS_TS="${DATA_BASE}/nnUNet_raw_data/${TASK_NAME}/labelsTs"
SUMMARY_JSON="${ENSEMBLE_DIR}/summary.json"
N_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${ENSEMBLE_DIR}"

cd "${REPO_DIR}"

echo "=== nnUNet Test Ensemble + Evaluation Round 1 ==="
echo "Trainer:     ${TRAINER}"
echo "Plans:       ${PLANS}"
echo "Fold dirs:   ${FOLD_PRED_BASE}/fold_0 ... fold_4"
echo "Output:      ${ENSEMBLE_DIR}"
echo "GT:          ${LABELS_TS}"
echo "Threads:     ${N_THREADS}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Started:     $(date)"
echo ""

export FOLD_PRED_BASE ENSEMBLE_DIR LABELS_TS SUMMARY_JSON RUN_DIR N_THREADS TASK ROUND_N

uv run python - << 'PYEOF'
import os, sys, json, shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

fold_pred_base = Path(os.environ['FOLD_PRED_BASE'])
ensemble_dir   = Path(os.environ['ENSEMBLE_DIR'])
gt_dir         = Path(os.environ['LABELS_TS'])
summary_json   = Path(os.environ['SUMMARY_JSON'])
run_dir        = Path(os.environ['RUN_DIR'])
n_threads      = int(os.environ['N_THREADS'])
task_id        = os.environ['TASK']
round_n        = os.environ['ROUND_N']

ensemble_dir.mkdir(parents=True, exist_ok=True)

# ── Collect case names from fold_0 ────────────────────────────────────────────
fold0_dir = fold_pred_base / 'fold_0'
case_npz  = sorted(fold0_dir.glob('*.npz'))
if not case_npz:
    print(f'ERROR: No .npz files in {fold0_dir}', file=sys.stderr)
    sys.exit(1)
case_names = [f.stem for f in case_npz]
n_folds    = 5
print(f'Found {len(case_names)} cases in fold_0.')

# ── Verify all fold directories present ──────────────────────────────────────
for f in range(n_folds):
    d = fold_pred_base / f'fold_{f}'
    if not d.exists():
        print(f'ERROR: Missing fold directory: {d}', file=sys.stderr)
        sys.exit(1)

# ── Per-case: average softmax, transpose (D,H,W)→(W,H,D), save outputs ───────
def process_case(case_name):
    arrays = []
    for f in range(n_folds):
        p = fold_pred_base / f'fold_{f}' / f'{case_name}.npz'
        if not p.exists():
            return case_name, f'MISSING fold_{f}'
        arrays.append(np.load(str(p))['softmax'].astype(np.float32))  # (C,D,H,W)

    avg  = np.mean(arrays, axis=0)                   # (C,D,H,W)
    hard = np.argmax(avg, axis=0).astype(np.uint8)   # (D,H,W)
    # NIfTI/nibabel expects (W,H,D) = (x,y,z) — nnUNet softmax is depth-first
    hard_nib = hard.transpose(2, 1, 0)               # (W,H,D)

    # Borrow affine + header from fold_0 hard label
    ref_path = fold0_dir / f'{case_name}.nii.gz'
    if ref_path.exists():
        ref = nib.load(str(ref_path))
        if hard_nib.shape != ref.shape:
            # 1-voxel rounding difference: fall back to fold_0 label
            shutil.copy2(str(ref_path), str(ensemble_dir / f'{case_name}.nii.gz'))
            return case_name, f'WARN: shape {hard_nib.shape} vs {ref.shape} — used fold_0 fallback'
        out_img = nib.Nifti1Image(hard_nib, ref.affine, ref.header)
    else:
        out_img = nib.Nifti1Image(hard_nib, np.eye(4))

    nib.save(out_img, str(ensemble_dir / f'{case_name}.nii.gz'))
    np.savez_compressed(str(ensemble_dir / f'{case_name}.npz'),
                        softmax=avg.astype(np.float16))
    return case_name, None

print(f'Averaging softmax across {n_folds} folds + saving ensemble ({n_threads} threads)...')
warnings, done = [], 0
with ThreadPoolExecutor(max_workers=n_threads) as ex:
    futs = {ex.submit(process_case, c): c for c in case_names}
    for fut in as_completed(futs):
        name, msg = fut.result()
        done += 1
        if msg:
            warnings.append(f'{name}: {msg}')
            print(f'  [{done}/{len(case_names)}] {msg}')
        else:
            print(f'  [{done}/{len(case_names)}] OK {name}')

if warnings:
    print(f'\n[WARN] {len(warnings)} cases had issues.')
print(f'\nEnsemble complete. Output: {ensemble_dir}')

# ── Evaluate against GT ───────────────────────────────────────────────────────
print('\nEvaluating against GT...')
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

# Auto-detect number of classes from softmax shape
sample_npz = ensemble_dir / f'{case_names[0]}.npz'
n_classes  = int(np.load(str(sample_npz))['softmax'].shape[0]) if sample_npz.exists() else 105
labels     = list(range(n_classes))

print(f'Evaluating {len(pairs)} cases, {n_classes} classes ({n_threads} threads)...')
aggregate_scores(
    pairs,
    labels=labels,
    json_output_file=str(summary_json),
    json_name=f'Task{task_id}_test_round{round_n}',
    json_description=f'5-fold softmax ensemble test evaluation — Round {round_n}',
    json_author='auto',
    json_task=f'Task{task_id}_test',
    num_threads=n_threads,
)
print(f'Done. Summary: {summary_json}')

# ── Print per-class Dice ──────────────────────────────────────────────────────
with open(summary_json) as fh:
    d = json.load(fh)
mean_vals = d['results']['mean']
print(f"\n{'Class':>5}  {'Mean Dice':>9}")
print('-' * 20)
for i in range(1, n_classes):
    dice = mean_vals.get(str(i), {}).get('Dice', float('nan'))
    print(f'{i:>5}  {dice:>9.4f}')
overall = mean_vals.get('mean', {}).get('Dice', float('nan'))
print('-' * 20)
print(f"{'Mean':>5}  {overall:>9.4f}")

# ── Copy summary to run_dir ───────────────────────────────────────────────────
dest = run_dir / 'test_summary.json'
shutil.copy2(str(summary_json), str(dest))
print(f'\nCopied → {dest}')
PYEOF

echo ""
echo "=== Ensemble+eval round 1 complete: $(date) ==="
echo "Hard labels: ${ENSEMBLE_DIR}/*.nii.gz"
echo "Summary:     ${SUMMARY_JSON}"
echo "Copied to:   ${RUN_DIR}/test_summary.json"
