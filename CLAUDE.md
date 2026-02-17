# Project: nnUNet Custom Pipeline for TotalSegmentator

## Project Overview
End-to-end nnUNet v1 pipeline for TotalSegmentator medical image segmentation on the Triton HPC cluster.

## Key Paths
- **Repo:** `/scratch/work/tianmid1/nnUNet_cust` (also mirrored at `/m/triton/work/tianmid1/nnUNet_cust`)
- **Data base:** `/m/triton/scratch/elec/t41026-hintlab/tianmid1/data`
  - Raw data: `.../data/nnUNet_raw_data/`
  - Preprocessed: `.../data/nnUNet_preprocessed/`
  - Results: `.../data/nnUNet_results/`
- **Scripts:** `scripts/` — numbered in pipeline order (00_setup → 01_preprocess → 02_train → 03_validate → 04_fuse)
- **Custom trainers:** `nnunet/training/network_training/`
- **Venv:** `.venv/` inside repo root — always run via `uv run` from `${REPO_DIR}`

## Active Tasks

| Task | Name | Classes | Description |
|------|------|---------|-------------|
| Task601 | TotalSegmentatorV1 | 104 | Full TotalSeg v1, standard trainer |
| Task602 | TotalSegMini | small | Mini subset for debugging |
| Task611 | TotalSegGroup1 | 13 | Lungs, GI tract (stomach, bowel, colon, esophagus, trachea, face) |
| Task612 | TotalSegGroup2 | 25 | Solid organs, vessels, muscles (liver, spleen, kidneys, pancreas, bladder, etc.) |
| Task613 | TotalSegGroup3 | 10 | Cardiovascular (aorta, heart chambers, pulmonary artery, iliac arteries) |
| Task614 | TotalSegGroup4 | 60 | Bones (vertebrae C1-L5, all ribs, humerus, scapula, femur, hip, sacrum) |

Tasks 611-614 use trainer: `nnUNetTrainerV2_fast`, plans: `nnUNetPlansv2.1`

## SLURM / HPC Conventions

**Partitions and GPU types on Triton:**
| Partition | GPU flag | Notes |
|-----------|----------|-------|
| `gpu-b300-288g-ellis` | `--gres=gpu:b300:1` | nodelist: gpu64 — primary for Task601 |
| `gpu-h200-141g-ellis` | `--gres=gpu:h200:1` | primary for Task611-614 (grouped) |
| `gpu-v100-16g` | `--gres=gpu:v100:1` | |
| `gpu-v100-32g` | `--gres=gpu:v100:1` | |
| `gpu-a100-80g` | `--gres=gpu:a100:1` | |
| `gpu-h100-80g` | `--gres=gpu:h100:1` | |

**Log pattern:** `--output=logs/<name>_fold%a_%j.out` (`%j` = job ID, `%a` = array index, `%x` = job name)

**Submitting grouped tasks (611-614):**
```bash
# Single task, all folds
sbatch --array=0-4 --export=ALL,TASK=613 scripts/02_train_grouped.slurm

# All 4 tasks × 5 folds
for T in 611 612 613 614; do
    sbatch --array=0-4 --export=ALL,TASK=$T scripts/02_train_grouped.slurm
done
```

**Always verify before submitting:**
1. `bash -n script.sh` — check shell syntax
2. GPU type matches partition
3. `logs/` directory exists
4. Validate with `sbatch --test-only` before real submission

## nnUNet Conventions
- **Version:** nnUNet v1 — use `nnUNet_train`, `nnUNet_predict`, `nnUNet_evaluate_folder`
- **Trainer invocation:** `uv run nnUNet_train 3d_fullres <TrainerClass> <TaskID> <fold> -p <PlansID>`
- **Validation flag:** `--validation_only` (NOT `--val` — does not exist)
- **Continue training:** `-c` flag
- **dataset.json format (v1):**
  - `"labels"` maps **integer strings** to class names: `{"0": "background", "1": "liver", ...}`
  - Labels must be sorted by integer value
  - File paths in `training`/`test` arrays do NOT include the `.nii.gz` extension
  - `"file_ending"` field does NOT exist in v1 format

## Python / Library Conventions
- **Always use `nibabel`** for NIfTI I/O — NEVER SimpleITK
  - SimpleITK fails on many CT scans due to strict orthonormality tolerance checks
  - nibabel is tolerant and works reliably with all TotalSegmentator data
- **dtype:** use `float64` for HU statistic accumulation to avoid overflow
- **Parallelism:** use `multiprocessing` for large dataset processing (1200+ scans)
- **Entry point:** all scripts run via `uv run` from repo root

## Custom Trainers
- `nnUNetTrainerV2` — standard trainer (Task601)
- `nnUNetTrainerV2_fast` — fast trainer: `batch_size=32`, 16 iters/epoch, mixed precision, cosine LR (Tasks 611-614)
- Custom trainers live in `nnunet/training/network_training/`

## Environment Variables (set in SLURM scripts)
```bash
export nnUNet_raw_data_base="${DATA_BASE}"
export nnUNet_preprocessed="${DATA_BASE}/nnUNet_preprocessed"
export RESULTS_FOLDER="${DATA_BASE}/nnUNet_results"
export nnUNet_n_proc_DA=16
```
