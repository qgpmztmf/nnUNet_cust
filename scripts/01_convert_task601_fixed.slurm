#!/bin/bash
#SBATCH --job-name=convert_601_fixed
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=logs/convert_601_fixed_%j.out
#SBATCH --error=logs/convert_601_fixed_%j.err

# Convert Task601 using the FIXED TotalSegmentator v1 dataset.
# Step 1: Build a merged complete dataset (symlinks for all 1204 cases,
#         fixed ct.nii.gz substituted for the 76 bad cases).
# Step 2: Run conversion on the merged dataset.
# Usage: sbatch scripts/01_convert_task601_fixed.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data"

ORIGINAL="${DATA_BASE}/TotalSegmentator_v1/Totalsegmentator_dataset"
FIXED_76="${DATA_BASE}/TotalSegmentator_v1/Totalsegmentator_dataset_fixed"
MERGED="${DATA_BASE}/TotalSegmentator_v1/Totalsegmentator_dataset_complete_fixed"
OUTPUT="${DATA_BASE}/nnUNet_raw_data/Task601_TotalSegmentatorV1_fixed"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"

echo "=== Convert Task601 TotalSegmentatorV1 (FIXED) ==="
echo "Original: ${ORIGINAL}"
echo "Fixed 76: ${FIXED_76}"
echo "Merged:   ${MERGED}"
echo "Output:   ${OUTPUT}"
echo "Cores:    ${SLURM_CPUS_PER_TASK}"
echo "Node:     ${SLURM_NODELIST}"
echo "Started:  $(date)"
echo ""

# Step 1: Build merged dataset
echo "--- Step 1: Building merged fixed dataset ---"
python3 - << EOF
import os, shutil
from pathlib import Path

original = Path("${ORIGINAL}")
fixed_76 = Path("${FIXED_76}")
merged   = Path("${MERGED}")

merged.mkdir(parents=True, exist_ok=True)

shutil.copy2(original / "meta.csv", merged / "meta.csv")

fixed_cases = {p.name for p in fixed_76.iterdir() if p.is_dir()}
all_cases   = [p for p in original.iterdir() if p.is_dir()]

for case_dir in sorted(all_cases):
    case = case_dir.name
    dst  = merged / case

    if case in fixed_cases:
        dst.mkdir(exist_ok=True)
        ct_dst = dst / "ct.nii.gz"
        if ct_dst.exists() or ct_dst.is_symlink():
            ct_dst.unlink()
        ct_dst.symlink_to(fixed_76 / case / "ct.nii.gz")
        seg_dst = dst / "segmentations"
        if seg_dst.is_symlink():
            seg_dst.unlink()
        if not seg_dst.exists():
            seg_dst.symlink_to(case_dir / "segmentations")
    else:
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(case_dir)

print(f"Merged dataset ready: {len(all_cases)} cases ({len(fixed_cases)} with fixed CT)")
EOF

echo ""
echo "--- Step 2: Running nnUNet conversion ---"
uv run python convert_totalseg_v1_to_nnunet.py \
    --input_dir  "${MERGED}" \
    --output_dir "${OUTPUT}" \
    --target_spacing 3.0 3.0 3.0 \
    --num_cores "${SLURM_CPUS_PER_TASK}"

echo ""
echo "Conversion complete: $(date)"
