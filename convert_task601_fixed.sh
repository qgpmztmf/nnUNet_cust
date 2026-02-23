#!/bin/bash
# Convert Task601 using the FIXED TotalSegmentator v1 dataset.
# Step 1: Build a merged complete dataset (symlinks for all cases, fixed ct.nii.gz for 76 bad cases).
# Step 2: Run the conversion script on the merged dataset.

set -e

ORIGINAL=/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset
FIXED_76=/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset_fixed
MERGED=/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset_complete_fixed
OUTPUT=/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_raw_data/Task601_TotalSegmentatorV1_fixed

echo "=== Step 1: Building merged fixed dataset at $MERGED ==="
python3 - << 'EOF'
import os, shutil
from pathlib import Path

original = Path(os.environ.get('ORIGINAL', '/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset'))
fixed_76 = Path(os.environ.get('FIXED_76', '/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset_fixed'))
merged   = Path(os.environ.get('MERGED',   '/home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset_complete_fixed'))

merged.mkdir(parents=True, exist_ok=True)

# Copy meta.csv
shutil.copy2(original / 'meta.csv', merged / 'meta.csv')

fixed_cases = {p.name for p in fixed_76.iterdir() if p.is_dir()}
all_cases   = [p for p in original.iterdir() if p.is_dir()]

for case_dir in sorted(all_cases):
    case = case_dir.name
    dst  = merged / case

    if case in fixed_cases:
        # Use fixed ct.nii.gz + original segmentations (symlink)
        dst.mkdir(exist_ok=True)
        ct_dst = dst / 'ct.nii.gz'
        if ct_dst.exists() or ct_dst.is_symlink():
            ct_dst.unlink()
        ct_dst.symlink_to(fixed_76 / case / 'ct.nii.gz')
        seg_dst = dst / 'segmentations'
        if seg_dst.exists() or seg_dst.is_symlink():
            if seg_dst.is_symlink():
                seg_dst.unlink()
        if not seg_dst.exists():
            seg_dst.symlink_to(case_dir / 'segmentations')
    else:
        # Symlink the whole case dir
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                dst.unlink()
        if not dst.exists():
            dst.symlink_to(case_dir)

print(f"Merged dataset ready: {len(all_cases)} cases ({len(fixed_cases)} with fixed CT)")
EOF

export ORIGINAL FIXED_76 MERGED

echo ""
echo "=== Step 2: Running nnUNet conversion ==="
uv run python /home/tianmid1/tianmid1/nnUNet_cust/convert_totalseg_v1_to_nnunet.py \
    --input_dir  "$MERGED" \
    --output_dir "$OUTPUT" \
    --target_spacing 3.0 3.0 3.0 \
    --num_cores 16
