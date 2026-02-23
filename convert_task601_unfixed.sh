#!/bin/bash
# Convert Task601 using the ORIGINAL (unfixed) TotalSegmentator v1 dataset.
# Corrupted out-of-FOV voxels are still present in 76 cases.

uv run python /home/tianmid1/tianmid1/nnUNet_cust/convert_totalseg_v1_to_nnunet.py \
    --input_dir  /home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset \
    --output_dir /home/tianmid1/tianmid1/t41026-hintlab/tianmid1/data/nnUNet_raw_data/Task601_TotalSegmentatorV1 \
    --target_spacing 3.0 3.0 3.0 \
    --num_cores 16
