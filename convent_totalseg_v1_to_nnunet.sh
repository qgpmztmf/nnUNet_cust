uv run python convert_totalseg_v1_to_nnunet.py \
        --input_dir  /m/triton/work/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset \
        --output_dir /m/triton/work/tianmid1/data/nnUNet_raw/Dataset601_TotalSegmentatorV1 \
        --target_spacing 3.0 3.0 3.0 \
        --num_cores 8