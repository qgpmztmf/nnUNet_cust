"""Compute Normalized Surface Dice (NSD) for nnUNet validation predictions.

Usage:
    python scripts/compute_nsd.py \
        --pred_dir /path/to/validation_raw \
        --gt_dir /path/to/gt_segmentations \
        --dataset_json /path/to/dataset.json \
        --threshold 1.0 \
        --num_workers 8
"""

import argparse
import json
import math
import sys
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import __surface_distances


def normalized_surface_dice(a, b, threshold, spacing, connectivity=1):
    """Compute NSD between two binary masks."""
    if not np.any(a) and not np.any(b):
        return float("nan")
    if not np.any(a) or not np.any(b):
        return 0.0

    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)
    return dc


def evaluate_case(args):
    """Evaluate NSD for a single case across all labels."""
    pred_path, gt_path, label_ids, threshold = args
    case_name = Path(pred_path).name

    pred_sitk = sitk.ReadImage(str(pred_path))
    gt_sitk = sitk.ReadImage(str(gt_path))

    pred = sitk.GetArrayFromImage(pred_sitk)
    gt = sitk.GetArrayFromImage(gt_sitk)
    spacing = np.array(pred_sitk.GetSpacing())[::-1]  # sitk is xyz, numpy is zyx

    results = {}
    for label_id in label_ids:
        pred_bin = (pred == label_id)
        gt_bin = (gt == label_id)

        if not np.any(pred_bin) and not np.any(gt_bin):
            results[label_id] = float("nan")
        else:
            results[label_id] = normalized_surface_dice(
                pred_bin, gt_bin, threshold, tuple(spacing)
            )

    print(f"  {case_name} done", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Surface distance threshold in mm (default: 1.0)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <pred_dir>/results_nsd.json)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    with open(args.dataset_json) as f:
        dataset = json.load(f)

    label_ids = [int(k) for k in dataset["labels"].keys() if k != "0"]

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Found {len(pred_files)} prediction files")
    print(f"Evaluating {len(label_ids)} classes with threshold={args.threshold}mm")

    task_args = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  WARNING: GT not found for {pred_path.name}, skipping")
            continue
        task_args.append((str(pred_path), str(gt_path), label_ids, args.threshold))

    # Run evaluation
    print(f"Running with {args.num_workers} workers...")
    with Pool(args.num_workers) as pool:
        all_results = pool.map(evaluate_case, task_args)

    # Aggregate: mean NSD per class (ignoring NaN)
    per_class = {lid: [] for lid in label_ids}
    for case_result in all_results:
        for lid in label_ids:
            val = case_result[lid]
            if not (isinstance(val, float) and math.isnan(val)):
                per_class[lid].append(val)

    mean_nsd = {}
    for lid in label_ids:
        vals = per_class[lid]
        if vals:
            mean_nsd[lid] = round(np.mean(vals), 3)

    # Map to reference names
    name_map = {
        "spleen": "organ_spleen", "kidney_right": "organ_right_kidney",
        "kidney_left": "organ_left_kidney", "gallbladder": "organ_gallbladder",
        "liver": "organ_liver", "stomach": "organ_stomach", "aorta": "organ_aorta",
        "inferior_vena_cava": "organ_inferior_vena_cava",
        "portal_vein_and_splenic_vein": "organ_portal_vein_and_splenic_vein",
        "pancreas": "organ_pancreas", "adrenal_gland_right": "organ_right_adrenal_gland",
        "adrenal_gland_left": "organ_left_adrenal_gland",
        "lung_upper_lobe_left": "lung_left_upper_lobe",
        "lung_lower_lobe_left": "lung_left_lower_lobe",
        "lung_upper_lobe_right": "lung_right_upper_lobe",
        "lung_middle_lobe_right": "lung_right_middle_lobe",
        "lung_lower_lobe_right": "lung_right_lower_lobe",
        "esophagus": "heart_esophagus", "trachea": "heart_trachea",
        "heart_myocardium": "dheart_myocardium",
        "heart_atrium_left": "dheart_left_atrium",
        "heart_ventricle_left": "dheart_left_ventricle",
        "heart_atrium_right": "dheart_right_atrium",
        "heart_ventricle_right": "dheart_right_ventricle",
        "pulmonary_artery": "dheart_pulmunary_artery",
    }

    labels = dataset["labels"]
    result = OrderedDict()
    for idx_str, ds_name in sorted(labels.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
        if not idx_str.isdigit() or idx_str == "0":
            continue
        lid = int(idx_str)
        if lid in mean_nsd:
            ref_name = name_map.get(ds_name, ds_name)
            result[ref_name] = mean_nsd[lid]

    output_path = args.output or str(pred_dir / "results_nsd.json")
    with open(output_path, "w") as f:
        json.dump({"normalized_surface_distance": result}, f, indent=4)

    print(f"\nResults saved to: {output_path}")
    print(f"Classes with NSD scores: {len(result)}")


if __name__ == "__main__":
    main()
