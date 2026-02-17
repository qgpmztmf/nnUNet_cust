"""
Demo: Verify the grouped segmentation pipeline is feasible.

This script runs 4 tests on a SINGLE real CT case to prove the round-trip works:

  Test 1 - Config Integrity:
    Verify all 104 classes are covered exactly once across 4 groups.

  Test 2 - HU Clipping:
    Show that each group's HU window clips the CT to the expected range.

  Test 3 - Label Round-Trip (Perfect Predictions):
    Convert global labels -> 4 local label maps -> fuse back to global.
    The fused result must be IDENTICAL to the original.

  Test 4 - Softmax Fusion with Priority:
    Create synthetic softmax with deliberate overlaps.
    Verify higher-priority group wins at contested voxels.

Usage:
    python demo_grouped_pipeline.py \
        --input_dir  /path/to/TotalSegmentator_v1/Totalsegmentator_dataset \
        --split_file /path/to/split_group4.json \
        --windows_file /path/to/hu_windows.json \
        --case s0001
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Import from sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from convert_totalseg_v1_grouped import (
    CLASS_MAP_ALL, EXCLUDE_CASES, GROUP_NAMES, GROUP_PRIORITY,
    load_group_config, read_image, resample_image, merge_masks,
    validate_with_nibabel,
)
from fuse_group_predictions import GroupFuser

try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: SimpleITK required. Install with: pip install SimpleITK")
    sys.exit(1)


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_pass(msg: str):
    print(f"  [PASS] {msg}")


def print_fail(msg: str):
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Test 1: Config Integrity
# ---------------------------------------------------------------------------

def test_config_integrity(groups, windows, local_maps, global_maps):
    print_header("Test 1: Config Integrity")

    all_ok = True

    # Check all 104 classes are assigned to exactly one group
    all_global_ids = set()
    for gid, members in groups.items():
        group_ids = {gid for gid, _name in members}
        overlap = all_global_ids & group_ids
        if overlap:
            print_fail(f"Group {gid} overlaps with previous groups: {overlap}")
            all_ok = False
        all_global_ids |= group_ids

    expected = set(range(1, 105))
    missing = expected - all_global_ids
    extra = all_global_ids - expected
    if missing:
        print_fail(f"Missing global IDs: {sorted(missing)}")
        all_ok = False
    if extra:
        print_fail(f"Extra global IDs: {sorted(extra)}")
        all_ok = False

    if not missing and not extra and all_ok:
        print_pass(f"All 104 classes assigned to exactly one group")

    # Check each group has a window
    for gid in groups:
        if gid not in windows:
            print_fail(f"Group {gid} has no HU window")
            all_ok = False

    # Check local_id ranges
    for gid in groups:
        n_cls = len(groups[gid])
        local_ids = set(local_maps[gid].values())
        expected_local = set(range(1, n_cls + 1))
        if local_ids != expected_local:
            print_fail(f"Group {gid}: local IDs {local_ids} != expected {expected_local}")
            all_ok = False

    # Print group summary
    total = 0
    for gid in sorted(groups.keys()):
        n = len(groups[gid])
        total += n
        low, high = windows[gid]
        print(f"  Group {gid} ({GROUP_NAMES.get(gid, ''):12s}): "
              f"{n:2d} classes, HU [{low:.1f}, {high:.1f}], priority={GROUP_PRIORITY[gid]}")
    print(f"  Total: {total} classes")

    if all_ok:
        print_pass("Config integrity verified")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: HU Clipping
# ---------------------------------------------------------------------------

def test_hu_clipping(scan_array: np.ndarray, windows: Dict):
    print_header("Test 2: HU Clipping")

    print(f"  Original CT range: [{scan_array.min():.1f}, {scan_array.max():.1f}]")

    all_ok = True
    for gid in sorted(windows.keys()):
        low, high = windows[gid]
        clipped = np.clip(scan_array, low, high)
        actual_min = clipped.min()
        actual_max = clipped.max()

        # Verify clipping
        if actual_min < low - 1e-3 or actual_max > high + 1e-3:
            print_fail(f"Group {gid}: range [{actual_min:.1f}, {actual_max:.1f}] "
                       f"exceeds [{low:.1f}, {high:.1f}]")
            all_ok = False
        else:
            # Show what fraction of voxels are clipped
            n_below = np.sum(scan_array < low)
            n_above = np.sum(scan_array > high)
            n_total = scan_array.size
            pct_clipped = 100.0 * (n_below + n_above) / n_total
            print(f"  Group {gid} ({GROUP_NAMES.get(gid, ''):12s}): "
                  f"HU [{low:.1f}, {high:.1f}] -> "
                  f"clipped [{actual_min:.1f}, {actual_max:.1f}], "
                  f"{pct_clipped:.1f}% voxels clipped "
                  f"({100*n_below/n_total:.1f}% low, {100*n_above/n_total:.1f}% high)")

    if all_ok:
        print_pass("HU clipping verified for all groups")
    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Label Round-Trip (Perfect Predictions)
# ---------------------------------------------------------------------------

def test_label_roundtrip(mask_array: np.ndarray,
                         local_maps: Dict,
                         global_maps: Dict,
                         groups: Dict,
                         mapping_files_content: Dict):
    print_header("Test 3: Label Round-Trip (Perfect Predictions)")

    # Step 1: Global -> Local for each group
    local_labels = {}
    for gid, lmap in local_maps.items():
        local_mask = np.zeros_like(mask_array, dtype=np.uint8)
        for global_id, local_id in lmap.items():
            local_mask[mask_array == global_id] = local_id
        local_labels[gid] = local_mask

        n_fg = np.sum(local_mask > 0)
        n_classes = len(np.unique(local_mask)) - 1  # exclude background
        print(f"  Group {gid}: {n_classes} classes present, "
              f"{n_fg} foreground voxels ({100*n_fg/mask_array.size:.2f}%)")

    # Step 2: Fuse back using hard label fusion
    fuser = GroupFuser.__new__(GroupFuser)
    fuser.mappings = mapping_files_content
    fuser.n_global_classes = 104
    fuser.priorities = {gid: GROUP_PRIORITY[gid] for gid in groups}
    fuser.local_to_global = {}
    fuser.global_to_group = {}
    for gid in groups:
        fuser.local_to_global[gid] = {
            int(lid): info["global_id"]
            for lid, info in mapping_files_content[gid]["local_to_global"].items()
        }
        for lid, info in mapping_files_content[gid]["local_to_global"].items():
            fuser.global_to_group[info["global_id"]] = gid

    fused = fuser.fuse_argmax(local_labels)

    # Step 3: Compare
    match = np.array_equal(fused, mask_array)
    n_diff = np.sum(fused != mask_array)

    if match:
        print_pass(f"Round-trip EXACT MATCH: fused == original ({mask_array.size} voxels)")
    else:
        print_fail(f"Round-trip MISMATCH: {n_diff} voxels differ "
                   f"({100*n_diff/mask_array.size:.4f}%)")
        # Show which classes differ
        diff_mask = fused != mask_array
        orig_at_diff = np.unique(mask_array[diff_mask])
        fused_at_diff = np.unique(fused[diff_mask])
        print(f"    Original classes at diff: {orig_at_diff}")
        print(f"    Fused classes at diff:    {fused_at_diff}")

    return match


# ---------------------------------------------------------------------------
# Test 4: Softmax Fusion with Priority
# ---------------------------------------------------------------------------

def test_softmax_priority(groups, local_maps, mapping_files_content):
    print_header("Test 4: Softmax Fusion with Priority")

    # Create a small synthetic volume: 4 voxels, each "contested" by 2 groups
    D, H, W = 1, 1, 4

    # Build the fuser
    fuser = GroupFuser.__new__(GroupFuser)
    fuser.mappings = mapping_files_content
    fuser.n_global_classes = 104
    fuser.priorities = {gid: GROUP_PRIORITY[gid] for gid in groups}
    fuser.local_to_global = {}
    fuser.global_to_group = {}
    for gid in groups:
        fuser.local_to_global[gid] = {
            int(lid): info["global_id"]
            for lid, info in mapping_files_content[gid]["local_to_global"].items()
        }
        for lid, info in mapping_files_content[gid]["local_to_global"].items():
            fuser.global_to_group[info["global_id"]] = gid

    # Scenario: each voxel is contested between two groups with EQUAL confidence
    # Higher-priority group should win
    test_cases = [
        (1, 2, "Group1 vs Group2 -> Group2 wins"),
        (1, 3, "Group1 vs Group3 -> Group3 wins"),
        (2, 3, "Group2 vs Group3 -> Group3 wins"),
        (3, 4, "Group3 vs Group4 -> Group4 wins"),
    ]

    all_ok = True
    for voxel_idx, (g_low, g_high, desc) in enumerate(test_cases):
        # Build softmax for all groups: mostly background
        softmax_dict = {}
        for gid in groups:
            n_local = len(groups[gid])
            sm = np.zeros((n_local + 1, D, H, W), dtype=np.float32)
            sm[0] = 1.0  # all background by default
            softmax_dict[gid] = sm

        # At voxel_idx: both g_low and g_high predict their first class with prob 0.6
        # (background gets 0.4)
        for g in [g_low, g_high]:
            softmax_dict[g][0, 0, 0, voxel_idx] = 0.4  # background
            softmax_dict[g][1, 0, 0, voxel_idx] = 0.6  # first local class

        fused = fuser.fuse_softmax(softmax_dict)

        # The winning class should be from g_high (higher priority)
        expected_global_id = fuser.local_to_global[g_high][1]
        actual = fused[0, 0, voxel_idx]

        if actual == expected_global_id:
            winner_name = CLASS_MAP_ALL.get(expected_global_id, "?")
            print_pass(f"{desc}: voxel={actual} ({winner_name})")
        else:
            loser_id = fuser.local_to_global[g_low][1]
            print_fail(f"{desc}: expected {expected_global_id}, got {actual}")
            all_ok = False

    if all_ok:
        print_pass("Priority rule verified for all conflict scenarios")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(input_dir: str, split_file: str, windows_file: str, case: str):
    input_dir = Path(input_dir)

    # Load configs
    groups, windows, local_maps, global_maps = load_group_config(split_file, windows_file)

    # Build mapping content (simulates reading class_mapping.json files)
    mapping_files_content = {}
    for gid in groups:
        mapping_files_content[gid] = {
            "group_id": gid,
            "group_name": GROUP_NAMES.get(gid, ""),
            "task_id": 611 + gid - 1,
            "hu_window": {"low": windows[gid][0], "high": windows[gid][1]},
            "priority": GROUP_PRIORITY[gid],
            "num_classes": len(groups[gid]),
            "local_to_global": {
                str(local_id): {"global_id": global_id, "name": name}
                for local_id, (global_id, name) in enumerate(groups[gid], start=1)
            },
            "global_to_local": {
                str(global_id): local_id
                for global_id, local_id in local_maps[gid].items()
            },
        }

    # ---- Test 1: Config only (no data needed) ----
    t1 = test_config_integrity(groups, windows, local_maps, global_maps)

    # ---- Test 4: Synthetic softmax (no data needed) ----
    t4 = test_softmax_priority(groups, local_maps, mapping_files_content)

    # ---- Tests 2 & 3: Need real data ----
    patient_dir = input_dir / case
    if not patient_dir.exists():
        print(f"\n[SKIP] Tests 2 & 3: case directory {patient_dir} not found")
        print(f"       Available cases: {sorted([p.name for p in input_dir.iterdir() if p.is_dir()])[:5]}...")
        t2 = t3 = None
    else:
        ct_path = patient_dir / "ct.nii.gz"
        if not validate_with_nibabel(ct_path):
            print(f"\n[SKIP] Tests 2 & 3: {case} has CRC error")
            t2 = t3 = None
        else:
            print(f"\nLoading case {case} ...")
            scan = read_image(ct_path)
            scan = resample_image(scan, [3.0, 3.0, 3.0], default_value=-1024,
                                  interpolator=sitk.sitkLinear)
            scan_array = sitk.GetArrayFromImage(scan).astype(np.float32)

            mask = merge_masks(patient_dir / "segmentations", CLASS_MAP_ALL)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(scan)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mask = resampler.Execute(mask)
            mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)

            print(f"  CT shape: {scan_array.shape}, Label shape: {mask_array.shape}")
            print(f"  Unique labels: {sorted(np.unique(mask_array).tolist())}")

            t2 = test_hu_clipping(scan_array, windows)
            t3 = test_label_roundtrip(mask_array, local_maps, global_maps,
                                      groups, mapping_files_content)

    # ---- Summary ----
    print_header("Summary")
    results = [
        ("Test 1: Config Integrity", t1),
        ("Test 2: HU Clipping", t2),
        ("Test 3: Label Round-Trip", t3),
        ("Test 4: Softmax Priority", t4),
    ]
    all_pass = True
    for name, result in results:
        if result is None:
            print(f"  {name}: SKIPPED")
        elif result:
            print(f"  {name}: PASS")
        else:
            print(f"  {name}: FAIL")
            all_pass = False

    if all_pass:
        print(f"\n  All tests passed! The grouped pipeline is feasible.")
    else:
        print(f"\n  Some tests failed. Check output above.")

    return all_pass


def parse_args():
    p = argparse.ArgumentParser(description="Demo: verify grouped segmentation pipeline.")
    p.add_argument("--input_dir", required=True,
                   help="Path to Totalsegmentator_dataset/ (contains sXXXX/ subdirs).")
    p.add_argument("--split_file", required=True,
                   help="Path to split_group4.json.")
    p.add_argument("--windows_file", required=True,
                   help="Path to hu_windows.json.")
    p.add_argument("--case", default="s0001",
                   help="Case directory name to use for tests (default: s0001).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_demo(
        input_dir=args.input_dir,
        split_file=args.split_file,
        windows_file=args.windows_file,
        case=args.case,
    )
    sys.exit(0 if success else 1)
