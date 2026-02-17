"""
Convert TotalSegmentator v1 dataset into 4 separate nnUNet v1 tasks, one per organ group.

Each task receives:
  - A single-channel CT image clipped to the group's HU window (nnUNet then applies z-score)
  - A label mask with ONLY the group's classes, remapped to local IDs 1..K

Group assignments and HU windows are read from external JSON config files.

The script reads each CT + mask ONCE and writes all 4 group outputs in a single pass
for efficiency.

Corrupted case s0864 is automatically excluded (CRC error in ct.nii.gz).

Usage:
    python convert_totalseg_v1_grouped.py \
        --input_dir   /path/to/TotalSegmentator_v1/Totalsegmentator_dataset \
        --output_base /path/to/nnUNet_raw_data \
        --split_file  /path/to/split_group4.json \
        --windows_file /path/to/hu_windows.json \
        --task_base_id 611 \
        --target_spacing 3.0 3.0 3.0 \
        --num_cores 8

This produces:
    Task611_TotalSegGroup1/  (lung/GI/hollow, 12 classes)
    Task612_TotalSegGroup2/  (soft tissue/organs, 24 classes)
    Task613_TotalSegGroup3/  (vascular/cardiac, 9 classes)
    Task614_TotalSegGroup4/  (bone/skeletal, 59 classes)
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDE_CASES = {"0864"}

GROUP_NAMES = {
    1: "LungGI",
    2: "SoftTissue",
    3: "Vascular",
    4: "Bone",
}

# Priority for fusion (higher = wins conflicts). Matches group number.
GROUP_PRIORITY = {1: 1, 2: 2, 3: 3, 4: 4}


# ---------------------------------------------------------------------------
# CLASS_MAP_ALL: global_id -> structure_name  (104 classes)
# ---------------------------------------------------------------------------

CLASS_MAP_ALL: Dict[int, str] = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "aorta", 8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein", 10: "pancreas",
    11: "adrenal_gland_right", 12: "adrenal_gland_left",
    13: "lung_upper_lobe_left", 14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right", 16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5", 19: "vertebrae_L4", 20: "vertebrae_L3",
    21: "vertebrae_L2", 22: "vertebrae_L1",
    23: "vertebrae_T12", 24: "vertebrae_T11", 25: "vertebrae_T10",
    26: "vertebrae_T9", 27: "vertebrae_T8", 28: "vertebrae_T7",
    29: "vertebrae_T6", 30: "vertebrae_T5", 31: "vertebrae_T4",
    32: "vertebrae_T3", 33: "vertebrae_T2", 34: "vertebrae_T1",
    35: "vertebrae_C7", 36: "vertebrae_C6", 37: "vertebrae_C5",
    38: "vertebrae_C4", 39: "vertebrae_C3", 40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus", 43: "trachea",
    44: "heart_myocardium", 45: "heart_atrium_left",
    46: "heart_ventricle_left", 47: "heart_atrium_right",
    48: "heart_ventricle_right", 49: "pulmonary_artery", 50: "brain",
    51: "iliac_artery_left", 52: "iliac_artery_right",
    53: "iliac_vena_left", 54: "iliac_vena_right",
    55: "small_bowel", 56: "duodenum", 57: "colon",
    58: "rib_left_1", 59: "rib_left_2", 60: "rib_left_3",
    61: "rib_left_4", 62: "rib_left_5", 63: "rib_left_6",
    64: "rib_left_7", 65: "rib_left_8", 66: "rib_left_9",
    67: "rib_left_10", 68: "rib_left_11", 69: "rib_left_12",
    70: "rib_right_1", 71: "rib_right_2", 72: "rib_right_3",
    73: "rib_right_4", 74: "rib_right_5", 75: "rib_right_6",
    76: "rib_right_7", 77: "rib_right_8", 78: "rib_right_9",
    79: "rib_right_10", 80: "rib_right_11", 81: "rib_right_12",
    82: "humerus_left", 83: "humerus_right",
    84: "scapula_left", 85: "scapula_right",
    86: "clavicula_left", 87: "clavicula_right",
    88: "femur_left", 89: "femur_right",
    90: "hip_left", 91: "hip_right", 92: "sacrum", 93: "face",
    94: "gluteus_maximus_left", 95: "gluteus_maximus_right",
    96: "gluteus_medius_left", 97: "gluteus_medius_right",
    98: "gluteus_minimus_left", 99: "gluteus_minimus_right",
    100: "autochthon_left", 101: "autochthon_right",
    102: "iliopsoas_left", 103: "iliopsoas_right",
    104: "urinary_bladder",
}


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_group_config(split_file: str, windows_file: str):
    """Load group assignments and HU windows from JSON files.

    Returns:
        groups:  dict  {group_id: [(global_id, name), ...]}  sorted by global_id
        windows: dict  {group_id: (low, high)}
        local_maps: dict  {group_id: {global_id: local_id}}
        global_maps: dict {group_id: {local_id: global_id}}
    """
    with open(split_file) as f:
        name_to_group = json.load(f)  # {"spleen": 2, ...}

    with open(windows_file) as f:
        win_raw = json.load(f)["HU_windows"]  # {"1": {"low": ..., "high": ...}, ...}

    # Build name -> global_id reverse lookup
    name_to_global = {v: k for k, v in CLASS_MAP_ALL.items()}

    # Build groups: {group_id: [(global_id, name), ...]}
    groups = {}
    for name, gid in name_to_group.items():
        gid = int(gid)
        if name not in name_to_global:
            print(f"[WARN] Unknown class '{name}' in split file, skipping", file=sys.stderr)
            continue
        groups.setdefault(gid, []).append((name_to_global[name], name))

    # Sort each group by global_id for deterministic local_id assignment
    for gid in groups:
        groups[gid].sort(key=lambda x: x[0])

    # Build local <-> global mappings
    # local_id starts at 1 (0 = background)
    local_maps = {}   # {group_id: {global_id: local_id}}
    global_maps = {}  # {group_id: {local_id: global_id}}
    for gid, members in groups.items():
        lm = {}
        gm = {}
        for local_id, (global_id, _name) in enumerate(members, start=1):
            lm[global_id] = local_id
            gm[local_id] = global_id
        local_maps[gid] = lm
        global_maps[gid] = gm

    # Parse HU windows
    windows = {}
    for gid_str, wdict in win_raw.items():
        windows[int(gid_str)] = (float(wdict["low"]), float(wdict["high"]))

    return groups, windows, local_maps, global_maps


# ---------------------------------------------------------------------------
# Robust NIfTI reader (same as original convert script)
# ---------------------------------------------------------------------------

def _nib_to_sitk(path: Path, pixel_type=None) -> sitk.Image:
    nib_img = nib.load(str(path))
    data = np.asarray(nib_img.dataobj)
    affine = nib_img.affine.astype(np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction_ras = affine[:3, :3] / spacing
    U, _, Vt = np.linalg.svd(direction_ras)
    direction_ras = U @ Vt
    flip = np.diag([-1.0, -1.0, 1.0])
    direction_lps = flip @ direction_ras
    origin_lps = flip @ affine[:3, 3]
    sitk_img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    sitk_img.SetSpacing(spacing.tolist())
    sitk_img.SetDirection(direction_lps.flatten().tolist())
    sitk_img.SetOrigin(origin_lps.tolist())
    if pixel_type is not None:
        sitk_img = sitk.Cast(sitk_img, pixel_type)
    return sitk_img


def read_image(path: Path, pixel_type=None) -> sitk.Image:
    try:
        if pixel_type is not None:
            return sitk.ReadImage(str(path), pixel_type)
        return sitk.ReadImage(str(path))
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            raise
        return _nib_to_sitk(path, pixel_type)


def validate_with_nibabel(path: Path) -> bool:
    try:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)
        _ = data.flat[0]
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_image(image, new_spacing, default_value, interpolator):
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [int(round(sz * sp / nsp))
                for sz, sp, nsp in zip(size, spacing, new_spacing)]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(),
                         default_value, image.GetPixelID())


def merge_masks(segmentations_dir: Path, class_map: Dict[int, str]) -> sitk.Image:
    combined: Optional[sitk.Image] = None
    for label_value, label_name in class_map.items():
        mask_path = segmentations_dir / f"{label_name}.nii.gz"
        if not mask_path.exists():
            continue
        mask = read_image(mask_path, sitk.sitkUInt8)
        mask = sitk.Cast(mask, sitk.sitkUInt8) * label_value
        if combined is None:
            combined = mask
            continue
        try:
            combined = sitk.Maximum(combined, mask)
        except RuntimeError:
            mask.CopyInformation(combined)
            combined = sitk.Maximum(combined, mask)
    if combined is None:
        raise RuntimeError(f"No masks found in {segmentations_dir}")
    return combined


def load_splits(meta_csv: Path) -> Tuple[set, set, set]:
    train_ids, val_ids, test_ids = set(), set(), set()
    with open(meta_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            case_id = row["image_id"].strip()[-4:]
            split = row["split"].strip()
            if split == "train":
                train_ids.add(case_id)
            elif split == "val":
                val_ids.add(case_id)
            elif split == "test":
                test_ids.add(case_id)
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Per-patient processing: produces outputs for ALL groups in one pass
# ---------------------------------------------------------------------------

def process_patient(patient_dir: Path,
                    output_dirs: Dict[int, Path],
                    target_spacing: List[float],
                    windows: Dict[int, Tuple[float, float]],
                    local_maps: Dict[int, Dict[int, int]],
                    test_ids: set) -> None:

    case_id = patient_dir.name[-4:]

    if case_id in EXCLUDE_CASES:
        return

    train_or_test = "Ts" if case_id in test_ids else "Tr"

    # Check if ALL outputs already exist (resume support)
    all_exist = True
    for gid, odir in output_dirs.items():
        scan_out = odir / f"images{train_or_test}" / f"TotalSeg_{case_id}_0000.nii.gz"
        mask_out = odir / f"labels{train_or_test}" / f"TotalSeg_{case_id}.nii.gz"
        if not scan_out.exists() or not mask_out.exists():
            all_exist = False
            break
    if all_exist:
        return

    # Validate with nibabel first
    ct_path = patient_dir / "ct.nii.gz"
    if not validate_with_nibabel(ct_path):
        print(f"\n[SKIP] {patient_dir.name}: nibabel CRC/read error", file=sys.stderr)
        return

    # ---- Read and resample CT ONCE ----
    scan = read_image(ct_path)
    scan = resample_image(scan, target_spacing, default_value=-1024,
                          interpolator=sitk.sitkLinear)
    scan_array = sitk.GetArrayFromImage(scan).astype(np.float32)

    # ---- Read and merge full mask ONCE ----
    mask = merge_masks(patient_dir / "segmentations", CLASS_MAP_ALL)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    mask = resampler.Execute(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask_array = sitk.GetArrayFromImage(mask)

    # ---- Write outputs for each group ----
    for gid, odir in output_dirs.items():
        scan_out = odir / f"images{train_or_test}" / f"TotalSeg_{case_id}_0000.nii.gz"
        mask_out = odir / f"labels{train_or_test}" / f"TotalSeg_{case_id}.nii.gz"

        if scan_out.exists() and mask_out.exists():
            continue

        # Clip CT to group's HU window
        low, high = windows[gid]
        clipped = np.clip(scan_array, low, high)
        clipped_img = sitk.GetImageFromArray(clipped)
        clipped_img.CopyInformation(scan)

        # Remap labels: global_id -> local_id, non-group classes -> 0
        lmap = local_maps[gid]
        local_mask = np.zeros_like(mask_array, dtype=np.uint8)
        for global_id, local_id in lmap.items():
            local_mask[mask_array == global_id] = local_id

        local_mask_img = sitk.GetImageFromArray(local_mask)
        local_mask_img.CopyInformation(scan)

        sitk.WriteImage(clipped_img, str(scan_out), useCompression=True)
        sitk.WriteImage(local_mask_img, str(mask_out), useCompression=True)


# Worker pool
_worker_args = {}

def _init_worker(output_dirs, target_spacing, windows, local_maps, test_ids):
    _worker_args["output_dirs"] = output_dirs
    _worker_args["target_spacing"] = target_spacing
    _worker_args["windows"] = windows
    _worker_args["local_maps"] = local_maps
    _worker_args["test_ids"] = test_ids

def _process_patient_worker(patient_dir: Path) -> None:
    try:
        process_patient(patient_dir, **_worker_args)
    except Exception as exc:
        print(f"\n[ERROR] {patient_dir.name}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_grouped_datasets(input_dir: str,
                            output_base: str,
                            split_file: str,
                            windows_file: str,
                            task_base_id: int = 611,
                            target_spacing: List[float] = None,
                            num_cores: int = -1) -> None:

    if target_spacing is None:
        target_spacing = [3.0, 3.0, 3.0]

    input_dir = Path(input_dir)
    output_base = Path(output_base)

    if num_cores == -1:
        num_cores = os.cpu_count() or 1

    # Load configs
    groups, windows, local_maps, global_maps = load_group_config(split_file, windows_file)

    print("=" * 70)
    print("Grouped TotalSegmentator Conversion")
    print("=" * 70)
    for gid in sorted(groups.keys()):
        low, high = windows[gid]
        n_cls = len(groups[gid])
        task_id = task_base_id + gid - 1
        print(f"  Group {gid} ({GROUP_NAMES.get(gid, '?'):12s}): "
              f"Task{task_id}, {n_cls:2d} classes, HU [{low:.1f}, {high:.1f}]")
    print()

    # Load split information
    meta_csv = input_dir / "meta.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv not found in {input_dir}")
    tr_ids, val_ids, test_ids = load_splits(meta_csv)

    # Enumerate patients (exclude known bad cases)
    patients = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name[-4:] not in EXCLUDE_CASES
    ])
    if not patients:
        raise RuntimeError(f"No case directories found in {input_dir}")

    n_tr = sum(1 for p in patients if p.name[-4:] not in test_ids)
    n_ts = len(patients) - n_tr
    print(f"Found {len(patients)} cases (excluding {len(EXCLUDE_CASES)} corrupted): "
          f"{n_tr} train+val, {n_ts} test")

    # Create output directories and write dataset.json + class_mapping.json
    output_dirs = {}
    for gid in sorted(groups.keys()):
        task_id = task_base_id + gid - 1
        task_name = f"Task{task_id}_TotalSegGroup{gid}"
        odir = output_base / task_name
        odir.mkdir(parents=True, exist_ok=True)
        for subdir in ("imagesTr", "imagesTs", "labelsTr", "labelsTs"):
            (odir / subdir).mkdir(parents=True, exist_ok=True)
        output_dirs[gid] = odir

        # Build label dict: {"0": "background", "1": "spleen", ...}  (nnUNet v1 format)
        labels_dict = {"0": "background"}
        for local_id, (global_id, name) in enumerate(groups[gid], start=1):
            labels_dict[str(local_id)] = name

        # nnUNet v1 dataset.json: modality "CT" -> nnUNet applies percentile clip + z-score
        dataset_json = {
            "name": task_name,
            "description": f"TotalSegmentator group {gid} ({GROUP_NAMES.get(gid, '')}) "
                           f"- HU clipped to [{windows[gid][0]:.1f}, {windows[gid][1]:.1f}]",
            "tensorImageSize": "4D",
            "reference": "",
            "licence": "",
            "release": "1.0",
            "modality": {"0": "CT"},
            "labels": labels_dict,
            "numTraining": n_tr,
            "numTest": n_ts,
            "file_ending": ".nii.gz",
            "training": [
                {
                    "image": f"./imagesTr/TotalSeg_{p.name[-4:]}.nii.gz",
                    "label": f"./labelsTr/TotalSeg_{p.name[-4:]}.nii.gz",
                }
                for p in patients if p.name[-4:] not in test_ids
            ],
            "test": [
                f"./imagesTs/TotalSeg_{p.name[-4:]}.nii.gz"
                for p in patients if p.name[-4:] in test_ids
            ],
        }
        with open(odir / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=4)

        # Save class mapping for use by fusion script
        # {local_id: {global_id, name}, ...}  + reverse mapping
        mapping = {
            "group_id": gid,
            "group_name": GROUP_NAMES.get(gid, ""),
            "task_id": task_id,
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
        with open(odir / "class_mapping.json", "w") as f:
            json.dump(mapping, f, indent=4)

        print(f"  Wrote {task_name}/dataset.json + class_mapping.json")

    print()

    # Process all cases
    if num_cores > 1:
        print(f"Processing with {num_cores} parallel workers ...")
        with multiprocessing.Pool(
            processes=num_cores,
            initializer=_init_worker,
            initargs=(output_dirs, target_spacing, windows, local_maps, test_ids),
        ) as pool:
            list(tqdm(
                pool.imap_unordered(_process_patient_worker, patients),
                total=len(patients),
                desc="Converting",
            ))
    else:
        print("Processing in single-process mode ...")
        for patient in tqdm(patients, desc="Converting"):
            process_patient(patient, output_dirs, target_spacing,
                            windows, local_maps, test_ids)

    print(f"\nDone. {len(output_dirs)} datasets written under: {output_base}")
    print("\nNext steps:")
    for gid in sorted(groups.keys()):
        tid = task_base_id + gid - 1
        print(f"  nnUNet_plan_and_preprocess -t {tid}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert TotalSegmentator v1 into grouped nnUNet v1 tasks."
    )
    p.add_argument("--input_dir", required=True,
                   help="Path to Totalsegmentator_dataset/ (contains sXXXX/ and meta.csv).")
    p.add_argument("--output_base", required=True,
                   help="Base directory for nnUNet raw data (e.g. .../nnUNet_raw_data).")
    p.add_argument("--split_file", required=True,
                   help="Path to split_group4.json (class -> group mapping).")
    p.add_argument("--windows_file", required=True,
                   help="Path to hu_windows.json (group -> HU window).")
    p.add_argument("--task_base_id", type=int, default=611,
                   help="Starting task ID. Groups 1-4 become task_base_id .. task_base_id+3. Default: 611.")
    p.add_argument("--target_spacing", nargs=3, type=float, default=[3.0, 3.0, 3.0],
                   metavar=("X", "Y", "Z"), help="Voxel spacing (mm). Default: 3 3 3.")
    p.add_argument("--num_cores", type=int, default=-1,
                   help="Worker processes. -1 = all CPUs.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_grouped_datasets(
        input_dir=args.input_dir,
        output_base=args.output_base,
        split_file=args.split_file,
        windows_file=args.windows_file,
        task_base_id=args.task_base_id,
        target_spacing=args.target_spacing,
        num_cores=args.num_cores,
    )
