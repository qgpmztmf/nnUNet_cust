"""
Convert TotalSegmentator v1 dataset to nnUNet v1 format with 4-channel HU windowing.

Instead of a single CT channel, this script produces 4 separate NIfTI files per case,
each with a different HU window applied and values normalised to [0, 1]:

  Channel 0: Brain        (WL=40,  WW=80)    -> HU [0, 80]
  Channel 1: Lung         (WL=-600, WW=1500) -> HU [-1350, 150]
  Channel 2: Bone         (WL=300, WW=1500)  -> HU [-450, 1050]
  Channel 3: Soft Tissue  (WL=50,  WW=350)   -> HU [-125, 225]

All modalities are set to "noNorm" in dataset.json so nnUNet passes the pre-windowed
[0, 1] values through without further normalisation.

Corrupted case s0864 (CRC error in ct.nii.gz) is automatically excluded.

Usage:
    python convert_totalseg_v1_to_nnunet_multiwindow.py \
        --input_dir  /path/to/TotalSegmentator_v1/Totalsegmentator_dataset \
        --output_dir /path/to/nnUNet_raw/nnUNet_raw_data/Task604_TotalSegmentatorV1 \
        --target_spacing 3.0 3.0 3.0 \
        --num_cores 8
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
# HU windows: (name, window_level, window_width)
# lower = WL - WW/2,  upper = WL + WW/2
# ---------------------------------------------------------------------------

HU_WINDOWS = [
    ("brain",       40,   80),    # HU [0, 80]
    ("lung",       -600, 1500),   # HU [-1350, 150]
    ("bone",        300, 1500),   # HU [-450, 1050]
    ("soft_tissue",  50,  350),   # HU [-125, 225]
]

# Cases to exclude (known data corruption)
EXCLUDE_CASES = {"0864"}


# ---------------------------------------------------------------------------
# Robust NIfTI reader
# ---------------------------------------------------------------------------

def _nib_to_sitk(path: Path, pixel_type=None) -> sitk.Image:
    """Read a NIfTI file via nibabel, orthonormalize direction cosines, return as sitk.Image."""
    nib_img = nib.load(str(path))
    data = np.asarray(nib_img.dataobj)
    affine = nib_img.affine.astype(np.float64)

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction_ras = affine[:3, :3] / spacing

    U, _, Vt = np.linalg.svd(direction_ras)
    direction_ras = U @ Vt

    flip = np.diag([-1.0, -1.0, 1.0])
    direction_lps = flip @ direction_ras
    origin_lps    = flip @ affine[:3, 3]

    sitk_img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    sitk_img.SetSpacing(spacing.tolist())
    sitk_img.SetDirection(direction_lps.flatten().tolist())
    sitk_img.SetOrigin(origin_lps.tolist())

    if pixel_type is not None:
        sitk_img = sitk.Cast(sitk_img, pixel_type)
    return sitk_img


def read_image(path: Path, pixel_type=None) -> sitk.Image:
    """Read a NIfTI image, falling back to nibabel if direction cosines are non-orthonormal."""
    try:
        if pixel_type is not None:
            return sitk.ReadImage(str(path), pixel_type)
        return sitk.ReadImage(str(path))
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            raise
        return _nib_to_sitk(path, pixel_type)


def validate_with_nibabel(path: Path) -> bool:
    """Check if nibabel can read the file without CRC or other errors."""
    try:
        img = nib.load(str(path))
        # Force reading a small chunk to trigger CRC check
        data = np.asarray(img.dataobj)
        _ = data.flat[0]
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Class map (104 TotalSegmentator classes)
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
# Helpers
# ---------------------------------------------------------------------------

def apply_hu_window(image_array: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Apply an HU window and normalise to [0, 1] as float32.

    Formula: clamp((I - lower) / WW, 0, 1)  where lower = WL - WW/2
    """
    lower = wl - ww / 2.0
    out = (image_array - lower) / ww
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32)


def resample_image(image: sitk.Image,
                   new_spacing: List[float],
                   default_value: float,
                   interpolator) -> sitk.Image:
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [int(round(sz * sp / nsp))
                for sz, sp, nsp in zip(size, spacing, new_spacing)]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        default_value,
        image.GetPixelID(),
    )


def merge_masks(segmentations_dir: Path, class_map: Dict[int, str]) -> sitk.Image:
    """Merge per-structure masks into a single multi-label mask."""
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
    """Return (train_ids, val_ids, test_ids) as 4-digit string sets from meta.csv."""
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
# Per-patient processing
# ---------------------------------------------------------------------------

def process_patient(patient_dir: Path,
                    output_dir: Path,
                    target_spacing: List[float],
                    class_map: Dict[int, str],
                    tr_ids: set,
                    val_ids: set,
                    test_ids: set) -> None:
    """Resample CT, apply 4 HU windows, and write to the appropriate nnUNet subdirectory."""

    case_id = patient_dir.name[-4:]  # 'sXXXX' -> 'XXXX'

    # Skip excluded cases
    if case_id in EXCLUDE_CASES:
        return

    train_or_test = "Ts" if case_id in test_ids else "Tr"

    # Check if all output files already exist (for resuming)
    channel_outs = [
        output_dir / f"images{train_or_test}" / f"TotalSegmentator_{case_id}_{ch:04d}.nii.gz"
        for ch in range(len(HU_WINDOWS))
    ]
    mask_out = output_dir / f"labels{train_or_test}" / f"TotalSegmentator_{case_id}.nii.gz"

    if all(p.exists() for p in channel_outs) and mask_out.exists():
        return

    # Validate with nibabel first to catch CRC errors
    ct_path = patient_dir / "ct.nii.gz"
    if not validate_with_nibabel(ct_path):
        print(f"\n[SKIP] {patient_dir.name}: nibabel CRC/read error", file=sys.stderr)
        return

    # ---- CT image ----
    scan = read_image(ct_path)
    scan = resample_image(scan, target_spacing,
                          default_value=-1024,
                          interpolator=sitk.sitkLinear)

    # Get the numpy array from the resampled scan
    scan_array = sitk.GetArrayFromImage(scan).astype(np.float64)

    # ---- Write windowed channels ----
    for ch_idx, (win_name, wl, ww) in enumerate(HU_WINDOWS):
        windowed = apply_hu_window(scan_array, wl, ww)

        # Create sitk image preserving spatial metadata from the resampled scan
        windowed_img = sitk.GetImageFromArray(windowed)
        windowed_img.CopyInformation(scan)

        sitk.WriteImage(windowed_img, str(channel_outs[ch_idx]), useCompression=True)

    # ---- Segmentation mask ----
    mask = merge_masks(patient_dir / "segmentations", class_map)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    mask = resampler.Execute(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    sitk.WriteImage(mask, str(mask_out), useCompression=True)


# Worker initialiser for multiprocessing
_worker_args = {}

def _init_worker(output_dir, target_spacing, class_map, tr_ids, val_ids, test_ids):
    _worker_args["output_dir"]     = output_dir
    _worker_args["target_spacing"] = target_spacing
    _worker_args["class_map"]      = class_map
    _worker_args["tr_ids"]         = tr_ids
    _worker_args["val_ids"]        = val_ids
    _worker_args["test_ids"]       = test_ids


def _process_patient_worker(patient_dir: Path) -> None:
    try:
        process_patient(patient_dir, **_worker_args)
    except Exception as exc:
        print(f"\n[ERROR] {patient_dir.name}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_dataset(input_dir: str,
                   output_dir: str,
                   target_spacing: List[float],
                   class_map: Dict[int, str] = CLASS_MAP_ALL,
                   num_cores: int = -1) -> None:
    """Convert TotalSegmentator v1 dataset to nnUNet v1 raw format with 4-channel HU windows."""

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if num_cores == -1:
        num_cores = os.cpu_count() or 1

    # Load split information from meta.csv
    meta_csv = input_dir / "meta.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv not found in {input_dir}")
    tr_ids, val_ids, test_ids = load_splits(meta_csv)

    # Enumerate patient directories (exclude known bad cases)
    patients = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name[-4:] not in EXCLUDE_CASES
    ])
    if not patients:
        raise RuntimeError(f"No case directories found in {input_dir}")

    n_tr = sum(1 for p in patients if p.name[-4:] not in test_ids)
    n_ts = len(patients) - n_tr
    print(f"Found {len(patients)} cases (excluding {len(EXCLUDE_CASES)} corrupted): "
          f"{n_tr} -> imagesTr, {n_ts} -> imagesTs")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("imagesTr", "imagesTs", "labelsTr", "labelsTs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Print window configuration
    print("\nHU Windows:")
    for ch_idx, (name, wl, ww) in enumerate(HU_WINDOWS):
        lower = wl - ww / 2
        upper = wl + ww / 2
        print(f"  Channel {ch_idx} ({name:12s}): WL={wl:5d}, WW={ww:5d}  ->  HU [{lower:.0f}, {upper:.0f}]")

    # Write dataset.json (nnUNet v1 format)
    labels_dict = {"background": 0}
    labels_dict.update({v: k for k, v in class_map.items()})

    dataset_json = {
        "name": "Task604_TotalSegmentatorV1",
        "description": "TotalSegmentator v1 with 4-channel HU windowing (brain/lung/bone/soft_tissue)",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": {
            "0": "noNorm",
            "1": "noNorm",
            "2": "noNorm",
            "3": "noNorm",
        },
        "labels": labels_dict,
        "numTraining": n_tr,
        "numTest": n_ts,
        "file_ending": ".nii.gz",
        "training": [
            {
                "image": f"./imagesTr/TotalSegmentator_{p.name[-4:]}",
                "label": f"./labelsTr/TotalSegmentator_{p.name[-4:]}.nii.gz",
            }
            for p in patients if p.name[-4:] not in test_ids
        ],
        "test": [
            f"./imagesTs/TotalSegmentator_{p.name[-4:]}"
            for p in patients if p.name[-4:] in test_ids
        ],
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"\nWrote dataset.json  (numTraining={n_tr}, modality=noNorm x4)")

    # Process cases
    if num_cores > 1:
        print(f"Processing with {num_cores} parallel workers ...")
        with multiprocessing.Pool(
            processes=num_cores,
            initializer=_init_worker,
            initargs=(output_dir, target_spacing, class_map, tr_ids, val_ids, test_ids),
        ) as pool:
            list(tqdm(
                pool.imap_unordered(_process_patient_worker, patients),
                total=len(patients),
                desc="Converting",
            ))
    else:
        print("Processing in single-process mode ...")
        for patient in tqdm(patients, desc="Converting"):
            process_patient(patient, output_dir, target_spacing, class_map, tr_ids, val_ids, test_ids)

    print(f"\nDone. Output written to: {output_dir}")
    print("Next steps:")
    print("  1. nnUNet_plan_and_preprocess -t 604")
    print("  2. Check dataset_properties.pkl shows noNorm for all 4 modalities")
    print("  3. nnUNet_train 3d_fullres nnUNetTrainerV2_ep2000_nomirror 604 FOLD")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TotalSegmentator v1 to nnUNet format with 4-channel HU windowing."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to Totalsegmentator_dataset/ directory (contains sXXXX/ subdirs).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Destination nnUNet raw dataset directory (will be created if absent).",
    )
    parser.add_argument(
        "--target_spacing", nargs=3, type=float, default=[3.0, 3.0, 3.0],
        metavar=("X", "Y", "Z"),
        help="Voxel spacing to resample to (mm). Default: 3 3 3.",
    )
    parser.add_argument(
        "--num_cores", type=int, default=-1,
        help="Number of worker processes. -1 = all CPUs (default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_spacing=args.target_spacing,
        class_map=CLASS_MAP_ALL,
        num_cores=args.num_cores,
    )
