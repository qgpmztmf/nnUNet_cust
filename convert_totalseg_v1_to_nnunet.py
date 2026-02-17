"""
Convert TotalSegmentator v1 dataset to nnUNet v2 format.

Splits are read from meta.csv (located inside input_dir):
  - train / val cases -> imagesTr / labelsTr
  - test cases        -> imagesTs / labelsTs

Note: meta.csv train+val assignments are identical to splits_final.json (verified).

Usage:
    python convert_totalseg_v1_to_nnunet.py \
        --input_dir  /m/triton/work/tianmid1/data/TotalSegmentator_v1/Totalsegmentator_dataset \
        --output_dir /m/triton/work/tianmid1/data/nnUNet_raw/Dataset601_TotalSegmentatorV1 \
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
# Robust NIfTI reader
# ---------------------------------------------------------------------------

def _nib_to_sitk(path: Path, pixel_type=None) -> sitk.Image:
    """Read a NIfTI file via nibabel, orthonormalize direction cosines, return as sitk.Image.

    SimpleITK >= 2.1 rejects NIfTI files whose direction matrix deviates even
    slightly from orthonormality (floating-point rounding in the scanner header).
    nibabel is more lenient; we use it as a fallback and fix the directions via SVD.
    """
    nib_img = nib.load(str(path))
    data = np.asarray(nib_img.dataobj)          # preserve original dtype
    affine = nib_img.affine.astype(np.float64)  # 4×4 RAS affine

    # Decompose affine: A = D_ras * diag(spacing)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)          # column norms
    direction_ras = affine[:3, :3] / spacing                   # normalize columns

    # Nearest orthonormal matrix (SVD polar decomposition)
    U, _, Vt = np.linalg.svd(direction_ras)
    direction_ras = U @ Vt

    # NIfTI/nibabel uses RAS; SimpleITK uses LPS  →  flip x and y
    flip = np.diag([-1.0, -1.0, 1.0])
    direction_lps = flip @ direction_ras
    origin_lps    = flip @ affine[:3, 3]

    # nibabel data layout: [i, j, k] = [x, y, z]
    # SimpleITK GetImageFromArray expects [z, y, x]
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


# ---------------------------------------------------------------------------
# Class map (104 TotalSegmentator classes)
# ---------------------------------------------------------------------------

CLASS_MAP_ALL: Dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "rib_left_1",
    59: "rib_left_2",
    60: "rib_left_3",
    61: "rib_left_4",
    62: "rib_left_5",
    63: "rib_left_6",
    64: "rib_left_7",
    65: "rib_left_8",
    66: "rib_left_9",
    67: "rib_left_10",
    68: "rib_left_11",
    69: "rib_left_12",
    70: "rib_right_1",
    71: "rib_right_2",
    72: "rib_right_3",
    73: "rib_right_4",
    74: "rib_right_5",
    75: "rib_right_6",
    76: "rib_right_7",
    77: "rib_right_8",
    78: "rib_right_9",
    79: "rib_right_10",
    80: "rib_right_11",
    81: "rib_right_12",
    82: "humerus_left",
    83: "humerus_right",
    84: "scapula_left",
    85: "scapula_right",
    86: "clavicula_left",
    87: "clavicula_right",
    88: "femur_left",
    89: "femur_right",
    90: "hip_left",
    91: "hip_right",
    92: "sacrum",
    93: "face",
    94: "gluteus_maximus_left",
    95: "gluteus_maximus_right",
    96: "gluteus_medius_left",
    97: "gluteus_medius_right",
    98: "gluteus_minimus_left",
    99: "gluteus_minimus_right",
    100: "autochthon_left",
    101: "autochthon_right",
    102: "iliopsoas_left",
    103: "iliopsoas_right",
    104: "urinary_bladder",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Merge per-structure masks into a single multi-label mask.

    When structures overlap, the higher label value wins (same behaviour as
    the reference implementation).
    """
    combined: Optional[sitk.Image] = None
    for label_value, label_name in class_map.items():
        mask_path = segmentations_dir / f"{label_name}.nii.gz"
        if not mask_path.exists():
            # Some cases may be missing a label; skip silently.
            continue

        mask = read_image(mask_path, sitk.sitkUInt8)
        mask = sitk.Cast(mask, sitk.sitkUInt8) * label_value

        if combined is None:
            combined = mask
            continue

        try:
            combined = sitk.Maximum(combined, mask)
        except RuntimeError:
            # Rare: different physical space across masks within one case.
            mask.CopyInformation(combined)
            combined = sitk.Maximum(combined, mask)

    if combined is None:
        raise RuntimeError(f"No masks found in {segmentations_dir}")
    return combined


def load_splits(meta_csv: Path) -> Tuple[set, set, set]:
    """Return (train_ids, val_ids, test_ids) as 4-digit string sets from meta.csv.

    meta.csv columns (semicolon-delimited): image_id;age;gender;institute;study_type;split
    split values: 'train', 'val', 'test'
    """
    train_ids, val_ids, test_ids = set(), set(), set()
    with open(meta_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            case_id = row["image_id"].strip()[-4:]  # 's0001' -> '0001'
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
    """Resample CT + merged mask and write to the appropriate nnUNet subdirectory."""

    case_id = patient_dir.name[-4:]   # 'sXXXX' -> 'XXXX'

    # test cases -> imagesTs; train + val -> imagesTr
    train_or_test = "Ts" if case_id in test_ids else "Tr"

    scan_out = output_dir / f"images{train_or_test}" / f"TotalSegmentator_{case_id}_0000.nii.gz"
    mask_out = output_dir / f"labels{train_or_test}" / f"TotalSegmentator_{case_id}.nii.gz"

    # Skip already-processed cases (allows resuming interrupted runs).
    if scan_out.exists() and mask_out.exists():
        return

    # ---- CT image ----
    scan = read_image(patient_dir / "ct.nii.gz")
    scan = resample_image(scan, target_spacing,
                          default_value=-1024,
                          interpolator=sitk.sitkLinear)

    # ---- Segmentation mask ----
    # Resample mask to the exact same grid as scan (not independently) to
    # avoid 1-voxel size mismatches from floating-point rounding.
    mask = merge_masks(patient_dir / "segmentations", class_map)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    mask = resampler.Execute(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    sitk.WriteImage(scan, str(scan_out), useCompression=True)
    sitk.WriteImage(mask, str(mask_out), useCompression=True)


# Worker initialiser for multiprocessing (avoids pickling issues with Path/dict).
_worker_args = {}

def _init_worker(output_dir, target_spacing, class_map, tr_ids, val_ids, test_ids):
    _worker_args["output_dir"]      = output_dir
    _worker_args["target_spacing"]  = target_spacing
    _worker_args["class_map"]       = class_map
    _worker_args["tr_ids"]          = tr_ids
    _worker_args["val_ids"]         = val_ids
    _worker_args["test_ids"]        = test_ids


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
    """Convert TotalSegmentator v1 dataset to nnUNet v2 raw format.

    Args:
        input_dir:      Path to Totalsegmentator_dataset/ (contains sXXXX/ dirs and meta.csv).
        output_dir:     Destination nnUNet raw dataset directory.
        target_spacing: Isotropic or anisotropic voxel spacing, e.g. [3, 3, 3].
        class_map:      Label index -> structure name mapping.
        num_cores:      Worker processes. -1 = all available CPUs.
    """

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    # Resolve worker count
    if num_cores == -1:
        num_cores = os.cpu_count() or 1

    # Load split information from meta.csv
    meta_csv = input_dir / "meta.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv not found in {input_dir}")
    tr_ids, val_ids, test_ids = load_splits(meta_csv)

    # Enumerate patient directories
    patients = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not patients:
        raise RuntimeError(f"No case directories found in {input_dir}")

    n_tr = sum(1 for p in patients if p.name[-4:] not in test_ids)
    n_ts = len(patients) - n_tr
    print(f"Found {len(patients)} cases: {n_tr} -> imagesTr, {n_ts} -> imagesTs")
    print(f"  (train={len(tr_ids)}, val={len(val_ids)}, test={len(test_ids)} in meta.csv)")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("imagesTr", "imagesTs", "labelsTr", "labelsTs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {v: k for k, v in class_map.items()} | {"background": 0},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4, sort_keys=True)
    print(f"Wrote dataset.json  (numTraining={n_tr})")

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
    print("  1. nnUNetv2_plan_and_preprocess -c 3d_fullres -d DATASET_ID --verify_dataset_integrity")
    print("  2. Copy splits_final.json into the nnUNet_preprocessed/DatasetXXX_*/ directory")
    print("  3. nnUNetv2_train DATASET_ID 3d_fullres 0")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TotalSegmentator v1 dataset to nnUNet v2 raw format."
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
